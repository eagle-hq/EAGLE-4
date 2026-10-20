# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Cost tables from whole-step timings (an engine without an attention-only timing, e.g. vLLM)."""

from __future__ import annotations

import pytest

from eagle_harness.core.calibration import (
    AttentionShape,
    CalibrationPaths,
    MeasureParams,
    StepTiming,
    attention_growth_records,
    fit_step_timings,
)
from eagle_harness.core.cost_table import load_cost_tables
from eagle_harness.core.limits import ComputeIntensity, draft_query_lens, verify_buckets

LAYERS = 16
STEP, STOP = 256, 4096


def timings(widths, fixed_us, per_token_us, per_layer_per_key_ns):
    """Step time = fixed + per-token + attention growing linearly with the cache length (per layer, per query token)."""
    lengths = [STEP * (1 + i) + off for i in range((STOP - STEP) // STEP) for off in (STEP // 4, 3 * STEP // 4)]
    return [
        StepTiming(q, length, fixed_us + per_token_us * q + LAYERS * per_layer_per_key_ns * length * q / 1000)
        for q in widths
        for length in lengths
    ]


def test_cache_cost_is_the_growth_over_the_shortest_cache_per_layer():
    rows = [StepTiming(4, 100, 500.0), StepTiming(4, 300, 580.0), StepTiming(4, 500, 660.0)]
    records = attention_growth_records(rows, attention_layers=8, category="cache_llm")
    per_layer_us = {r.cache_length: r.time_us / r.measure_count for r in records}
    assert per_layer_us[300] == pytest.approx(10.0, abs=0.01)
    assert per_layer_us[500] == pytest.approx(20.0, abs=0.01)
    assert per_layer_us[100] == pytest.approx(0.001, abs=0.001)  # floor of one record unit, never zero


def test_fits_write_tables_the_cost_model_loads(tmp_path):
    paths = CalibrationPaths(str(tmp_path))
    params = MeasureParams(
        context_step=STEP, context_stop=STOP, batch_size=1, verify_num=64, node_top_k=8, loop_top_m=64, repeat_cnt=1
    )
    draft = (AttentionShape(1, 1, 4096), timings(draft_query_lens(1, ComputeIntensity.HIGH), 300.0, 15.0, 2.0))
    target = (AttentionShape(LAYERS), timings(sorted(verify_buckets(1, ComputeIntensity.HIGH)), 4500.0, 20.0, 1.0))
    fit_step_timings(paths, params, ComputeIntensity.HIGH, draft, target, tp_size=4)
    table = load_cost_tables(paths.cost)[1]
    short, long = table.interval_index(300), table.interval_index(3900)
    verify = table.intervals[long].target
    assert verify.value[verify.index.index(63)] > verify.value[verify.index.index(0)]  # a wider verify costs more
    assert table.intervals[long].target.value[0] > table.intervals[short].target.value[0]  # a longer cache costs more
    assert (
        table.intervals[long].draft.value[0] < table.intervals[long].target.value[0]
    )  # a draft step is cheaper than a verify


@pytest.mark.parametrize(
    "intensity,verify_width,draft_width",
    [(ComputeIntensity.HIGH, 16, 4), (ComputeIntensity.LOW, 32, 8), (ComputeIntensity.CHAIN, 8, 1)],
)
def test_fits_respect_measured_width_limits(tmp_path, intensity, verify_width, draft_width):
    paths = CalibrationPaths(str(tmp_path))
    params = MeasureParams(
        context_step=STEP,
        context_stop=STOP,
        batch_size=1,
        verify_num=verify_width,
        node_top_k=draft_width,
        loop_top_m=verify_width,
        repeat_cnt=1,
    )
    draft_widths = [q for q in draft_query_lens(1, intensity) if q <= draft_width]
    target_widths = sorted(q for q in verify_buckets(1, intensity) if q <= verify_width)
    fit_step_timings(
        paths,
        params,
        intensity,
        (AttentionShape(1), timings(draft_widths, 300.0, 15.0, 2.0)),
        (AttentionShape(LAYERS), timings(target_widths, 4500.0, 20.0, 1.0)),
    )
    table = load_cost_tables(paths.cost)[1]
    for interval in table.intervals:
        assert interval.draft.index == tuple(q - 1 for q in draft_widths)
        assert interval.target.index == tuple(q - 1 for q in target_widths)
        assert all(value > 0 for value in (*interval.draft.value, *interval.target.value))


def test_chain_fits_price_a_chain_width_per_batch_size(tmp_path):
    """Chain tables at a weight-bound and a compute-bound batch size, fitted from whole-step timings (what the vLLM
    driver measures), give the widest chain at batch size 1 and a narrower one at 512."""
    from eagle_harness.core.chain_width import AcceptProfile, chain_width_schedule

    paths = CalibrationPaths(str(tmp_path))
    for bs, per_token_us in ((1, 2.0), (512, 900.0)):  # per request-token: weight reads dominate at 1, compute at 512
        params = MeasureParams(
            context_step=STEP,
            context_stop=STOP,
            batch_size=bs,
            verify_num=16,
            node_top_k=1,
            loop_top_m=16,
            repeat_cnt=1,
        )
        draft = (
            AttentionShape(1, 1, 4096),
            timings(draft_query_lens(bs, ComputeIntensity.CHAIN), 150.0, 0.1 * bs, 0.5),
        )
        widths = sorted(verify_buckets(bs, ComputeIntensity.CHAIN))
        target = (AttentionShape(LAYERS), timings(widths, 4500.0, per_token_us, 1.0))
        fit_step_timings(paths, params, ComputeIntensity.CHAIN, draft, target, tp_size=4)
    tables = load_cost_tables(paths.cost)
    profile = AcceptProfile.from_counts(1000, [900 - 50 * d for d in range(15)])
    schedule = chain_width_schedule(tables, profile, 16, 512)
    assert schedule[0][:2] == (1, 256) and schedule[0][2] >= 8
    assert schedule[-1][1] == 512 and schedule[-1][2] < schedule[0][2]
