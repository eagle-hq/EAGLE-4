# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""TokenSpeed calibrate mode: the measured widths, round slots and batch lifecycle, and the driver's records through
the core fits into tables serve mode accepts.

Needs TokenSpeed importable (its kernel package refuses a host without a GPU), so it runs in the GPU test jobs; the
measurement lifecycle and attention sweeps also need CUDA.
"""

from __future__ import annotations

import argparse
import os

import pytest
import torch

try:
    import tokenspeed.runtime.execution.drafter.tree_round.device_policy  # noqa: F401
except (ImportError, RuntimeError) as exc:  # RuntimeError: tokenspeed-kernel on a host without a GPU
    pytest.skip(f"TokenSpeed is not importable here: {exc}", allow_module_level=True)

from eagle_harness.core.calibration import (  # noqa: E402
    AttentionShape,
    CacheSweepPlan,
    CalibrationPaths,
    MeasureRecord,
    fit_batch,
    load_jsonl,
    query_lens,
    save_jsonl,
    save_model_timings,
)
from eagle_harness.core.config import build_cost_model  # noqa: E402
from eagle_harness.core.cost_table import cost_model_problem, load_cost_tables  # noqa: E402
from eagle_harness.core.limits import ComputeIntensity, cost_batch_sizes  # noqa: E402
from eagle_harness.tokenspeed.calibrate import attention_records, lane_mask, measure_params  # noqa: E402
from eagle_harness.tokenspeed.config import parse_algorithm_config  # noqa: E402
from eagle_harness.tokenspeed.measurement import (  # noqa: E402
    FORCED_TOKEN,
    MEASURED_ROUNDS,
    CostMeasurement,
    MeasurePolicy,
)

K, DEPTH, N = 4, 12, 16
TREES = (ComputeIntensity.HIGH, ComputeIntensity.LOW)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA events and kernels")


def _policy(intensity=ComputeIntensity.HIGH) -> MeasurePolicy:
    return MeasurePolicy(verify_num=N, node_top_k=K, tree_depth=DEPTH, intensity=intensity, device=torch.device("cpu"))


def _args(**overrides) -> argparse.Namespace:
    values = dict(context_step=64, context_stop=8192, num_draft_tokens=N, topk=K)
    return argparse.Namespace(**{**values, **overrides})


@pytest.mark.parametrize("intensity", TREES, ids=lambda i: i.value)
@pytest.mark.parametrize("batch_size", cost_batch_sizes(32))
def test_measured_widths_are_the_widths_the_fits_read(intensity, batch_size):
    policy = _policy(intensity)
    drafts, verifies = set(), set()
    for _ in range(64):
        policy.next_widths(batch_size, N)
        drafts.add(policy.draft_width)
        verifies.add(policy.verify_width)
    assert drafts == set(query_lens("model_slm", batch_size, intensity, K))
    assert verifies == set(query_lens("model_llm", batch_size, intensity, N))


def test_a_measured_round_drafts_every_depth_at_its_widths():
    policy = _policy()
    policy.next_widths(1, N)
    policy.next_widths(1, N)
    policy.write_round()
    widths = policy.widths.tolist()
    # Depth 1 at the previous round's width (decided before this round's), the rest at this round's.
    assert widths[1] == policy.previous_draft_width
    assert widths[2:DEPTH] == [policy.draft_width] * (DEPTH - 2)
    pool = 1 + K + K * sum(widths[1:DEPTH])
    assert policy.cumu_starts[1].item() == 1 + K
    assert policy.past_verify_num.item() == N
    assert policy.cumu_count.item() == pool
    assert policy.verify_count.item() == min(N, pool)
    assert policy.verify_num.item() == min(policy.verify_width, N, pool)
    assert [policy.go_ready(d, 1) for d in (1, DEPTH - 1, DEPTH)] == [True, True, False]
    assert not policy.deltas_depth(2) and policy.table_width(8) == 0


def test_the_capture_round_drafts_every_depth_at_full_width():
    policy = _policy()
    policy.write_full_round()
    assert policy.widths.tolist()[1:DEPTH] == [K] * (DEPTH - 1)
    assert policy.verify_num.item() == N


def test_chains_are_refused():
    with pytest.raises(NotImplementedError, match="measures trees"):
        _policy(ComputeIntensity.CHAIN)


def test_lane_masks():
    # A verify chain: row r sees window keys 0..r.
    assert lane_mask(3, 16, causal=True) == [0b1, 0b11, 0b111]
    # Two depths of four lanes: lane r sees its own slot at each depth.
    assert lane_mask(4, 8, causal=False) == [0b10001, 0b100010, 0b1000100, 0b10001000]
    # 64 lanes over two depths: lane r sees keys r (word 0) and 64 + r (word 1); key 63 is word 0's sign bit.
    words = lane_mask(64, 128, causal=False)
    assert words[:2] == [1, 1] and words[126:] == [-(1 << 63), -(1 << 63)]
    assert lane_mask(64, 64, causal=True)[63] == -1


def _synthetic_batch(paths: CalibrationPaths, batch_size: int, intensity: ComputeIntensity, window: int) -> None:
    """Records shaped as calibrate mode and the driver write them: stage samples over the measured contexts, one
    attention timing per length shared by every width that runs its rows."""
    params = measure_params(_args(), batch_size)
    plan = CacheSweepPlan.from_params(params)
    drafts = query_lens("model_slm", batch_size, intensity, K)
    verifies = query_lens("model_llm", batch_size, intensity, N)
    records = []
    for i in range(MEASURED_ROUNDS):
        cache_len, draft, verify = 418 + i, drafts[i % len(drafts)], verifies[i % len(verifies)]
        records.append(MeasureRecord(110 * (DEPTH - 1) + i % 5, DEPTH - 1, draft, cache_len, "model_slm", batch_size))
        records.append(MeasureRecord(3600 + 40 * batch_size + i % 7, 1, verify, cache_len, "model_llm", batch_size))
    save_model_timings(paths, params, records)
    for model, rows_cost in (("llm", 15.0), ("slm", 12.0)):
        widths = query_lens(f"cache_{model}", batch_size, intensity, N if model == "llm" else K)
        timings = [
            (kv, int(plan.num_layers * (rows_cost + 0.02 * kv) + j % 3))
            for j, kv in enumerate(plan.cache_lengths())
            if kv >= plan.kv_step
        ]
        for width in widths:
            save_jsonl(
                os.path.join(paths.raw(batch_size, "cache", model), f"cache_qo_len_{width}.jsonl"),
                [MeasureRecord(t, plan.num_layers, width, kv, f"cache_{model}", batch_size) for kv, t in timings],
            )
    fit_batch(paths, params, intensity, AttentionShape(1, 1, window), AttentionShape(16), tp_size=4)


@pytest.mark.parametrize("intensity", TREES, ids=lambda i: i.value)
def test_measured_records_fit_into_tables_serve_mode_accepts(tmp_path, intensity):
    paths = CalibrationPaths(str(tmp_path))
    for batch_size in (1, 2, 4):
        _synthetic_batch(paths, batch_size, intensity, window=2048)
    tables = load_cost_tables(paths.cost)
    assert sorted(tables) == [1, 2, 4]
    for table in tables.values():
        assert cost_model_problem(table) is None
        assert table.cache_length_step == 64 and table.max_interval_id == 8192 // 64 - 2
    config = parse_algorithm_config({"mode": "serve", "intensity": intensity.value, "data_path": str(tmp_path)})
    model = build_cost_model(config, node_top_k=K, tree_depth=DEPTH, verify_num=N, target_is_moe=False)
    assert sorted(model.tables) == [1, 2, 4]


@needs_cuda
def test_a_measured_batch_records_its_rounds_then_ends_with_an_eos(tmp_path):
    paths = CalibrationPaths(str(tmp_path))
    paths.write_measure_params(measure_params(_args(), 2))
    measurement = CostMeasurement(
        data_path=str(tmp_path), tree_depth=DEPTH, tp_group=None, rank=0, eos_token=7, device=torch.device("cuda")
    )
    measurement.on_extend()
    assert measurement.params.batch_size == 2 and measurement.commit_token.item() == FORCED_TOKEN
    for i in range(MEASURED_ROUNDS):
        assert not measurement.complete()
        measurement.begin_draft()
        torch.cuda._sleep(20_000)
        measurement.end_draft(2, draft_width=2, verify_width=16 if i % 2 else 8, cache_len=400 + i)
        measurement.verify_start.record()
        torch.cuda._sleep(20_000)
        measurement.verify_end.record()
        measurement.record_verify(2)
    assert measurement.commit_token.item() == 7
    assert measurement.complete()
    test_dir = measurement.write_model_records()
    assert not measurement.complete()
    draft = load_jsonl(os.path.join(test_dir, "model_slm", "model_qo_len_2.jsonl"))
    verify8 = load_jsonl(os.path.join(test_dir, "model_llm", "model_qo_len_8.jsonl"))
    verify16 = load_jsonl(os.path.join(test_dir, "model_llm", "model_qo_len_16.jsonl"))
    assert len(draft) == MEASURED_ROUNDS and len(verify8) + len(verify16) == MEASURED_ROUNDS
    assert all(r["measure_count"] == DEPTH - 1 and r["time_intervals"] > 0 for r in draft)
    assert [r["cache_length"] for r in verify8] == list(range(400, 400 + MEASURED_ROUNDS, 2))


@needs_cuda
def test_rounds_of_another_batch_size_are_not_recorded(tmp_path):
    paths = CalibrationPaths(str(tmp_path))
    paths.write_measure_params(measure_params(_args(), 4))
    measurement = CostMeasurement(
        data_path=str(tmp_path), tree_depth=DEPTH, tp_group=None, rank=0, eos_token=7, device=torch.device("cuda")
    )
    measurement.on_extend()
    measurement.begin_draft()
    measurement.end_draft(2, draft_width=2, verify_width=16, cache_len=400)
    measurement.record_verify(2)
    assert measurement.records == [] and measurement.verify_rounds == 0


@needs_cuda
@pytest.mark.parametrize(
    "rows, window, sliding", [(N, N, 0), (1, 0, 0), (K, (DEPTH - 1) * K, 2048)], ids=["verify", "one-row", "lanes"]
)
def test_attention_records_time_every_length_for_every_width(rows, window, sliding):
    plan = CacheSweepPlan.from_params(measure_params(_args(context_step=256, context_stop=4096), 2))
    records = attention_records(
        batch_size=2, rows=rows, window=window, num_q_heads=6, num_kv_heads=1, head_dim=256, sliding_window=sliding,
        plan=plan, widths=[1, 8], category="cache_llm",
    )  # fmt: skip
    lengths = [kv for kv in plan.cache_lengths() if kv >= plan.kv_step]
    assert [(r.query_length, r.cache_length) for r in records] == [(w, kv) for kv in lengths for w in (1, 8)]
    assert all(r.measure_count == plan.num_layers and r.time_us > 0 for r in records)
    # Widths that run the same rows share one timing per length.
    assert all(a.time_us == b.time_us for a, b in zip(records[::2], records[1::2]))
