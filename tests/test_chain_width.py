# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Chain width per batch size from cost tables and an acceptance profile (:mod:`eagle_harness.core.chain_width`)."""

from __future__ import annotations

import json
import os

import pytest

from eagle_harness.core.chain_width import (
    AcceptProfile,
    chain_width,
    chain_width_schedule,
    chain_widths,
    load_accept_profile,
    load_round_times,
    save_accept_profile,
    save_round_times,
    served_chain_width,
    served_width_schedule,
    width_at,
)
from eagle_harness.core.cost_table import BatchCostTable, CostCurve, IntervalCost, load_cost_tables

WIDTHS = (1, 2, 4, 8, 16)


def _curve(index, value):
    return CostCurve(
        value=tuple(value), ratio=tuple(v / value[0] for v in value), index=tuple(index), array_query=(), array_cache=()
    )


def _table(bs: int, verify_us, draft_us: float | None, widths=WIDTHS) -> BatchCostTable:
    """Two context intervals with the same costs: ``verify_us(width)`` for the target, ``draft_us`` per draft step."""
    target = _curve([w - 1 for w in widths], [verify_us(w) for w in widths])
    draft = _curve([0], [draft_us]) if draft_us is not None else _curve([1], [1.0])
    rows = tuple(IntervalCost(target=target, draft=draft) for _ in range(2))
    return BatchCostTable(bs, 256, 256, 16, 16, 1, rows, tuple(2.0 for _ in rows))


PROFILE = AcceptProfile.from_counts(1000, [900 - 50 * d for d in range(15)])  # a_d = 0.90, 0.85, ..., 0.20


def test_profile_counts_and_expected_tokens():
    assert PROFILE.rates[:2] == (0.9, 0.85)
    assert PROFILE.expected_tokens(1) == 1.0
    assert PROFILE.expected_tokens(3) == pytest.approx(1.0 + 0.9 + 0.85)
    with pytest.raises(ValueError):
        PROFILE.expected_tokens(17)  # 15 measured positions cover widths 1..16
    with pytest.raises(ValueError):
        AcceptProfile.from_counts(10, [5, 7])  # a later position cannot accept more often


def test_weight_bound_batch_takes_the_widest_chain():
    # Verify cost flat in the width (weight reads), cheap draft steps: every drafted token pays.
    assert chain_width(_table(1, lambda w: 5000.0 + w, 20.0), PROFILE, 16) == 16


def test_compute_bound_batch_falls_back_to_plain_decoding():
    # Verify cost proportional to the tokens verified: a drafted token costs a full token and is accepted < 1.
    assert chain_width(_table(512, lambda w: 3000.0 * w, 300.0), PROFILE, 16) == 1


def test_width_balances_draft_cost_and_acceptance():
    # 5000 us verify, 600 us per draft step; tokens / ms: width 1 0.20, 2 0.34, 4 0.52, 8 0.68 (6.25 / 9.2), 16 0.66.
    assert chain_width(_table(64, lambda w: 5000.0, 600.0), PROFILE, 16) == 8


def test_only_timed_and_profiled_widths_are_candidates():
    short = AcceptProfile.from_counts(10, [9, 8, 7])  # widths 1..4
    assert chain_width(_table(1, lambda w: 5000.0, 1.0), short, 16) == 4
    assert chain_width(_table(1, lambda w: 5000.0, 1.0), PROFILE, 8) == 8  # max_width caps
    assert chain_width(_table(1, lambda w: 5000.0, None), PROFILE, 16) is None  # no draft step timed


def test_schedule_uses_the_nearest_calibrated_batch_size():
    tables = {1: _table(1, lambda w: 5000.0 + w, 20.0), 512: _table(512, lambda w: 3000.0 * w, 300.0)}
    assert chain_width_schedule(tables, PROFILE, 16, 512) == [(1, 256, 16), (257, 512, 1)]
    assert chain_width_schedule({}, PROFILE, 16, 512) is None


def test_profile_and_tables_round_trip(tmp_path):
    save_accept_profile(str(tmp_path), 1000, [900 - 50 * d for d in range(15)])
    assert load_accept_profile(str(tmp_path)) == PROFILE
    assert load_accept_profile(str(tmp_path / "missing")) is None
    write_tables(str(tmp_path), {1: _table(1, lambda w: 5000.0 + w, 20.0)})
    loaded = load_cost_tables(str(tmp_path / "cost"))
    assert chain_width(loaded[1], PROFILE, 16) == 16


def write_tables(data_path: str, tables) -> None:
    """Write tables in the on-disk layout :func:`load_cost_tables` reads (``cost/bs_<N>/model_{llm,slm}/``)."""
    for bs, table in tables.items():
        for model, pick in (("llm", lambda i: i.target), ("slm", lambda i: i.draft)):
            folder = os.path.join(data_path, "cost", f"bs_{bs}", f"model_{model}")
            os.makedirs(folder, exist_ok=True)
            for n, interval in enumerate(table.intervals):
                curve = pick(interval)
                data = {
                    "value": list(curve.value),
                    "ratio": list(curve.ratio),
                    "index": list(curve.index),
                    "array_query": [],
                    "array_cache": [],
                    "cache_length_start": table.cache_length_start,
                    "cache_length_step": table.cache_length_step,
                    "max_query_length_query": 16,
                    "max_query_length_cache": 16,
                }
                with open(os.path.join(folder, f"model_interval_{n}.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f)


# ------------------------------------------------------------------ served round times
def test_served_width_takes_the_most_tokens_per_microsecond():
    # tokens / ms: width 1 0.100 (10 ms), 2 0.158 (1.9 / 12 ms), 4 0.203 (3.65 / 18 ms), 8 0.196 (6.25 / 32 ms).
    assert served_chain_width({1: 10000.0, 2: 12000.0, 4: 18000.0, 8: 32000.0}, PROFILE, 8) == 4
    assert served_chain_width({1: 10000.0, 2: 12000.0, 4: 18000.0, 8: 32000.0}, PROFILE, 2) == 2  # max_width caps


def test_served_width_stops_speculating_when_every_chain_loses():
    # A compute-bound batch: each drafted token costs more than it gains, so plain decoding wins.
    assert served_chain_width({1: 10000.0, 2: 21000.0, 4: 40000.0}, PROFILE, 8) == 1


def test_served_width_needs_the_plain_decoding_round():
    assert served_chain_width({2: 12000.0, 4: 18000.0}, PROFILE, 8) is None
    assert served_width_schedule({1: {1: 10.0, 2: 9.0}, 64: {2: 9.0}}, PROFILE, 8, 64) is None


def test_served_schedule_ranges_follow_the_nearest_measured_batch_size():
    rounds = {1: {1: 10000.0, 4: 12000.0}, 64: {1: 20000.0, 4: 30000.0}, 256: {1: 40000.0, 4: 160000.0}}
    schedule = served_width_schedule(rounds, PROFILE, 8, 256)
    assert schedule == [(1, 159, 4), (160, 256, 1)]  # 160 is a tie between 64 and 256: the larger wins
    assert width_at(schedule, 1) == 4 and width_at(schedule, 200) == 1
    with pytest.raises(ValueError):
        width_at(schedule, 257)


def test_round_times_round_trip_and_win_over_the_tables(tmp_path):
    data = str(tmp_path)
    assert load_round_times(data) is None
    write_tables(str(tmp_path), {1: _table(1, lambda w: 5000.0 + w, 20.0)})  # the tables alone would pick width 8
    save_accept_profile(data, 1000, [900 - 50 * d for d in range(15)])
    assert chain_widths(data, 8, 4) == [(1, 4, 8)]
    save_round_times(
        data,
        [
            {"batch_size": 1, "width": 1, "round_us": 1000.0, "count": 9},
            {"batch_size": 1, "width": 8, "round_us": 9000.0, "count": 9},
        ],
    )
    assert load_round_times(data) == {1: {1: 1000.0, 8: 9000.0}}
    assert chain_widths(data, 8, 4) == [(1, 4, 1)]  # the served rounds price width 8 at a loss
    with pytest.raises(ValueError):
        save_round_times(data, [{"batch_size": 1, "width": 1, "round_us": 0.0, "count": 1}])


def test_chain_widths_need_the_acceptance_profile(tmp_path):
    save_round_times(str(tmp_path), [{"batch_size": 1, "width": 1, "round_us": 1000.0, "count": 9}])
    assert chain_widths(str(tmp_path), 8, 4) is None
