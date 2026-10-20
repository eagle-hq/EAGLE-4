# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Chain calibration's round recorder (:mod:`eagle_harness.sglang.chain_rounds`) and the driver's fits, on CPU."""

from __future__ import annotations

import json

import pytest
import torch

from eagle_harness.core.chain_width import load_accept_profile, load_round_times
from eagle_harness.sglang import calibrate as sglang_calibrate
from eagle_harness.sglang import chain_rounds
from eagle_harness.sglang.chain_rounds import BLOCK_ROUNDS, MIN_ROUNDS, RAW_FILE, ChainCalibration, accept_counts


def _calibration(tmp_path, widths=(1, 2, 4)):
    return ChainCalibration(str(tmp_path), widths, torch.device("cpu"), writer=True)


def test_widths_cycle_in_blocks():
    cal = ChainCalibration("/unused", (4, 1, 2), torch.device("cpu"), writer=False)
    plan = [cal.width(1) for _ in range(3 * BLOCK_ROUNDS + 1)]
    assert plan == [1] * BLOCK_ROUNDS + [2] * BLOCK_ROUNDS + [4] * BLOCK_ROUNDS + [1]


def test_only_rounds_between_two_like_rounds_are_recorded(tmp_path, monkeypatch):
    clock = iter(range(0, 10**9, 1000))  # every round_start 1 us after the previous one
    monkeypatch.setattr(chain_rounds.time, "perf_counter_ns", lambda: next(clock))
    cal = _calibration(tmp_path)
    for width in (1, 1, 1, 1, 2, 2, 2):
        cal.round_start(8, width, 100)
    # Width 1: rounds 2 and 3 are recorded (round 4 ends a block); width 2: round 6 needs round 7's start.
    assert cal.rounds == [(8, 1, 1.0, 100), (8, 1, 1.0, 100), (8, 2, 1.0, 100)]
    cal.interrupt()  # a prefill: the period in progress is not a decode round's
    cal.round_start(8, 2, 100)
    cal.round_start(8, 2, 100)
    assert len(cal.rounds) == 3
    cal.round_start(4, 2, 100)  # another batch size
    cal.round_start(4, 2, 100)
    assert len(cal.rounds) == 3


def test_interrupt_writes_the_rounds_and_the_widest_accept_lengths(tmp_path):
    cal = _calibration(tmp_path)
    cal.accepted(4, torch.tensor([1, 4, 4, 2], dtype=torch.int32))
    cal.accepted(2, torch.tensor([2, 2]))  # not the widest: not counted
    for _ in range(4):
        cal.round_start(1, 4, 50)
    cal.interrupt()
    with open(tmp_path / RAW_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["accept_lens"] == [0, 1, 1, 0, 2] and raw["max_width"] == 4 and len(raw["rounds"]) == 2


def test_flush_writes_the_round_times_of_complete_batch_sizes_and_the_profile(tmp_path):
    cal = _calibration(tmp_path, widths=(1, 2, 4))
    cal.rounds = [(1, w, us, 300) for w, us in ((1, 10000.0), (2, 12000.0), (4, 15000.0)) for _ in range(MIN_ROUNDS)]
    cal.rounds += [(256, w, us, 300) for w, us in ((1, 40000.0), (2, 90000.0), (4, 170000.0)) for _ in range(8)]
    cal.rounds += [(255, 1, 39000.0, 300)] * 20  # a tail batch size that did not time every width
    cal.rounds += [(64, w, 1.0, 300) for w in (1, 2, 4) for _ in range(MIN_ROUNDS - 1)]  # too few rounds
    cal.accepted(4, torch.tensor([1] * 10 + [2] * 20 + [3] * 30 + [4] * 40))
    cal.flush()
    assert load_round_times(str(tmp_path)) == {
        1: {1: 10000.0, 2: 12000.0, 4: 15000.0},
        256: {1: 40000.0, 2: 90000.0, 4: 170000.0},
    }
    # 100 request-rounds; accepting drafted token 1: accept length >= 2 (90), token 2: >= 3 (70), token 3: 4 (40).
    assert load_accept_profile(str(tmp_path)).rates == pytest.approx((0.9, 0.7, 0.4))
    with open(tmp_path / "rounds" / "chain.json", encoding="utf-8") as f:
        assert {r["count"] for r in json.load(f)["rounds"]} == {MIN_ROUNDS}


def test_accept_counts_from_the_histogram():
    assert accept_counts([0, 10, 20, 30, 40]) == (100, [90, 70, 40])
    assert accept_counts([0, 5, 0]) == (5, [0])


def test_report_prints_the_widths(tmp_path, capsys):
    cal = _calibration(tmp_path, widths=(1, 2))
    cal.rounds = [(8, 1, 10000.0, 9)] * MIN_ROUNDS + [(8, 2, 11000.0, 9)] * MIN_ROUNDS
    cal.accepted(2, torch.tensor([2, 2, 2, 1]))
    cal.flush()
    sglang_calibrate.report(str(tmp_path), 2, 16)
    assert "chain widths per batch size: [(1, 16, 2)]" in capsys.readouterr().out
