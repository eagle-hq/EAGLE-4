# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The served-round recorder of the vLLM tree speculator (CPU tensors, a fake clock)."""

from __future__ import annotations

import torch

from eagle_harness.core.chain_width import load_tree_rounds
from eagle_harness.vllm.round_recorder import ServedRoundRecorder


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _logits(rows_per_req):
    """``cu_num_logits`` for requests verifying ``rows_per_req[i]`` rows each (1: no drafts, e.g. a prefill)."""
    return torch.tensor([0, *rows_per_req], dtype=torch.int32).cumsum(0)


def _feed(rec, clock, rounds, num_reqs, count, sampled, period_s, prefill=False):
    for _ in range(rounds):
        clock.now += period_s
        rec.round(num_reqs, count, prefill, torch.full((num_reqs + 3,), sampled, dtype=torch.int32), _logits([4] * num_reqs))


def test_rows_are_the_median_period_and_mean_accept_of_steady_rounds(tmp_path):
    clock = Clock()
    rec = ServedRoundRecorder(str(tmp_path), writer=True, flush_rounds=10_000, min_rounds=4, clock=clock)
    # The first call of a shape opens it; the next 5 rounds of 64 requests at count 3 are width-4 rounds of 20 ms.
    _feed(rec, clock, 6, 64, 3, sampled=3, period_s=0.020)
    rows = rec.rows()
    assert rows == [{"batch_size": 64, "width": 4, "round_us": 20_000.0, "accept": 3.0, "count": 5}]


def test_a_period_counts_only_between_two_decode_rounds_of_one_shape(tmp_path):
    clock = Clock()
    rec = ServedRoundRecorder(str(tmp_path), writer=True, flush_rounds=10_000, min_rounds=1, clock=clock)
    _feed(rec, clock, 3, 8, 15, sampled=5, period_s=0.010)  # 2 periods at (8, 16)
    _feed(rec, clock, 1, 8, 15, sampled=5, period_s=0.500, prefill=True)  # a prefill round: no period
    _feed(rec, clock, 1, 8, 15, sampled=5, period_s=0.400)  # follows a prefill: no period
    _feed(rec, clock, 1, 9, 15, sampled=5, period_s=0.300)  # batch size changed: no period (accept still counts)
    _feed(rec, clock, 1, 9, 3, sampled=5, period_s=0.011)  # the previous round drafted 15: a width-16 round of 9
    _feed(rec, clock, 1, 9, 3, sampled=2, period_s=0.007)  # a width-4 round of 9
    rows = {(r["batch_size"], r["width"]): r for r in rec.rows()}
    assert set(rows) == {(8, 16), (9, 16), (9, 4)}
    assert rows[(8, 16)]["round_us"] == 10_000.0 and rows[(8, 16)]["count"] == 2
    assert rows[(9, 16)]["round_us"] == 11_000.0 and rows[(9, 16)]["accept"] == 5.0
    assert rows[(9, 4)]["round_us"] == 7_000.0 and rows[(9, 4)]["accept"] == 2.0


def test_accept_is_the_served_mean_over_every_verify_of_the_cell_prefills_included(tmp_path):
    """A batch that also prefills verifies its drafts as chains and commits fewer tokens; vLLM's spec-decode counters
    count those rounds, so the cell's accept does too, over the requests that verified drafts only."""
    clock = Clock()
    rec = ServedRoundRecorder(str(tmp_path), writer=True, flush_rounds=10_000, min_rounds=1, clock=clock)
    _feed(rec, clock, 3, 4, 15, sampled=6, period_s=0.010)  # 2 tree rounds of 4 requests: 6 tokens each
    clock.now += 0.050
    # 4 decode requests verify 16 rows as chains (2 tokens each) beside one new request's prefill (1 row, 1 token)
    sampled = torch.tensor([2, 2, 2, 2, 1], dtype=torch.int32)
    rec.round(5, 15, True, sampled, _logits([16, 16, 16, 16, 1]))
    (row,) = rec.rows()
    assert (row["batch_size"], row["width"], row["count"]) == (4, 16, 2)
    assert row["round_us"] == 10_000.0  # the prefill round is not timed
    assert row["accept"] == round((2 * 4 * 6 + 4 * 2) / (3 * 4), 4)  # 56 tokens over 12 request-rounds


def test_cells_below_min_rounds_are_not_written(tmp_path):
    clock = Clock()
    rec = ServedRoundRecorder(str(tmp_path), writer=True, flush_rounds=10_000, min_rounds=8, clock=clock)
    _feed(rec, clock, 5, 64, 3, sampled=3, period_s=0.020)
    assert rec.rows() == []


def test_flush_merges_into_the_data_path_and_only_the_writer_writes(tmp_path):
    clock = Clock()
    rec = ServedRoundRecorder(str(tmp_path), writer=True, flush_rounds=4, min_rounds=2, clock=clock)
    _feed(rec, clock, 5, 64, 15, sampled=4, period_s=0.048)  # the 4th recorded round flushes
    assert load_tree_rounds(str(tmp_path)) == {64: {16: (48_000.0, 4.0, 4)}}
    other = tmp_path / "other"
    quiet = ServedRoundRecorder(str(other), writer=False, flush_rounds=1, min_rounds=1, clock=clock)
    _feed(quiet, clock, 3, 64, 15, sampled=4, period_s=0.048)
    assert load_tree_rounds(str(other)) is None
