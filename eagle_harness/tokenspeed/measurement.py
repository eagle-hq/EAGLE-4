# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Calibrate mode on TokenSpeed: the measured rounds behind a tree's cost tables, engine side.

The driver (:mod:`eagle_harness.tokenspeed.calibrate`) writes ``config/measure_config.json``
(:class:`~eagle_harness.core.calibration.MeasureParams`) and sends ``batch_size`` identical greedy requests. Every
decode round then drafts every depth at one draft width and verifies at one verify width, both cycled from a counter
(:class:`MeasurePolicy`), and commits exactly one fixed token per request (the drafter's ``rewrite_verify_outputs``), so
the context grows by one token a round. Rank 0 times each round's draft depths (wall time between drained GPUs and a
TP barrier) and the next forward's verify (target forward through the accept, events recorded inside the decode graph),
and keeps the samples as ``model_slm`` / ``model_llm`` records for :func:`~eagle_harness.core.calibration.fit_batch`.

A tree verifies its whole window whatever the swept width, as serving does, so one decode graph width serves every
measured round. Only the rounds of a batch at the configured size count, and each prefill starts them afresh. Every rank
counts them; once the batch has its :data:`MEASURED_ROUNDS` verifies, every request commits an EOS token instead, which
ends the batch at the same round on every rank.
"""

from __future__ import annotations

import os
from dataclasses import fields

import torch
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.execution.drafter.tree_round.device_policy import write_round_slots

from eagle_harness.core.calibration import CalibrationPaths, MeasureParams, MeasureRecord, load_jsonl, save_model_timings
from eagle_harness.core.limits import MAX_BATCH_SIZE_TREE, ComputeIntensity, draft_query_lens, verify_buckets

# The token every request commits each measured round.
FORCED_TOKEN = 888
# Verify (and draft) samples a batch records before it ends.
MEASURED_ROUNDS = 512


def swept_draft_widths(batch_size: int, intensity: ComputeIntensity, node_top_k: int) -> list[int]:
    """Draft widths a batch measures: those the draft fit reads (``query_lens`` of ``model_slm``)."""
    return [w for w in draft_query_lens(batch_size, intensity) if w <= node_top_k]


def swept_verify_widths(batch_size: int, intensity: ComputeIntensity, verify_num: int) -> list[int]:
    """Verify widths a batch measures, widest first: those the target fit reads (``query_lens`` of ``model_llm``)."""
    return [w for w in verify_buckets(batch_size, intensity) if w <= verify_num]


def load_measure_params(paths: CalibrationPaths) -> MeasureParams | None:
    """The batch the driver asked for, None before it wrote one."""
    path = os.path.join(paths.config, "measure_config.json")
    if not os.path.exists(path):
        return None
    row = load_jsonl(path)[0]
    return MeasureParams(**{f.name: row[f.name] for f in fields(MeasureParams) if f.name in row})


class MeasurePolicy:
    """Calibrate-mode round decisions, taken on the host into the device slots ``DynamicDrafter`` reads: every depth
    at the round's draft width (the first at the previous round's, as the first step's width is decided before the
    round's), no early stop, every request verified at the round's verify width. Nothing is left for the device to
    decide, and it is no ``DevicePolicy``, so rounds are never device-chained."""

    chain = False
    # Measured rounds verify at the width written before the round.
    eager_select_verify = False

    def __init__(
        self, *, verify_num: int, node_top_k: int, tree_depth: int, intensity: ComputeIntensity, device: torch.device
    ) -> None:
        if intensity is ComputeIntensity.CHAIN:
            raise NotImplementedError("TokenSpeed calibrate mode measures trees; chains need a variable verify width")
        self.verify_num_limit = verify_num
        self.node_top_k = node_top_k
        self.tree_depth = tree_depth
        self.intensity = intensity
        self.counter = 0
        self.draft_width = node_top_k
        self.previous_draft_width = node_top_k
        self.verify_width = verify_num
        # Per depth: width, pool start, work list size, go (DevicePolicy's slot layout).
        self._slots = torch.zeros((4, tree_depth + 1), dtype=torch.int32, device=device)
        self.widths, self.cumu_starts, self.work_sizes, self.go = self._slots.unbind(0)
        self._counts = torch.zeros((4,), dtype=torch.int32, device=device)
        self.past_verify_num, self.cumu_count, self.verify_count, self.verify_num = self._counts.split(1)

    def table_width(self, batch_size: int) -> int:
        """No width table: every round may draft every depth."""
        return 0

    def observe_batch_size(self, batch_size: int) -> None:
        if not 0 <= batch_size <= MAX_BATCH_SIZE_TREE:
            raise ValueError(f"a measured tree round needs 0..{MAX_BATCH_SIZE_TREE} live requests, got {batch_size}")

    def next_widths(self, batch_size: int, loop_top_m: int) -> None:
        """Advance the counter and pick the round's widths."""
        self.counter += 1
        drafts = swept_draft_widths(batch_size, self.intensity, self.node_top_k)
        verifies = swept_verify_widths(batch_size, self.intensity, min(loop_top_m, self.verify_num_limit))
        self.previous_draft_width = self.draft_width
        self.draft_width = drafts[self.counter % len(drafts)]
        self.verify_width = verifies[self.counter % len(verifies)]

    def write_round(self) -> None:
        """The round's slots at the current widths."""
        self._write([self.previous_draft_width] + [self.draft_width] * (self.tree_depth - 2), self.verify_width)

    def write_full_round(self) -> None:
        """Slots of a round drafting every depth at full width (graph capture's warm-up)."""
        self._write([self.node_top_k] * (self.tree_depth - 1), self.verify_num_limit)

    def _write(self, widths: list[int], verify_num: int) -> None:
        start = write_round_slots(self._slots, self.node_top_k, widths, lambda depth, start: 0)
        verify_count = min(self.verify_num_limit, start)
        # The tree holds only the ranked nodes; the rest of its window repeats the root.
        self._counts.copy_(
            torch.tensor([self.verify_num_limit, start, verify_count, min(verify_num, verify_count)], dtype=torch.int32)
        )

    def go_ready(self, depth: int, round_seq: int) -> bool:
        return depth < self.tree_depth

    def deltas_depth(self, depth: int) -> bool:
        return False

    def begin_round(self, live_rows: torch.Tensor, frontier: torch.Tensor, sampled: torch.Tensor) -> None:
        """The round's slots are written before it."""

    def decide(self, depth: int, work_deltas: torch.Tensor) -> None:
        """Measured widths are decided before the round."""

    def select_verify(self, top_scores: torch.Tensor) -> None:
        """Measured rounds verify at the width written before the round."""


class CostMeasurement:
    """Calibrate-mode sampling: the batch's parameters, the round timers and the recorded samples (rank 0 keeps and
    writes them)."""

    def __init__(
        self, *, data_path: str, tree_depth: int, tp_group: tuple[int, ...] | None, rank: int, eos_token: int,
        device: torch.device,
    ) -> None:
        self.paths = CalibrationPaths(data_path)
        self.tree_depth = tree_depth
        self.tp_group = tp_group
        self._barrier = torch.zeros((1,), dtype=torch.int32, device=device)
        self.recorder = rank == 0
        self.params: MeasureParams | None = None
        self.records: list[MeasureRecord] = []
        # The token every request commits this round, read by the decode graph: FORCED_TOKEN, then the EOS once the
        # batch has its rounds.
        self.commit_token = torch.full((1,), FORCED_TOKEN, dtype=torch.int32, device=device)
        self.eos_token = eos_token
        # Recorded inside the decode graph around the target forward and verify.
        self.verify_start = torch.cuda.Event(enable_timing=True, external=True)
        self.verify_end = torch.cuda.Event(enable_timing=True, external=True)
        self.draft_start = torch.cuda.Event(enable_timing=True)
        self.draft_end = torch.cuda.Event(enable_timing=True)
        # (verify width, context) of the tree the next forward verifies.
        self.pending_verify: tuple[int, int] | None = None
        # Rounds of the batch counted on every rank (rank 0 also keeps their samples).
        self.verify_rounds = 0
        self.draft_rounds = 0
        self.written = False

    def on_extend(self) -> None:
        """A prefill: reload the parameters and start the batch afresh."""
        self.params = load_measure_params(self.paths)
        self.records = []
        self.pending_verify = None
        self.verify_rounds = 0
        self.draft_rounds = 0
        self.written = False
        self.commit_token.fill_(FORCED_TOKEN)

    @property
    def finished(self) -> bool:
        """Whether the batch has its rounds (the same on every rank)."""
        return self.verify_rounds >= MEASURED_ROUNDS

    def recording(self, batch_size: int) -> bool:
        params = self.params
        return params is not None and batch_size == params.batch_size and not self.finished

    def _drain(self) -> None:
        if self.tp_group is not None:
            all_reduce(self._barrier, self.tp_group)
        torch.cuda.synchronize()

    def record_verify(self, batch_size: int) -> None:
        """The verify this forward ran, if it verified a measured round."""
        pending, self.pending_verify = self.pending_verify, None
        if pending is None or not self.recording(batch_size):
            return
        self.verify_rounds += 1
        if self.finished:
            self.commit_token.fill_(self.eos_token)
        if self.recorder:
            self.verify_end.synchronize()
            elapsed = int(self.verify_start.elapsed_time(self.verify_end) * 1000)
            self._append(elapsed, 1, pending[0], pending[1], "model_llm", batch_size)

    def begin_draft(self) -> None:
        self._drain()
        self.draft_start.record()

    def end_draft(self, batch_size: int, draft_width: int, verify_width: int, cache_len: int) -> None:
        """Sample the round's draft; the next forward verifies its tree at ``verify_width``."""
        self._drain()
        self.draft_end.record()
        if not self.recording(batch_size) or self.draft_rounds >= MEASURED_ROUNDS:
            return
        self.draft_rounds += 1
        self.pending_verify = (verify_width, cache_len)
        if self.recorder:
            self.draft_end.synchronize()
            elapsed = int(self.draft_start.elapsed_time(self.draft_end) * 1000)
            self._append(elapsed, self.tree_depth - 1, draft_width, cache_len, "model_slm", batch_size)

    def _append(self, elapsed: int, count: int, width: int, cache_len: int, category: str, batch_size: int) -> None:
        if elapsed <= 0:
            raise RuntimeError(f"{category} sample of {elapsed} us")
        self.records.append(MeasureRecord(elapsed, count, width, cache_len, category, batch_size))

    def complete(self) -> bool:
        """Whether rank 0 holds the finished batch's samples, not written yet."""
        return self.recorder and self.finished and not self.written

    def write_model_records(self) -> str:
        """Write the batch's raw stage samples under ``test/bs_<n>``; returns that directory."""
        save_model_timings(self.paths, self.params, self.records)
        self.written = True
        return os.path.join(self.paths.test, f"bs_{self.params.batch_size}")
