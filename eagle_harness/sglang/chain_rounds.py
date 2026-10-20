# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Fixed-width chain rounds: the served width per batch size, and the calibration that times it.

A chain (``--speculative-eagle-topk 1``, ``chain_intensity``) with served round times and an acceptance profile
under ``data_path`` verifies a fixed width per batch size (:func:`eagle_harness.core.chain_width.chain_widths`),
width 0 (native decode without draft extend) where speculation loses. ``"mode": "calibrate"`` measures that choice
while real requests are served: it forces every chain width the target has a verify graph for, in blocks of
:data:`BLOCK_ROUNDS` rounds, and records

- the period between consecutive decode rounds at the same batch size (wall clock, so the scheduler, the draft
  extend and the launches are in it), per batch size and width;
- the accept lengths of the widest rounds, per request (a histogram kept on the GPU, read when the file is written).

A round is recorded only between two decode rounds of the same batch size and width, so the first and last round
of a block (whose periods hold another width's prelaunched draft) and the rounds next to a prefill are not.

Rank 0 writes what it has every :data:`FLUSH_ROUNDS` rounds and at the next prefill: the raw rounds
(``<data_path>/raw/sglang_rounds.json``), the median period of every batch size that timed each width at least
:data:`MIN_ROUNDS` times (``rounds/chain.json``) and the acceptance profile (``accept/chain.json``), the two files a
serve-mode chain server with the same ``data_path`` reads.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import torch

from eagle_harness.core.chain_width import save_accept_profile, save_round_times, width_at

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

BLOCK_ROUNDS = 8
FLUSH_ROUNDS = 256
MIN_ROUNDS = 8
RAW_FILE = os.path.join("raw", "sglang_rounds.json")


def round_cells(
    rounds: Sequence[Sequence[float]], widths: Sequence[int], min_rounds: int = MIN_ROUNDS
) -> Dict[int, Dict[int, Tuple[float, int]]]:
    """``{batch size: {width: (median period us, rounds)}}`` of the batch sizes that timed every one of ``widths`` at
    least ``min_rounds`` times."""
    periods: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for bs, width, us, _context in rounds:
        periods[(int(bs), int(width))].append(us)
    cells: Dict[int, Dict[int, Tuple[float, int]]] = defaultdict(dict)
    for (bs, width), values in periods.items():
        if len(values) >= min_rounds:
            cells[bs][width] = (statistics.median(values), len(values))
    return {bs: dict(sorted(row.items())) for bs, row in sorted(cells.items()) if set(row) == set(widths)}


def accept_counts(accept_lens: Sequence[int]) -> Tuple[int, List[int]]:
    """Request-rounds and those accepting drafted token ``d`` (``d = 1 .. max``) from a histogram of accept lengths
    (tokens committed per request and round: 1 + accepted drafts)."""
    per_position = [sum(accept_lens[d + 1 :]) for d in range(1, len(accept_lens) - 1)]
    return sum(accept_lens), per_position


class ChainCalibration:
    """The forced width of each round (``width``) and the recorder of the served rounds (``round_start``,
    ``accepted``)."""

    def __init__(
        self, data_path: str, widths: Sequence[int], device: torch.device, *, writer: bool, native_decode: bool = False
    ):
        """``widths``: the chain widths to time (each has a target verify graph); ``writer``: this rank writes."""
        self.widths = sorted(widths)
        self.max_width = self.widths[-1]
        self._native_decode = native_decode
        self.last_width = self.widths[0]
        if native_decode:
            self.widths.append(0)
        self._data_path = data_path
        self._writer = writer
        self._planned = 0  # rounds whose width was handed out
        self._started = 0  # decode rounds started
        # The last two decode rounds since the last other step: (bs, width, start ns, mean context).
        self._recent: List[Tuple[int, int, int, int]] = []
        self.rounds: List[Tuple[int, int, float, int]] = []  # (bs, width, period us, mean context)
        self._flushed = 0  # rounds in the file
        self._accept = torch.zeros(self.max_width + 1, dtype=torch.int64, device=device)

    def width(self, bs: int) -> int:
        """The width of the next round (one call per round, in round order)."""
        if self._native_decode:
            # Collect enough settled intervals in one pass, then stay OFF: its
            # draft KV is no longer current and must not be consumed again.
            index = min(self._planned // (BLOCK_ROUNDS + MIN_ROUNDS), len(self.widths) - 1)
        else:
            index = (self._planned // BLOCK_ROUNDS) % len(self.widths)
        self.last_width = self.widths[index]
        self._planned += 1
        return self.last_width

    def round_start(self, bs: int, width: int, context: int) -> None:
        """A decode round of ``bs`` requests at ``width`` starts now; the previous round's period ends."""
        now = time.perf_counter_ns()
        recent = self._recent
        if len(recent) == 2 and all(r[:2] == (bs, width) for r in recent):
            self.rounds.append((bs, width, (now - recent[1][2]) / 1000.0, recent[1][3]))
        self._recent = recent[-1:] + [(bs, width, now, context)]
        self._started += 1
        if self._started % FLUSH_ROUNDS == 0:
            self.flush()
            self._recent = []  # the write is in this round's period

    def interrupt(self) -> None:
        """A non-decode step runs: the period in progress is not a decode round's. Writes the rounds recorded since
        the last write (a prefill after a batch, e.g. the calibration driver's next level, flushes it)."""
        self._recent = []
        if self._flushed != len(self.rounds):
            self.flush()
        if self._native_decode:
            self._planned = 0
            self.last_width = self.widths[0]

    def accepted(self, width: int, accept_lens: torch.Tensor) -> None:
        """Count the round's accept lengths (tokens committed per request, 1..width) when it is the widest."""
        if width == self.max_width:
            self._accept += torch.bincount(accept_lens.to(torch.int64), minlength=self.max_width + 1)

    def flush(self) -> None:
        """Write the raw rounds, the round times and the acceptance profile (rank 0); reads the accept histogram from
        the GPU."""
        self._flushed = len(self.rounds)
        if not self._writer:
            return
        accept_lens = self._accept.tolist()
        raw = {"max_width": self.max_width, "widths": self.widths, "accept_lens": accept_lens}
        raw["rounds"] = [list(r) for r in self.rounds]
        path = os.path.join(self._data_path, RAW_FILE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(f"{path}.tmp", "w", encoding="utf-8") as f:
            json.dump(raw, f)
        os.replace(f"{path}.tmp", path)
        cells = round_cells(self.rounds, self.widths)
        if cells:
            rows = [
                {"batch_size": bs, "width": w, "round_us": us, "count": n}
                for bs, row in cells.items()
                for w, (us, n) in row.items()
            ]
            save_round_times(self._data_path, rows)
        drafts, per_position = accept_counts(accept_lens)
        if drafts:
            save_accept_profile(self._data_path, drafts, per_position)


class ChainRounds:
    """The served/calibrated chain width and optional recording of completed rounds.

    Drafting asks for ``draft_width`` once when it queues a round. Native decode
    reads ``plain_width`` without advancing that plan, including a prelaunched
    choice of OFF. Context is read only for calibration, without a device sync.
    """

    def __init__(
        self,
        schedule: Sequence[tuple[int, int, int]] | None,
        calibration: ChainCalibration | None,
        average_context: Callable[[ScheduleBatch], int],
    ):
        self._schedule = schedule
        self._calibration = calibration
        self._average_context = average_context

    def draft_width(self, bs: int) -> int:
        width = self._calibration.width(bs) if self._calibration is not None else width_at(self._schedule, bs)
        return max(1, width)

    def plain_width(self, bs: int) -> int:
        return self._calibration.last_width if self._calibration is not None else width_at(self._schedule, bs)

    def interrupt(self) -> None:
        if self._calibration is not None:
            self._calibration.interrupt()

    def round_start(self, bs: int, width: int, batch: ScheduleBatch) -> None:
        if self._calibration is not None:
            self._calibration.round_start(bs, width, self._average_context(batch))

    def accepted(self, width: int, accept_lens: torch.Tensor) -> None:
        if self._calibration is not None:
            self._calibration.accepted(width, accept_lens)
