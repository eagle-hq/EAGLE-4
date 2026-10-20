# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Served decode rounds of the vLLM tree speculator, timed for :func:`eagle_harness.core.chain_width.served_tree_schedule`.

vLLM fixes a round's verify width from its batch size, so a server times only the widths its schedule runs: calibrate
a width by serving the workload once with ``max_verify_width`` set to it and ``record_rounds`` on, then the next
width; every server merges its cells into ``<data_path>/rounds/tree.json`` and keeps the others.

The speculator calls :meth:`ServedRoundRecorder.round` once per ``propose``. The period from one call to the next
holds the draft growth of ``count`` rows and the verify of ``count + 1`` rows that follows it, plus the scheduler and
the launches between them, so it is recorded as one round of width ``count + 1`` when both calls saw the same batch
size, the previous call's ``count`` and no prefill. ``num_sampled`` (tokens each request committed in the verify just
finished) is summed on the device; only :meth:`rows` reads it back, once per flush.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from typing import ContextManager

import torch

from eagle_harness.core.chain_width import merge_tree_rounds

FLUSH_ROUNDS = 256
MIN_ROUNDS = 8


class ServedRoundRecorder:
    def __init__(
        self,
        data_path: str,
        *,
        writer: bool,
        flush_rounds: int = FLUSH_ROUNDS,
        min_rounds: int = MIN_ROUNDS,
        clock: Callable[[], float] = time.perf_counter,
        sync_allowed: Callable[[], ContextManager] = contextlib.nullcontext,
    ):
        """``writer``: this rank writes the file (one rank of a tensor-parallel group); ``sync_allowed``: the context
        under which a flush may read the device back (vLLM's ``gpu_sync_allowed``)."""
        self.data_path = data_path
        self.writer = writer
        self.flush_rounds = flush_rounds
        self.min_rounds = min_rounds
        self.clock = clock
        self.sync_allowed = sync_allowed
        self._last: tuple[int, int] | None = None  # (batch size, count) of the previous decode call
        self._last_t = 0.0
        self._periods: dict[tuple[int, int], list[float]] = defaultdict(list)
        self._tokens: dict[tuple[int, int], torch.Tensor] = {}  # committed tokens, on the device
        self._recorded = 0

    def round(self, num_reqs: int, count: int, has_prefill: bool, num_sampled: torch.Tensor) -> None:
        """One ``propose`` call: ``count`` rows to draft now, ``num_sampled[:num_reqs]`` from the verify just done."""
        now = self.clock()
        last = self._last
        if not has_prefill and last is not None and last[0] == num_reqs:
            key = (num_reqs, last[1] + 1)
            self._periods[key].append(now - self._last_t)
            committed = num_sampled[:num_reqs].sum(dtype=torch.int64)
            if key in self._tokens:
                self._tokens[key] += committed
            else:
                self._tokens[key] = committed.clone()
            self._recorded += 1
            if self.writer and self._recorded % self.flush_rounds == 0:
                self.flush()
        self._last = None if has_prefill else (num_reqs, count)
        self._last_t = now

    def rows(self) -> list[dict]:
        """The cells timed at least ``min_rounds`` times (reads the device sums back)."""
        out = []
        for (bs, width), periods in sorted(self._periods.items()):
            if len(periods) < self.min_rounds:
                continue
            accept = int(self._tokens[(bs, width)].item()) / (bs * len(periods))
            out.append(
                {
                    "batch_size": bs,
                    "width": width,
                    "round_us": round(statistics.median(periods) * 1e6, 1),
                    "accept": round(accept, 4),
                    "count": len(periods),
                }
            )
        return out

    def flush(self) -> None:
        if not self.writer:
            return
        with self.sync_allowed():
            rows = self.rows()
        if rows:
            merge_tree_rounds(self.data_path, rows)
