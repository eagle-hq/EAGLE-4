# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Per-batch serving decisions the speculator answers through vLLM's optional speculator hooks.

vLLM review PR 3 lets a speculator decide two things per scheduled batch (:func:`host_speculator_hooks` finds them):

- ``suspend_drafting(num_reqs, req_ids)``: native target decode (native-OFF) for the batch sizes where the priced
  chain schedule picked width 0 (:func:`~eagle_harness.vllm.config.native_decode_ranges`).
- ``verify_budget(req_ids, num_draft_tokens)``: how many scheduled drafts each request verifies. The scheduler sizes a
  request's drafts from the batch size of the step that drafted them; this verifies at most the width the schedule
  gives the batch being verified, so the measured high-load width (``served_tree_schedule``, the chain widths) applies
  from the first step at that batch size instead of one step later. A tree's verify rows put each parent before its
  children, so a prefix of them is a tree.

On a vLLM without these hooks (the 0.1.0 release engine) nothing here runs: the native ranges go through
``SpeculativeConfig.native_decode_batch_sizes`` and the schedule alone sets the width, as before.
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Callable, FrozenSet, Iterable, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# hook -> its parameters after ``self``: a host whose hook has other parameters does not count as having it
SPECULATOR_HOOKS = {
    "suspend_drafting": ("num_reqs", "req_ids"),
    "verify_budget": ("req_ids", "num_draft_tokens"),
}
LOG_INTERVAL_S = 60.0


def host_speculator_hooks() -> FrozenSet[str]:
    """The optional ``BaseSpeculator`` hooks this vLLM calls (empty without vLLM or on the release engine)."""
    try:
        from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
    except ImportError:
        return frozenset()
    found = set()
    for name, params in SPECULATOR_HOOKS.items():
        hook = getattr(BaseSpeculator, name, None)
        if callable(hook) and tuple(inspect.signature(hook).parameters)[1:] == params:
            found.add(name)
    return frozenset(found)


class BatchPolicy:
    """Answers ``suspend_drafting`` and ``verify_budget`` from static per-batch-size tables (host only, no device
    reads) and logs what they did at most once a minute."""

    def __init__(
        self,
        native_ranges: Iterable[Tuple[int, int]],
        schedule: Optional[Sequence[Sequence[int]]],
        max_num_reqs: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.native_ranges = tuple((int(first), int(last)) for first, last in native_ranges)
        # drafts verified at each batch size (None: the scheduler's count stands)
        self.width_drafts: Optional[np.ndarray] = None
        if schedule:
            drafts = np.full(max_num_reqs + 1, np.iinfo(np.int32).max, dtype=np.int64)
            for first, last, count in schedule:
                drafts[max(int(first), 0) : min(int(last), max_num_reqs) + 1] = int(count)
            self.width_drafts = drafts
        self.clock = clock
        self.batches = self.suspended = self.trimmed = 0
        self._logged: Tuple[int, int, int] = (0, 0, 0)
        self._next_log = 0.0

    def suspend(self, num_reqs: int) -> bool:
        suspend = any(first <= num_reqs <= last for first, last in self.native_ranges)
        if suspend:
            self.suspended += 1
            self._report()
        return suspend

    def verified_drafts(self, num_reqs: int, num_draft_tokens: np.ndarray) -> np.ndarray:
        """``num_draft_tokens`` itself when nothing is trimmed (vLLM then keeps the batch as scheduled)."""
        self.batches += 1
        if self.width_drafts is None or not len(num_draft_tokens):
            return num_draft_tokens
        width = self.width_drafts[min(num_reqs, len(self.width_drafts) - 1)]
        if int(num_draft_tokens.max()) <= width:
            return num_draft_tokens
        self.trimmed += 1
        self._report()
        return np.minimum(num_draft_tokens, width)

    def _report(self) -> None:
        now = self.clock()
        counts = (self.batches, self.trimmed, self.suspended)
        if now < self._next_log or counts == self._logged:
            return
        self._next_log, self._logged = now + LOG_INTERVAL_S, counts
        logger.info(
            "eagle_harness: verify_budget trimmed %d of %d draft batches to their batch size's width; "
            "suspend_drafting suspended %d batches",
            self.trimmed,
            self.batches,
            self.suspended,
        )
