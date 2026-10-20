# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Rank-major all-gather of the draft's packed candidates over NVLink multicast.

Every draft step all-gathers each rank's packed candidates (a few KB). NCCL's ring all-gather is ~10-12 us there at
TP4 on B300; ``torch.ops.symm_mem.multimem_all_gather_out`` into a symmetric buffer is one kernel. Where symmetric
memory or multicast is missing the gather falls back to vLLM's (NCCL) all-gather. Both give the same bytes in the same
rank-major order, so the outputs do not change.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _multimem_all_gather_out(x: torch.Tensor, group_name: str, out: torch.Tensor) -> None:
    """``out`` (``[world * rows, width]``, a view of a symmetric buffer) = the rank-major gather of ``x`` ``[rows, width]``.

    The op wants ``out`` with the input's rank and ``out.shape[0] == world * x.shape[0]``."""
    torch.ops.symm_mem.multimem_all_gather_out(x, group_name, out)


class CandidateGather:
    """``all_gather(x, dim=0)`` of the TP group for int64 ``[rows, width]`` with ``rows <= max_rows``.

    The symmetric buffer is allocated at the first call (a warm-up run, which every rank makes at the same time;
    never inside a CUDA graph capture) and holds the result until the next call.
    """

    def __init__(self, group: Any, max_rows: int):
        self.group = group  # vLLM's GroupCoordinator of the TP group
        self.max_rows = max_rows
        self.out: torch.Tensor | None = None
        self._tried = False

    def _allocate(self, x: torch.Tensor) -> None:
        self._tried = True
        try:
            import torch.distributed._symmetric_memory as symm_mem

            out = symm_mem.empty(self.group.world_size * self.max_rows * x.shape[1], dtype=x.dtype, device=x.device)
            handle = symm_mem.rendezvous(out, self.group.device_group.group_name)
        except (ImportError, RuntimeError) as error:
            logger.info("candidate all-gather on NCCL: no symmetric memory (%s)", error)
            return
        if handle.multicast_ptr == 0:
            logger.info("candidate all-gather on NCCL: no multicast")
            return
        self.out = out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self._tried and not torch.cuda.is_current_stream_capturing():
            self._allocate(x)
        rows, width = x.shape
        world = self.group.world_size
        if self.out is None or rows > self.max_rows or x.dtype != self.out.dtype or not x.is_contiguous():
            return self.group.all_gather(x, dim=0)
        out = self.out[: world * rows * width].view(world * rows, width)
        _multimem_all_gather_out(x, self.group.device_group.group_name, out)
        return out
