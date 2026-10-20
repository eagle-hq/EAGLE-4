# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""GPU timing sweeps of attention cost vs context length (calibration step 1).

:func:`sweep` times every (query length, context length) of a plan through a
``Timer`` (CUDA events around ``num_layers`` repeated calls, warm-up lengths
below ``kv_step`` dropped) and writes the jsonl records
:func:`eagle_harness.core.calibration.fit_cache_costs` reads. The FlashInfer
timer calls the kernel directly, not through an engine.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Protocol, Sequence

import torch

from eagle_harness.core.calibration import CacheSweepPlan, MeasureRecord, save_jsonl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeadLayout:
    """Per-rank attention heads of a model under tensor parallelism."""

    num_qo_heads: int
    num_kv_heads: int
    head_dim: int

    @classmethod
    def for_rank(
        cls, num_qo_heads_total: int, num_kv_heads_total: int, head_dim: int, tp_size: int, *, replicate_kv: bool = True
    ) -> "HeadLayout":
        """KV heads fewer than the ranks are replicated (one per rank) when ``replicate_kv``."""
        if num_qo_heads_total % tp_size != 0:
            raise ValueError("Total number of query heads must be divisible by tp_size.")
        if replicate_kv:
            if num_kv_heads_total % tp_size != 0 and tp_size % num_kv_heads_total != 0:
                raise ValueError("KV heads and tp_size must divide one another.")
            kv = max(1, num_kv_heads_total // tp_size)
        else:
            if num_kv_heads_total % tp_size != 0:
                raise ValueError("Total number of KV heads must be divisible by tp_size.")
            kv = num_kv_heads_total // tp_size
        return cls(num_qo_heads_total // tp_size, kv, head_dim)


class Timer(Protocol):
    """Times batched attention calls for the sweeps."""

    def time_us(self, batch_size: int, qo_len: int, kv_len: int, num_layers: int) -> int:
        """Microseconds for ``num_layers`` attention calls at this shape."""


def _timed(fn, num_layers: int) -> int:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(num_layers):
        fn()
    end.record()
    torch.cuda.synchronize()
    return int(start.elapsed_time(end) * 1000)


class FlashInferTimer:
    """Paged prefill with a causal custom mask (the tree-verify kernel family)."""

    def __init__(self, heads: HeadLayout, tp_size: int, device: str = "cuda", page_size: int = 1):
        import flashinfer

        self.heads, self.device, self.page_size = heads, device, page_size
        workspace = torch.empty(min(8, tp_size) * 128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

    def time_us(self, batch_size: int, qo_len: int, kv_len: int, num_layers: int) -> int:
        """Microseconds of ``num_layers`` paged-prefill calls at this shape."""
        h, dev, page = self.heads, self.device, self.page_size
        nnz_qo = qo_len * batch_size
        pages_per_seq = (kv_len + page - 1) // page
        last_page_len = kv_len - (pages_per_seq - 1) * page
        num_pages = pages_per_seq * batch_size
        mask = torch.tril(torch.ones(qo_len, kv_len, dtype=torch.bool, device=dev), diagonal=kv_len - qo_len)
        self.wrapper.plan(
            torch.arange(0, nnz_qo + 1, qo_len, dtype=torch.int32, device=dev),
            torch.arange(0, num_pages + 1, pages_per_seq, dtype=torch.int32, device=dev),
            torch.arange(num_pages, dtype=torch.int32, device=dev),
            torch.full((batch_size,), last_page_len, dtype=torch.int32, device=dev),
            h.num_qo_heads,
            h.num_kv_heads,
            h.head_dim,
            page,
            custom_mask=mask.expand(batch_size, -1, -1).flatten(),
        )
        q = torch.randn(nnz_qo, h.num_qo_heads, h.head_dim, dtype=torch.float16, device=dev)
        kv = torch.randn(num_pages, 2, page, h.num_kv_heads, h.head_dim, dtype=torch.float16, device=dev)
        return _timed(lambda: self.wrapper.run(q, kv), num_layers)


def sweep(
    timer: Timer,
    *,
    category: str,
    batch_size: int,
    qo_lens: Sequence[int],
    plan: CacheSweepPlan,
    save_dir: str,
    device: str = "cuda",
) -> List[str]:
    """Time every (qo_len, kv_len) of the plan; write ``cache_qo_len_<q>.jsonl`` per qo_len."""
    written = []
    for qo_len in qo_lens:
        records = []
        for kv_len in plan.cache_lengths():
            kv_len = max(kv_len, 1)
            us = timer.time_us(batch_size, qo_len, kv_len, plan.num_layers)
            if kv_len >= plan.kv_step:  # shorter lengths are warm-up
                records.append(MeasureRecord(us, plan.num_layers, qo_len, kv_len, category, batch_size))
            _release_if_low(device)
        if not records:
            raise RuntimeError(f"No valid measurements collected for qo_len = {qo_len}.")
        path = os.path.join(save_dir, f"cache_qo_len_{qo_len}.jsonl")
        save_jsonl(path, records)
        written.append(path)
    return written


def _release_if_low(device: str) -> None:
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info(device)
    if free / total < 0.05:
        logger.warning("Low GPU memory during the attention sweep; timings may be slower.")
        torch.cuda.empty_cache()
