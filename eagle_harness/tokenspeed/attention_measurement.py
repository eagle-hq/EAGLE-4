# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""TokenSpeed tree-cascade attention timings for calibration."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.ops.attention.mha.flashinfer import trtllm_batch_decode_with_kv_cache
from tokenspeed_kernel.ops.attention.tree import tree_window_attention
from tokenspeed.runtime.layers.attention.backends.paged.trtllm import canonicalize_stride
from tokenspeed.runtime.layers.attention.kernel_page_sizes import TRTLLM_MHA_PAGE_SIZE

from eagle_harness.core.calibration import MeasureRecord

_WORKSPACE_BYTES = 256 << 20


def attention_samples(
    *,
    batch_size: int,
    rows: int,
    window: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int,
    context_step: int,
    context_stop: int,
    fast: bool,
    repeat: int,
    widths: list[int],
    category: str,
) -> list[MeasureRecord]:
    """One attention layer of the decode tree cascade over the context range:
    ``rows`` query rows per request against its ``kv_len``-key prefix (the
    trtllm-gen decode; a sliding window decodes each row alone over the
    window ending there) plus its ``window``-key tree window.

    Serving forwards all ``rows`` whatever the round's width, so one timing
    per context stands for every swept width.

    Returns:
        A sample per (context, width): microseconds over ``repeat`` layers.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    page = TRTLLM_MHA_PAGE_SIZE
    interval = context_step // (16 if fast else 32)
    lengths = [
        kv for kv in range(context_step // 2, context_stop + 1, interval)
        if kv >= context_step
    ]  # fmt: skip
    pages_per_req = math.ceil((context_stop + window) / page)
    num_pages = batch_size * pages_per_req
    k_rows = torch.randn(
        (num_pages * page, num_kv_heads, head_dim), device=device, dtype=dtype
    )
    v_rows = torch.randn_like(k_rows)

    def paged(rows_: torch.Tensor) -> torch.Tensor:
        cache = rows_.view(-1, page, num_kv_heads, head_dim).permute(0, 2, 1, 3)
        return canonicalize_stride(cache) if num_kv_heads == 1 else cache

    k_cache, v_cache = paged(k_rows), paged(v_rows)
    page_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(
        batch_size, pages_per_req
    )
    q = torch.randn(
        (batch_size * rows, num_q_heads, head_dim), device=device, dtype=dtype
    )
    workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    scale = head_dim**-0.5
    causal = sliding_window <= 0
    # Row r sees its path's window keys: a chain for verify rows, one ancestor per depth for lanes.
    if causal:
        bits = [(1 << (r + 1)) - 1 for r in range(rows)]
    else:
        bits = [
            sum(1 << (d * rows + r) for d in range(window // rows)) for r in range(rows)
        ]
    # Window key j is bit j % 64 of the row's word j // 64; bit 63 is the sign.
    words = [
        (b >> (64 * w)) & ((1 << 64) - 1)
        for b in bits
        for w in range((window + 63) // 64)
    ]
    mask = torch.tensor(
        [x - (1 << 64) if x >= 1 << 63 else x for x in words] * batch_size,
        dtype=torch.int64,
        device=device,
    )
    samples = []
    for kv_len in lengths:
        prefix = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
        seq_lens = prefix + window
        if causal:
            tables, prefix_lens, q_len = page_table, prefix, rows
        else:
            tables = page_table.repeat_interleave(rows, dim=0)
            prefix_lens, q_len = prefix.repeat_interleave(rows), 1

        def layer() -> torch.Tensor:
            out, lse = trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=(k_cache, v_cache),
                workspace_buffer=workspace,
                block_tables=tables,
                seq_lens=prefix_lens,
                max_seq_len=context_stop + window,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                out_dtype=dtype,
                q_len_per_req=q_len,
                window_left=-1 if causal else sliding_window - 1,
                return_lse=True,
            )
            return tree_window_attention(
                q, k_rows, v_rows, page_table, seq_lens, mask, out, lse,
                rows_per_req=rows, window=window, causal_prefix=causal,
                page_size=page, sm_scale=scale,
            )  # fmt: skip

        layer()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeat):
                layer()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = max(int(start.elapsed_time(end) * 1000), 1)
        samples += [
            MeasureRecord(elapsed, repeat, width, kv_len, category, batch_size)
            for width in widths
        ]
    return samples
