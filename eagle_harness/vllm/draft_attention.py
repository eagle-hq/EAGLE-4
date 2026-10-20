# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Attention of the draft tree's steps on vLLM.

Draft step ``s`` runs ``K`` frontier rows per request, each at position ``c + s - 1`` (``c``: the committed
length), writing its draft KV to a slot past ``c`` and attending to

- the request's committed draft KV inside the draft's sliding window (the same keys for all rows), and
- the row's own ``s`` tree ancestors, itself last, whose slots :attr:`DraftTree.ancestors` keeps.

:func:`tree_step_attention` reads those keys through the block table and loads each committed key once for all the
rows of a request (:mod:`eagle_harness.kernels.row_attention`, the kernel sglang's draft uses). The per-row pair
:func:`build_tree_rows` + :func:`row_attention` (each row a one-token decode over an explicit list of KV slots) is
its reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from eagle_harness.kernels.row_attention import RowsLaunch, program_splits, rows_attention, rows_padded, window_start

KEYS_PER_SPLIT = 512
MAX_SPLITS = 16
BLOCK_KEYS = 64
_BUILD_BLOCK = 1024


@triton.jit
def _tree_rows_kernel(
    block_table_ptr,  # [num_reqs, max_blocks] int32 of the draft KV group, by batch row
    stride_table,
    block_size,
    committed_ptr,  # [num_reqs] committed draft length c
    ancestors_ptr,  # [num_reqs, K, max_depth] int32: draft KV slot of each frontier row's ancestors
    keys_ptr,  # [rows, max_keys] int32 out
    stride_keys,
    num_keys_ptr,  # [rows] int32 out
    step,
    window,
    max_depth: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    b = row // top_k
    committed = tl.load(committed_ptr + b).to(tl.int32)
    start = window_start(committed + step - 1, window)
    start = tl.minimum(start, committed)
    prefix = committed - start
    total = prefix + step
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    position = start + offs
    block = tl.load(block_table_ptr + b * stride_table + position // block_size, mask=offs < prefix, other=0)
    slot = block.to(tl.int32) * block_size + position % block_size
    ancestor = tl.load(
        ancestors_ptr + row * max_depth + (offs - prefix), mask=(offs >= prefix) & (offs < total), other=0
    )
    tl.store(keys_ptr + row * stride_keys + offs, tl.where(offs < prefix, slot, ancestor), mask=offs < total)
    if chunk == 0:
        tl.store(num_keys_ptr + row, total)


def build_tree_rows(
    block_table: torch.Tensor,
    block_size: int,
    committed: torch.Tensor,
    ancestors: torch.Tensor,
    keys: torch.Tensor,
    num_keys: torch.Tensor,
    *,
    step: int,
    window: int,
) -> None:
    """Key lists of draft step ``step`` for the ``bs * K`` frontier rows (``ancestors``: ``[bs, K, depth]``)."""
    bs, top_k, max_depth = ancestors.shape
    _tree_rows_kernel[(bs * top_k, triton.cdiv(keys.shape[1], _BUILD_BLOCK))](
        block_table,
        block_table.stride(0),
        block_size,
        committed,
        ancestors,
        keys,
        keys.stride(0),
        num_keys,
        step,
        window,
        max_depth=max_depth,
        top_k=top_k,
        BLOCK=_BUILD_BLOCK,
    )


@triton.jit
def _row_attention_split_kernel(
    q_ptr,  # [rows, H, D]
    kv_ptr,  # KV cache base
    keys_ptr,  # [rows, max_keys] int32 KV slots
    stride_keys,
    num_keys_ptr,  # [rows]
    partial_ptr,  # [rows, H, SPLITS, D] fp32
    partial_lse_ptr,  # [rows, H, SPLITS] fp32
    stride_qr,
    stride_qh,
    stride_block,
    stride_head,
    stride_slot,
    block_size,
    scale,
    D: tl.constexpr,
    H: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    PER_SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    num_keys = tl.load(num_keys_ptr + row)
    lo = split * PER_SPLIT
    hi = tl.minimum(lo + PER_SPLIT, num_keys)
    dims = tl.arange(0, D)
    q = tl.load(q_ptr + row * stride_qr + head * stride_qh + dims).to(tl.float32)
    peak = tl.full([], float("-inf"), dtype=tl.float32)
    total = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)
    for first in range(lo, hi, BLOCK):
        offs = first + tl.arange(0, BLOCK)
        valid = offs < hi
        slot = tl.load(keys_ptr + row * stride_keys + offs, mask=valid, other=0).to(tl.int64)
        at = (
            (slot // block_size) * stride_block
            + (head // QUERY_GROUP_SIZE) * stride_head
            + (slot % block_size) * stride_slot
        )
        k = tl.load(kv_ptr + at[:, None] + dims[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], 1) * scale
        scores = tl.where(valid, scores, float("-inf"))
        new_peak = tl.maximum(peak, tl.max(scores, 0))
        weights = tl.exp(scores - new_peak)
        rescale = tl.exp(peak - new_peak)
        v = tl.load(kv_ptr + at[:, None] + D + dims[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * rescale + tl.sum(weights[:, None] * v, 0)
        total = total * rescale + tl.sum(weights, 0)
        peak = new_peak
    out = (row * H + head) * SPLITS + split
    tl.store(partial_ptr + out * D + dims, acc / total)
    tl.store(partial_lse_ptr + out, peak + tl.log(total))


@triton.jit
def _row_attention_merge_kernel(
    partial_ptr,
    partial_lse_ptr,
    num_keys_ptr,
    out_ptr,  # [rows, H, D]
    H: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    PER_SPLIT: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    num_splits = tl.cdiv(tl.load(num_keys_ptr + row), PER_SPLIT)
    splits = tl.arange(0, SPLITS)
    base = (row * H + head) * SPLITS
    lse = tl.load(partial_lse_ptr + base + splits, mask=splits < num_splits, other=float("-inf"))
    weights = tl.exp(lse - tl.max(lse, 0))
    parts = tl.load(
        partial_ptr + (base + splits)[:, None] * D + tl.arange(0, D)[None, :],
        mask=(splits < num_splits)[:, None],
        other=0.0,
    )
    out = tl.sum(parts * (weights / tl.sum(weights, 0))[:, None], 0)
    tl.store(out_ptr + (row * H + head) * D + tl.arange(0, D), out.to(out_ptr.dtype.element_ty))


def row_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    keys: torch.Tensor,
    num_keys: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """One-token attention of ``rows`` queries ``[rows, H, D]`` over the KV slots in ``keys``.

    ``key_cache``: the layer's KV cache viewed ``[blocks, slots in a block, heads, 2 * D]`` (K then V of a token).
    """
    rows, num_heads, head_dim = q.shape
    kv_heads = key_cache.shape[2]
    if kv_heads <= 0 or num_heads % kv_heads:
        raise ValueError(f"query heads ({num_heads}) must be divisible by KV heads ({kv_heads})")
    block_size = key_cache.shape[1]
    max_splits = triton.cdiv(keys.shape[1], KEYS_PER_SPLIT)
    splits = min(triton.next_power_of_2(max_splits), MAX_SPLITS)
    per_split = triton.cdiv(keys.shape[1], splits)
    partial = torch.empty((rows, num_heads, splits, head_dim), dtype=torch.float32, device=q.device)
    partial_lse = torch.empty((rows, num_heads, splits), dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    _row_attention_split_kernel[(rows, num_heads, splits)](
        q,
        key_cache,
        keys,
        keys.stride(0),
        num_keys,
        partial,
        partial_lse,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(2),
        key_cache.stride(1),
        block_size,
        scale,
        D=head_dim,
        H=num_heads,
        QUERY_GROUP_SIZE=num_heads // kv_heads,
        SPLITS=splits,
        PER_SPLIT=per_split,
        BLOCK=BLOCK_KEYS,
    )
    _row_attention_merge_kernel[(rows, num_heads)](
        partial, partial_lse, num_keys, out, H=num_heads, D=head_dim, SPLITS=splits, PER_SPLIT=per_split
    )
    return out


# Keys per iteration of :func:`tree_step_attention` (a [rows, 128] x [128, D] tile per step).
STEP_BLOCK = 128


def tree_step_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    table_block: int,
    committed: torch.Tensor,
    ancestors: torch.Tensor,
    scale: float,
    *,
    step: int,
    window: int,
) -> torch.Tensor:
    """Attention of draft step ``step``'s ``bs * K`` rows ``q`` ``[bs * K, H, D]`` (no key lists).

    ``key_cache``: the layer's KV cache viewed ``[blocks, slots, heads, 2 * D]``; ``block_table[b]`` maps request
    ``b``'s positions to blocks of ``table_block`` tokens; ``ancestors`` ``[bs, K, depth]`` holds flat token slots.
    The window is cut into about one program per SM in all. Same keys as :func:`build_tree_rows` +
    :func:`row_attention`.
    """
    _, heads, head_dim = q.shape
    bs, top_k, _ = ancestors.shape
    launch = RowsLaunch(splits=program_splits(bs * heads, key_cache.device), block_n=STEP_BLOCK)
    rows_p = rows_padded(top_k)
    part = torch.empty(bs, heads, launch.splits + 1, rows_p, head_dim, dtype=torch.float32, device=q.device)
    part_lse = torch.empty(bs, heads, launch.splits + 1, rows_p, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    rows_attention(
        q,
        key_cache,
        None,
        out,
        part,
        part_lse,
        block_table,
        committed,
        ancestors,
        rows_per_request=top_k,
        step=step,
        window=window,
        scale=scale,
        launch=launch,
        table_page=table_block,
    )
    return out
