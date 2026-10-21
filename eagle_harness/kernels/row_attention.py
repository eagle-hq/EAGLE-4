# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Attention of decode-time draft rows whose keys are mostly shared (both engines' draft steps, sglang's draft extend).

A request's rows see one range of its committed keys, read through its page table, and the rows of a draft step
also see their own tree ancestors:

- draft step ``step`` (``rows`` frontier rows per request, all at position ``seq_len + step - 1``): the committed keys
  of the window ending at the round's last committed position, ``[seq_len - window, seq_len - 1]`` (frozen for the
  round, as the draft was trained), the same for every row, plus the row's ``step`` ancestors (itself last);
- draft extend (``step == 0``; row ``j`` at ``seq_len + j``): the keys ``[pos - window + 1, pos]`` of its own position.

One program per (request, head, key split) loads each key and value of the request's range once for all its rows
(a ``tl.dot`` over the rows) and masks the keys outside a row's own range; a draft step's ancestors are one more
split. A second kernel merges the splits by their log-sum-exp. Lengths are read on the device: no host planning,
CUDA-graph safe.

The engines differ only in how a key position becomes a KV address, which the launcher takes as arguments: the page
of position ``p`` is ``table[row, p // table_page]`` (``row``: ``table_rows[b]``, or ``b`` without ``table_rows``),
its slot ``page * table_page + p % table_page``, and a slot ``s`` sits at ``(s // cache_page, s % cache_page, head)``
of the KV cache.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from eagle_harness.kernels.online_softmax import online_block


@dataclass(frozen=True)
class RowsLaunch:
    """Launch config of :func:`rows_attention`: key splits of a request's shared range (a draft step's tree
    ancestors are one more), keys per iteration, Triton warps and pipeline stages."""

    splits: int
    block_n: int
    num_warps: int = 4
    num_stages: int = 3


def rows_padded(rows_per_request: int) -> int:
    """Rows of a request's ``tl.dot`` tile: a power of two, at least 16 (``tl.dot``'s smallest tile)."""
    return max(16, triton.next_power_of_2(rows_per_request))


@functools.cache
def program_splits(programs: int, device: torch.device) -> int:
    """Key splits per (request, head) program: the power of two that gives every SM at least one program."""
    if device.type != "cuda":
        return 4
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return triton.next_power_of_2(triton.cdiv(sms, programs))


@triton.jit
def window_start(pos, window):
    """First key position a query at ``pos`` sees (``window <= 0``: the whole context)."""
    start = pos * 0
    if window > 0:
        start = tl.maximum(pos - window + 1, 0)
    return start


@triton.jit
def _kv_rows(kv, slot, h, d, mask, kv_strides, cache_page):
    """Keys and values ``[len(slot), HEAD_DIM]`` of head ``h`` at the KV cache slots ``slot`` (int64)."""
    k_cache, v_cache = kv
    key_strides, value_strides = kv_strides
    stride_page, stride_slot, stride_head = key_strides
    at = (slot // cache_page) * stride_page + (slot % cache_page) * stride_slot + h * stride_head
    k = tl.load(k_cache + at[:, None] + d[None, :], mask=mask[:, None], other=0.0)
    if v_cache is None:  # packed [K, V] in the last dimension
        v = tl.load(k_cache + at[:, None] + d.shape[0] + d[None, :], mask=mask[:, None], other=0.0)
    else:
        v_page, v_slot, v_head = value_strides
        v_at = (slot // cache_page) * v_page + (slot % cache_page) * v_slot + h * v_head
        v = tl.load(v_cache + v_at[:, None] + d[None, :], mask=mask[:, None], other=0.0)
    return k, v


@triton.jit
def _rows_attention_kernel(
    q,  # [bs * ROWS, heads, HEAD_DIM]
    kv,  # (keys, values): KV cache views, a slot at (slot // cache_page, slot % cache_page, head)
    parts,  # (out [bs, heads, NUM_SPLITS + 1, ROWS_P, HEAD_DIM] fp32, log-sum-exp [..., ROWS_P] fp32, log2 domain)
    table,  # page table of the key positions, a row per request
    table_rows,  # [bs] the requests' rows of ``table``, or None: row b
    seq_lens,  # [bs] committed length
    ancestors,  # [bs, ROWS, MAX_DEPTH] int32 KV slots of each row's tree ancestors (a draft step)
    q_strides,  # (row, head)
    kv_strides,  # (page, slot in a page, head) for keys and for values
    paging,  # (table row stride, table page, cache page)
    step,
    window,
    scale_log2,
    num_heads,
    QUERY_GROUP_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    ROWS_P: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EXTEND: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    ANC_P: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    kv_head = h // QUERY_GROUP_SIZE
    split = tl.program_id(2)
    stride_table, table_page, cache_page = paging
    stride_q_row, stride_q_head = q_strides
    r = tl.arange(0, ROWS_P)
    d = tl.arange(0, HEAD_DIM)
    row_ok = r < ROWS
    seq_len = tl.load(seq_lens + b).to(tl.int32)
    if EXTEND:  # row j at seq_len + j sees its window up to itself
        pos = seq_len + r
        lo = window_start(pos, window)
        hi = pos
    qv = tl.load(
        q + (b * ROWS + r)[:, None] * stride_q_row + h * stride_q_head + d[None, :], mask=row_ok[:, None], other=0.0
    )
    m = tl.full((ROWS_P,), float("-inf"), tl.float32)
    denom = tl.zeros((ROWS_P,), tl.float32)
    acc = tl.zeros((ROWS_P, HEAD_DIM), tl.float32)
    if split < NUM_SPLITS:
        if EXTEND:
            first = tl.min(tl.where(row_ok, lo, 2147483647), 0)
            last = tl.max(tl.where(row_ok, hi, -1), 0)
        else:  # every row sees the committed keys of the window frozen at the last committed position
            first = window_start(seq_len - 1, window)
            last = seq_len - 1
        if table_rows is None:
            row_pages = table + b * stride_table
        else:
            row_pages = table + tl.load(table_rows + b).to(tl.int64) * stride_table
        per_split = tl.cdiv(tl.cdiv(last + 1 - first, NUM_SPLITS), BLOCK_N) * BLOCK_N
        start = first + split * per_split
        end = tl.minimum(start + per_split, last + 1)
        for block in range(start, end, BLOCK_N):
            p = block + tl.arange(0, BLOCK_N)
            in_range = p < end
            page = tl.load(row_pages + p // table_page, mask=in_range, other=0).to(tl.int64)
            k, v = _kv_rows(kv, page * table_page + p % table_page, kv_head, d, in_range, kv_strides, cache_page)
            ok = row_ok[:, None] & in_range[None, :]
            if EXTEND:
                ok = ok & (p[None, :] >= lo[:, None]) & (p[None, :] <= hi[:, None])
            m, denom, acc = online_block(qv, k, v, ok, m, denom, acc, scale_log2)
    else:  # a draft step's tree ancestors: column c is ancestor c % MAX_DEPTH of row c // MAX_DEPTH
        c = tl.arange(0, ANC_P)
        owner = c // MAX_DEPTH
        valid = (owner < ROWS) & (c % MAX_DEPTH < step)
        slot = tl.load(ancestors + (b * ROWS + owner) * MAX_DEPTH + c % MAX_DEPTH, mask=valid, other=0).to(tl.int64)
        k, v = _kv_rows(kv, slot, kv_head, d, valid, kv_strides, cache_page)
        ok = valid[None, :] & (owner[None, :] == r[:, None])
        m, denom, acc = online_block(qv, k, v, ok, m, denom, acc, scale_log2)
    part_out, part_lse = parts
    part = ((b * num_heads + h) * (NUM_SPLITS + 1) + split) * ROWS_P + r
    denom_safe = tl.maximum(denom, 1e-30)
    tl.store(part_lse + part, tl.where(denom > 0, m + tl.log2(denom_safe), float("-inf")))
    tl.store(part_out + part[:, None] * HEAD_DIM + d[None, :], acc / denom_safe[:, None])


@triton.jit
def _merge_splits_kernel(
    parts,  # (out, log-sum-exp) of _rows_attention_kernel
    out,  # [bs * ROWS, heads, HEAD_DIM]
    out_strides,  # (row, head)
    num_heads,
    ROWS: tl.constexpr,
    ROWS_P: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLITS_P: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    part_out, part_lse = parts
    stride_o_row, stride_o_head = out_strides
    s = tl.arange(0, SPLITS_P)
    d = tl.arange(0, HEAD_DIM)
    used = s < SPLITS
    part = (((row // ROWS) * num_heads + h) * (NUM_SPLITS + 1) + s) * ROWS_P + row % ROWS
    lse = tl.load(part_lse + part, mask=used, other=float("-inf"))
    top = tl.max(lse, 0)
    weight = tl.where(lse == float("-inf"), 0.0, tl.exp2(lse - top))
    parts_d = tl.load(part_out + part[:, None] * HEAD_DIM + d[None, :], mask=used[:, None], other=0.0)
    merged = tl.sum(weight[:, None] * parts_d, 0) / tl.sum(weight, 0)
    tl.store(out + row * stride_o_row + h * stride_o_head + d, merged.to(out.dtype.element_ty))


def _kv_strides(cache: torch.Tensor) -> tuple:
    """(page, slot, head) strides of a ``[slots, heads, D]`` (page 1) or ``[pages, page, heads, D]`` cache view."""
    if cache.dim() == 3:
        return cache.stride(0), 0, cache.stride(1)
    return cache.stride(0), cache.stride(1), cache.stride(2)


def rows_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor | None,
    out: torch.Tensor,
    part_out: torch.Tensor,
    part_lse: torch.Tensor,
    table: torch.Tensor,
    seq_lens: torch.Tensor,
    ancestors: torch.Tensor | None,
    *,
    rows_per_request: int,
    step: int,
    window: int,
    scale: float,
    launch: RowsLaunch,
    table_rows: torch.Tensor | None = None,
    table_page: int = 1,
) -> None:
    """Attention of ``q`` ``[bs * rows_per_request, heads, D]`` into ``out`` (same shape).

    ``step == 0``: extend rows (row ``j`` of request ``b`` at ``seq_lens[b] + j``, causal); else the rows of draft
    step ``step`` over the committed window plus ``ancestors[b, row, :step]`` (KV slots). ``keys`` / ``values``:
    the layer's KV cache as ``[slots, kv_heads, D]`` or ``[pages, page, kv_heads, D]`` views.
    Consecutive groups of ``heads // kv_heads`` query heads share one KV head.
    For a packed cache, pass ``keys`` with last dimension ``2 * D`` and ``values=None``.
    The key at
    position ``p`` of request ``b`` is slot ``table[table_rows[b], p // table_page] * table_page + p % table_page``
    (row ``b`` without ``table_rows``). ``part_out`` / ``part_lse``: scratch of at least
    ``bs * heads * (launch.splits + 1) * rows_padded(rows_per_request)`` rows.
    """
    rows, heads, head_dim = q.shape
    kv_heads = keys.shape[-2]
    if kv_heads <= 0 or heads % kv_heads:
        raise ValueError(f"query heads ({heads}) must be divisible by KV heads ({kv_heads})")
    kv_strides = (_kv_strides(keys), None if values is None else _kv_strides(values))
    extend = step == 0
    max_depth = 1 if extend else ancestors.shape[-1]
    rows_p = rows_padded(rows_per_request)
    splits = launch.splits + (0 if extend else 1)
    parts = (part_out, part_lse)
    _rows_attention_kernel[(rows // rows_per_request, heads, splits)](
        q,
        (keys, values),
        parts,
        table,
        table_rows,
        seq_lens,
        table if extend else ancestors,  # unread by an extend
        (q.stride(0), q.stride(1)),
        kv_strides,
        (table.stride(0), table_page, 1 if keys.dim() == 3 else keys.shape[1]),
        step,
        window,
        scale * math.log2(math.e),
        heads,
        QUERY_GROUP_SIZE=heads // kv_heads,
        ROWS=rows_per_request,
        ROWS_P=rows_p,
        HEAD_DIM=head_dim,
        EXTEND=extend,
        MAX_DEPTH=max_depth,
        ANC_P=rows_padded(rows_per_request * max_depth),
        NUM_SPLITS=launch.splits,
        BLOCK_N=launch.block_n,
        num_warps=launch.num_warps,
        num_stages=launch.num_stages,
    )
    _merge_splits_kernel[(rows, heads)](
        parts,
        out,
        (out.stride(0), out.stride(1)),
        heads,
        ROWS=rows_per_request,
        ROWS_P=rows_p,
        HEAD_DIM=head_dim,
        SPLITS=splits,
        SPLITS_P=triton.next_power_of_2(splits),
        NUM_SPLITS=launch.splits,
    )
