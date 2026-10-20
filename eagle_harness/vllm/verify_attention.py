# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Attention of a tree verify forward's full-attention layers, split over the committed context.

A request's ``W`` verify rows attend to its committed context in the paged KV cache (the same keys for every row)
and to their own tree ancestors among the ``W`` new tokens. One program per (request, query head, split) loads its
part of the context once for all ``W`` rows (a ``tl.dot`` over the rows); one more program per (request, query
head) takes the new tokens with the rows' ancestor bit masks. A second kernel merges the parts by their
log-sum-exp. The context is cut into as many equal parts as give every SM a program, so a long request at a small
batch reads its keys with the whole GPU, not one SM per KV head.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from eagle_harness.kernels.online_softmax import online_block
from eagle_harness.kernels.row_attention import program_splits

BLOCK = 64  # context keys per inner step


@triton.jit
def _verify_attention_kernel(
    q_ptr,  # [tokens, Hq, D]
    new_kv,  # (keys, values) of the new tokens, [tokens, Hkv, D] each
    cache_kv,  # (keys, values): [blocks, page, Hkv, D] strided views of the paged cache
    block_table_ptr,  # [num_reqs, max_blocks]
    context_lens_ptr,  # [num_reqs] committed tokens of each request
    query_start_loc_ptr,  # [num_reqs + 1]
    masks_ptr,  # [num_reqs, W] int64: bit j set iff row j is row r or one of its ancestors
    part_ptr,  # [num_reqs, Hq, SPLITS + 1, ROWS_P, D] fp32
    part_lse_ptr,  # [num_reqs, Hq, SPLITS + 1, ROWS_P] fp32, log2 domain
    q_strides,  # (token, head)
    new_strides,  # (token, head) of the new keys, then of the new values
    cache_strides,  # (block, slot, head) of the cached keys, then of the cached values
    stride_table,
    page,
    scale_log2,
    HQ: tl.constexpr,
    G: tl.constexpr,  # query heads per KV head
    D: tl.constexpr,
    W: tl.constexpr,
    ROWS_P: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    k_new_ptr, v_new_ptr = new_kv
    k_cache_ptr, v_cache_ptr = cache_kv
    stride_qt, stride_qh = q_strides
    stride_knt, stride_knh, stride_vnt, stride_vnh = new_strides
    stride_kb, stride_ks, stride_kh, stride_vb, stride_vs, stride_vh = cache_strides
    b, h, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kv_head = h // G
    start = tl.load(query_start_loc_ptr + b)
    width = tl.load(query_start_loc_ptr + b + 1) - start
    r = tl.arange(0, ROWS_P)
    d = tl.arange(0, D)
    row_ok = r < width
    qv = tl.load(q_ptr + (start + r)[:, None] * stride_qt + h * stride_qh + d[None, :], mask=row_ok[:, None], other=0.0)
    peak = tl.full((ROWS_P,), float("-inf"), tl.float32)
    denom = tl.zeros((ROWS_P,), tl.float32)
    acc = tl.zeros((ROWS_P, D), tl.float32)
    if split < SPLITS:  # an equal part of the committed context
        context = tl.load(context_lens_ptr + b).to(tl.int32)
        per_split = tl.cdiv(tl.cdiv(context, SPLITS), BLOCK) * BLOCK
        lo = split * per_split
        hi = tl.minimum(lo + per_split, context)
        for first in range(lo, hi, BLOCK):
            p = first + tl.arange(0, BLOCK)
            in_range = p < hi
            block = tl.load(block_table_ptr + b * stride_table + p // page, mask=in_range, other=0).to(tl.int64)
            slot = p % page
            k = tl.load(
                k_cache_ptr + (block * stride_kb + slot * stride_ks + kv_head * stride_kh)[:, None] + d[None, :],
                mask=in_range[:, None],
                other=0.0,
            )
            v = tl.load(
                v_cache_ptr + (block * stride_vb + slot * stride_vs + kv_head * stride_vh)[:, None] + d[None, :],
                mask=in_range[:, None],
                other=0.0,
            )
            peak, denom, acc = online_block(qv, k, v, row_ok[:, None] & in_range[None, :], peak, denom, acc, scale_log2)
    else:  # the new tokens: each row sees itself and its ancestors
        j = tl.arange(0, ROWS_P)
        key_ok = j < width
        k = tl.load(
            k_new_ptr + (start + j)[:, None] * stride_knt + kv_head * stride_knh + d[None, :],
            mask=key_ok[:, None],
            other=0.0,
        )
        v = tl.load(
            v_new_ptr + (start + j)[:, None] * stride_vnt + kv_head * stride_vnh + d[None, :],
            mask=key_ok[:, None],
            other=0.0,
        )
        row_mask = tl.load(masks_ptr + b * W + r, mask=row_ok, other=0)
        visible = ((row_mask[:, None] >> j[None, :].to(tl.int64)) & 1) != 0
        peak, denom, acc = online_block(
            qv, k, v, row_ok[:, None] & key_ok[None, :] & visible, peak, denom, acc, scale_log2
        )
    part = ((b * HQ + h) * (SPLITS + 1) + split) * ROWS_P + r
    safe = tl.maximum(denom, 1e-30)
    tl.store(part_lse_ptr + part, tl.where(denom > 0, peak + tl.log2(safe), float("-inf")))
    tl.store(part_ptr + part[:, None] * D + d[None, :], acc / safe[:, None])


@triton.jit
def _verify_merge_kernel(
    part_ptr,
    part_lse_ptr,
    out_ptr,  # [tokens, Hq, D]
    query_start_loc_ptr,
    stride_ot,
    stride_oh,
    HQ: tl.constexpr,
    D: tl.constexpr,
    ROWS_P: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLITS_P: tl.constexpr,
):
    b, r, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(query_start_loc_ptr + b)
    if r < tl.load(query_start_loc_ptr + b + 1) - start:
        s = tl.arange(0, SPLITS_P)
        d = tl.arange(0, D)
        used = s < SPLITS + 1
        part = ((b * HQ + h) * (SPLITS + 1) + s) * ROWS_P + r
        lse = tl.load(part_lse_ptr + part, mask=used, other=float("-inf"))
        weight = tl.where(lse == float("-inf"), 0.0, tl.exp2(lse - tl.max(lse, 0)))
        parts = tl.load(part_ptr + part[:, None] * D + d[None, :], mask=used[:, None], other=0.0)
        merged = tl.sum(weight[:, None] * parts, 0) / tl.sum(weight, 0)
        tl.store(out_ptr + (start + r) * stride_ot + h * stride_oh + d, merged.to(out_ptr.dtype.element_ty))


def verify_attention(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    masks: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    *,
    max_query_len: int,
) -> torch.Tensor:
    """Tree verify attention into ``out`` ``[tokens, Hq, D]`` (rows past the last request's are left alone).

    ``q`` ``[tokens, Hq, D]``; ``k_new`` / ``v_new`` ``[tokens, Hkv, D]`` the new tokens' keys and values;
    ``key_cache`` / ``value_cache`` ``[blocks, page, Hkv, D]`` views of the paged cache, which holds each request's
    first ``context_lens[b]`` tokens through ``block_table[b]``; ``masks[b, r]`` has bit ``j`` set iff new token
    ``j`` is row ``r`` or one of its ancestors. Every request has at most ``max_query_len`` rows.
    """
    num_reqs = query_start_loc.shape[0] - 1
    heads, head_dim = q.shape[1], q.shape[2]
    rows_p = max(16, triton.next_power_of_2(max_query_len))  # tl.dot's smallest tile
    splits = program_splits(num_reqs * heads, q.device)
    part = torch.empty(num_reqs, heads, splits + 1, rows_p, head_dim, dtype=torch.float32, device=q.device)
    part_lse = torch.empty(num_reqs, heads, splits + 1, rows_p, dtype=torch.float32, device=q.device)
    _verify_attention_kernel[(num_reqs, heads, splits + 1)](
        q,
        (k_new, v_new),
        (key_cache, value_cache),
        block_table,
        context_lens,
        query_start_loc,
        masks,
        part,
        part_lse,
        (q.stride(0), q.stride(1)),
        (k_new.stride(0), k_new.stride(1), v_new.stride(0), v_new.stride(1)),
        (*key_cache.stride()[:3], *value_cache.stride()[:3]),
        block_table.stride(0),
        key_cache.shape[1],
        scale * math.log2(math.e),
        HQ=heads,
        G=heads // key_cache.shape[2],
        D=head_dim,
        W=masks.shape[1],
        ROWS_P=rows_p,
        SPLITS=splits,
        BLOCK=BLOCK,
    )
    _verify_merge_kernel[(num_reqs, max_query_len, heads)](
        part,
        part_lse,
        out,
        query_start_loc,
        out.stride(0),
        out.stride(1),
        HQ=heads,
        D=head_dim,
        ROWS_P=rows_p,
        SPLITS=splits,
        SPLITS_P=triton.next_power_of_2(splits + 1),
    )
    return out
