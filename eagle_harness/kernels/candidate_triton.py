# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Triton kernels of the EAGLE-4 candidate head: shard top-k (chunk + merge) and the candidate tail.

Portable fallback of the native CUDA kernels (``csrc/candidate.cu``) and the only implementation on AMD GPUs and
NVIDIA before sm_90. Bit-identical to the torch ops named below on NVIDIA; no engine imports.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# EAGLE4 efficient lm_head candidate path (draft step tail).
#
# These kernels replace, bit for bit, the torch ops the draft step used to run
# after the low-rank scores (launchers: candidate_ops.shard_topk / candidate_tail):
#   torch.topk(scores, K, sorted=False)                       -> shard top-K
#   torch.topk(gathered, K, sorted=False) + gather + .float() -> candidate select
#   torch.softmax(fp32, -1) -> torch.topk(k) / torch.max -> gather -> top-k tail
# "Bit for bit" includes torch's ordering: torch.topk(sorted=False) emits the
# elements whose radix key is strictly above the K-th largest key in ascending
# position order, then the K-th-key ties in ascending position order until K
# are taken; torch.softmax over a <= 1024-wide fp32 row is the warp-persistent
# softmax (lane l sums elements l, l+32, ... in order, then an xor butterfly);
# torch.topk(sorted=True) with k <= 32 runs the gathered k through a 32-wide
# bitonic network (SmallBitonicSort) that is not stable on exact ties.
# ---------------------------------------------------------------------------


@triton.jit
def _eagle4_radix_key(x, KEY_BITS: tl.constexpr):
    # torch.topk's order-preserving unsigned key (TopKTypeConfig::convert); NaN
    # maps to the maximum. int32 for 16-bit floats, int64 for fp32, so that -1
    # is free as the empty-slot sentinel.
    if KEY_BITS == 16:
        b = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        key = tl.where((b & 0x8000) != 0, b ^ 0xFFFF, b ^ 0x8000)
        key = tl.where(x != x, 0xFFFF, key)
    else:
        b = x.to(tl.uint32, bitcast=True)
        key = tl.where((b >> 31) != 0, ~b, b | 0x80000000)
        key = tl.where(x != x, 0xFFFFFFFF, key)
        key = key.to(tl.int64)
    return key


@triton.jit
def _eagle4_radix_key_to_fp32(key):
    # Inverse of _eagle4_radix_key for KEY_BITS == 32.
    kb = key.to(tl.uint32)
    b = tl.where((kb >> 31) != 0, kb ^ 0x80000000, ~kb)
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _eagle4_kth_key(key, K, KEY_BITS: tl.constexpr):
    # The K-th largest key (largest t with count(key >= t) >= K), two bits per
    # step: a step's three candidate counts (and, in the first step, the valid
    # count) share one reduction as 16-bit fields of an int64. Rows must hold
    # < 32768 keys. -1 when fewer than K keys are valid (>= 0).
    kth = tl.max(key, axis=0) * 0
    n_valid = kth.to(tl.int64)
    for i in tl.static_range(KEY_BITS // 2):
        c1 = kth | (tl.full([], 1, key.dtype) << (KEY_BITS - 2 - 2 * i))
        c2 = kth | (tl.full([], 2, key.dtype) << (KEY_BITS - 2 - 2 * i))
        c3 = kth | (tl.full([], 3, key.dtype) << (KEY_BITS - 2 - 2 * i))
        packed = (
            (key >= c1).to(tl.int64)
            + ((key >= c2).to(tl.int64) << 16)
            + ((key >= c3).to(tl.int64) << 32)
        )
        if i == 0:
            packed += (key >= 0).to(tl.int64) << 48
        counts = tl.sum(packed, axis=0)
        if i == 0:
            n_valid = counts >> 48
        kth = tl.where(
            ((counts >> 32) & 0xFFFF) >= K,
            c3,
            tl.where(((counts >> 16) & 0xFFFF) >= K, c2, tl.where((counts & 0xFFFF) >= K, c1, kth)),
        )
    return tl.where(n_valid >= K, kth, -1)


@triton.jit
def _eagle4_split_at_kth(key, kth, valid):
    # Keys above the K-th key and K-th-key ties, with their exclusive ranks in
    # position order from one packed cumsum (16-bit fields; < 32768 keys).
    gt = (key > kth) & valid
    eq = (key == kth) & valid
    flags = gt.to(tl.int32) + (eq.to(tl.int32) << 16)
    before = tl.cumsum(flags, axis=0) - flags
    total = tl.sum(flags, axis=0)
    return gt, eq, before & 0xFFFF, before >> 16, total & 0xFFFF, total >> 16


@triton.jit
def _eagle4_topk_slots(key, K, KEY_BITS: tl.constexpr):
    # torch.topk(sorted=False) placement: returns (selected, dest) where the
    # selected keys are the K largest (ties at the K-th key resolved to the
    # smallest positions) and dest is the output slot: keys above the K-th key
    # first, in position order, then the K-th-key ties in position order.
    # `key` carries -1 in empty positions and must hold >= K valid keys.
    kth = _eagle4_kth_key(key, K, KEY_BITS)
    gt, eq, gt_rank, eq_rank, n_gt, n_eq = _eagle4_split_at_kth(key, kth, key >= 0)
    selected = gt | (eq & (eq_rank < K - n_gt))
    dest = tl.where(gt, gt_rank, n_gt + eq_rank)
    return selected, dest


@triton.jit
def eagle4_shard_topk_chunk_kernel(
    scores_ptr,       # [M, N] low-rank scores of this vocab shard (16-bit float)
    stride_sm,
    N,
    cand_keys_ptr,    # [M, NB, K] int32 radix keys, -1 = empty
    cand_idx_ptr,     # [M, NB, K] int32 shard-local positions
    K: tl.constexpr,
    CH: tl.constexpr,
    KEY_BITS: tl.constexpr,
):
    """Stage 1 of the shard top-k: the top K radix keys of each vocab chunk."""
    # Stage 1 of the shard top-K: every CH-wide chunk keeps its own top-K under
    # the row order (key desc, position asc), compacted in position order. That
    # set contains every chunk element of the row's top-K, so stage 2 only
    # re-selects among NB*K candidates that are already in position order.
    m = tl.program_id(0)
    c = tl.program_id(1)
    nb = tl.num_programs(1)
    offs = c * CH + tl.arange(0, CH)
    valid = offs < N
    x = tl.load(scores_ptr + m.to(tl.int64) * stride_sm + offs, mask=valid, other=0.0)
    key = tl.where(valid, _eagle4_radix_key(x, KEY_BITS), -1)
    kth = _eagle4_kth_key(key, K, KEY_BITS)
    gt, eq, gt_rank, eq_rank, n_gt, n_eq = _eagle4_split_at_kth(key, kth, valid)
    need = K - n_gt
    selected = gt | (eq & (eq_rank < need))
    pos = gt_rank + tl.minimum(eq_rank, need)
    base = (m * nb + c).to(tl.int64) * K
    tl.store(cand_keys_ptr + base + pos, key, mask=selected)
    tl.store(cand_idx_ptr + base + pos, offs, mask=selected)
    slots = tl.arange(0, K)
    tl.store(
        cand_keys_ptr + base + slots,
        tl.full([K], -1, tl.int32),
        mask=slots >= n_gt + tl.minimum(n_eq, need),
    )


@triton.jit
def eagle4_shard_topk_merge_kernel(
    cand_keys_ptr,    # [M, NBK] int32 (stage-1 output, position ordered)
    cand_idx_ptr,     # [M, NBK] int32
    NBK,
    local_idx_ptr,    # [M, K] int64 shard-local ids (fused_gather_dot input)
    stride_lm,
    global_idx_ptr,   # [M, K] int64 global token ids (local + vocab_start)
    stride_gm,
    vocab_start,
    K: tl.constexpr,
    NBK_P: tl.constexpr,
    KEY_BITS: tl.constexpr,
):
    """Stage 2 of the shard top-k: merge the chunks' keys into the shard's top K ids."""
    # Stage 2: the row's top-K among the stage-1 candidates, placed exactly
    # where torch.topk(sorted=False) places them.
    m = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, NBK_P)
    valid = offs < NBK
    key = tl.load(cand_keys_ptr + m * NBK + offs, mask=valid, other=-1)
    idx = tl.load(cand_idx_ptr + m * NBK + offs, mask=valid, other=0).to(tl.int64)
    selected, dest = _eagle4_topk_slots(key, K, KEY_BITS)
    tl.store(local_idx_ptr + m * stride_lm + dest, idx, mask=selected)
    tl.store(global_idx_ptr + m * stride_gm + dest, idx + vocab_start, mask=selected)


@triton.jit
def _eagle4_softmax_fp32(v, P: tl.constexpr):
    # torch.softmax(fp32 row, -1) for a row of <= 1024 elements, bit exact:
    # softmax_warp_forward keeps elements l, l+32, l+64, ... in lane l, sums
    # them in that order from 0, reduces lanes with an xor butterfly
    # (16, 8, 4, 2, 1) and divides with IEEE division. `v` is the row padded to
    # P (power of two >= 32) with -inf.
    ITER: tl.constexpr = P // 32
    e = libdevice.exp(v - tl.max(v, axis=0))
    e2 = tl.reshape(e, [ITER, 32])
    it = tl.arange(0, ITER)[:, None]
    lane_sum = tl.zeros([32], tl.float32)
    for i in tl.static_range(ITER):
        lane_sum = lane_sum + tl.sum(tl.where(it == i, e2, 0.0), axis=0)
    s16 = tl.sum(tl.reshape(lane_sum, [2, 16]), axis=0)
    s8 = tl.sum(tl.reshape(s16, [2, 8]), axis=0)
    s4 = tl.sum(tl.reshape(s8, [2, 4]), axis=0)
    s2 = tl.sum(tl.reshape(s4, [2, 2]), axis=0)
    total = tl.sum(s2, axis=0)
    return tl.math.div_rn(e, tl.broadcast_to(total, e.shape))


@triton.jit
def _eagle4_bitonic_round(g, STRIDE: tl.constexpr, HALF: tl.constexpr):
    # One bitonicSwap round of torch's SmallBitonicSort (32 slots, descending
    # GTOp) on packed entries (key << 11 | slot << 1 | valid): thread tx pairs
    # pos = 2*tx - (tx & (STRIDE-1)) with pos + STRIDE and swaps when
    # ((kA > kB && validA) || !validB) == ((tx & HALF) != 0); HALF == 0 is the
    # final merge (flag always false).
    NO: tl.constexpr = 32 // (2 * STRIDE)
    side = tl.arange(0, 2)[None, :, None]
    g3 = tl.reshape(g, [NO, 2, STRIDE])
    a = tl.sum(tl.where(side == 0, g3, 0), axis=1)
    b = tl.sum(tl.where(side == 1, g3, 0), axis=1)
    tx = tl.arange(0, NO)[:, None] * STRIDE + tl.arange(0, STRIDE)[None, :]
    flag = (tx & HALF) != 0
    swap = (((a >> 11) > (b >> 11)) & ((a & 1) != 0)) | ((b & 1) == 0)
    do = swap == flag
    return tl.reshape(
        tl.where(side == 0, tl.where(do, b, a)[:, None, :], tl.where(do, a, b)[:, None, :]),
        [32],
    )


@triton.jit
def _eagle4_small_bitonic_desc(g):
    # torch SmallBitonicSort<32> descending: sizes 2..16, then the final merge.
    g = _eagle4_bitonic_round(g, 1, 1)
    g = _eagle4_bitonic_round(g, 2, 2)
    g = _eagle4_bitonic_round(g, 1, 2)
    g = _eagle4_bitonic_round(g, 4, 4)
    g = _eagle4_bitonic_round(g, 2, 4)
    g = _eagle4_bitonic_round(g, 1, 4)
    g = _eagle4_bitonic_round(g, 8, 8)
    g = _eagle4_bitonic_round(g, 4, 8)
    g = _eagle4_bitonic_round(g, 2, 8)
    g = _eagle4_bitonic_round(g, 1, 8)
    g = _eagle4_bitonic_round(g, 16, 0)
    g = _eagle4_bitonic_round(g, 8, 0)
    g = _eagle4_bitonic_round(g, 4, 0)
    g = _eagle4_bitonic_round(g, 2, 0)
    g = _eagle4_bitonic_round(g, 1, 0)
    return g


@triton.jit
def _eagle4_probs_topk(probs, slot, n, TOPK: tl.constexpr, P: tl.constexpr):
    # fast_topk(probs, TOPK) for one row (slots < n valid, P <= 1024): returns
    # (probs, slots) of the winners on a 32-wide vector, the first TOPK entries
    # meaningful. TOPK == 1 is torch.max(dim) (first maximal index); TOPK > 1 is
    # torch.topk(sorted=True): gather in topk(sorted=False) order, then the
    # SmallBitonicSort network, ties included.
    valid = slot < n
    pkey = _eagle4_radix_key(probs, 32)
    # (key desc, slot asc) as one int64, so every max is unique
    ranked = tl.where(valid, (pkey << 10) | (P - 1 - slot), -1)
    d = tl.arange(0, 32)
    if TOPK == 1:
        best = tl.max(ranked, axis=0)
        out_key = tl.where(d == 0, best >> 10, 0)
        out_slot = tl.where(d == 0, (P - 1 - (best & 1023)).to(tl.int32), 0)
    else:
        for i in tl.static_range(TOPK):
            best = tl.max(ranked, axis=0)
            ranked = tl.where(ranked == best, -1, ranked)
        kth = best >> 10
        gt, eq, gt_rank, eq_rank, n_gt, n_eq = _eagle4_split_at_kth(pkey, kth, valid)
        selected = gt | (eq & (eq_rank < TOPK - n_gt))
        dest = tl.where(gt, gt_rank, n_gt + eq_rank)
        match = (dest[None, :] == d[:, None]) & selected[None, :]
        packed = (pkey << 11) | (slot.to(tl.int64) << 1) | 1
        g = _eagle4_small_bitonic_desc(tl.sum(tl.where(match, packed[None, :], 0), axis=1))
        out_key = g >> 11
        out_slot = ((g >> 1) & 1023).to(tl.int32)
    return _eagle4_radix_key_to_fp32(out_key), out_slot


@triton.jit
def eagle4_candidate_tail_kernel(
    cand_logits_ptr,    # [R, M, NCL] candidate logits (16-bit or fp32, col stride 1)
    stride_cl_r,
    stride_cl_m,
    cand_tokens_ptr,    # [R, M, NCL] int64 or int32 global token ids
    stride_ct_r,
    stride_ct_m,
    logits_out_ptr,     # [M, NC] fp32 next_token_logits
    stride_lo_m,
    tokens_out_ptr,     # [M, NC] int64 candidate ids (R > 1 only)
    stride_to_m,
    topk_probs_ptr,     # [M, TOPK] fp32
    stride_tp_m,
    topk_tokens_ptr,    # [M, TOPK] int64
    stride_tt_m,
    R: tl.constexpr,
    NCL: tl.constexpr,
    NC: tl.constexpr,
    CP: tl.constexpr,
    P: tl.constexpr,
    KEY_BITS: tl.constexpr,
    TOPK: tl.constexpr,
):
    """Candidate tail kernel: merge of the ranks' candidates, softmax and top-k of a row."""
    # One row per program. R > 1: keep the NC best of the R*NCL gathered
    # candidates in torch.topk(sorted=False) order and write them as fp32
    # logits + ids; R == 1: just widen the NC logits to fp32. Then, for
    # TOPK > 0, the draft tail softmax + top-k + id gather on those NC logits.
    m = tl.program_id(0).to(tl.int64)
    slot = tl.arange(0, P)
    in_row = slot < NC
    if R > 1:
        offs = tl.arange(0, CP)
        valid = offs < R * NCL
        r = offs // NCL
        j = offs % NCL
        x = tl.load(
            cand_logits_ptr + r * stride_cl_r + m * stride_cl_m + j, mask=valid, other=0.0
        )
        key = tl.where(valid, _eagle4_radix_key(x, KEY_BITS), -1)
        selected, dest = _eagle4_topk_slots(key, NC, KEY_BITS)
        tokens = tl.load(
            cand_tokens_ptr + r * stride_ct_r + m * stride_ct_m + j, mask=selected, other=0
        )
        tl.store(logits_out_ptr + m * stride_lo_m + dest, x.to(tl.float32), mask=selected)
        tl.store(tokens_out_ptr + m * stride_to_m + dest, tokens.to(tl.int64), mask=selected)
        row_tokens_ptr = tokens_out_ptr + m * stride_to_m
        # The compaction went through global memory: make it visible to the
        # whole CTA and read it back in slot order (from L2, bypassing L1).
        tl.debug_barrier()
        v = tl.load(
            logits_out_ptr + m * stride_lo_m + slot,
            mask=in_row, other=float("-inf"), cache_modifier=".cg",
        )
    else:
        v = tl.load(
            cand_logits_ptr + m * stride_cl_m + slot, mask=in_row, other=float("-inf")
        ).to(tl.float32)
        tl.store(logits_out_ptr + m * stride_lo_m + slot, v, mask=in_row)
        row_tokens_ptr = cand_tokens_ptr + m * stride_ct_m
    if TOPK > 0:
        probs = _eagle4_softmax_fp32(v, P)
        top_probs, top_slots = _eagle4_probs_topk(probs, slot, NC, TOPK, P)
        d = tl.arange(0, 32)
        out = d < TOPK
        top_tokens = tl.load(row_tokens_ptr + top_slots, mask=out, other=0, cache_modifier=".cg").to(tl.int64)
        tl.store(topk_probs_ptr + m * stride_tp_m + d, top_probs, mask=out)
        tl.store(topk_tokens_ptr + m * stride_tt_m + d, top_tokens, mask=out)
