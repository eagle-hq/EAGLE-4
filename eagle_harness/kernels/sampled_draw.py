# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
"""One-launch recorded-q proposal draw: softmax over the candidates, Gumbel-top-K children, q recorded in place.

Replaces the torch chain of :func:`eagle_harness.core.sampled_tree.draw_children` (softmax, rand, two logs, two
top-k, gather, five slice copies) and the per-step TP broadcast of its uniforms. The uniforms come from counter-based
Philox keyed by ``(seed, round, launch, element)``: every TP rank holds the same seed (sglang broadcasts
``random_seed``) and replays the same graphs, so every rank draws the same tree without a collective. With
``uniforms`` given (tests), the draw equals :func:`draw_children`'s on the same uniforms.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _sample_children(
    LOGITS,
    TOKENS,
    ROUND,
    UNIF,
    Q,
    QIDS,
    POS,
    OUT_TOKENS,
    SCORES,
    REQ,
    FILTERS,
    seed,
    launch,
    stride_l,
    stride_t,
    rows_per_req,
    q_batch_stride,
    q_slot,
    pos_batch_stride,
    pos_start,
    NC: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_UNIFORMS: tl.constexpr,
    USE_FILTERS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    req = row // rows_per_req
    local = row - req * rows_per_req
    store = tl.load(REQ + req)  # the request's row in the recorded-q stores
    c = tl.arange(0, BLOCK)
    mask = c < NC
    logits = tl.load(LOGITS + row * stride_l + c, mask=mask, other=float("-inf")).to(tl.float32)
    tokens = tl.load(TOKENS + row * stride_t + c, mask=mask, other=0).to(tl.int64)
    if USE_FILTERS:
        # The request's sampling settings (see filter_proposal_logits): 1/temperature, top-k, top-p, min-p.
        inv_t = tl.load(FILTERS + store * 4)
        logits = logits * inv_t
    peak = tl.max(logits, 0)
    e = tl.exp(logits - peak)
    e = tl.where(mask, e, 0.0)
    total = tl.sum(e, 0)
    probs = e / total
    keep = mask
    if USE_FILTERS:
        top_k = tl.load(FILTERS + store * 4 + 1)
        top_p = tl.load(FILTERS + store * 4 + 2)
        min_p = tl.load(FILTERS + store * 4 + 3)
        # Rank and the mass ranked above each candidate, ties broken by index, chunk by chunk.
        rank = tl.zeros((BLOCK,), dtype=tl.float32)
        above = tl.zeros((BLOCK,), dtype=tl.float32)
        for j0 in tl.static_range(0, BLOCK, CHUNK):
            cj = j0 + tl.arange(0, CHUNK)
            lj = tl.load(LOGITS + row * stride_l + cj, mask=cj < NC, other=float("-inf")).to(tl.float32) * inv_t
            pj = tl.where(cj < NC, tl.exp(lj - peak), 0.0) / total
            higher = (pj[None, :] > probs[:, None]) | ((pj[None, :] == probs[:, None]) & (cj[None, :] < c[:, None]))
            higher = higher & (cj[None, :] != c[:, None]) & (cj[None, :] < NC)
            rank += tl.sum(higher.to(tl.float32), 1)
            above += tl.sum(tl.where(higher, pj[None, :], 0.0), 1)
        # Apply the request's joint filters to the draft probabilities, not the target's token support.
        keep = (rank < top_k) & (above < top_p) & (probs >= min_p * tl.max(probs, 0))
        keep = mask & keep
        kept = tl.where(keep, probs, 0.0)
        probs = tl.where(tl.sum(keep.to(tl.int32), 0) == NC, probs, kept / tl.sum(kept, 0))
    if USE_UNIFORMS:
        u = tl.load(UNIF + row * NC + c, mask=mask, other=0.5)
    else:
        rnd = tl.load(ROUND)
        element = (row * NC + c).to(tl.uint32)
        r0, _r1, _r2, _r3 = tl.philox(
            seed,
            element,
            tl.full((BLOCK,), rnd, tl.uint32),
            tl.full((BLOCK,), launch, tl.uint32),
            tl.zeros((BLOCK,), tl.uint32),
        )
        u = tl.uint_to_uniform_float(r0)
    # rand is [0, 1): clamping only its possible zero avoids log(0); matches draw_children.
    u = tl.maximum(u, 1.1754944e-38)
    keys = tl.log(probs) - tl.log(-tl.log(u))
    # A filtered candidate (q = 0) ranks below every kept one but stays distinct from the drawn and padding slots, so
    # draws past the kept support land on distinct inert candidates.
    keys = tl.where(keep, keys, -1e30)
    keys = tl.where(mask, keys, float("-inf"))
    qbase = Q + store * q_batch_stride + (q_slot + local) * NC
    tl.store(qbase + c, probs, mask=mask)
    tl.store(QIDS + store * q_batch_stride + (q_slot + local) * NC + c, tokens, mask=mask)
    pbase = POS + store * pos_batch_stride + pos_start + local * K
    ranked = tl.where(mask, probs, -1.0)
    for j in tl.static_range(K):
        best = tl.max(keys, 0)
        # the first index attaining the max (a tie needs equal fp keys; torch.topk's tie order is unspecified too)
        pos = tl.min(tl.where(keys == best, c, BLOCK), 0)
        tl.store(pbase + j, pos.to(tl.int64))
        token = tl.sum(tl.where(c == pos, tokens, 0), 0)
        tl.store(OUT_TOKENS + row * K + j, token)
        keys = tl.where(c == pos, float("-inf"), keys)
        # The planner's score is q's j-th order statistic, never the draw's own q (that would bias the pruning).
        top = tl.max(ranked, 0)
        tl.store(SCORES + row * K + j, top)
        first = tl.min(tl.where(ranked == top, c, BLOCK), 0)
        ranked = tl.where(c == first, -1.0, ranked)


def sample_children(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    count: int,
    *,
    seed: int,
    launch: int,
    counter: torch.Tensor,
    q: torch.Tensor,
    q_ids: torch.Tensor,
    positions: torch.Tensor,
    slot: int,
    start: int,
    rows_per_req: int,
    uniforms: Optional[torch.Tensor] = None,
    req_index: Optional[torch.Tensor] = None,
    filters: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gumbel-top-``count`` ordered draws per row, recorded in place; returns ``(scores [rows, count] fp32, tokens
    [rows, count] i64)``.

    ``q``/``q_ids`` are ``[bs, slots, NC]`` stores: request ``b``'s row ``i`` (``i < rows_per_req``) is written at
    slot ``slot + i``. ``positions`` is ``[bs, nodes]``: child ``j`` of that row is written at ``start + i * count +
    j``. ``counter`` is a device int32 round counter (advanced by the caller at step 0); ``launch`` distinguishes the
    draws of one round. The scores are q's order statistics. ``req_index`` (int64 ``[bs]``) maps request ``b`` to its
    row in the stores (default: the batch position). ``filters`` (fp32 ``[store rows, 4]``: 1/temperature, top-k,
    top-p, min-p per store row) apply the request's sampling settings to q; the draw then
    equals :func:`draw_children`'s on :func:`~eagle_harness.core.sampled_tree.filter_proposal_logits`.
    """
    rows, nc = logits.shape
    if token_ids.shape != logits.shape or not 1 <= count <= nc or rows % rows_per_req:
        raise ValueError("expected matching [rows, candidates] logits/ids, count within the width, whole requests")
    bs = rows // rows_per_req
    if req_index is None:
        req_index = torch.arange(bs, dtype=torch.int64, device=logits.device)
    if q.shape != q_ids.shape or q.dim() != 3 or q.shape[1] < slot + rows_per_req or q.shape[2] != nc:
        raise ValueError("q store must be [bs, slots, candidates] with room for the written slots")
    if positions.dim() != 2 or positions.shape[0] != q.shape[0] or positions.shape[1] < start + rows_per_req * count:
        raise ValueError("positions must be [bs, nodes] with room for the written children")
    if not (q.is_contiguous() and q_ids.is_contiguous() and positions.is_contiguous()):
        raise ValueError("recorded-q stores must be contiguous")
    if filters is not None and (filters.dtype != torch.float32 or filters.dim() != 2 or filters.shape[1] != 4
                                or not filters.is_contiguous()):
        raise ValueError("filters must be a contiguous fp32 [rows, 4] store (1/temperature, top-k, top-p, min-p)")
    device = logits.device
    scores = torch.empty((rows, count), dtype=torch.float32, device=device)
    tokens = torch.empty((rows, count), dtype=torch.int64, device=device)
    if rows == 0:
        return scores, tokens
    _sample_children[(rows,)](
        logits,
        token_ids,
        counter,
        uniforms if uniforms is not None else logits,
        q,
        q_ids,
        positions,
        tokens,
        scores,
        req_index,
        filters if filters is not None else logits,
        int(seed) & 0xFFFFFFFF,
        int(launch),
        logits.stride(0),
        token_ids.stride(0),
        rows_per_req,
        q.stride(0),
        slot,
        positions.stride(0),
        start,
        NC=nc,
        K=count,
        BLOCK=triton.next_power_of_2(nc),
        USE_UNIFORMS=uniforms is not None,
        USE_FILTERS=filters is not None,
        CHUNK=min(32, triton.next_power_of_2(nc)),
        num_warps=4 if nc <= 256 else 8,
    )
    return scores, tokens
