# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Triton launches for the bookkeeping of a draft tree: its steps (:meth:`DraftTree.step`, work deltas
included) and the ranking and selection of its verify set.

:func:`tree_step`, per batch row (one program): the layer scores ``front_score * prob``, the
layer's tokens, its top-``K`` (the next frontier, ties to the lower node),
the frontier's parents, tokens, hidden rows and ancestor slots, and the new
best-score set (top ``work_cap`` of the old set and the layer). At the tree's
delta slots the same program also writes the cost model's work deltas (the
prefix sums of the best set merged with the layer's best ``j`` scores, for every
verify budget and ``j``), which the torch version spends about ten launches on.
The torch version in :class:`~eagle_harness.core.draft_tree.DraftTree` is the
reference (``tests/test_tree_step_gpu.py``). :func:`tree_begin` seeds depth 1
from the root; :func:`work_deltas` computes the begin slot's deltas. :func:`tree_rank` and
:func:`tree_select` replace the ten or so small torch launches of :meth:`DraftTree.rank_reference` and
the node sort and token gather of :meth:`DraftTree.select`; :func:`select_rows` turns the grown tree into verify rows
with the row of each row's parent (:meth:`DraftTree.verify_rows`; was ~30 torch kernels, ~90 us per round).

The ``treestep`` version (:data:`TREESTEP`, ktest's ``harness=treestep``) keeps every output bit but the deltas' (fp32
rounding): a step and begin are one launch each, deltas and their batch mean included, and the frontier's hidden rows
are copied by programs of their own when the batch is small (H200, served tree: 90 -> 55 us a 12-level round at one
request, 120 -> 67 at four, 915 -> 276 at 512).
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from eagle_harness.kernels import variant

# The tree's merges by rank instead of sorts, the batch mean of the work deltas and begin's deltas in the step's own
# launch, the step's KV slots in one copy.
TREESTEP = variant.name("treestep")


@triton.jit
def _tree_step_kernel(
    front_scores_ptr,  # [bs, K] fp32, read then overwritten
    layer,  # the step's draft outputs: probs [bs * K, K] (top-k probabilities of each frontier row), tokens
    # [bs * K, K] int64 (their token ids), hidden [bs * K, H] (the rows' draft hidden states)
    tree,  # the tree's buffers, see the unpacking below
    layer_strides,  # row strides of probs, tokens, hidden
    tree_strides,  # row strides of scores, node_tokens, parents, front_hidden, best, deltas
    H,
    start,
    pstart,
    K: tl.constexpr,
    K_P: tl.constexpr,
    KK_P: tl.constexpr,
    D: tl.constexpr,
    D_P: tl.constexpr,
    CAP: tl.constexpr,
    MERGE_P: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DELTAS: tl.constexpr,
    V: tl.constexpr,
    J_P: tl.constexpr,
    DELTA_P: tl.constexpr,
):
    probs_ptr, tokens_ptr, hidden_ptr = layer
    (
        scores_ptr,  # [bs, capacity] fp32 node scores
        node_tokens_ptr,  # [bs, capacity] int64 node tokens
        parents_ptr,  # [bs, parent_capacity] int64
        front_tokens_ptr,  # [bs * K] int64
        front_hidden_ptr,  # [bs * K, H]
        ancestors_ptr,  # [bs, K, D] int32, permuted in place
        best_ptr,  # [bs, CAP] fp32, read then overwritten
        best_prev_ptr,  # [bs, CAP] fp32
        layer_top_ptr,  # [bs, K] fp32
        deltas_ptr,  # [bs or 1, V, K + 1] fp32 work deltas of this row (DELTAS only)
    ) = tree
    probs_stride, tokens_stride, hidden_stride = layer_strides
    scores_stride, node_tokens_stride, parents_stride, front_hidden_stride, best_stride, deltas_stride = tree_strides
    b = tl.program_id(0).to(tl.int64)
    # ---- every read of in-place buffers first
    n = tl.arange(0, KK_P)
    valid = n < K * K
    row = b * K + n // K
    fs = tl.load(front_scores_ptr + b * K + n // K, mask=valid, other=0.0)
    p = tl.load(probs_ptr + row * probs_stride + n % K, mask=valid, other=0.0).to(tl.float32)
    layer = fs * p
    tok = tl.load(tokens_ptr + row * tokens_stride + n % K, mask=valid, other=0)
    m = tl.arange(0, MERGE_P)
    in_best = m < CAP
    in_layer = (m >= CAP) & (m < CAP + K * K)
    old_best = tl.load(best_ptr + b * best_stride + m, mask=in_best, other=0.0)
    ml = tl.where(in_layer, m - CAP, 0)
    fs_m = tl.load(front_scores_ptr + b * K + ml // K, mask=in_layer, other=0.0)
    p_m = tl.load(probs_ptr + (b * K + ml // K) * probs_stride + ml % K, mask=in_layer, other=0.0).to(tl.float32)
    merged = tl.where(in_best, old_best, tl.where(in_layer, fs_m * p_m, float("-inf")))
    # Top K of the layer: non-negative fp32 bits order like the values; the low word breaks
    # ties toward the lower node.
    key = (layer.to(tl.int32, bitcast=True).to(tl.int64) << 32) | (KK_P - 1 - n).to(tl.int64)
    key = tl.where(valid, key, -1)
    top_key = tl.topk(key, K_P)
    r = tl.arange(0, K_P)
    r_ok = r < K
    pick = KK_P - 1 - (top_key & 0xFFFFFFFF)
    top = (top_key >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    lane = pick // K
    d = tl.arange(0, D_P)
    anc_mask = r_ok[:, None] & (d[None, :] < D)
    anc = tl.load(ancestors_ptr + (b * K + lane)[:, None] * D + d[None, :], mask=anc_mask, other=0)
    front_tok = tl.load(tokens_ptr + (b * K + lane) * tokens_stride + pick % K, mask=r_ok, other=0)
    new_best = tl.sort(merged, descending=True)
    tl.debug_barrier()
    # ---- writes
    tl.store(scores_ptr + b * scores_stride + start + n, layer, mask=valid)
    tl.store(node_tokens_ptr + b * node_tokens_stride + start + n, tok, mask=valid)
    tl.store(parents_ptr + b * parents_stride + pstart + r, pick + start, mask=r_ok)
    tl.store(front_scores_ptr + b * K + r, top, mask=r_ok)
    tl.store(layer_top_ptr + b * K + r, top, mask=r_ok)
    tl.store(front_tokens_ptr + b * K + r, front_tok, mask=r_ok)
    tl.store(ancestors_ptr + (b * K + r)[:, None] * D + d[None, :], anc, mask=anc_mask)
    tl.store(best_prev_ptr + b * best_stride + m, old_best, mask=in_best)
    tl.store(best_ptr + b * best_stride + m, new_best, mask=in_best)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + tl.arange(0, BLOCK_H)
        hm = r_ok[:, None] & (h[None, :] < H)
        rows = tl.load(hidden_ptr + (b * K + lane)[:, None] * hidden_stride + h[None, :], mask=hm)
        tl.store(front_hidden_ptr + (b * K + r)[:, None] * front_hidden_stride + h[None, :], rows, mask=hm)
    if DELTAS:
        # deltas[v, j] = sum of the top v of (old best set, layer top j): row j of the tile merges the old set with
        # the layer's best j scores (the other extras are zero, scores are non-negative), sorted, then prefix-summed.
        tl.debug_barrier()  # the old set and the layer top were just stored by this program
        j = tl.arange(0, J_P)[:, None]
        c = tl.arange(0, DELTA_P)[None, :]
        e = c - CAP
        old = tl.load(best_prev_ptr + b * best_stride + c, mask=c < CAP, other=0.0, cache_modifier=".cg")
        # J_P rounds K + 1 up to a power of two: rows j > K are never stored, and must not read past the row's K.
        extra = tl.load(layer_top_ptr + b * K + e, mask=(e >= 0) & (e < j) & (e < K), other=0.0, cache_modifier=".cg")
        sums = tl.cumsum(tl.sort(old + extra, dim=1, descending=True), axis=1)
        tl.store(deltas_ptr + b * deltas_stride + c * (K + 1) + j, sums, mask=(j <= K) & (c < V))


def tree_step(
    tree, bs: int, step: int, probs: torch.Tensor, tokens: torch.Tensor, hidden: torch.Tensor, deltas: bool = False
) -> None:
    """:meth:`DraftTree.step` in one launch; ``deltas``: also store ``tree.deltas[step]`` (the step's work deltas)."""
    shape = tree.shape
    k = shape.top_k
    hidden = hidden.view(bs * k, -1)
    if probs.stride(-1) != 1 or tokens.stride(-1) != 1 or hidden.stride(-1) != 1:
        raise ValueError("tree_step needs unit column strides")
    if tokens.dtype != torch.int64:
        tokens = tokens.long()
    if variant.current == TREESTEP:
        return _tree_step_merge(tree, bs, step, probs, tokens, hidden, deltas)
    depth = tree.ancestors.shape[-1]
    # One row writes its deltas straight into the slot; several rows write theirs side by side, then their mean.
    rows_out = tree.deltas[step] if bs == 1 else tree.delta_rows
    _tree_step_kernel[(bs,)](
        tree.front_scores,
        (probs, tokens, hidden),
        (
            tree.scores,
            tree.tokens,
            tree.parents,
            tree.front_tokens,
            tree.front_hidden,
            tree.ancestors,
            tree.best,
            tree.best_prev,
            tree.layer_top,
            rows_out,
        ),
        (probs.stride(0), tokens.stride(0), hidden.stride(0)),
        (
            tree.scores.stride(0),
            tree.tokens.stride(0),
            tree.parents.stride(0),
            tree.front_hidden.stride(0),
            tree.best.stride(0),
            0 if bs == 1 else rows_out.stride(0),
        ),
        hidden.shape[1],
        shape.layer_start(step + 1),
        shape.num_parents(step + 1),
        K=k,
        K_P=max(triton.next_power_of_2(k), 2),  # tl.topk needs k >= 2; the lanes past K are masked
        KK_P=max(triton.next_power_of_2(k * k), 2),
        D=depth,
        D_P=triton.next_power_of_2(depth),
        CAP=shape.work_cap,
        MERGE_P=triton.next_power_of_2(shape.work_cap + k * k),
        BLOCK_H=1024,
        DELTAS=deltas,
        V=tree.max_verify,
        J_P=triton.next_power_of_2(k + 1),
        DELTA_P=triton.next_power_of_2(shape.work_cap + k),
        num_warps=4,
    )
    if deltas and bs > 1:
        torch.mean(tree.delta_rows[:bs], dim=0, out=tree.deltas[step])


@triton.jit
def _work_deltas_kernel(
    best_prev_ptr,  # [bs, CAP] fp32 best scores before the layer, sorted
    layer_top_ptr,  # [bs, K] fp32 the layer's best scores, sorted
    deltas_ptr,  # [V, K + 1] fp32 out
    bs,
    best_stride,
    V,
    K: tl.constexpr,
    CAP: tl.constexpr,
    MERGE_P: tl.constexpr,
):
    # Program j: the layer's best j scores join the old best set; the batch mean of the prefix sums of
    # the merged set's sorted scores is the delta of every budget (row v - 1: the top v).
    j = tl.program_id(0)
    m = tl.arange(0, MERGE_P)
    extra = m - CAP
    in_best = m < CAP
    in_extra = (extra >= 0) & (extra < j)
    total = tl.zeros([MERGE_P], dtype=tl.float32)
    for b in range(0, bs):
        old = tl.load(best_prev_ptr + b * best_stride + m, mask=in_best, other=0.0)
        new = tl.load(layer_top_ptr + b * K + extra, mask=in_extra, other=0.0)
        total += tl.cumsum(tl.sort(old + new, descending=True), axis=0)
    tl.store(deltas_ptr + m * (K + 1) + j, total / bs, mask=m < V)


def work_deltas(tree, bs: int, slot: int) -> None:
    """:meth:`DraftTree._store_deltas` in one launch: ``tree.deltas[slot]`` from ``best_prev`` and ``layer_top``."""
    k = tree.shape.top_k
    cap = tree.shape.work_cap
    _work_deltas_kernel[(k + 1,)](
        tree.best_prev,
        tree.layer_top,
        tree.deltas[slot],
        bs,
        tree.best_prev.stride(0),
        tree.max_verify,
        K=k,
        CAP=cap,
        MERGE_P=triton.next_power_of_2(cap + k),
        num_warps=1,
    )


@triton.jit
def _tree_begin_kernel(
    root,  # the draft extend's outputs: probs [bs, K] fp32 (top-K probabilities), tokens [bs, K] int64, hidden [bs, H]
    tree,  # scores, node_tokens, front_scores, front_tokens, front_hidden, best, best_prev, layer_top
    root_hidden_stride,
    tree_strides,  # row strides of scores, node_tokens, front_hidden, best
    H,
    root_score,
    K: tl.constexpr,
    K_P: tl.constexpr,
    CAP: tl.constexpr,
    CAP_P: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    root_probs_ptr, root_tokens_ptr, root_hidden_ptr = root
    (
        scores_ptr,
        node_tokens_ptr,
        front_scores_ptr,
        front_tokens_ptr,
        front_hidden_ptr,
        best_ptr,
        best_prev_ptr,
        layer_top_ptr,
    ) = tree
    scores_stride, node_tokens_stride, front_hidden_stride, best_stride = tree_strides
    b = tl.program_id(0).to(tl.int64)
    n = tl.arange(0, K_P)
    valid = n < K
    p = tl.load(root_probs_ptr + b * K + n, mask=valid, other=0.0)
    key = (p.to(tl.int32, bitcast=True).to(tl.int64) << 32) | (K_P - 1 - n).to(tl.int64)
    top_key = tl.topk(tl.where(valid, key, -1), K_P)  # descending, ties to the lower index
    pick = K_P - 1 - (top_key & 0xFFFFFFFF)
    top = (top_key >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    tok = tl.load(root_tokens_ptr + b * K + pick, mask=valid, other=0)
    tl.store(scores_ptr + b * scores_stride + n, top, mask=valid)
    tl.store(node_tokens_ptr + b * node_tokens_stride + n, tok, mask=valid)
    tl.store(front_scores_ptr + b * K + n, top, mask=valid)
    tl.store(layer_top_ptr + b * K + n, top, mask=valid)
    tl.store(front_tokens_ptr + b * K + n, tok, mask=valid)
    tl.debug_barrier()
    m = tl.arange(0, CAP_P)
    in_cap = m < CAP
    first = tl.load(front_scores_ptr + b * K + (m - 1), mask=(m >= 1) & (m <= K), other=0.0)
    tl.store(best_prev_ptr + b * best_stride + m, tl.where(m == 0, root_score, 0.0), mask=in_cap)
    tl.store(best_ptr + b * best_stride + m, tl.where(m == 0, root_score, first), mask=in_cap)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + tl.arange(0, BLOCK_H)
        hm = valid[:, None] & (h[None, :] < H)
        row = tl.load(root_hidden_ptr + b * root_hidden_stride + h, mask=h < H)
        tl.store(front_hidden_ptr + (b * K + n)[:, None] * front_hidden_stride + h[None, :], row[None, :], mask=hm)


def tree_begin(tree, bs: int, root_score: float, deltas: bool = False) -> None:
    """:meth:`DraftTree.begin` in one launch; ``deltas``: also store ``tree.deltas[0]`` (another launch, but in the
    treestep version)."""
    if variant.current == TREESTEP:
        return _tree_begin_merge(tree, bs, root_score, deltas)
    shape = tree.shape
    k = shape.top_k
    hidden = tree.root_hidden
    _tree_begin_kernel[(bs,)](
        (tree.root_probs, tree.root_tokens, hidden),
        (
            tree.scores,
            tree.tokens,
            tree.front_scores,
            tree.front_tokens,
            tree.front_hidden,
            tree.best,
            tree.best_prev,
            tree.layer_top,
        ),
        hidden.stride(0),
        (tree.scores.stride(0), tree.tokens.stride(0), tree.front_hidden.stride(0), tree.best.stride(0)),
        hidden.shape[1],
        root_score,
        K=k,
        K_P=triton.next_power_of_2(k),
        CAP=shape.work_cap,
        CAP_P=triton.next_power_of_2(shape.work_cap),
        BLOCK_H=1024,
        num_warps=4,
    )
    if deltas:
        work_deltas(tree, bs, 0)


@triton.jit
def _tree_rank_kernel(
    scores_ptr,  # [bs, capacity] fp32 node scores
    decay_ptr,  # [capacity] fp32 depth decay
    ranked_ptr,  # [bs, COUNT] int64 out, best first
    cum_ptr,  # [bs, COUNT + 1] fp32 out, cumulative verify scores (root first)
    scores_stride,
    root_score,
    N: tl.constexpr,
    N_P: tl.constexpr,
    COUNT: tl.constexpr,
    COUNT_P: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    n = tl.arange(0, N_P)
    valid = n < N
    decayed = tl.load(scores_ptr + b * scores_stride + n, mask=valid, other=0.0) * tl.load(
        decay_ptr + n, mask=valid, other=0.0
    )
    # Non-negative fp32 bits order like the values; the low word breaks ties toward the lower node.
    key = (decayed.to(tl.int32, bitcast=True).to(tl.int64) << 32) | (N_P - 1 - n).to(tl.int64)
    top_key = tl.topk(tl.where(valid, key, -1), COUNT_P)  # descending
    r = tl.arange(0, COUNT_P)
    used = r < COUNT
    top = tl.where(used, (top_key >> 32).to(tl.int32).to(tl.float32, bitcast=True), 0.0)
    tl.store(ranked_ptr + b * COUNT + r, N_P - 1 - (top_key & 0xFFFFFFFF), mask=used)
    tl.store(cum_ptr + b * (COUNT + 1), root_score)
    tl.store(cum_ptr + b * (COUNT + 1) + 1 + r, root_score + tl.cumsum(top, axis=0), mask=used)


def tree_rank(tree, bs: int, depth: int, count: int, root_score: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """:meth:`DraftTree.rank` in one launch (``count`` at most the tree's node count)."""
    n = tree.shape.num_nodes(depth)
    ranked = torch.empty(bs, count, dtype=torch.int64, device=tree.scores.device)
    cum = torch.empty(bs, count + 1, dtype=torch.float32, device=tree.scores.device)
    _tree_rank_kernel[(bs,)](
        tree.scores,
        tree.decay,
        ranked,
        cum,
        tree.scores.stride(0),
        root_score,
        N=n,
        N_P=triton.next_power_of_2(n),
        COUNT=count,
        COUNT_P=max(triton.next_power_of_2(count), 2),
        num_warps=4,
    )
    return ranked, cum


@triton.jit
def _tree_select_kernel(
    chosen_ptr,  # [bs, M] int64 node indices
    tokens_ptr,  # [bs, capacity] int64 node tokens
    index_ptr,  # [bs, M] int64 out, the nodes ascending
    draft_ptr,  # [bs, M] int64 out, their tokens
    chosen_stride,
    tokens_stride,
    M: tl.constexpr,
    M_P: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, M_P)
    used = i < M
    node = tl.sort(tl.load(chosen_ptr + b * chosen_stride + i, mask=used, other=2147483647))  # padding sorts last
    tl.store(index_ptr + b * M + i, node, mask=used)
    tl.store(draft_ptr + b * M + i, tl.load(tokens_ptr + b * tokens_stride + node, mask=used, other=0), mask=used)


def tree_select(chosen: torch.Tensor, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The nodes ``chosen`` ``[bs, M]`` sorted ascending and their tokens, in one launch."""
    bs, m = chosen.shape
    index = torch.empty(bs, m, dtype=torch.int64, device=chosen.device)
    draft = torch.empty_like(index)
    if m:
        _tree_select_kernel[(bs,)](
            chosen,
            tokens,
            index,
            draft,
            chosen.stride(0),
            tokens.stride(0),
            M=m,
            M_P=max(triton.next_power_of_2(m), 2),
            num_warps=1,
        )
    return index, draft


@triton.jit
def _select_rows_kernel(
    scores,
    decay,
    tokens,
    parents,
    out_tokens,
    out_parent_rows,
    row_of,
    stride_scores,
    stride_tokens,
    stride_parents,
    count,
    NODES: tl.constexpr,
    N_BLOCK: tl.constexpr,
    K: tl.constexpr,
):
    b = tl.program_id(0)
    node = tl.arange(0, N_BLOCK)
    valid = node < NODES
    s = tl.load(scores + b * stride_scores + node, mask=valid, other=0.0) * tl.load(decay + node, mask=valid, other=0.0)
    s = tl.where(valid, s, -1.0)
    threshold = tl.sum(tl.where(node == count - 1, tl.sort(s, descending=True), 0.0))
    greater = s > threshold
    tied = s == threshold  # the lowest-numbered tied nodes fill the set, as torch.topk does
    need = count - tl.sum(greater.to(tl.int32))
    chosen = greater | (tied & (tl.cumsum(tied.to(tl.int32), 0) <= need))
    row = tl.cumsum(chosen.to(tl.int32), 0) - 1  # rows in node order: a parent before its children
    tl.store(out_tokens + b * count + row, tl.load(tokens + b * stride_tokens + node, mask=chosen), mask=chosen)
    tl.store(row_of + b * N_BLOCK + node, tl.where(chosen, row, -1))
    tl.debug_barrier()
    # parent node: depth 1 hangs from the root; a node of layer L >= 1 from frontier entry ``local // K`` of layer L - 1
    rel = node - K
    layer = 1 + rel // (K * K)
    base = tl.where(layer == 1, 1, K + 1 + (layer - 2) * K)
    column = base + (rel % (K * K)) // K
    inner = chosen & (node >= K)
    parent = tl.load(parents + b * stride_parents + column, mask=inner, other=-1)
    parent_row = tl.load(row_of + b * N_BLOCK + parent, mask=inner & (parent >= 0), other=-1)
    tl.store(out_parent_rows + b * (count + 1) + 1 + row, tl.where(parent_row >= 0, parent_row + 1, 0), mask=chosen)
    tl.store(out_parent_rows + b * (count + 1), -1)


def select_rows(
    scores: torch.Tensor,
    decay: torch.Tensor,
    tokens: torch.Tensor,
    parents: torch.Tensor,
    count: int,
    top_k: int,
    row_of: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``count`` best nodes by decayed score as verify rows, in node order: ``(tokens [bs, count] int64,
    parent_rows [bs, count + 1] int32)`` with row 0 the root (-1) and 0 for a node whose parent is the root or not
    chosen. ``scores`` / ``tokens`` ``[bs, nodes]``, ``parents`` the tree's frontier list, ``row_of`` int32 scratch
    ``[>= bs, next_power_of_2(nodes)]``. Ties at the last row go to the lowest node number."""
    bs, nodes = scores.shape
    out_tokens = torch.empty(bs, count, dtype=torch.int64, device=scores.device)
    out_parent_rows = torch.empty(bs, count + 1, dtype=torch.int32, device=scores.device)
    n_block = triton.next_power_of_2(nodes)
    assert row_of.shape[1] == n_block and row_of.is_contiguous()
    _select_rows_kernel[(bs,)](
        scores,
        decay,
        tokens,
        parents,
        out_tokens,
        out_parent_rows,
        row_of,
        scores.stride(0),
        tokens.stride(0),
        parents.stride(0),
        count,
        NODES=nodes,
        N_BLOCK=n_block,
        K=top_k,
    )
    return out_tokens, out_parent_rows


# ---- the treestep version: no sorts, one launch a step (the batch mean included), begin's deltas in begin's launch,
# and a few rows' hidden copies in copier programs beside the row's own


@triton.jit
def _merged_sums(
    old_ptr,  # the row's old best set [CAP] fp32, sorted descending (unless ROOTED)
    root_score,
    top,  # [K_P] fp32 the layer's best K scores, sorted descending (lanes past K unused)
    K: tl.constexpr,
    K_P: tl.constexpr,
    CAP: tl.constexpr,
    CAP_P: tl.constexpr,
    J_P: tl.constexpr,
    V_P: tl.constexpr,
    ROOTED: tl.constexpr,  # the old set is the root alone (begin)
):
    """[J_P, V_P]: row j holds the prefix sums of the old best set merged with ``top[:j]``, the work deltas of every
    budget: both lists are sorted, so each of top's scores lands after the old scores at least as high, and every other
    place takes the next old score."""
    c = tl.arange(0, CAP_P)
    if ROOTED:
        old = tl.where(c == 0, root_score, 0.0)
    else:
        old = tl.load(old_ptr + c, mask=c < CAP, other=0.0)
    e = tl.arange(0, K_P)
    place = e + tl.sum(((old[None, :] >= top[:, None]) & (c[None, :] < CAP)).to(tl.int32), axis=1)
    j = tl.arange(0, J_P)[:, None]
    q = tl.arange(0, V_P)[None, :]
    placed = tl.zeros([J_P, V_P], dtype=tl.int32)  # top's scores placed before q
    joined = tl.zeros([J_P, V_P], dtype=tl.float32)
    is_top = placed != 0
    for i in tl.static_range(K):
        at = tl.sum(tl.where(e == i, place, 0))
        score = tl.sum(tl.where(e == i, top, 0.0))
        used = i < j
        placed += (used & (at < q)).to(tl.int32)
        here = used & (at == q)
        is_top |= here
        joined = tl.where(here, score, joined)
    src = q - placed
    if ROOTED:
        old_q = tl.where(src == 0, root_score, 0.0)
    else:
        old_q = tl.load(old_ptr + src, mask=src < CAP, other=0.0)
    return tl.cumsum(tl.where(is_top, joined, old_q), axis=1)


@triton.jit
def _store_sums(sums, row_ptr, K: tl.constexpr, J_P: tl.constexpr, V, V_P: tl.constexpr):
    j = tl.arange(0, J_P)[:, None]
    q = tl.arange(0, V_P)[None, :]
    tl.store(row_ptr + q * (K + 1) + j, sums, mask=(j <= K) & (q < V))


@triton.jit
def _batch_mean(rows_ptr, out_ptr, done_ptr, bs, N, N_P: tl.constexpr, ROWS: tl.constexpr):
    """The program that stores its row last averages the ``bs`` rows of ``N`` floats into ``out_ptr``, in row order
    whichever program that is, and resets the count for the next launch."""
    tl.debug_barrier()  # every thread's part of this row is stored
    if tl.atomic_add(done_ptr, 1, sem="acq_rel", scope="gpu") == bs - 1:
        i = tl.arange(0, N_P)
        total = tl.zeros([N_P], dtype=tl.float32)
        for r0 in tl.range(0, bs, ROWS, num_stages=2):
            r = r0 + tl.arange(0, ROWS)[:, None]
            rows = tl.load(rows_ptr + r * N + i[None, :], mask=(r < bs) & (i[None, :] < N), other=0.0,
                           cache_modifier=".cg")
            total += tl.sum(rows, axis=0)
        tl.store(out_ptr + i, total / bs, mask=i < N)
        tl.store(done_ptr, 0)


@triton.jit
def _copy_rows(src_ptr, src_rows, src_stride, dst_ptr, dst_rows, dst_stride, ok, H, part, chunk,
               BLOCK_H: tl.constexpr):
    """Columns ``[chunk * part, + chunk)`` of the rows ``src_rows`` into the rows ``dst_rows``."""
    lo = part * chunk
    for h0 in range(lo, tl.minimum(lo + chunk, H), BLOCK_H):
        h = h0 + tl.arange(0, BLOCK_H)
        hm = ok[:, None] & (h[None, :] < H)
        rows = tl.load(src_ptr + src_rows[:, None] * src_stride + h[None, :], mask=hm)
        tl.store(dst_ptr + dst_rows[:, None] * dst_stride + h[None, :], rows, mask=hm)


@triton.jit
def _layer_top(fs, probs_ptr, probs_stride, b, K: tl.constexpr, K_P: tl.constexpr, KK_P: tl.constexpr):
    """The layer's scores ``fs * prob`` and the keys of its top K."""
    n = tl.arange(0, KK_P)
    valid = n < K * K
    p = tl.load(probs_ptr + (b * K + n // K) * probs_stride + n % K, mask=valid, other=0.0).to(tl.float32)
    layer = fs * p
    # Non-negative fp32 bits order like the values; the low word breaks ties toward the lower node.
    key = (layer.to(tl.int32, bitcast=True).to(tl.int64) << 32) | (KK_P - 1 - n).to(tl.int64)
    return layer, tl.topk(tl.where(valid, key, -1), K_P)


@triton.jit
def _tree_step_merge_kernel(
    front_scores_ptr,  # [bs, K] fp32, read then overwritten
    layer,  # probs [bs * K, K], tokens [bs * K, K] int64, hidden [bs * K, H]: as _tree_step_kernel's
    tree,  # as _tree_step_kernel's, then the batch mean's: delta rows [bs, V, K + 1] fp32, the count [1] int32
    layer_strides,  # row strides of probs, tokens, hidden
    tree_strides,  # row strides of scores, node_tokens, parents, front_hidden, best
    H,
    chunk,  # hidden columns a copier copies
    start,
    pstart,
    fstart,  # the frontier's column in parents
    bs,
    V,
    K: tl.constexpr,
    K_P: tl.constexpr,
    KK_P: tl.constexpr,
    D: tl.constexpr,
    D_P: tl.constexpr,
    CAP: tl.constexpr,
    CAP_P: tl.constexpr,
    BLOCK_H: tl.constexpr,
    COPIERS: tl.constexpr,  # programs 1.. of a row copy its hidden rows (else program 0 does)
    DELTAS: tl.constexpr,
    MEAN: tl.constexpr,  # several rows: each stores its deltas in a delta row, the last one stores their mean
    J_P: tl.constexpr,
    V_P: tl.constexpr,
    N_P: tl.constexpr,
    ROWS: tl.constexpr,
):
    probs_ptr, tokens_ptr, hidden_ptr = layer
    (scores_ptr, node_tokens_ptr, parents_ptr, front_tokens_ptr, front_hidden_ptr, ancestors_ptr, best_ptr,
     best_prev_ptr, layer_top_ptr, deltas_ptr, delta_rows_ptr, done_ptr) = tree
    probs_stride, tokens_stride, hidden_stride = layer_strides
    scores_stride, node_tokens_stride, parents_stride, front_hidden_stride, best_stride = tree_strides
    b = tl.program_id(0).to(tl.int64)
    n = tl.arange(0, KK_P)
    valid = n < K * K
    r = tl.arange(0, K_P)
    r_ok = r < K
    if COPIERS:
        if tl.program_id(1) > 0:
            # A copier reads the frontier's scores where this launch writes nothing: at its nodes (front_scores, which
            # program 0 overwrites, holds the same bits).
            node = tl.load(parents_ptr + b * parents_stride + fstart + n // K, mask=valid, other=0)
            fs = tl.load(scores_ptr + b * scores_stride + node, mask=valid, other=0.0)
            _, top_key = _layer_top(fs, probs_ptr, probs_stride, b, K, K_P, KK_P)
            lane = (KK_P - 1 - (top_key & 0xFFFFFFFF)) // K
            _copy_rows(hidden_ptr, b * K + lane, hidden_stride, front_hidden_ptr, b * K + r, front_hidden_stride, r_ok,
                       H, tl.program_id(1) - 1, chunk, BLOCK_H)
            return
    # ---- every read of in-place buffers first
    fs = tl.load(front_scores_ptr + b * K + n // K, mask=valid, other=0.0)
    layer, top_key = _layer_top(fs, probs_ptr, probs_stride, b, K, K_P, KK_P)
    row = b * K + n // K
    tok = tl.load(tokens_ptr + row * tokens_stride + n % K, mask=valid, other=0)
    pick = KK_P - 1 - (top_key & 0xFFFFFFFF)
    top = (top_key >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    lane = pick // K
    d = tl.arange(0, D_P)
    anc_mask = r_ok[:, None] & (d[None, :] < D)
    anc = tl.load(ancestors_ptr + (b * K + lane)[:, None] * D + d[None, :], mask=anc_mask, other=0)
    front_tok = tl.load(tokens_ptr + (b * K + lane) * tokens_stride + pick % K, mask=r_ok, other=0)
    # The new best set, the top CAP of the old set and the layer: a layer score lands after the old scores at least as
    # high and the layer's higher ones (ties to the lower node); every other place takes the next old score.
    best_row = best_ptr + b * best_stride
    c = tl.arange(0, CAP_P)
    old_best = tl.load(best_row + c, mask=c < CAP, other=0.0)
    over_old = tl.sum(((old_best[None, :] >= layer[:, None]) & (c[None, :] < CAP)).to(tl.int32), axis=1)
    ahead = (layer[None, :] > layer[:, None]) | ((layer[None, :] == layer[:, None]) & (n[None, :] < n[:, None]))
    place = tl.where(valid, over_old + tl.sum((ahead & valid[None, :]).to(tl.int32), axis=1), CAP_P)
    placed = tl.sum((place[None, :] < c[:, None]).to(tl.int32), axis=1)
    here = place[None, :] == c[:, None]
    from_layer = tl.sum(tl.where(here, layer[None, :], 0.0), axis=1)
    from_old = tl.load(best_row + c - placed, mask=c < CAP, other=0.0)
    new_best = tl.where(tl.max(here.to(tl.int32), axis=1) > 0, from_layer, from_old)
    if DELTAS:
        sums = _merged_sums(best_row, 0.0, top, K, K_P, CAP, CAP_P, J_P, V_P, False)
    tl.debug_barrier()
    # ---- writes
    tl.store(scores_ptr + b * scores_stride + start + n, layer, mask=valid)
    tl.store(node_tokens_ptr + b * node_tokens_stride + start + n, tok, mask=valid)
    tl.store(parents_ptr + b * parents_stride + pstart + r, pick + start, mask=r_ok)
    tl.store(front_scores_ptr + b * K + r, top, mask=r_ok)
    tl.store(layer_top_ptr + b * K + r, top, mask=r_ok)
    tl.store(front_tokens_ptr + b * K + r, front_tok, mask=r_ok)
    tl.store(ancestors_ptr + (b * K + r)[:, None] * D + d[None, :], anc, mask=anc_mask)
    tl.store(best_prev_ptr + b * best_stride + c, old_best, mask=c < CAP)
    tl.store(best_row + c, new_best, mask=c < CAP)
    if not COPIERS:
        _copy_rows(hidden_ptr, b * K + lane, hidden_stride, front_hidden_ptr, b * K + r, front_hidden_stride, r_ok, H,
                   0, chunk, BLOCK_H)
    if DELTAS:
        if MEAN:
            _store_sums(sums, delta_rows_ptr + b * V * (K + 1), K, J_P, V, V_P)
            _batch_mean(delta_rows_ptr, deltas_ptr, done_ptr, bs, V * (K + 1), N_P, ROWS)
        else:
            _store_sums(sums, deltas_ptr, K, J_P, V, V_P)


@triton.jit
def _tree_begin_merge_kernel(
    root,  # probs [bs, K] fp32, tokens [bs, K] int64, hidden [bs, H]: as _tree_begin_kernel's
    tree,  # as _tree_begin_kernel's, then deltas [V, K + 1], delta rows [bs, V, K + 1] fp32, the count [1] int32
    root_hidden_stride,
    tree_strides,  # row strides of scores, node_tokens, front_hidden, best
    H,
    chunk,  # hidden columns a program copies
    root_score,
    bs,
    V,
    K: tl.constexpr,
    K_P: tl.constexpr,
    CAP: tl.constexpr,
    CAP_P: tl.constexpr,
    BLOCK_H: tl.constexpr,
    COPIERS: tl.constexpr,
    DELTAS: tl.constexpr,
    MEAN: tl.constexpr,
    J_P: tl.constexpr,
    V_P: tl.constexpr,
    N_P: tl.constexpr,
    ROWS: tl.constexpr,
):
    root_probs_ptr, root_tokens_ptr, root_hidden_ptr = root
    (scores_ptr, node_tokens_ptr, front_scores_ptr, front_tokens_ptr, front_hidden_ptr, best_ptr, best_prev_ptr,
     layer_top_ptr, deltas_ptr, delta_rows_ptr, done_ptr) = tree
    scores_stride, node_tokens_stride, front_hidden_stride, best_stride = tree_strides
    b = tl.program_id(0).to(tl.int64)
    n = tl.arange(0, K_P)
    valid = n < K
    if COPIERS:
        if tl.program_id(1) > 0:
            _copy_rows(root_hidden_ptr, b + 0 * n, root_hidden_stride, front_hidden_ptr, b * K + n,
                       front_hidden_stride, valid, H, tl.program_id(1) - 1, chunk, BLOCK_H)
            return
    else:
        _copy_rows(root_hidden_ptr, b + 0 * n, root_hidden_stride, front_hidden_ptr, b * K + n, front_hidden_stride,
                   valid, H, 0, chunk, BLOCK_H)
    p = tl.load(root_probs_ptr + b * K + n, mask=valid, other=0.0)
    key = (p.to(tl.int32, bitcast=True).to(tl.int64) << 32) | (K_P - 1 - n).to(tl.int64)
    top_key = tl.topk(tl.where(valid, key, -1), K_P)  # descending, ties to the lower index
    pick = K_P - 1 - (top_key & 0xFFFFFFFF)
    top = (top_key >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    tok = tl.load(root_tokens_ptr + b * K + pick, mask=valid, other=0)
    tl.store(scores_ptr + b * scores_stride + n, top, mask=valid)
    tl.store(node_tokens_ptr + b * node_tokens_stride + n, tok, mask=valid)
    tl.store(front_scores_ptr + b * K + n, top, mask=valid)
    tl.store(layer_top_ptr + b * K + n, top, mask=valid)
    tl.store(front_tokens_ptr + b * K + n, tok, mask=valid)
    # best: the root, then the layer (sorted); best_prev: the root alone
    m = tl.arange(0, CAP_P)
    first = tl.sum(tl.where((n[None, :] == m[:, None] - 1) & valid[None, :], top[None, :], 0.0), axis=1)
    tl.store(best_prev_ptr + b * best_stride + m, tl.where(m == 0, root_score, 0.0), mask=m < CAP)
    tl.store(best_ptr + b * best_stride + m, tl.where(m == 0, root_score, first), mask=m < CAP)
    if DELTAS:
        sums = _merged_sums(best_prev_ptr, root_score, top, K, K_P, CAP, CAP_P, J_P, V_P, True)
        if MEAN:
            _store_sums(sums, delta_rows_ptr + b * V * (K + 1), K, J_P, V, V_P)
            _batch_mean(delta_rows_ptr, deltas_ptr, done_ptr, bs, V * (K + 1), N_P, ROWS)
        else:
            _store_sums(sums, deltas_ptr, K, J_P, V, V_P)


def _mean_meta(tree, bs: int) -> dict:
    k, v = tree.shape.top_k, tree.max_verify
    return dict(MEAN=bs > 1, J_P=triton.next_power_of_2(k + 1), V_P=triton.next_power_of_2(v),
                N_P=triton.next_power_of_2(v * (k + 1)), ROWS=min(triton.next_power_of_2(bs), MERGE["rows"]),
                BLOCK_H=MERGE["block_h"], num_warps=MERGE["warps"])


def _copy_grid(bs: int, h: int) -> Tuple[Tuple[int, int], int, bool]:
    """``(grid, chunk, copiers)``: a few rows copy their hidden rows in copier programs, a chunk of columns each, as
    many as fill the GPU; many rows copy them in their own program."""
    block = MERGE["block_h"]
    split = min(triton.cdiv(h, block), MERGE["programs"] // bs)
    if split < 2:
        return (bs, 1), max(triton.cdiv(h, block), 1) * block, False
    chunk = triton.cdiv(triton.cdiv(h, split), block) * block
    return (bs, 1 + triton.cdiv(h, chunk)), chunk, True


MERGE = dict(rows=64, block_h=1024, warps=4, programs=128)  # the merge kernels' launch


def _tree_step_merge(tree, bs, step, probs, tokens, hidden, deltas):
    shape = tree.shape
    k = shape.top_k
    depth = tree.ancestors.shape[-1]
    grid, chunk, copiers = _copy_grid(bs, hidden.shape[1])
    _tree_step_merge_kernel[grid](
        tree.front_scores,
        (probs, tokens, hidden),
        (tree.scores, tree.tokens, tree.parents, tree.front_tokens, tree.front_hidden, tree.ancestors, tree.best,
         tree.best_prev, tree.layer_top, tree.deltas[step], tree.delta_rows, tree.delta_count),
        (probs.stride(0), tokens.stride(0), hidden.stride(0)),
        (tree.scores.stride(0), tree.tokens.stride(0), tree.parents.stride(0), tree.front_hidden.stride(0),
         tree.best.stride(0)),
        hidden.shape[1],
        chunk,
        shape.layer_start(step + 1),
        shape.num_parents(step + 1),
        1 if step == 1 else shape.num_parents(step),
        bs,
        tree.max_verify,
        K=k,
        K_P=max(triton.next_power_of_2(k), 2),  # tl.topk needs k >= 2; the lanes past K are masked
        KK_P=max(triton.next_power_of_2(k * k), 2),
        D=depth,
        D_P=triton.next_power_of_2(depth),
        CAP=shape.work_cap,
        CAP_P=triton.next_power_of_2(shape.work_cap),
        COPIERS=copiers,
        DELTAS=deltas,
        **_mean_meta(tree, bs),
    )


def _tree_begin_merge(tree, bs, root_score, deltas):
    shape = tree.shape
    k = shape.top_k
    hidden = tree.root_hidden
    grid, chunk, copiers = _copy_grid(bs, hidden.shape[1])
    _tree_begin_merge_kernel[grid](
        (tree.root_probs, tree.root_tokens, hidden),
        (tree.scores, tree.tokens, tree.front_scores, tree.front_tokens, tree.front_hidden, tree.best, tree.best_prev,
         tree.layer_top, tree.deltas[0], tree.delta_rows, tree.delta_count),
        hidden.stride(0),
        (tree.scores.stride(0), tree.tokens.stride(0), tree.front_hidden.stride(0), tree.best.stride(0)),
        hidden.shape[1],
        chunk,
        root_score,
        bs,
        tree.max_verify,
        K=k,
        K_P=triton.next_power_of_2(k),
        CAP=shape.work_cap,
        CAP_P=triton.next_power_of_2(shape.work_cap),
        COPIERS=copiers,
        DELTAS=deltas,
        **_mean_meta(tree, bs),
    )


def set_step_slots(tree, bs: int, step: int, slots: torch.Tensor) -> None:
    """:meth:`DraftTree.set_step_slots` on CUDA: the treestep version casts and writes the column in one copy."""
    if variant.current == TREESTEP:
        tree.ancestors[:bs, :, step - 1].copy_(slots)
    else:
        tree.ancestors[:bs, :, step - 1] = slots.to(torch.int32)
