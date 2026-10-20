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
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


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
        extra = tl.load(layer_top_ptr + b * K + e, mask=(e >= 0) & (e < j), other=0.0, cache_modifier=".cg")
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


def tree_begin(tree, bs: int, root_score: float) -> None:
    """:meth:`DraftTree.begin` (without the work deltas) in one launch."""
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
