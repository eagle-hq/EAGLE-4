# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
"""Sparse-q recursive tree rejection; the host retains target filtering/commit."""

import torch
import triton
import triton.language as tl


@triton.jit
def _walk(
    P,
    Q,
    QIDS,
    POS,
    ROOT_VALID,
    NODES,
    TOKENS,
    INDEX,
    CHILD,
    SIBLING,
    COINS,
    PREDICT,
    PATH,
    LENGTH,
    FINAL_ROW,
    FINAL_IDS,
    FINAL_RESIDUAL,
    REQ,
    V: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    store = tl.load(REQ + b)  # the request's slot in the recorded-q stores (follows it across batch changes)
    c = tl.arange(0, BLOCK)
    current = tl.full((), 0, tl.int32)
    length = tl.full((), 1, tl.int32)
    running = tl.full((), 1, tl.int32)
    row = tl.load(INDEX + b * W)
    tl.store(PATH + b * D, row)
    # Entries beyond the accepted path are initialized by the launcher.
    while running != 0:
        child = tl.load(CHILD + b * W + current)
        raw = tl.load(NODES + b * (W - 1) + tl.maximum(child - 1, 0), mask=child >= 0, other=0)
        valid = (child >= 0) & ((current != 0) | tl.load(ROOT_VALID + store))
        slot = tl.where(raw < K, 0, 1 + (raw - K) // K)
        slot = tl.maximum(slot, 0)
        ids = tl.load(QIDS + (store * S + slot) * C + c, mask=(c < C) & valid, other=0)
        ids = tl.where(valid, ids, c)
        q = tl.load(Q + (store * S + slot) * C + c, mask=(c < C) & valid, other=0)
        residual = tl.load(P + row * V + ids, mask=c < C, other=0).to(tl.float32)
        outside = tl.maximum(1.0 - tl.sum(residual, 0), 0.0)
        found = tl.full((), 0, tl.int32)
        while (child >= 0) & valid & (found == 0) & (length < D):
            raw = tl.load(NODES + b * (W - 1) + child - 1)
            position = tl.load(POS + store * N + raw)
            qsum = tl.sum(q, 0)
            qconditional = q / tl.where(qsum > 0.0, qsum, 1.0)
            qy = tl.sum(tl.where(c == position, qconditional, 0.0), 0)
            py = tl.sum(tl.where(c == position, residual, 0.0), 0)
            mass = outside + tl.sum(residual, 0)
            coin = tl.load(COINS + b * W + child)
            accept = (qy > 0.0) & (py > 0.0) & (coin * mass * qy < py)
            if accept:
                token = tl.load(TOKENS + b * W + child)
                tl.store(PREDICT + row, token.to(tl.int32))
                current = child.to(tl.int32)
                row = tl.load(INDEX + b * W + child)
                tl.store(PATH + b * D + length, row.to(tl.int32))
                length += 1
                found = 1
            else:
                residual = tl.maximum(residual - mass * qconditional, 0.0)
                q = tl.where(c == position, 0.0, q)
                child = tl.load(SIBLING + b * W + child)
        if found == 0:
            running = 0
            tl.store(FINAL_ROW + b, row)
            tl.store(FINAL_IDS + b * C + c, ids, mask=c < C)
            tl.store(FINAL_RESIDUAL + b * C + c, residual, mask=c < C)
    tl.store(LENGTH + b, length)


def sample_tree(
    target_probs,
    q,
    q_ids,
    positions,
    root_valid,
    nodes,
    tokens,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    uniforms,
    final_uniforms,
    *,
    top_k,
    max_depth,
    req_index=None,
):
    """Return node-indexed predictions, accept lengths, and global accepted rows.

    q/ids are [B, expansion slots, candidates]; positions are [B, raw nodes].
    ``nodes`` maps non-root verify rows to raw draft nodes ([B, W - 1]).
    Siblings must retain an ordered prefix of the original draws. Every target
    row is normalized. All buffers are contiguous; vocabulary ids are valid.
    ``root_valid=False`` emits one target token, used when a prefill/batch change
    supplied deterministic roots without a recorded proposal distribution.
    ``req_index``: int64 ``[B]`` row of each request in ``q``/``q_ids``/``positions``/``root_valid`` (default: the
    batch position), so recorded proposals follow a request across batch changes.
    """
    bs, width, vocab = target_probs.shape
    slots, candidates = q.shape[1:]
    device = target_probs.device
    if req_index is None:
        req_index = torch.arange(bs, dtype=torch.int64, device=device)
    predict = torch.zeros(bs * width, dtype=torch.int32, device=device)
    path = torch.full((bs, max_depth), -1, dtype=torch.int32, device=device)
    lengths = torch.empty(bs, dtype=torch.int32, device=device)
    final_rows = torch.empty(bs, dtype=torch.int64, device=device)
    final_ids = torch.empty((bs, candidates), dtype=torch.int64, device=device)
    final_residual = torch.empty((bs, candidates), dtype=torch.float32, device=device)
    _walk[(bs,)](
        target_probs,
        q,
        q_ids,
        positions,
        root_valid,
        nodes,
        tokens,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        uniforms,
        predict,
        path,
        lengths,
        final_rows,
        final_ids,
        final_residual,
        req_index,
        V=vocab,
        W=width,
        D=max_depth,
        K=top_k,
        C=candidates,
        S=slots,
        N=positions.shape[1],
        BLOCK=triton.next_power_of_2(candidates),
    )
    residual = target_probs.reshape(-1, vocab).index_select(0, final_rows)
    residual.scatter_(1, final_ids, final_residual)
    cdf = residual.cumsum(dim=-1)
    threshold = final_uniforms.reshape(bs, 1) * cdf[:, -1:]
    bonus = torch.searchsorted(cdf, threshold.contiguous(), right=True).flatten().clamp_max(vocab - 1)
    predict.scatter_(0, final_rows, bonus.to(torch.int32))
    return predict, lengths, path
