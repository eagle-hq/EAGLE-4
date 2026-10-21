# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Launchers of the EAGLE-4 candidate top-k kernels (shard top-k, candidate tail) and the fused candidate head.

Each runs the native CUDA kernel of
:mod:`eagle_harness.kernels.candidate` where the GPU and the shape allow it (shard top-k: sm_90+, tail: sm_80+),
else the Triton kernels of :mod:`eagle_harness.kernels.candidate_triton` (any supported GPU); both are
bit-identical to the NVIDIA torch reference in their docstrings, equal values included. On ROCm the
selection is mathematically equivalent, but torch's unsorted/tied order and fp32 softmax rounding can differ.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton

from eagle_harness.kernels import candidate
from eagle_harness.kernels.candidate_triton import (
    eagle4_candidate_tail_kernel,
    eagle4_shard_topk_chunk_kernel,
    eagle4_shard_topk_merge_kernel,
)
from eagle_harness.kernels.gather_dot import gather_dot, gather_dot_packed

# Past ~8M scores the Triton kernels (ALU bound) lose to torch's radix select.
_SHARD_TOPK_MAX_ELEMS = 8_000_000
_16BIT_FLOATS = (torch.float16, torch.bfloat16)


def shard_topk(
    scores: torch.Tensor,
    k: int,
    vocab_start: int,
    *,
    cuda_kernels: bool = True,
    max_triton_elements: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices of the ``k`` largest scores per row (NVIDIA: torch's unsorted order).

    Returns ``(local_ids, global_ids)``, int64 ``[M, k]``; ``global_ids`` is
    ``local_ids + vocab_start``.
    """
    rows, n = scores.shape
    if scores.stride(-1) != 1 or not 0 < k <= n:
        raise ValueError(f"shard_topk needs unit column stride and 0 < k <= {n}, got k={k}")
    limit = _SHARD_TOPK_MAX_ELEMS if max_triton_elements is None else max_triton_elements
    if rows * n > limit or triton.cdiv(n, 8192) * k >= 32768:
        local_ids = torch.topk(scores, k, dim=-1, sorted=False).indices
        return local_ids, local_ids + vocab_start
    local_ids = torch.empty((rows, k), dtype=torch.int64, device=scores.device)
    global_ids = torch.empty_like(local_ids)
    if rows == 0:
        return local_ids, global_ids
    if candidate.shard_topk(scores, k, vocab_start, local_ids, global_ids, enabled=cuda_kernels):
        return local_ids, global_ids
    chunk = 2048 if n <= 32768 else (4096 if n <= 131072 else 8192)
    num_chunks = triton.cdiv(n, chunk)
    cand_keys = torch.empty((rows, num_chunks, k), dtype=torch.int32, device=scores.device)
    cand_idx = torch.empty_like(cand_keys)
    eagle4_shard_topk_chunk_kernel[(rows, num_chunks)](
        scores, scores.stride(0), n, cand_keys, cand_idx, K=k, CH=chunk, KEY_BITS=16, num_warps=8
    )
    num_cand = num_chunks * k
    eagle4_shard_topk_merge_kernel[(rows,)](
        cand_keys,
        cand_idx,
        num_cand,
        local_ids,
        local_ids.stride(0),
        global_ids,
        global_ids.stride(0),
        vocab_start,
        K=k,
        NBK_P=triton.next_power_of_2(num_cand),
        KEY_BITS=16,
        num_warps=8,
    )
    return local_ids, global_ids


def _tail_num_warps(width: int) -> int:
    if width <= 256:
        return 1
    if width <= 512:
        return 2
    return 4 if width <= 1024 else 8


def candidate_tail(
    cand_logits: torch.Tensor,
    cand_tokens: torch.Tensor,
    num_rows: int,
    top_k: int,
    *,
    cuda_kernels: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Everything after the candidate all-gather, in one launch.

    ``cand_logits`` / ``cand_tokens``: ``[R * M, NC]`` rank-major (an all-gather
    along dim 0), or ``[R, M, NC]`` views of any row and rank strides; tokens
    int64, or int32 with fp32 logits (the packed gather of :func:`candidate_topk`).
    Equivalent (bit for bit on NVIDIA) to::

        logits, pos = torch.topk(gathered_logits, NC, sorted=False)  # R > 1 only
        tokens = torch.gather(gathered_tokens, -1, pos)
        logits = logits.float()
        probs, pick = torch.topk(torch.softmax(logits, -1), top_k)
        topk_tokens = torch.gather(tokens, -1, pick)

    on the ``[M, R * NC]`` rank-concatenated view. Returns ``(logits [M, NC]
    fp32, tokens [M, NC] int64, (probs, topk_tokens) or None when top_k == 0)``.
    """
    if cand_tokens.shape != cand_logits.shape:
        raise ValueError(f"candidate_tail: logits {tuple(cand_logits.shape)} vs tokens {tuple(cand_tokens.shape)}")
    if cand_logits.dim() == 3:
        num_ranks, rows, nc = cand_logits.shape
        stride_cl_r, stride_cl_m = cand_logits.stride(0), cand_logits.stride(1)
        stride_ct_r, stride_ct_m = cand_tokens.stride(0), cand_tokens.stride(1)
    else:
        rows, nc = num_rows, cand_logits.shape[1]
        num_ranks = cand_logits.shape[0] // max(rows, 1)
        stride_cl_r, stride_cl_m = rows * cand_logits.stride(0), cand_logits.stride(0)
        stride_ct_r, stride_ct_m = rows * cand_tokens.stride(0), cand_tokens.stride(0)
    if rows != num_rows or num_ranks * rows != cand_logits.shape[:-1].numel():
        raise ValueError(
            f"candidate_tail: {tuple(cand_logits.shape)} is not [R * {num_rows}, NC] or [R, {num_rows}, NC]"
        )
    if (
        cand_tokens.dtype not in (torch.int64, torch.int32)
        or cand_logits.stride(-1) != 1
        or cand_tokens.stride(-1) != 1
    ):
        raise ValueError("candidate_tail needs int64 or int32 tokens and unit column strides")
    device = cand_logits.device
    logits_out = torch.empty((rows, nc), dtype=torch.float32, device=device)
    tokens_out = (
        torch.empty((rows, nc), dtype=torch.int64, device=device)
        if num_ranks > 1
        else cand_tokens.as_strided((rows, nc), (stride_ct_m, 1))
    )
    topk = None
    if top_k > 0:
        topk = (
            torch.empty((rows, top_k), dtype=torch.float32, device=device),
            torch.empty((rows, top_k), dtype=torch.int64, device=device),
        )
    if rows == 0:
        return logits_out, tokens_out, topk
    topk_probs, topk_tokens = topk if topk is not None else (logits_out, tokens_out)
    views = (
        cand_logits.as_strided((num_ranks, rows, nc), (stride_cl_r, stride_cl_m, 1)),
        cand_tokens.as_strided((num_ranks, rows, nc), (stride_ct_r, stride_ct_m, 1)),
    )
    if candidate.candidate_tail(
        views[0], views[1], nc, top_k, logits_out, tokens_out, topk_probs, topk_tokens, enabled=cuda_kernels
    ):
        return logits_out, tokens_out, topk
    width = num_ranks * nc
    eagle4_candidate_tail_kernel[(rows,)](
        cand_logits,
        stride_cl_r,
        stride_cl_m,
        cand_tokens,
        stride_ct_r,
        stride_ct_m,
        logits_out,
        logits_out.stride(0),
        tokens_out,
        tokens_out.stride(0),
        topk_probs,
        topk_probs.stride(0),
        topk_tokens,
        topk_tokens.stride(0),
        R=num_ranks,
        NCL=nc,
        NC=nc,
        CP=triton.next_power_of_2(width),
        P=triton.next_power_of_2(nc),
        KEY_BITS=16 if cand_logits.dtype in _16BIT_FLOATS else 32,
        TOPK=top_k,
        num_warps=_tail_num_warps(width),
    )
    return logits_out, tokens_out, topk


def candidate_head(
    scores: torch.Tensor,
    lm_head: torch.Tensor,
    hidden: torch.Tensor,
    *,
    num_candidates: int,
    vocab_start: int,
    top_k: int,
    all_gather: Callable[[torch.Tensor], torch.Tensor] | None = None,
    cuda_kernels: bool = True,
    max_triton_elements: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Candidate logits, ids and top-k probabilities for each row (None for ``top_k == 0``).

    ``scores`` [rows, shard vocab] are this rank's low-rank scores (padding columns already masked), ``lm_head``
    [shard vocab, H] its exact weights. Shard top-k, exact re-scoring (:func:`gather_dot`), the candidate tail; under TP
    ``all_gather`` merges the ranks' packed candidates (``[M, n] -> [R, M, n]`` in rank order, any strides, so a gather
    along either dim is a view), the logits and ids written by one launch (:func:`gather_dot_packed`) and read in place
    by the tail.
    """
    rows = hidden.shape[0]
    local_ids, global_ids = shard_topk(
        scores,
        num_candidates,
        vocab_start,
        cuda_kernels=cuda_kernels,
        max_triton_elements=max_triton_elements,
    )
    if all_gather is None:
        logits, tokens = gather_dot(lm_head, local_ids, hidden), global_ids
    else:
        ranks = all_gather(gather_dot_packed(lm_head, local_ids, hidden, vocab_start))
        logits, tokens = ranks[..., :num_candidates].view(torch.float32), ranks[..., num_candidates:]
    return candidate_tail(logits, tokens, rows, top_k, cuda_kernels=cuda_kernels)


def candidate_topk(
    scores: torch.Tensor,
    lm_head: torch.Tensor,
    hidden: torch.Tensor,
    *,
    num_candidates: int,
    vocab_start: int,
    top_k: int,
    all_gather: Callable[[torch.Tensor], torch.Tensor] | None = None,
    cuda_kernels: bool = True,
    max_triton_elements: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k probabilities and ids; use candidate_head to retain proposal logits."""
    return candidate_head(
        scores, lm_head, hidden, num_candidates=num_candidates,
        vocab_start=vocab_start, top_k=top_k, all_gather=all_gather,
        cuda_kernels=cuda_kernels,
        max_triton_elements=max_triton_elements,
    )[2]
