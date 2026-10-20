# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Launchers of the native CUDA candidate kernels (cluster shard top-k, candidate tail).

Bit-identical to the Triton kernels of :mod:`eagle_harness.kernels.candidate_triton`. Each call returns False when it
launched nothing (hardware without the feature, library unavailable, shape not covered) and the caller runs the Triton
path; ``enabled`` is ``HarnessConfig.draft_model.cuda_candidate_kernels``. A CUDA error raises.
"""

from __future__ import annotations

from typing import Optional

import torch

from eagle_harness.kernels import arch, native

_NOT_COVERED = -1
_DTYPE = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}


def _launch(result: int, what: str) -> bool:
    if result == _NOT_COVERED:
        return False
    if result != 0:
        raise RuntimeError(f"eagle-harness native {what} failed with CUDA error {result}")
    return True


def _stream(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


def shard_topk(
    scores: torch.Tensor,
    k: int,
    vocab_start: int,
    local_ids: torch.Tensor,
    global_ids: torch.Tensor,
    *,
    enabled: bool = True,
) -> bool:
    """Fill ``local_ids`` / ``global_ids`` ([M, k] int64, unit column stride) like torch.topk(sorted=False)."""
    if not (enabled and arch.clusters(scores.device) and scores.dtype in (torch.bfloat16, torch.float16)):
        return False
    lib = native.library()
    if lib is None:
        return False
    rows, n = scores.shape
    return _launch(
        lib.eagle4_shard_topk(
            scores.device.index,
            _DTYPE[scores.dtype],
            scores.data_ptr(),
            scores.stride(0),
            rows,
            n,
            k,
            local_ids.data_ptr(),
            local_ids.stride(0),
            global_ids.data_ptr(),
            global_ids.stride(0),
            vocab_start,
            _stream(scores.device),
        ),
        "shard top-k",
    )


def candidate_tail(
    cand_logits: torch.Tensor,
    cand_tokens: torch.Tensor,
    num_candidates: int,
    top_k: int,
    logits_out: torch.Tensor,
    tokens_out: torch.Tensor,
    topk_probs: Optional[torch.Tensor],
    topk_tokens: Optional[torch.Tensor],
    *,
    enabled: bool = True,
) -> bool:
    """Candidate selection + softmax + top-k on [R, M, NCL] views (int64 tokens, or int32 with fp32 logits).

    False where not covered (R * NCL > 1024, NC not in 32/64/128/256, top_k > 32).
    """
    if not (enabled and arch.native_candidate_tail(cand_logits.device) and cand_logits.dtype in _DTYPE):
        return False
    lib = native.library()
    if lib is None:
        return False
    if topk_probs is None:  # top_k == 0: the kernel does not touch them
        topk_probs, topk_tokens = logits_out, tokens_out
    ranks, rows, ncl = cand_logits.shape
    return _launch(
        lib.eagle4_candidate_tail(
            cand_logits.device.index,
            _DTYPE[cand_logits.dtype],
            cand_tokens.element_size(),
            cand_logits.data_ptr(),
            cand_logits.stride(0),
            cand_logits.stride(1),
            cand_tokens.data_ptr(),
            cand_tokens.stride(0),
            cand_tokens.stride(1),
            logits_out.data_ptr(),
            logits_out.stride(0),
            tokens_out.data_ptr(),
            tokens_out.stride(0),
            topk_probs.data_ptr(),
            topk_probs.stride(0),
            topk_tokens.data_ptr(),
            topk_tokens.stride(0),
            ranks,
            rows,
            ncl,
            num_candidates,
            top_k,
            _stream(cand_logits.device),
        ),
        "candidate tail",
    )
