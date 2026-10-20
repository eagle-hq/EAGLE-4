# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Score gathered LM-head rows: ``out[b, k] = weight[indices[b, k]] . hidden[b]``, products and sum in fp32.

Products and tile sums are fp32: bf16 ones give a logit error of 2.7% on average (20x a bf16 GEMM's), fp32 products are
exact. The per-row-count tile table is tuned on B300 (static, so CUDA-graph safe). :func:`gather_dot` stores the hidden
dtype. :func:`gather_dot_packed` is the TP variant that fuses the candidate packing of the all-gather into the same
launch: ``[rows, 2K]`` int32 holding the logits (rounded to the hidden dtype, as fp32 bits) and then
``indices + vocab_start``, the layout :func:`~eagle_harness.kernels.candidate_ops.candidate_topk` all-gathers and the
candidate tail reads, in place of four elementwise launches.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl


class _Tiles(NamedTuple):
    block_k: int
    block_h: int
    num_warps: int
    num_stages: int


def _tiles(rows: int) -> _Tiles:
    # One candidate per program and two warps up to 16 rows (the draft step's 4 to 16), two candidates per program
    # beyond: the fp32 products are register-heavy, so wide-warp tiles lost. Swept on B300 (62080 x 5120 bf16,
    # block_k 1-16, block_h 256-2048, warps 2-8, stages 2-4); static, so CUDA-graph safe.
    if rows <= 16:
        return _Tiles(1, 2048, 2, 3)
    if rows <= 32:
        return _Tiles(2, 2048, 2, 3)
    return _Tiles(2, 1024, 2, 3)


@triton.jit
def _gather_dot_kernel(
    weight_ptr,  # [V, H] lm_head weight
    indices_ptr,  # [B, K] candidate vocab ids
    hidden_ptr,  # [B, H]
    output_ptr,  # [B, K] hidden dtype, or [B, 2K] int32 when PACKED
    stride_w_v,
    stride_w_h,
    stride_i_b,
    stride_i_k,
    stride_h_b,
    stride_h_h,
    stride_o_b,
    stride_o_k,
    vocab_start,
    K,
    H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PACKED: tl.constexpr,
):
    # Each program handles BLOCK_K candidates of one row, so hidden[b, :] is loaded once per tile instead of once per
    # candidate (the K-per-program variant spent half its bandwidth re-reading hidden).
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    # int64: vocab_id * stride_w_v can exceed int32 for large vocab * hidden.
    vocab_ids = tl.load(indices_ptr + pid_b * stride_i_b + k_offs * stride_i_k, mask=k_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    EVEN_H: tl.constexpr = H % BLOCK_H == 0
    for off_h in range(0, H, BLOCK_H):
        cols = off_h + tl.arange(0, BLOCK_H)
        if EVEN_H:
            h_val = tl.load(hidden_ptr + pid_b * stride_h_b + cols * stride_h_h)
            w_val = tl.load(
                weight_ptr + vocab_ids[:, None] * stride_w_v + cols[None, :] * stride_w_h,
                mask=k_mask[:, None],
                other=0.0,
            )
        else:
            h_mask = cols < H
            h_val = tl.load(hidden_ptr + pid_b * stride_h_b + cols * stride_h_h, mask=h_mask, other=0.0)
            w_val = tl.load(
                weight_ptr + vocab_ids[:, None] * stride_w_v + cols[None, :] * stride_w_h,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
        acc += tl.sum(w_val.to(tl.float32) * h_val.to(tl.float32)[None, :], axis=1)

    out_row = output_ptr + pid_b * stride_o_b
    if PACKED:
        logits = acc.to(hidden_ptr.dtype.element_ty).to(tl.float32)
        tl.store(out_row + k_offs * stride_o_k, logits.to(tl.int32, bitcast=True), mask=k_mask)
        tl.store(out_row + (K + k_offs) * stride_o_k, (vocab_ids + vocab_start).to(tl.int32), mask=k_mask)
    else:
        tl.store(out_row + k_offs * stride_o_k, acc.to(output_ptr.dtype.element_ty), mask=k_mask)


def gather_dot_reference(weight: torch.Tensor, indices: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """The same product in torch (fp32 accumulate); for CPU and tests."""
    rows = weight[indices.long()].float()  # [B, K, H]
    return torch.einsum("bkh,bh->bk", rows, hidden.float()).to(hidden.dtype)


def pack_reference(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """``[rows, 2K]`` int32: the fp32 logit bits, then the token ids (vocab ids fit in int32)."""
    return torch.cat([logits.float().view(torch.int32), token_ids.to(torch.int32)], dim=-1)


def _launch(
    weight: torch.Tensor,
    indices: torch.Tensor,
    hidden: torch.Tensor,
    output: torch.Tensor,
    vocab_start: int,
    packed: bool,
) -> None:
    rows, k = indices.shape
    h = weight.shape[1]
    tiles = _tiles(rows)
    _gather_dot_kernel[(rows, triton.cdiv(k, tiles.block_k))](
        weight, indices, hidden, output,
        weight.stride(0), weight.stride(1), indices.stride(0), indices.stride(1),
        hidden.stride(0), hidden.stride(1), output.stride(0), output.stride(1),
        vocab_start, k,
        H=h, BLOCK_K=tiles.block_k, BLOCK_H=min(tiles.block_h, triton.next_power_of_2(h)), PACKED=packed,
        num_warps=tiles.num_warps, num_stages=tiles.num_stages,
    )  # fmt: skip


def _check(weight: torch.Tensor, indices: torch.Tensor, hidden: torch.Tensor) -> None:
    if hidden.shape != (indices.shape[0], weight.shape[1]):
        raise ValueError(f"hidden {tuple(hidden.shape)} != ({indices.shape[0]}, {weight.shape[1]})")


def gather_dot(
    weight: torch.Tensor, indices: torch.Tensor, hidden: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``weight`` [V, H], ``indices`` [B, K] (row ids into ``weight``), ``hidden`` [B, H] -> [B, K]."""
    _check(weight, indices, hidden)
    rows, k = indices.shape
    if not weight.is_cuda:
        result = gather_dot_reference(weight, indices, hidden)
        if out is None:
            return result
        out.copy_(result)
        return out
    output = torch.empty((rows, k), device=indices.device, dtype=hidden.dtype) if out is None else out
    if output.shape != (rows, k) or output.dtype != hidden.dtype:
        raise ValueError(f"out {tuple(output.shape)} {output.dtype} != ({rows}, {k}) {hidden.dtype}")
    if rows and k:
        _launch(weight, indices, hidden, output, 0, packed=False)
    return output


def gather_dot_packed(
    weight: torch.Tensor, indices: torch.Tensor, hidden: torch.Tensor, vocab_start: int
) -> torch.Tensor:
    """:func:`gather_dot` fused with :func:`pack_reference` of ``(result, indices + vocab_start)``: [B, 2K] int32."""
    _check(weight, indices, hidden)
    rows, k = indices.shape
    if not weight.is_cuda:
        return pack_reference(gather_dot_reference(weight, indices, hidden), indices + vocab_start)
    output = torch.empty((rows, 2 * k), device=indices.device, dtype=torch.int32)
    if rows and k:
        _launch(weight, indices, hidden, output, vocab_start, packed=True)
    return output
