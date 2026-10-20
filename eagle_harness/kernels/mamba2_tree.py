# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao, Albert Gu
# Derived from vLLM (Apache-2.0): the per-token update of _selective_scan_update_kernel in
# vllm/model_executor/layers/mamba/ops/mamba_ssm.py (itself adapted from mamba_ssm, Apache-2.0), with the parent-state
# reload of sglang's tree-mode selective_state_update (python/sglang/kernels/ops/mamba/triton_ops/mamba_ssm.py).
# Modified by Eagle AI Corp: row slots and parent tables for tree verify, per-head parameters, launch rule.
# Notices and license texts: THIRD_PARTY_NOTICES.md and LICENSES/.
"""Mamba2 tree verify: the selective state update of every tree node, each continuing from its parent.

A request's verify rows are tree nodes ordered parent first; row ``t`` of request ``b`` updates the SSM state of its
parent row (row 0: the request's state) with its own token and stores the result in its own state slot, the layout
vLLM's chain verify uses. One program walks a request's rows for one head and a block of head dims; a row whose parent
is the previous row continues from the state in registers, any other row reloads its parent's state from the parent's
slot. On a chain the kernel is therefore vLLM's chain kernel step for step (same operations in the same order).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from eagle_harness.kernels.gdn_tree import PAD_SLOT_ID


@triton.jit
def _exp(x):
    # exp through the hardware exp2, as vLLM's chain kernel computes it
    LOG2E = tl.constexpr(1.4426950408889634)
    return tl.math.exp2(LOG2E * x)


@triton.jit
def _softplus(dt):
    return tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)


@triton.jit
def _tree_ssu_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    dt_bias_ptr,
    out_ptr,
    slots_ptr,  # [batch, >= W] int32: the state slot of each row
    parents_ptr,  # [>= batch, W] int32: each row's parent row (row 0: -1)
    num_accepted_ptr,  # [batch]: request b starts from slots[b, num_accepted - 1]
    query_start_loc_ptr,  # [batch + 1]
    head_dim,
    dstate,
    heads_per_group,
    stride_state_slot,
    stride_state_head,
    stride_state_dim,
    stride_x_tok,
    stride_x_head,
    stride_dt_tok,
    stride_dt_head,
    stride_B_tok,
    stride_B_group,
    stride_C_tok,
    stride_C_group,
    stride_out_tok,
    stride_out_head,
    stride_slots,
    stride_parents,
    PAD_SLOT_ID: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_NUM_ACCEPTED: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m, b, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(query_start_loc_ptr + b).to(tl.int64)
    seq_len = tl.load(query_start_loc_ptr + b + 1).to(tl.int64) - bos
    if seq_len == 0:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < head_dim
    mask_n = offs_n < dstate
    mask = mask_m[:, None] & mask_n[None, :]
    tile = offs_m[:, None] * stride_state_dim + offs_n[None, :]
    head_state = state_ptr + h * stride_state_head
    slots = slots_ptr + b * stride_slots
    parents = parents_ptr + b * stride_parents

    init_row = 0
    if HAS_NUM_ACCEPTED:
        init_row = tl.maximum(tl.load(num_accepted_ptr + b) - 1, 0)
    slot = tl.load(slots + init_row).to(tl.int64)
    state = tl.load(head_state + slot * stride_state_slot + tile, mask=mask & (slot != PAD_SLOT_ID), other=0.0)
    state = state.to(tl.float32)

    A = tl.load(A_ptr + h).to(tl.float32)
    if HAS_D:
        D = tl.load(D_ptr + h).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + h).to(tl.float32)
    x_ptr += bos * stride_x_tok + h * stride_x_head
    dt_ptr += bos * stride_dt_tok + h * stride_dt_head
    B_ptr += bos * stride_B_tok + (h // heads_per_group) * stride_B_group
    C_ptr += bos * stride_C_tok + (h // heads_per_group) * stride_C_group
    out_ptr += bos * stride_out_tok + h * stride_out_head

    for t in range(seq_len):
        parent = tl.load(parents + t)
        if (t > 0) & (parent != t - 1):
            # A branch: continue from the parent row's stored state (this program stored it; the barrier orders the
            # other threads' stores of the tile before the reload).
            tl.debug_barrier()
            parent_slot = tl.load(slots + parent).to(tl.int64)
            state = tl.load(
                head_state + parent_slot * stride_state_slot + tile,
                mask=mask & (parent_slot != PAD_SLOT_ID),
                other=0.0,
            ).to(tl.float32)
        x = tl.load(x_ptr + t * stride_x_tok + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        dt = tl.load(dt_ptr + t * stride_dt_tok).to(tl.float32)
        if HAS_DT_BIAS:
            dt += dt_bias
        if DT_SOFTPLUS:
            dt = _softplus(dt)
        dA = _exp(A * dt)
        B = tl.load(B_ptr + t * stride_B_tok + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        C = tl.load(C_ptr + t * stride_C_tok + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        dB = B * dt
        state = state * dA + dB * x[:, None]
        row_slot = tl.load(slots + t).to(tl.int64)
        if row_slot != PAD_SLOT_ID:
            tl.store(head_state + row_slot * stride_state_slot + tile, state.to(state_ptr.dtype.element_ty), mask=mask)
        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        tl.store(out_ptr + t * stride_out_tok + offs_m, out, mask=mask_m)


def _launch_shape(head_dim: int, dstate: int, programs_per_block: int, device: torch.device) -> tuple[int, int]:
    """(BLOCK_M, num_warps): one warp per program holding ~32 state elements per thread, and half that while the grid
    has fewer than 16 programs per SM. The walk is latency-bound, so many small programs win: on a B300 with NemotronH
    Nano's layer (64 heads x 64 x 128, fp32 state) this was 1.3-1.9x faster than 4-8 warp programs over 16-64 head
    dims at batch 1-64, and 1.4-1.9x faster than vLLM's chain kernel at batch 1-16."""
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count if device.type == "cuda" else 1
    block_n = triton.next_power_of_2(dstate)
    block_m = max(1, min(triton.next_power_of_2(head_dim), 1024 // block_n))
    if block_m > max(1, 512 // block_n) and triton.cdiv(head_dim, block_m) * programs_per_block < 16 * sm_count:
        block_m //= 2
    return block_m, 1


def mamba2_tree_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    state_indices: torch.Tensor,
    parents: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    dt_softplus: bool = True,
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Mamba2's selective state update of a tree verify, in place.

    Shapes (per tensor-parallel rank): ``state`` (slots, heads, head_dim, dstate) with a contiguous last dim; ``x``
    and ``out`` (tokens, heads, head_dim) and ``dt`` (tokens, heads), any token and head strides with contiguous head
    dims; ``B`` / ``C`` (tokens, groups, dstate), contiguous last dim; ``A`` (float32, ``-exp(A_log)``), ``D`` and
    ``dt_bias`` (heads,). Request ``b``'s rows are tokens ``query_start_loc[b]:query_start_loc[b + 1]``; row ``t``
    continues from row ``parents[b, t]`` (``-1`` for row 0: the state at ``state_indices[b, num_accepted_tokens[b] -
    1]``, or column 0 without accepted counts) and stores its state at ``state_indices[b, t]``; a slot equal to
    ``pad_slot_id`` is neither read (zeros) nor written. ``out[t] = C_t . state_t + D * x_t`` with
    ``state_t = exp(A dt_t) state_parent + dt_t B_t x_t`` and ``dt_t = softplus(dt + dt_bias)`` (``dt_softplus``).
    """
    tokens, heads, head_dim = x.shape
    groups, dstate = B.shape[1], B.shape[2]
    batch = query_start_loc.shape[0] - 1
    assert state.shape[1:] == (heads, head_dim, dstate) and state.stride(-1) == 1
    assert dt.shape == (tokens, heads) and out.shape == x.shape and C.shape == B.shape == (tokens, groups, dstate)
    assert x.stride(-1) == 1 and out.stride(-1) == 1 and B.stride(-1) == 1 and C.stride(-1) == 1
    assert A.shape == (heads,) and A.dtype == torch.float32 and heads % groups == 0
    assert state_indices.shape[0] >= batch and parents.shape[0] >= batch
    assert parents.stride(-1) == 1 and state_indices.stride(-1) == 1
    block_m, num_warps = _launch_shape(head_dim, dstate, batch * heads, x.device)
    _tree_ssu_kernel[(triton.cdiv(head_dim, block_m), batch, heads)](
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        out,
        state_indices,
        parents,
        num_accepted_tokens,
        query_start_loc,
        head_dim,
        dstate,
        heads // groups,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        x.stride(0),
        x.stride(1),
        dt.stride(0),
        dt.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        out.stride(0),
        out.stride(1),
        state_indices.stride(0),
        parents.stride(0),
        PAD_SLOT_ID=pad_slot_id,
        HAS_D=D is not None,
        HAS_DT_BIAS=dt_bias is not None,
        HAS_NUM_ACCEPTED=num_accepted_tokens is not None,
        DT_SOFTPLUS=dt_softplus,
        BLOCK_M=block_m,
        BLOCK_N=triton.next_power_of_2(dstate),
        num_warps=num_warps,
    )
    return out
