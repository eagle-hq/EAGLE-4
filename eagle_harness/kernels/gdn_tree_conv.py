# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao
# Derived from sglang (Apache-2.0): the tree conv from python/sglang/kernels/ops/mamba/causal_conv1d_triton.py (itself
# adapted from vLLM and causal-conv1d, BSD-3-Clause) and the recurrence from
# python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py (sglang's fla kernels adapt
# flash-linear-attention, MIT). Modified by Eagle AI Corp: ancestor tables, slot tables for both engines, path replay.
# Notices and license texts: THIRD_PARTY_NOTICES.md and LICENSES/.
"""GDN tree verify: the width-4 causal conv1d of every tree node through the ancestor table.

Each node's window is its three nearest ancestors' inputs (or the request's history columns above the root), read
through the table ``gdn_tree.gdn_tree_ancestors`` builds once per forward instead of chasing the tree links in every
layer and channel block. Output, windows and state are bit-identical to the tree path of sglang's
``causal_conv1d_update``. Two kernels compute the same values: the gather kernel takes all nodes of a request as one
tile, the serial kernel walks the nodes over a wide channel block, which wins only for many short trees.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import triton
import triton.language as tl

from eagle_harness.kernels import arch
from eagle_harness.kernels.gdn_tree import PAD_SLOT_ID


@triton.jit
def _conv_request(
    conv_state_ptr,
    conv_state_indices_ptr,
    stride_state_indices,
    num_accepted_ptr,
    query_start_loc_ptr,
    stride_conv_state_seq,
    pad_slot_id: tl.constexpr,
    seqlen: tl.constexpr,
    HAS_NUM_ACCEPTED: tl.constexpr,
    HAS_QUERY_START_LOC: tl.constexpr,
):
    # request idx_seq: its conv state row, first token, node count and history column; live: not padding, has nodes
    idx_seq = tl.program_id(0)
    slot = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices).to(tl.int64)
    zero = slot * 0
    bos = zero
    num_tokens = zero + seqlen
    history = zero
    if HAS_QUERY_START_LOC:
        bos = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
        num_tokens = tl.load(query_start_loc_ptr + idx_seq + 1).to(tl.int64) - bos
    if HAS_NUM_ACCEPTED:
        history = tl.load(num_accepted_ptr + idx_seq).to(tl.int64) - 1
    live = (slot > pad_slot_id) & (num_tokens > 0)
    return bos, num_tokens, conv_state_ptr + slot * stride_conv_state_seq, history, live


@triton.jit
def _conv_weights(w_ptr, idx_feats, mask_w, stride_w_dim, stride_w_width):
    # the 4 weight columns of this channel block
    w_base = w_ptr + idx_feats * stride_w_dim
    w_col0 = tl.load(w_base + 0 * stride_w_width, mask_w, other=0.0)
    w_col1 = tl.load(w_base + 1 * stride_w_width, mask_w, other=0.0)
    w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w, other=0.0)
    w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w, other=0.0)
    return w_col0, w_col1, w_col2, w_col3


@triton.jit
def _conv_history(conv_states_base, history, mask_w, stride_conv_state_tok):
    # the request's 3 history columns (from column `history`)
    col0 = tl.load(conv_states_base + history * stride_conv_state_tok, mask_w, 0.0)
    col1 = tl.load(conv_states_base + (history + 1) * stride_conv_state_tok, mask_w, 0.0)
    col2 = tl.load(conv_states_base + (history + 2) * stride_conv_state_tok, mask_w, 0.0)
    return col0, col1, col2


@triton.jit
def _conv_pdl_wait(USE_GDC: tl.constexpr):
    # Under PDL the launch starts while the input projection's GEMM runs: everything above this point reads only
    # tables, weights and metadata written before that GEMM. Padded programs wait too, so the grid never completes
    # early. The dependent launch is released at the end (_conv_pdl_release): released here, the recurrence's
    # programs sit on the SMs through the conv and the chain measured ~1.3 us per layer slower than this.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()


@triton.jit
def _conv_pdl_release(USE_GDC: tl.constexpr):
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _window_value(anc, x_anc, col0, col1, col2):
    # an ancestor code's input: node anc (x_anc) for anc >= 0, else history column 2 / 1 / 0 for -1 / -2 / -3
    return tl.where(anc >= 0, x_anc, tl.where(anc == -1, col2, tl.where(anc == -2, col1, col0)))


@triton.jit
def _conv_roll(
    conv_states_base,
    x_base,
    col0,
    col1,
    col2,
    num_tokens,
    mask_w,
    stride_conv_state_tok: tl.constexpr,
    stride_x_token,
    STATE_LEN: tl.constexpr,
):
    # The state row keeps the newest min(STATE_LEN, nodes + 2) entries of (history ++ inputs): the entries a later
    # window may still need. With 3 columns (sglang) that is upstream's in-place shift (the last 3 inputs); with room
    # for every input (vLLM) it is the chain kernel's roll [history[1:], x_0, .., x_{n-1}], which the commit compacts.
    # Runs after every history read.
    tl.debug_barrier()
    kept = tl.minimum(num_tokens + 2, STATE_LEN)
    for j in tl.static_range(STATE_LEN):
        entry = num_tokens + 3 - kept + j  # index into (col0, col1, col2, x_0, ..)
        if j < kept:
            state_dtype = conv_states_base.dtype.element_ty
            if entry < 3:
                new_col = tl.where(entry == 0, col0, tl.where(entry == 1, col1, col2)).to(state_dtype)
            else:
                new_col = tl.load(x_base + (entry - 3) * stride_x_token, mask=mask_w, other=0.0).to(state_dtype)
            tl.store(conv_states_base + j * stride_conv_state_tok, new_col, mask=mask_w)


@triton.jit
def _conv_gather_kernel(
    x_ptr,  # (batch, dim, seqlen), or (1, dim, tokens) with query_start_loc
    w_ptr,  # (dim, 4)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_ptr,
    query_start_loc_ptr,
    intermediate_conv_window_ptr,
    intermediate_state_indices_ptr,
    ancestors_ptr,  # (batch, 3, seqlen) int32
    o_ptr,  # like x
    dim,
    stride_x_seq,
    stride_x_dim: tl.constexpr,
    stride_x_token,
    stride_w_dim,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices,
    stride_inter_seq,
    stride_inter_step,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_intermediate_state_indices,
    stride_o_seq,
    stride_o_dim: tl.constexpr,
    stride_o_token,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_NUM_ACCEPTED: tl.constexpr,
    HAS_QUERY_START_LOC: tl.constexpr,
    HAS_WINDOWS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    seqlen: tl.constexpr,
    NP2_SEQLEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    # all nodes of a request at once: a [NP2_SEQLEN, BLOCK_N] tile whose ancestor rows are gathered through the table
    idx_seq = tl.program_id(0)
    bos, num_tokens, conv_states_base, history, live = _conv_request(
        conv_state_ptr,
        conv_state_indices_ptr,
        stride_state_indices,
        num_accepted_ptr,
        query_start_loc_ptr,
        stride_conv_state_seq,
        pad_slot_id,
        seqlen,
        HAS_NUM_ACCEPTED,
        HAS_QUERY_START_LOC,
    )
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_tokens = tl.arange(0, NP2_SEQLEN)
    mask_w = idx_feats < dim
    mask_t = idx_tokens < num_tokens
    mask_2d = mask_t[:, None] & mask_w[None, :]
    w_col0, w_col1, w_col2, w_col3 = _conv_weights(w_ptr, idx_feats, mask_w, stride_w_dim, stride_w_width)
    anc_base = ancestors_ptr + idx_seq * 3 * seqlen + idx_tokens
    anc1 = tl.load(anc_base, mask=mask_t, other=0)[:, None]
    anc2 = tl.load(anc_base + seqlen, mask=mask_t, other=0)[:, None]
    anc3 = tl.load(anc_base + 2 * seqlen, mask=mask_t, other=0)[:, None]
    if HAS_BIAS:
        acc = tl.broadcast_to(
            tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)[None, :],
            (NP2_SEQLEN, BLOCK_N),
        )
    else:
        acc = tl.zeros((NP2_SEQLEN, BLOCK_N), dtype=tl.float32)
    _conv_pdl_wait(USE_GDC)
    if live == 0:
        return

    conv_states_base += idx_feats * stride_conv_state_dim
    col0, col1, col2 = _conv_history(conv_states_base, history, mask_w, stride_conv_state_tok)

    x_feats = x_ptr + idx_seq * stride_x_seq + bos * stride_x_token + idx_feats * stride_x_dim
    x_base = x_feats[None, :]
    x_self = tl.load(x_base + (idx_tokens * stride_x_token)[:, None], mask=mask_2d, other=0.0)
    x_anc1 = tl.load(x_base + anc1 * stride_x_token, mask=mask_2d & (anc1 >= 0), other=0.0)
    x_anc2 = tl.load(x_base + anc2 * stride_x_token, mask=mask_2d & (anc2 >= 0), other=0.0)
    x_anc3 = tl.load(x_base + anc3 * stride_x_token, mask=mask_2d & (anc3 >= 0), other=0.0)
    h0 = col0[None, :]
    h1 = col1[None, :]
    h2 = col2[None, :]
    v_anc1 = _window_value(anc1, x_anc1, h0, h1, h2)
    v_anc2 = _window_value(anc2, x_anc2, h0, h1, h2)
    v_anc3 = _window_value(anc3, x_anc3, h0, h1, h2)

    # the upstream products and order: self * w3, parent * w2, grandparent * w1, great-grandparent * w0
    acc += x_self * w_col3[None, :]
    acc += v_anc1 * w_col2[None, :]
    acc += v_anc2 * w_col1[None, :]
    acc += v_anc3 * w_col0[None, :]
    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))
    tl.store(
        o_ptr
        + idx_seq * stride_o_seq
        + ((bos + idx_tokens) * stride_o_token)[:, None]
        + (idx_feats * stride_o_dim)[None, :],
        acc,
        mask=mask_2d,
    )

    if HAS_WINDOWS:
        # windows as one [NP2_SEQLEN, BLOCK_N, 4] tile: slot 2 self, slot 1 parent, slot 0 grandparent, slot 3 masked
        intermediate_state_batch_coord = tl.load(
            intermediate_state_indices_ptr + idx_seq * stride_intermediate_state_indices
        ).to(tl.int64)
        win_dtype = intermediate_conv_window_ptr.dtype.element_ty
        win_self = x_self.to(win_dtype)
        win_anc1 = v_anc1.to(win_dtype)
        win_anc2 = v_anc2.to(win_dtype)
        # join(join(a, b), join(c, d)) reshaped to 4 orders the slots (a, c, b, d)
        windows = tl.reshape(
            tl.join(tl.join(win_anc2, win_self), tl.join(win_anc1, win_self)),
            (NP2_SEQLEN, BLOCK_N, 4),
        )
        idx_win = tl.arange(0, 4)
        tl.store(
            intermediate_conv_window_ptr
            + intermediate_state_batch_coord * stride_inter_seq
            + (idx_tokens * stride_inter_step)[:, None, None]
            + (idx_feats * stride_inter_dim)[None, :, None]
            + (idx_win * stride_inter_win)[None, None, :],
            windows,
            mask=mask_2d[:, :, None] & (idx_win < 3)[None, None, :],
        )

    # the roll of _conv_roll, with the inputs stored from the tile: entry e goes to column e + shift
    tl.debug_barrier()
    shift = tl.minimum(num_tokens + 2, STATE_LEN) - num_tokens - 3
    state_dtype = conv_states_base.dtype.element_ty
    tl.store(conv_states_base + shift * stride_conv_state_tok, col0, mask=mask_w & (shift >= 0))
    tl.store(conv_states_base + (shift + 1) * stride_conv_state_tok, col1, mask=mask_w & (shift + 1 >= 0))
    tl.store(conv_states_base + (shift + 2) * stride_conv_state_tok, col2, mask=mask_w & (shift + 2 >= 0))
    dest = idx_tokens + 3 + shift
    tl.store(
        conv_states_base[None, :] + (dest * stride_conv_state_tok)[:, None],
        x_self.to(state_dtype),
        mask=mask_2d & (dest >= 0)[:, None],
    )
    _conv_pdl_release(USE_GDC)


@triton.jit
def _conv_serial_kernel(
    x_ptr,  # (batch, dim, seqlen), or (1, dim, tokens) with query_start_loc
    w_ptr,  # (dim, 4)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_ptr,
    query_start_loc_ptr,
    intermediate_conv_window_ptr,
    intermediate_state_indices_ptr,
    ancestors_ptr,  # (batch, 3, seqlen) int32
    o_ptr,  # like x
    dim,
    stride_x_seq,
    stride_x_dim: tl.constexpr,
    stride_x_token,
    stride_w_dim,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices,
    stride_inter_seq,
    stride_inter_step,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_intermediate_state_indices,
    stride_o_seq,
    stride_o_dim: tl.constexpr,
    stride_o_token,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_NUM_ACCEPTED: tl.constexpr,
    HAS_QUERY_START_LOC: tl.constexpr,
    HAS_WINDOWS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    seqlen: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    # one node at a time over a wide channel block (many requests)
    idx_seq = tl.program_id(0)
    bos, num_tokens, conv_states_base, history, live = _conv_request(
        conv_state_ptr,
        conv_state_indices_ptr,
        stride_state_indices,
        num_accepted_ptr,
        query_start_loc_ptr,
        stride_conv_state_seq,
        pad_slot_id,
        seqlen,
        HAS_NUM_ACCEPTED,
        HAS_QUERY_START_LOC,
    )
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_w = idx_feats < dim
    w_col0, w_col1, w_col2, w_col3 = _conv_weights(w_ptr, idx_feats, mask_w, stride_w_dim, stride_w_width)
    if HAS_BIAS:
        acc_preload = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
    _conv_pdl_wait(USE_GDC)
    if live == 0:
        return

    conv_states_base += idx_feats * stride_conv_state_dim
    col0, col1, col2 = _conv_history(conv_states_base, history, mask_w, stride_conv_state_tok)

    x_base = x_ptr + idx_seq * stride_x_seq + bos * stride_x_token + idx_feats * stride_x_dim
    o_base = o_ptr + idx_seq * stride_o_seq + bos * stride_o_token + idx_feats * stride_o_dim
    anc_base = ancestors_ptr + idx_seq * 3 * seqlen
    if HAS_WINDOWS:
        intermediate_state_batch_coord = tl.load(
            intermediate_state_indices_ptr + idx_seq * stride_intermediate_state_indices
        ).to(tl.int64)
        win_base = (
            intermediate_conv_window_ptr
            + intermediate_state_batch_coord * stride_inter_seq
            + idx_feats * stride_inter_dim
        )
    for idx_token in range(num_tokens):
        anc1 = tl.load(anc_base + idx_token)
        anc2 = tl.load(anc_base + seqlen + idx_token)
        anc3 = tl.load(anc_base + 2 * seqlen + idx_token)
        x_self = tl.load(x_base + idx_token * stride_x_token, mask=mask_w, other=0.0)
        x_anc1 = tl.load(x_base + anc1 * stride_x_token, mask=mask_w & (anc1 >= 0), other=0.0)
        x_anc2 = tl.load(x_base + anc2 * stride_x_token, mask=mask_w & (anc2 >= 0), other=0.0)
        x_anc3 = tl.load(x_base + anc3 * stride_x_token, mask=mask_w & (anc3 >= 0), other=0.0)
        v_anc1 = _window_value(anc1, x_anc1, col0, col1, col2)
        v_anc2 = _window_value(anc2, x_anc2, col0, col1, col2)
        v_anc3 = _window_value(anc3, x_anc3, col0, col1, col2)

        acc = acc_preload
        acc += x_self * w_col3
        acc += v_anc1 * w_col2
        acc += v_anc2 * w_col1
        acc += v_anc3 * w_col0
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        tl.store(o_base + idx_token * stride_o_token, acc, mask=mask_w)

        if HAS_WINDOWS:
            win_ptr = win_base + idx_token * stride_inter_step
            tl.store(win_ptr + 2 * stride_inter_win, x_self, mask=mask_w)
            tl.store(win_ptr + 1 * stride_inter_win, v_anc1, mask=mask_w)
            tl.store(win_ptr + 0 * stride_inter_win, v_anc2, mask=mask_w)

    _conv_roll(
        conv_states_base, x_base, col0, col1, col2, num_tokens, mask_w, stride_conv_state_tok, stride_x_token, STATE_LEN
    )
    _conv_pdl_release(USE_GDC)


# The serial kernel wins only for many short trees.
CONV_SERIAL_MIN_BATCH_DIM = 81920
CONV_SERIAL_MAX_WIDTH = 16


def gdn_tree_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Union[bool, str, None],
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    tree_ancestors: torch.Tensor,
    pad_slot_id: int = PAD_SLOT_ID,
    out: Optional[torch.Tensor] = None,
    *,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``causal_conv1d_update`` of a tree verify (width-4 kernel, no accept offsets) whose ancestor table came from
    ``gdn_tree_ancestors`` (or a table with its codes). x: (batch, dim, seqlen) with any strides (e.g. a token-strided
    view of the in_proj output); weight: (dim, 4). out (same shape, any strides) defaults to ``torch.empty_like(x)``.

    Request n's history is the 3 columns of ``conv_state[conv_state_indices[n]]`` from column 0, or from column
    ``num_accepted_tokens[n] - 1`` (vLLM's chain-kernel convention); a row whose index is <= ``pad_slot_id`` is
    padding. With ``query_start_loc`` x is (1, dim, tokens) and request n's nodes are tokens
    ``query_start_loc[n]:query_start_loc[n + 1]`` (at most ``tree_ancestors.shape[2]``). Each node's window goes to
    ``intermediate_conv_window[intermediate_state_indices[n], node]`` if given; the state row then keeps the newest
    ``min(conv_state.shape[-1], nodes + 2)`` entries of (history ++ inputs).

    16-bit inputs only: their products are rounded before the fp32 accumulation, so the sums match the upstream
    kernel whichever multiply-adds the compiler fuses; with fp32 inputs the upstream fusion varies with seqlen.
    """
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    batch, dim, seqlen = x.shape
    width = tree_ancestors.shape[2]
    if query_start_loc is not None:
        assert batch == 1
        batch, seqlen = query_start_loc.shape[0] - 1, width
    assert x.element_size() == 2 and weight.element_size() == 2 and conv_state.element_size() == 2
    assert weight.shape[1] == 4 and tree_ancestors.shape[0] >= batch and tree_ancestors.shape[1:] == (3, seqlen)
    assert tree_ancestors.is_contiguous() and conv_state.shape[-1] >= 3
    has_windows = intermediate_conv_window is not None
    if out is None:
        out = torch.empty_like(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    args = (
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        intermediate_conv_window,
        intermediate_state_indices,
        tree_ancestors,
        out,
        dim,
        0 if query_start_loc is not None else x.stride(0),
        *x.stride()[1:],
        *weight.stride(),
        *conv_state.stride(),
        conv_state_indices.stride(0),
        *(intermediate_conv_window.stride() if has_windows else (0, 0, 0, 0)),
        intermediate_state_indices.stride(0) if has_windows else 0,
        0 if query_start_loc is not None else out.stride(0),
        *out.stride()[1:],
    )
    meta = dict(
        pad_slot_id=pad_slot_id,
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_NUM_ACCEPTED=num_accepted_tokens is not None,
        HAS_QUERY_START_LOC=query_start_loc is not None,
        HAS_WINDOWS=has_windows,
        STATE_LEN=conv_state.shape[-1],
        seqlen=seqlen,
    )
    # PDL: the launch overlaps the input projection's GEMM (weights and tables are read ahead of the wait, see
    # _conv_pdl_wait); the recurrence behind it is released as the programs finish.
    pdl = arch.pdl(x.device)
    meta["USE_GDC"] = pdl
    launch = {"launch_pdl": True} if pdl else {}
    if batch * dim >= CONV_SERIAL_MIN_BATCH_DIM and seqlen <= CONV_SERIAL_MAX_WIDTH:
        block_n = 128
        _conv_serial_kernel[(batch, triton.cdiv(dim, block_n))](*args, BLOCK_N=block_n, num_warps=4, **meta, **launch)
    else:
        block_n = 16 if batch == 1 and seqlen >= 16 else 32
        _conv_gather_kernel[(batch, triton.cdiv(dim, block_n))](
            *args,
            NP2_SEQLEN=triton.next_power_of_2(seqlen),
            BLOCK_N=block_n,
            num_warps=block_n // 8,
            **meta,
            **launch,
        )
    return out
