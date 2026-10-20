# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao
# Derived from sglang (Apache-2.0): the tree conv from python/sglang/kernels/ops/mamba/causal_conv1d_triton.py (itself
# adapted from vLLM and causal-conv1d, BSD-3-Clause) and the recurrence from
# python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py (sglang's fla kernels adapt
# flash-linear-attention, MIT). Modified by Eagle AI Corp: ancestor tables, slot tables for both engines, path replay.
# Notices and license texts: THIRD_PARTY_NOTICES.md and LICENSES/.
"""Gated delta rule recurrence over a draft tree: the SSM half of a GDN layer's tree verify.

The math
--------
A gated delta net layer keeps one state matrix ``h`` of shape [K, V] per value head. Each token updates it and reads
it out (the math of sglang's ``fused_sigmoid_gating_delta_rule_update``)::

    g    = -exp(A_log) * softplus(a + dt_bias)      # log decay, per value head
    beta = sigmoid(b)                               # write strength, rounded to b's dtype
    q, k = q / |q|, k / |k|;  q *= scale             # L2 norms over K
    h    = h * exp(g)
    v'   = (v - h^T k) * beta
    h    = h + k v'^T
    o    = h^T q

A chain (plain decode) applies this left to right. In a tree verify every node continues from its *parent's* state,
so a node's state is the fold of the update over its root path. Every node's output ``o`` and state (its
*snapshot*) are written, because the accepted path is only known after verification; the request's own state is
never written back here (the engines copy the accepted node's snapshot afterwards).

Data layout
-----------
- ``q``, ``k``: (1, tokens, H, K); ``v``: (1, tokens, HV, V); ``a``, ``b``: (tokens, HV). Strided along tokens only:
  serving passes views into one packed projection. ``HV`` is a multiple of ``H`` (grouped value heads).
- A request's nodes are rows ``cu_seqlens[n]:cu_seqlens[n + 1]`` in node order, parents first. Row 0 is the root.
- States and snapshots are stored per value head as [V, K] with K contiguous: ``[..., HV, V, K]``. Kernels hold the
  state transposed, as a [K, V-tile] register tile.
- Snapshots are addressed through slot tables, which lets one kernel serve both engines:

  ============  =======================================  ==========================================
  engine        snapshot buffer                          slot table
  ============  =======================================  ==========================================
  sglang        [slots, steps, HV, V, K] scratch          [N]: one scratch row per request
  vLLM          [slots, HV, V, K] (the state pool)        [N, S]: one pool slot per verify row
  ============  =======================================  ==========================================

  A slot ``<= pad_slot_id`` is padding (sglang -1, vLLM's null block 0): nothing is stored there, and an initial
  state read from it is zero.

The kernels
-----------
- ``_walk_kernel`` visits a request's nodes in order. A node starts from its parent's snapshot, reloaded from memory;
  a chain child (parent = previous node) with an fp32 state keeps the registers instead, which is bitwise the same
  value. One program owns a 4-column V tile, so each step of the serial loop is short and many programs run in
  parallel. It is compiled without FMA contraction and writes its multiply-adds as explicit FMAs, so its Gluon twin
  (``gdn_tree_gluon``), which writes the same FMAs, reproduces it bitwise.
- ``_path_replay_kernel`` gives each first-child chain one program that replays the chain's whole root path from the
  request state in registers, with no snapshot reads. It wins when the batch has few (request, value head) rows
  (``sglang.gdn_tree_verify`` decides) and needs an fp32 state, so that skipping the reloads changes nothing. It is
  compiled with default FMA contraction, so it is not bitwise equal to the walk; the tests compare the two within
  tolerance.

:func:`gdn_tree_recurrent_update` validates a launch and picks the kernel: the path replay when given path tables and
an fp32 state, else the Gluon twin where its dispatcher admits the launch, else the walk.
"""

from __future__ import annotations

import functools
import importlib
import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from eagle_harness.kernels import arch
from eagle_harness.kernels.gdn_tree import PAD_SLOT_ID

logger = logging.getLogger(__name__)

# V columns per program. The walk is latency-bound on its serial loop over the nodes: a narrow tile keeps each step
# of that loop short and gives eight times the programs of the upstream kernel's 32-wide tile. The replay carries two
# steps of inputs in flight per program and uses a wider tile.
_WALK_BLOCK_V = 4
_REPLAY_BLOCK_V = 8
# Added to |q|^2 and |k|^2 before the square root, as the upstream kernel's L2 norm does.
_L2_NORM_EPS = tl.constexpr(1e-6)


@triton.jit
def _load_node_inputs(ptrs, strides, row, mask_k, mask_v, valid):
    """Loads the raw inputs of the node ``row`` rows past ``ptrs``, zero where ``valid`` is false.

    Args:
        ptrs: (q, k, v, a, b) pointers of one row: K-vectors of q and k (one key head), a V tile of v, and the a and b
            scalars of one value head.
        strides: their token strides.
        row: rows to advance, a scalar.
        mask_k, mask_v: in-bounds masks of the K and V offsets.
        valid: scalar; false for a node past the end of the request.

    Returns:
        (q, k, v, a, b) in the input dtypes.
    """
    p_q, p_k, p_v, p_a, p_b = ptrs
    stride_q, stride_k, stride_v, stride_a, stride_b = strides
    q = tl.load(p_q + row * stride_q, mask=mask_k & valid, other=0)
    k = tl.load(p_k + row * stride_k, mask=mask_k & valid, other=0)
    v = tl.load(p_v + row * stride_v, mask=mask_v & valid, other=0)
    b = tl.load(p_b + row * stride_b, mask=valid, other=0)
    a = tl.load(p_a + row * stride_a, mask=valid, other=0)
    return q, k, v, a, b


@triton.jit
def _log_decay(a, A_log, dt_bias, softplus_beta, softplus_threshold):
    """Returns a node's log decay ``g = -exp(A_log) * softplus(a + dt_bias)``, fp32.

    ``a`` is the node's raw gate input; ``A_log`` and ``dt_bias`` are fp32. ``softplus`` is linear above
    ``softplus_threshold`` (in units of ``softplus_beta * x``), like ``torch.nn.functional.softplus``.
    """
    x = a.to(tl.float32) + dt_bias
    beta_x = softplus_beta * x
    softplus_x = tl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    return -tl.exp(A_log) * softplus_x


@triton.jit
def _write_strength(b, input_dtype: tl.constexpr):
    """Returns a node's ``beta = sigmoid(b)``, fp32, rounded through ``input_dtype`` (the dtype of the ``b`` input).

    The GDN prefill and decode kernels round beta to the input dtype; the verify must too, or its states drift from
    the ones decode would compute for the same tokens.
    """
    beta = 1.0 / (1.0 + tl.exp(-b.to(tl.float32)))
    return beta.to(input_dtype).to(tl.float32)


@triton.jit
def _normalized_query_key(q, k, scale):
    """Returns ``q / |q| * scale`` and ``k / |k|`` as fp32 [K] vectors (raw q and k in)."""
    q = q.to(tl.float32)
    k = k.to(tl.float32)
    q = q / (tl.sqrt(tl.sum(q * q) + _L2_NORM_EPS))
    k = k / (tl.sqrt(tl.sum(k * k) + _L2_NORM_EPS))
    q = q * scale
    return q, k


@triton.jit
def _dot_k(h, k):
    """Returns ``h^T k`` (sum over K of ``h[K, V] * k[K]``) as a [V] vector, with an FMA per pair of K rows.

    Adjacent K rows sit in one thread's registers; folding each pair into one written FMA fixes the rounding, so the
    Gluon twin's ``_dot_k``, which pairs the same rows, gives the same bits. K must be even.
    """
    kk = tl.broadcast_to(k[:, None], h.shape)
    h_even, h_odd = tl.split(tl.permute(tl.reshape(h, [h.shape[0] // 2, 2, h.shape[1]]), (0, 2, 1)))
    k_even, k_odd = tl.split(tl.permute(tl.reshape(kk, [h.shape[0] // 2, 2, h.shape[1]]), (0, 2, 1)))
    return tl.sum(tl.fma(h_even, k_even, h_odd * k_odd), 0)


@triton.jit
def _delta_rule_step(h, q, k, v, g, beta):
    """Applies one node's gated delta rule update to the state tile and reads it out (the walk's rounding).

    Args:
        h: [K, BV] fp32 state tile of the node's parent.
        q, k: [K] fp32, normalized (:func:`_normalized_query_key`).
        v: [BV] fp32 values of the tile.
        g, beta: fp32 scalars from :func:`_log_decay` and :func:`_write_strength`.

    Returns:
        (h, o): the node's [K, BV] state and its [BV] fp32 output ``h^T q``.
    """
    h *= tl.exp(g)
    v -= _dot_k(h, k)
    v *= beta
    h = tl.fma(k[:, None], v[None, :], h)
    o = tl.sum(h * q[:, None], 0)
    return h, o


@triton.jit
def _publish_slot_table(
    publish, stride_publish, p_slots, stride_snap_indices_row, i_n, num_slot_cols, SP: tl.constexpr
):
    """Copies request ``i_n``'s row of the per-row slot table to ``publish``.

    vLLM reads the table after the verify to find the accepted row's slot; writing it here saves a launch. Called by
    one program per request.
    """
    o_s = tl.arange(0, SP)
    m_s = o_s < num_slot_cols
    row = tl.load(p_slots + o_s * stride_snap_indices_row, mask=m_s)
    tl.store(publish + i_n * stride_publish + o_s, row, mask=m_s)


@triton.jit
def _walk_kernel(
    inputs,  # (q, k, v, a, b): see gdn_tree_recurrent_update
    input_strides,  # token strides of (q, k, v, a, b)
    gate_params,  # (A_log [HV], dt_bias [HV], softplus_beta, softplus_threshold)
    output,  # (o like v, its token stride)
    parents,  # (parent table [N, >= T] int32: the parent row of each row, its request stride, its row stride)
    # State addressing: named as in the Gluon twin's kernel, which takes the same gdn_tree_state_args.
    h0_source,
    h0_indices,
    stride_h0_source,
    stride_h0_indices_seq,
    stride_h0_indices_col,
    num_accepted,
    cu_seqlens,
    intermediate_states_buffer,
    intermediate_state_indices,
    stride_snap_indices_seq,
    stride_snap_indices_row,
    stride_snap_slot,
    stride_snap_row,
    publish,
    stride_publish,
    num_slot_cols,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SP: tl.constexpr,
    SKIP_CHAIN_RELOAD: tl.constexpr,
    ROW_SLOTS: tl.constexpr,
    HAS_NUM_ACCEPTED: tl.constexpr,
    PUBLISH: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    """Runs the recurrence over every node of one request, for one value head and one BV-wide V tile.

    Grid: (ceil(V / BV), requests, HV). Each node starts from its parent's snapshot (row 0 from the initial state; rows of
    a request without snapshot slots run as one chain) and writes its output and snapshot. The next node's parent and
    inputs are loaded one step ahead to hide their latency behind the current node's math.

    Snapshot addressing: with ``ROW_SLOTS`` row ``r`` of request ``n`` lives at slot ``intermediate_state_indices[n,
    r]``; otherwise every row of request ``n`` lives in scratch row ``intermediate_state_indices[n]``, at step ``r``.
    ``SKIP_CHAIN_RELOAD`` keeps a chain child's state in registers; the launcher sets it only for an fp32 state, where
    the registers equal the reload bitwise.
    """
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    q, k, v, a, b = inputs
    stride_q, stride_k, stride_v, stride_a, stride_b = input_strides
    A_log, dt_bias, softplus_beta, softplus_threshold = gate_params
    o, stride_o = output
    parent_table, stride_parent_seq, stride_parent_row = parents
    i_h = i_hv // (HV // H)
    p_slots = intermediate_state_indices + i_n * stride_snap_indices_seq
    if PUBLISH:
        if (i_v == 0) & (i_hv == 0):
            _publish_slot_table(publish, stride_publish, p_slots, stride_snap_indices_row, i_n, num_slot_cols, SP)

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_a = a + bos * stride_a + i_hv
    p_o = o + bos * stride_o + i_hv * V + o_v
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # the initial state; with HAS_NUM_ACCEPTED (vLLM's chain-kernel convention) its slot is in the column of the last
    # token the previous round accepted
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_idx = h0_indices + i_n * stride_h0_indices_seq
    if HAS_NUM_ACCEPTED:
        p_idx += (tl.load(num_accepted + i_n).to(tl.int64) - 1) * stride_h0_indices_col
    idx = tl.load(p_idx).to(tl.int64)
    if idx > PAD_SLOT_ID:
        b_h += tl.load(
            h0_source + idx * stride_h0_source + i_hv * K * V + o_v[None, :] * K + o_k[:, None],
            mask=mask_h,
            other=0,
        ).to(tl.float32)

    cache_idx = tl.load(p_slots).to(tl.int64)  # the request's scratch row (ROW_SLOTS: row 0's slot)
    o_cache = i_hv * K * V + o_v[None, :] * K + o_k[:, None]
    p_parent = parent_table + i_n * stride_parent_seq
    n_parent = tl.load(p_parent, mask=T > 0, other=0)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    n_q, n_k, n_v, n_a, n_b = _load_node_inputs((p_q, p_k, p_v, p_a, p_b), input_strides, 0, mask_k, mask_v, T > 0)

    step_idx = 0
    for _ in range(0, T):
        parent_step_idx = n_parent
        n_parent = tl.load(p_parent + (step_idx + 1) * stride_parent_row, mask=step_idx + 1 < T, other=0)
        slot = cache_idx
        parent_slot = cache_idx
        if ROW_SLOTS:
            slot = tl.load(p_slots + step_idx * stride_snap_indices_row).to(tl.int64)
            # the root's parent is no row (-1 in vLLM's tables): read nothing for it
            parent_slot = tl.load(
                p_slots + parent_step_idx * stride_snap_indices_row, mask=step_idx != 0, other=PAD_SLOT_ID
            ).to(tl.int64)
        reload_parent = step_idx != 0 and parent_slot > PAD_SLOT_ID
        if SKIP_CHAIN_RELOAD:
            reload_parent = reload_parent and parent_step_idx != step_idx - 1
        if reload_parent:
            b_h = tl.load(
                intermediate_states_buffer
                + parent_slot * stride_snap_slot
                + parent_step_idx * stride_snap_row
                + o_cache,
                mask=mask_h,
                other=0,
            ).to(tl.float32)

        has_next = step_idx + 1 < T
        b_q = n_q.to(tl.float32)
        b_k = n_k.to(tl.float32)
        b_v = n_v.to(tl.float32)
        b_b = n_b.to(tl.float32)
        b_a = n_a.to(tl.float32)
        n_q, n_k, n_v, n_a, n_b = _load_node_inputs(
            (p_q, p_k, p_v, p_a, p_b), input_strides, 1, mask_k, mask_v, has_next
        )

        b_g = _log_decay(b_a, b_A_log, b_dt_bias, softplus_beta, softplus_threshold)
        b_beta = _write_strength(b_b, b.dtype.element_ty)
        b_q, b_k = _normalized_query_key(b_q, b_k, scale)
        b_h, b_o = _delta_rule_step(b_h, b_q, b_k, b_v, b_g, b_beta)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        if slot > PAD_SLOT_ID:
            tl.store(
                intermediate_states_buffer + slot * stride_snap_slot + step_idx * stride_snap_row + o_cache,
                b_h.to(intermediate_states_buffer.dtype.element_ty),
                mask=mask_h,
            )

        step_idx += 1
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += stride_o
        p_a += stride_a


@triton.jit
def _path_token(toks, offs_d, d, linear):
    """Returns the row offset of the node at depth ``d`` of a root path: ``toks[d]``, or ``d`` for a ``linear`` request
    (one without snapshot slots, which runs its rows in order as one chain, as the walk does)."""
    return tl.where(linear, d, tl.sum(tl.where(offs_d == d, toks, 0)))


@triton.jit
def _path_replay_kernel(
    inputs,  # (q, k, v, a, b): see gdn_tree_recurrent_update
    input_strides,  # token strides of (q, k, v, a, b)
    gate_params,  # (A_log [HV], dt_bias [HV], softplus_beta, softplus_threshold)
    output,  # (o like v, its token stride)
    initial_state,  # (state pool [slots, HV, V, K], slot per request [N] int32, pool slot stride)
    snapshots,  # (scratch [slots, steps, HV, V, K], scratch row per request [N] int32, allocated steps per row)
    paths,  # (path table [N, P, 2 + DMAX] int32, its request stride, its path stride)
    cu_seqlens,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DMAX: tl.constexpr,
    DMAX_P2: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    """Replays one first-child chain's whole root path from the request state, for one value head and V tile.

    Grid: (ceil(V / BV), requests * HV, P). Path row ``p`` of request ``n`` is ``[length, write_from, root path...]``
    (:func:`~eagle_harness.kernels.gdn_tree.gdn_tree_ancestors` builds it). The program folds the update over the
    path in registers and writes outputs and snapshots only for depths ``>= write_from``: the chain the row owns, so
    every node is written by exactly one program.

    The loop is software-pipelined to shorten the serial chain per step to the state update: raw inputs are loaded
    two steps ahead, and the state-independent terms (norms, gate, beta) are formed one step ahead.
    """
    i_v, i_nh, i_p = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    q, k, v, a, b = inputs
    stride_q, stride_k, stride_v, stride_a, stride_b = input_strides
    A_log, dt_bias, softplus_beta, softplus_threshold = gate_params
    o, stride_o = output
    h0_source, h0_indices, stride_h0_source = initial_state
    intermediate_states_buffer, intermediate_state_indices, cache_steps = snapshots
    tree_paths, stride_paths_seq, stride_paths_path = paths
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)
    p_path = tree_paths + i_n * stride_paths_seq + i_p * stride_paths_path
    # a request without snapshots never reloads a parent: the walk runs its rows as one chain, here path 0
    linear = cache_idx < 0
    L = tl.where(linear, tl.where(i_p == 0, (eos - bos).to(tl.int32), 0), tl.load(p_path))
    wfrom = tl.where(linear, 0, tl.load(p_path + 1))
    if L > 0:
        offs_d = tl.arange(0, DMAX_P2)
        toks = tl.load(p_path + 2 + offs_d, mask=(offs_d < DMAX) & (offs_d < L) & (cache_idx >= 0)).to(tl.int64)
        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]
        o_cache = i_hv * K * V + o_v[None, :] * K + o_k[:, None]
        p_q = q + i_h * K + o_k
        p_k = k + i_h * K + o_k
        p_v = v + i_hv * V + o_v
        p_b = b + i_hv
        p_a = a + i_hv
        b_A_log = tl.load(A_log + i_hv).to(tl.float32)
        b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

        ptrs = (p_q, p_k, p_v, p_a, p_b)
        # raw inputs of step 0 (formed below) and step 1 (pending)
        t0 = bos + _path_token(toks, offs_d, 0, linear)
        # step 0 always exists here (L > 0), so its loads need no validity mask
        r_q = tl.load(p_q + t0 * stride_q, mask=mask_k, other=0)
        r_k = tl.load(p_k + t0 * stride_k, mask=mask_k, other=0)
        r_v = tl.load(p_v + t0 * stride_v, mask=mask_v, other=0)
        r_b = tl.load(p_b + t0 * stride_b)
        r_a = tl.load(p_a + t0 * stride_a)
        has1 = L > 1
        n_t = bos + _path_token(toks, offs_d, 1, linear)
        n_q, n_k, n_v, n_a, n_b = _load_node_inputs(ptrs, input_strides, n_t, mask_k, mask_v, has1)

        # step 0's state-independent terms
        c_eg = tl.exp(_log_decay(r_a, b_A_log, b_dt_bias, softplus_beta, softplus_threshold))
        c_beta = _write_strength(r_b, b.dtype.element_ty)
        c_q, c_k = _normalized_query_key(r_q, r_k, scale)
        c_v = r_v.to(tl.float32)
        c_t = t0

        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            b_h += tl.load(h0_source + idx * stride_h0_source + o_cache, mask=mask_h, other=0).to(tl.float32)
        p_cache = intermediate_states_buffer + cache_idx * cache_steps * HV * K * V

        for d in range(0, L):
            u_eg = c_eg
            u_beta = c_beta
            u_q = c_q
            u_k = c_k
            u_v = c_v
            u_t = c_t
            # step d + 1: state-independent terms from the pending raw inputs
            c_eg = tl.exp(_log_decay(n_a, b_A_log, b_dt_bias, softplus_beta, softplus_threshold))
            c_beta = _write_strength(n_b, b.dtype.element_ty)
            c_q, c_k = _normalized_query_key(n_q, n_k, scale)
            c_v = n_v.to(tl.float32)
            c_t = n_t
            # step d + 2: raw loads
            has2 = d + 2 < L
            n_t = bos + _path_token(toks, offs_d, d + 2, linear)
            n_q, n_k, n_v, n_a, n_b = _load_node_inputs(ptrs, input_strides, n_t, mask_k, mask_v, has2)
            # step d: the update (contracted by the compiler, unlike the walk's written FMAs)
            b_h *= u_eg
            b_v = u_v - tl.sum(b_h * u_k[:, None], 0)
            b_v *= u_beta
            b_h += u_k[:, None] * b_v[None, :]
            if d >= wfrom:
                b_o = tl.sum(b_h * u_q[:, None], 0)
                tl.store(o + u_t * stride_o + i_hv * V + o_v, b_o.to(o.dtype.element_ty), mask=mask_v)
                if cache_idx >= 0:
                    tl.store(
                        p_cache + (u_t - bos) * HV * K * V + o_cache,
                        b_h.to(intermediate_states_buffer.dtype.element_ty),
                        mask=mask_h,
                    )


# The Gluon twin of the walk is imported on first use, so callers that never ask for it never import Gluon. An import
# error (a Triton build without Gluon) keeps the Triton walk for the rest of the process.
_gluon_module_cache = None


def _gluon_module():
    """Returns the ``gdn_tree_gluon`` module, or False when it cannot be imported (logged once)."""
    global _gluon_module_cache
    if _gluon_module_cache is None:
        try:
            _gluon_module_cache = importlib.import_module("eagle_harness.kernels.gdn_tree_gluon")
        except Exception as e:
            logger.warning(
                "GDN tree verify: Gluon kernel unavailable (%s: %s); using the Triton walk.", type(e).__name__, e
            )
            _gluon_module_cache = False
    return _gluon_module_cache


def gdn_tree_state_args(
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    num_accepted_tokens: Optional[torch.Tensor],
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    publish: Optional[torch.Tensor],
) -> dict:
    """Returns the state-addressing kernel arguments of the walk and its Gluon twin, after checking the layout.

    Arguments are as in :func:`gdn_tree_recurrent_update`. A tensor argument a launch does not use (``num_accepted``
    without ``num_accepted_tokens``, ``publish`` when no table is published) is filled with another valid pointer so
    the signature stays fixed; its constexpr flag keeps the kernel from reading it.

    Raises:
        AssertionError: the slot table's rank does not match the buffer's (a [N] table needs a 5-D scratch buffer, a
            [N, S] table a 4-D pool), ``num_accepted_tokens`` is given without a 2-D initial-state table or the
            reverse, or ``publish`` is given with a per-request table.
    """
    row_slots = intermediate_state_indices.ndim == 2
    if row_slots:
        assert intermediate_states_buffer.ndim == 4, "one slot per row addresses a [slots, HV, V, K] buffer"
    else:
        assert intermediate_states_buffer.ndim == 5, "one slot per request addresses a [slots, steps, HV, V, K] buffer"
    assert (initial_state_indices.ndim == 2) == (num_accepted_tokens is not None)
    assert publish is None or row_slots, "publish copies a table of per-row slots"
    return dict(
        stride_h0_source=initial_state_source.stride(0),
        stride_h0_indices_seq=initial_state_indices.stride(0),
        stride_h0_indices_col=initial_state_indices.stride(1) if initial_state_indices.ndim == 2 else 0,
        num_accepted=initial_state_indices if num_accepted_tokens is None else num_accepted_tokens,
        stride_snap_indices_seq=intermediate_state_indices.stride(0),
        stride_snap_indices_row=intermediate_state_indices.stride(1) if row_slots else 0,
        stride_snap_slot=intermediate_states_buffer.stride(0),
        stride_snap_row=0 if row_slots else intermediate_states_buffer.stride(1),
        publish=intermediate_state_indices if publish is None else publish,
        stride_publish=0 if publish is None else publish.stride(0),
        num_slot_cols=intermediate_state_indices.shape[-1],
        SP=triton.next_power_of_2(intermediate_state_indices.shape[-1]),
        ROW_SLOTS=row_slots,
        HAS_NUM_ACCEPTED=num_accepted_tokens is not None,
        PUBLISH=publish is not None,
    )


def gdn_tree_recurrent_update(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    retrieve_parent_token: torch.Tensor,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    out: Optional[torch.Tensor] = None,
    publish: Optional[torch.Tensor] = None,
    tree_paths: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    gluon: bool = False,
) -> torch.Tensor:
    """Runs the gated delta rule over every node of a verify tree and stores each node's state as a snapshot.

    The tree counterpart of ``fused_sigmoid_gating_delta_rule_update`` (q/k L2 norm in the kernel, per-node
    snapshots, no write-back of the request state). Row ``r`` of request ``n`` continues from the snapshot of its
    parent row ``retrieve_parent_token[n, r]``; row 0 continues from the request's initial state. The module
    docstring describes the math and the layouts.

    Args:
        A_log, dt_bias: [HV] gate parameters of the layer.
        q, k: (1, tokens, H, K); K a power of two.
        v: (1, tokens, HV, V).
        a, b: (tokens, HV) gate inputs; ``beta`` is rounded to ``b``'s dtype.
        initial_state_source: the state pool, [slots, HV, V, K].
        initial_state_indices: [N] int32 pool slot of each request's state, or with ``num_accepted_tokens`` [N, C]
            (vLLM), where the state is at column ``num_accepted_tokens[n] - 1``.
        cu_seqlens: [N + 1] int32 row offsets of the requests; rows are in node order, parents first.
        intermediate_states_buffer: snapshot buffer, [slots, steps, HV, V, K] (sglang scratch) or [slots, HV, V, K]
            (vLLM; may be ``initial_state_source`` itself).
        intermediate_state_indices: [N] int32 scratch row per request (5-D buffer) or [N, S] slot per row (4-D).
        retrieve_parent_token: [N, >= rows] int32 parent row of each row, within the request; entry 0 is ignored.
        num_accepted_tokens: [N] int32, see ``initial_state_indices``.
        pad_slot_id: slots ``<= pad_slot_id`` are padding (sglang -1, vLLM 0): never written, read as zero.
        out: optional (1, tokens, HV, V) output, token-strided, contiguous over (HV, V).
        publish: optional [N, >= S] int32; receives the [N, S] slot table (per-row slots only).
        tree_paths: optional [N, P, 2 + DMAX] int32 path table (``gdn_tree_ancestors(paths=...)``); with an fp32 pool
            and scratch it selects the path replay. sglang's per-request tables and ``pad_slot_id`` -1 only.
        scale: q scale, default ``K ** -0.5``.
        softplus_beta, softplus_threshold: parameters of the gate's softplus.
        gluon: let the Gluon twin of the walk serve the launches
            :func:`~eagle_harness.kernels.gdn_tree_gluon.gluon_dispatch` admits (bitwise equal output).

    Returns:
        ``out`` when given, otherwise a tensor shaped like ``v`` with ``q``'s dtype and device.

    Raises:
        AssertionError: on a layout the kernels do not address (see :func:`gdn_tree_state_args`), a K that is not a
            power of two, or an ``out`` that is not contiguous over (HV, V).
    """
    _, _, H, K = k.shape
    V = v.shape[-1]
    HV = v.shape[2]
    N = cu_seqlens.shape[0] - 1
    BK = triton.next_power_of_2(K)
    assert BK == K, "K must be a power of two"
    if scale is None:
        scale = K**-0.5
    # Registers kept across a chain step, or a replayed path, equal the reloaded snapshot only while both stay fp32: a
    # bf16 scratch rounds the reload, and a bf16 pool adds a second load layout (another K summation order).
    fp32_state = intermediate_states_buffer.dtype == torch.float32 and initial_state_source.dtype == torch.float32
    if out is None:
        out = q.new_empty(v.shape)
    assert out.shape == v.shape and out.stride(3) == 1 and out.stride(2) == V
    use_paths = tree_paths is not None and fp32_state
    if gluon and not use_paths and arch.gluon_gdn(q.device):
        o = _gluon_or_none(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a,
            b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            scale=scale,
            cu_seqlens=cu_seqlens,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            retrieve_parent_token=retrieve_parent_token,
            pad_slot_id=pad_slot_id,
            out=out,
            publish=publish,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
        )
        if o is not None:
            return o
    # PDL: the kernel waits for its predecessor in-kernel (gdc_wait), so its launch overlaps the predecessor's tail.
    pdl = {"USE_GDC": True, "launch_pdl": True} if arch.pdl(q.device) else {}
    common = dict(
        inputs=(q, k, v, a, b),
        input_strides=(q.stride()[1], k.stride()[1], v.stride()[1], a.stride()[-2], b.stride()[-2]),
        gate_params=(A_log, dt_bias, softplus_beta, softplus_threshold),
        output=(out, out.stride(1)),
        cu_seqlens=cu_seqlens,
        scale=scale,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        **pdl,
    )
    if use_paths:
        assert tree_paths.shape[0] >= N and tree_paths.dtype == torch.int32
        assert initial_state_indices.ndim == 1 and intermediate_state_indices.ndim == 1 and publish is None
        assert pad_slot_id == PAD_SLOT_ID, "the path replay takes sglang's per-request tables"
        _launch_path_replay(
            common,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            tree_paths=tree_paths,
        )
    else:
        _launch_walk(
            common,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            retrieve_parent_token=retrieve_parent_token,
            publish=publish,
            pad_slot_id=pad_slot_id,
            skip_chain_reload=fp32_state,
        )
    return out


def _gluon_or_none(A_log, dt_bias, q, k, v, a, b, *, retrieve_parent_token, **launch) -> Optional[torch.Tensor]:
    """Runs the Gluon twin of the walk if its dispatcher admits the launch; returns o, or None for the Triton walk.

    ``launch`` holds :func:`gdn_tree_recurrent_update`'s keyword arguments with ``scale`` resolved and ``out`` set.
    """
    gluon_kernel = _gluon_module()
    if not gluon_kernel:
        return None
    return gluon_kernel.gluon_dispatch(
        # the dispatcher's self-check reference: this launcher restricted to the Triton walk
        functools.partial(gdn_tree_recurrent_update, gluon=False),
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        # a verify graph's parent buffer is as wide as its widest verify; the Gluon kernel sizes its precompute by it
        retrieve_parent_token=retrieve_parent_token[:, : q.shape[1]],
        use_pdl=arch.pdl(q.device),
        **launch,
    )


def _launch_walk(
    common: dict,
    *,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    num_accepted_tokens: Optional[torch.Tensor],
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    retrieve_parent_token: torch.Tensor,
    publish: Optional[torch.Tensor],
    pad_slot_id: int,
    skip_chain_reload: bool,
) -> None:
    """Launches :func:`_walk_kernel`, one program per (V tile, request, value head).

    ``common`` holds the arguments both kernels take; the others are :func:`gdn_tree_recurrent_update`'s.
    """
    V, HV = common["V"], common["HV"]
    BV = min(triton.next_power_of_2(V), _WALK_BLOCK_V)
    _walk_kernel[(triton.cdiv(V, BV), common["cu_seqlens"].shape[0] - 1, HV)](
        **common,
        parents=(retrieve_parent_token, retrieve_parent_token.stride(0), retrieve_parent_token.stride(1)),
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        **gdn_tree_state_args(
            initial_state_source,
            initial_state_indices,
            num_accepted_tokens,
            intermediate_states_buffer,
            intermediate_state_indices,
            publish,
        ),
        BV=BV,
        SKIP_CHAIN_RELOAD=skip_chain_reload,
        PAD_SLOT_ID=pad_slot_id,
        # One warp per program: the loop over the nodes is serial, so a 4-column tile has too little work for more.
        num_warps=1,
        num_stages=3,
        # Only the written FMAs fuse, so the Gluon twin, which writes the same ones, rounds alike on any Triton.
        enable_fp_fusion=False,
    )


def _launch_path_replay(
    common: dict,
    *,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    tree_paths: torch.Tensor,
) -> None:
    """Launches :func:`_path_replay_kernel`, one program per (V tile, request x value head, path row).

    ``common`` holds the arguments both kernels take; the others are :func:`gdn_tree_recurrent_update`'s.
    """
    V, HV, K = common["V"], common["HV"], common["K"]
    dmax = tree_paths.shape[2] - 2
    BV = min(triton.next_power_of_2(V), _REPLAY_BLOCK_V)
    _path_replay_kernel[(triton.cdiv(V, BV), (common["cu_seqlens"].shape[0] - 1) * HV, tree_paths.shape[1])](
        **common,
        initial_state=(initial_state_source, initial_state_indices, initial_state_source.stride(0)),
        # snapshot rows are strided by the buffer's allocated step count, which can exceed this forward's width
        snapshots=(
            intermediate_states_buffer,
            intermediate_state_indices,
            intermediate_states_buffer.stride(0) // (HV * K * V),
        ),
        paths=(tree_paths, tree_paths.stride(0), tree_paths.stride(1)),
        BV=BV,
        DMAX=dmax,
        DMAX_P2=triton.next_power_of_2(dmax),
        num_warps=1,
        num_stages=1,
    )
