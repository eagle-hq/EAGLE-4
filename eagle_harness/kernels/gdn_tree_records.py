# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
# Modified by Eagle AI Corp.
"""AMD tree recurrence records and accepted-path replay for ROCm."""

import functools

import torch
import triton
import triton.language as tl

from eagle_harness.kernels.gdn_tree_recurrence import _path_token as _gdn_tree_path_token


@triton.jit
def _gdn_tree_records_kernel(
    q_pre,  # (tokens, H, K) fp32: normalized q * scale (_gdn_tree_pre_kernel)
    k_pre,  # (tokens, H, K) fp32: normalized k
    gb_pre,  # (tokens, HV, 2) fp32: exp(g), beta
    v,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    records,  # (rows, steps, HV, REC) fp32: delta [0, V), key [V, V + K), decay at V + K
    record_indices,
    stride_records_row,
    tree_paths,  # (batch, P, 2 + DMAX) int32: length, write_from, root path
    stride_paths_seq,
    stride_paths_path,
    stride_v,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REC: tl.constexpr,
    DMAX: tl.constexpr,
    DMAX_P2: tl.constexpr,
    ROUND_STATE: tl.constexpr,
):
    # The path replay with replay records instead of snapshots. ROUND_STATE (a bf16 scratch): every node but the
    # first starts from its parent's state rounded to bf16, the state the serial walk reloads from the parent's bf16
    # snapshot. Each owned node writes its record: delta (its V block), and from V block 0 the key and the decay.
    # The per-token math (normalized q and k, decay, beta) comes from the pre kernel: computed once per token and
    # head instead of in every V block's serial loop.
    i_v, i_nh, i_p = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    rec_idx = tl.load(record_indices + i_n).to(tl.int64)
    p_path = tree_paths + i_n * stride_paths_seq + i_p * stride_paths_path
    # a row without records never reloads a parent in the serial walk: one chain (path 0), no rounding
    linear = rec_idx < 0
    L = tl.where(linear, tl.where(i_p == 0, (eos - bos).to(tl.int32), 0), tl.load(p_path))
    wfrom = tl.where(linear, 0, tl.load(p_path + 1))
    if L > 0:
        offs_d = tl.arange(0, DMAX_P2)
        toks = tl.load(p_path + 2 + offs_d, mask=(offs_d < DMAX) & (offs_d < L) & (rec_idx >= 0)).to(tl.int64)
        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]
        o_state = i_hv * K * V + o_v[None, :] * K + o_k[:, None]
        p_v = v + i_hv * V + o_v
        p_rec = records + rec_idx * stride_records_row + i_hv * REC

        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            b_h += tl.load(h0_source + idx * stride_h0_source + o_state, mask=mask_h, other=0).to(tl.float32)

        for d in range(0, L):
            u_t = bos + _gdn_tree_path_token(toks, offs_d, d, linear)
            u_q = tl.load(q_pre + (u_t * H + i_h) * K + o_k, mask=mask_k, other=0)
            u_k = tl.load(k_pre + (u_t * H + i_h) * K + o_k, mask=mask_k, other=0)
            u_eg = tl.load(gb_pre + (u_t * HV + i_hv) * 2)
            u_beta = tl.load(gb_pre + (u_t * HV + i_hv) * 2 + 1)
            r_v = tl.load(p_v + u_t * stride_v, mask=mask_v, other=0).to(tl.float32)
            # the serial kernel's per-node math, in its order
            if ROUND_STATE:
                if (d > 0) & (rec_idx >= 0):
                    b_h = b_h.to(h0_source.dtype.element_ty).to(tl.float32)
            b_h *= u_eg
            b_v = r_v - tl.sum(b_h * u_k[:, None], 0)
            b_v *= u_beta
            b_h += u_k[:, None] * b_v[None, :]
            if d >= wfrom:
                b_o = tl.sum(b_h * u_q[:, None], 0)
                tl.store(o + (u_t * HV + i_hv) * V + o_v, b_o.to(o.dtype.element_ty), mask=mask_v)
                if rec_idx >= 0:
                    p_node = p_rec + (u_t - bos) * HV * REC
                    tl.store(p_node + o_v, b_v, mask=mask_v)
                    if i_v == 0:
                        tl.store(p_node + V + o_k, u_k, mask=mask_k)
                        tl.store(p_node + V + K, u_eg)


@triton.jit
def _gdn_tree_pre_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    b,
    q_pre,  # (tokens, H, K) fp32: normalized q * scale
    k_pre,  # (tokens, H, K) fp32: normalized k
    gb_pre,  # (tokens, HV, 2) fp32: exp(g), beta
    scale,
    stride_a,
    stride_q,
    stride_k,
    stride_b,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    # The serial walk's per-token math, once per (token, v-head) instead of in every V block's serial loop:
    # the same expressions, so the walk that reads them computes the same values.
    i_t, i_hv = tl.program_id(0), tl.program_id(1)
    i_h = i_hv // (HV // H)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    b_a = tl.load(a + i_t * stride_a + i_hv).to(tl.float32)
    b_b = tl.load(b + i_t * stride_b + i_hv).to(tl.float32)
    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = tl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    b_g = -tl.exp(b_A_log) * softplus_x
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))
    b_beta = b_beta.to(b.dtype.element_ty).to(tl.float32)
    tl.store(gb_pre + (i_t * HV + i_hv) * 2, tl.exp(b_g))
    tl.store(gb_pre + (i_t * HV + i_hv) * 2 + 1, b_beta)
    if i_hv % (HV // H) == 0:
        o_k = tl.arange(0, BK)
        mask_k = o_k < K
        b_q = tl.load(q + i_t * stride_q + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(k + i_t * stride_k + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        tl.store(q_pre + (i_t * H + i_h) * K + o_k, b_q, mask=mask_k)
        tl.store(k_pre + (i_t * H + i_h) * K + o_k, b_k, mask=mask_k)


@triton.jit
def _gdn_tree_records_commit_kernel(
    state,  # (L, slots, HV, V, K) pool
    stride_state_layer,
    stride_state_slot,
    records,  # (L, rows, steps, HV, REC) fp32
    stride_records_layer,
    stride_records_row,
    state_indices,
    record_indices,
    commit_steps,
    parent,  # (batch, width) int32: parent node of every node, node 0 the root
    stride_parent_seq,
    stride_parent_token,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REC: tl.constexpr,
    DMAX: tl.constexpr,
    DMAX_P2: tl.constexpr,
    ROUND_STATE: tl.constexpr,
):
    # One program per (V block, request * v-head, layer): the accepted node's state, replayed over its root path from
    # the request's pool state (the same fold, roundings included, as the verify), written back to the pool.
    i_v, i_nh, i_l = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    step = tl.load(commit_steps + i_n).to(tl.int32)
    slot = tl.load(state_indices + i_n).to(tl.int64)
    row = tl.load(record_indices + i_n).to(tl.int64)
    if (step >= 0) & (slot >= 0) & (row >= 0):
        # root path, accepted node first: path[j] = parent^j(step) while >= 0
        offs_d = tl.arange(0, DMAX_P2)
        path = tl.full([DMAX_P2], -1, dtype=tl.int32)
        node = step
        p_par = parent + i_n * stride_parent_seq
        for j in range(DMAX):
            path = tl.where(offs_d == j, node, path)
            # int32 whatever the parent table's dtype: sglang's eager verify builds it int64, its graphs int32.
            nxt = tl.load(p_par + node * stride_parent_token, mask=node > 0, other=-1).to(tl.int32)
            node = tl.where(node > 0, nxt, -1)
        L = tl.sum((path >= 0).to(tl.int32))

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]
        p_state = state + i_l * stride_state_layer + slot * stride_state_slot + i_hv * K * V
        o_state = o_v[None, :] * K + o_k[:, None]
        b_h = tl.load(p_state + o_state, mask=mask_h, other=0).to(tl.float32)
        p_rec = records + i_l * stride_records_layer + row * stride_records_row + i_hv * REC
        for d in range(0, L):
            n = tl.sum(tl.where(offs_d == L - 1 - d, path, 0))
            p_node = p_rec + n.to(tl.int64) * HV * REC
            u_eg = tl.load(p_node + V + K)
            u_k = tl.load(p_node + V + o_k, mask=mask_k, other=0)
            u_d = tl.load(p_node + o_v, mask=mask_v, other=0)
            if ROUND_STATE:
                if d > 0:
                    b_h = b_h.to(state.dtype.element_ty).to(tl.float32)
            b_h *= u_eg
            b_h += u_k[:, None] * u_d[None, :]
        tl.store(p_state + o_state, b_h.to(state.dtype.element_ty), mask=mask_h)


def gdn_tree_records_commit(
    state: torch.Tensor,
    records: torch.Tensor,
    state_indices: torch.Tensor,
    record_indices: torch.Tensor,
    commit_steps: torch.Tensor,
    parent: torch.Tensor,
    round_state: bool,
) -> None:
    """Commit the accepted node of every request into ``state`` (all layers, one launch) by replaying its root path's
    records. state: (L, slots, HV, V, K); records: (L, rows, steps, HV, REC); parent: (batch, width), node 0 the root;
    a negative step, slot or row skips the request."""
    L_layers, _, HV, V, K = state.shape
    n = commit_steps.shape[0]
    if n == 0:
        return
    assert records.shape[0] == L_layers and records.shape[3] == HV and state.stride(-1) == 1
    assert state.stride(3) == K and state.stride(2) == V * K, "contiguous (V, K) state per head"
    dmax = parent.shape[1]
    BV = min(triton.next_power_of_2(V), 32)
    _gdn_tree_records_commit_kernel[(triton.cdiv(V, BV), n * HV, L_layers)](
        state,
        state.stride(0),
        state.stride(1),
        records,
        records.stride(0),
        records.stride(1),
        state_indices,
        record_indices,
        commit_steps,
        parent,
        parent.stride(0),
        parent.stride(1),
        HV=HV,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        REC=records.shape[4],
        DMAX=dmax,
        DMAX_P2=triton.next_power_of_2(dmax),
        ROUND_STATE=round_state,
        num_warps=4,
    )


@triton.jit
def _gdn_tree_chunk_kernel(
    q_pre,  # (tokens, H, K) fp32: normalized q * scale (_gdn_tree_pre_kernel)
    k_pre,  # (tokens, H, K) fp32: normalized k
    gb_pre,  # (tokens, HV, 2) fp32: exp(g), beta
    v,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    parent,  # (batch, >= tokens per request) parent node of every node, node 0 the root
    stride_parent,
    records,  # (rows, steps, HV, REC) fp32: delta [0, V), key [V, V + K), decay at V + K
    record_indices,
    stride_records_row,
    stride_v,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    NP: tl.constexpr,
    REC: tl.constexpr,
    LOG2N: tl.constexpr,
    VSPLIT: tl.constexpr = 1,
):
    # The tree as one chunk of the gated delta rule with the ancestor mask in place of the causal one, per
    # (request, v-head); no per-node state exists (the commit replays the records, without per-node rounding):
    #   R = strict-ancestor closure of the parent links, G_t = prod of exp(g) along the root path of t,
    #   D = (I + diag(beta) (R o G_t/G_j o K K^T))^-1 diag(beta) (V - diag(G) K S0)   (inverse by doubling),
    #   O = diag(G) Q S0 + ((R + I) o G_t/G_j o Q K^T) D.
    # VSPLIT programs per (request, v-head) share the V columns (few requests: more programs in flight).
    i_n, i_hv, i_vs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    T = (tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos).to(tl.int32)
    rn = tl.arange(0, NP)
    cn = tl.arange(0, NP)
    o_k = tl.arange(0, K)
    m_n = rn < T
    tok = bos + rn
    Kt = tl.load(k_pre + (tok[:, None] * H + i_h) * K + o_k[None, :], mask=m_n[:, None], other=0.0)
    Qt = tl.load(q_pre + (tok[:, None] * H + i_h) * K + o_k[None, :], mask=m_n[:, None], other=0.0)
    eg = tl.load(gb_pre + (tok * HV + i_hv) * 2, mask=m_n, other=1.0)
    beta = tl.load(gb_pre + (tok * HV + i_hv) * 2 + 1, mask=m_n, other=0.0)
    par = tl.load(parent + i_n * stride_parent + rn, mask=m_n & (rn > 0), other=-1)
    R = (cn[None, :] == par[:, None]).to(tl.float32)
    for _ in tl.static_range(LOG2N):
        R = tl.minimum(R + tl.dot(R, R, input_precision="ieee"), 1.0)
    eye = (rn[:, None] == cn[None, :]).to(tl.float32)
    M = R + eye
    logG = tl.sum(M * tl.log(tl.maximum(eg, 1e-30))[None, :], 1)
    G = tl.exp(logG)
    # G_t / G_j where j is an ancestor-or-self of t (elsewhere it can overflow)
    ratio = tl.where(M > 0, tl.exp(logG[:, None] - logG[None, :]), 0.0)
    KK = tl.dot(Kt, tl.trans(Kt), input_precision="ieee")
    QK = tl.dot(Qt, tl.trans(Kt), input_precision="ieee")
    L = beta[:, None] * (R * ratio * KK)
    # (I + L)^-1 = (I - L)(I + L^2)(I + L^4)... (L is nilpotent: strict ancestors only)
    Tinv = eye - L
    P = tl.dot(L, L, input_precision="ieee")
    for _ in tl.static_range(LOG2N - 1):
        Tinv = tl.dot(Tinv, eye + P, input_precision="ieee")
        P = tl.dot(P, P, input_precision="ieee")
    W = ratio * QK
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    row = tl.load(record_indices + i_n).to(tl.int64)
    p_rec = records + row * stride_records_row + (rn[:, None] * HV + i_hv) * REC
    for vb in tl.static_range(V // BV // VSPLIT):
        o_v = (i_vs * (V // BV // VSPLIT) + vb) * BV + tl.arange(0, BV)
        S0 = tl.load(
            h0_source + idx * stride_h0_source + i_hv * K * V + o_v[None, :] * K + o_k[:, None],
            mask=(idx >= 0),
            other=0.0,
        ).to(tl.float32)
        KS = tl.dot(Kt, S0, input_precision="ieee")
        QS = tl.dot(Qt, S0, input_precision="ieee")
        Vt = tl.load(v + tok[:, None] * stride_v + i_hv * V + o_v[None, :], mask=m_n[:, None], other=0.0)
        rhs = beta[:, None] * (Vt.to(tl.float32) - G[:, None] * KS)
        D = tl.dot(Tinv, rhs, input_precision="ieee")
        O = G[:, None] * QS + tl.dot(W, D, input_precision="ieee")  # noqa: E741 - preserve AMD arithmetic notation
        tl.store(o + (tok[:, None] * HV + i_hv) * V + o_v[None, :], O.to(o.dtype.element_ty), mask=m_n[:, None])
        tl.store(p_rec + o_v[None, :], D, mask=m_n[:, None] & (row >= 0))
    if i_vs == 0:
        tl.store(p_rec + V + o_k[None, :], Kt, mask=m_n[:, None] & (row >= 0))
        tl.store(records + row * stride_records_row + (rn * HV + i_hv) * REC + V + K, eg, mask=m_n & (row >= 0))


def gdn_tree_records_update(
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    *,
    state,
    state_indices,
    query_start_loc,
    parent,
    tree_paths,
    records,
    record_indices,
    width,
):
    """Verify with compact replay records, preserving the pool's state precision.

    BF16/FP16 pools require ``tree_paths`` to round each parent before its descendants read it.
    FP32 pools may use the parallel chunk recurrence and commit without intermediate rounding.
    """
    if tree_paths is None:
        if state.dtype != torch.float32:
            raise ValueError("tree_paths are required for low-precision recurrent state")
        return gdn_tree_chunk_update(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a,
            b,
            state=state,
            state_indices=state_indices,
            query_start_loc=query_start_loc,
            parent=parent,
            records=records,
            record_indices=record_indices,
            width=width,
        )
    _, _, heads, key_dim = k.shape
    value_heads, value_dim = v.shape[2:]
    batch = query_start_loc.numel() - 1
    out = torch.empty_like(v)
    # One wave per program; the pre kernel runs one wave too so its reductions order like the walk's did.
    q_pre, k_pre, gb_pre = _gdn_tree_pre(A_log, dt_bias, q, k, a, b, key_dim**-0.5, num_warps=1)
    paths = tree_paths.shape[1]
    value_tile = _records_value_tile(batch * value_heads * paths, value_dim, q.device)
    _gdn_tree_records_kernel[(triton.cdiv(value_dim, value_tile), batch * value_heads, paths)](
        q_pre,
        k_pre,
        gb_pre,
        v,
        out,
        state,
        state_indices,
        state.stride(0),
        query_start_loc,
        records,
        record_indices,
        records.stride(0),
        tree_paths,
        tree_paths.stride(0),
        tree_paths.stride(1),
        v.stride(1),
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BK=triton.next_power_of_2(key_dim),
        BV=value_tile,
        REC=records.shape[-1],
        DMAX=width,
        DMAX_P2=triton.next_power_of_2(width),
        ROUND_STATE=True,
        num_warps=1,
        num_stages=1,
    )
    return out


@functools.cache
def _wave_slots(device: torch.device) -> int:
    """Waves the device holds at once: its compute units times the waves one unit schedules (4 SIMDs x 8 on CDNA)."""
    return torch.cuda.get_device_properties(device).multi_processor_count * 32


def _records_value_tile(paths: int, value_dim: int, device: torch.device) -> int:
    """V block of the path replay: 8 columns (one wave, the latency-bound batch-1 case), doubled up to 32 while
    the programs would need more than two rounds of the device's wave slots (the throughput-bound batched case:
    a wider block replays each path in fewer waves). Every width gives the same results: one wave per program."""
    tile = min(triton.next_power_of_2(value_dim), 8)
    while tile < 32 and paths * triton.cdiv(value_dim, tile) > 2 * _wave_slots(device):
        tile *= 2
    return tile


def _gdn_tree_pre(A_log, dt_bias, q, k, a, b, scale, *, num_warps):
    """The per-token values of the gated delta rule (normalized q * scale, normalized k, exp(g), beta), fp32."""
    _, tokens, heads, key_dim = k.shape
    value_heads = a.shape[-1]
    q_pre = torch.empty((tokens, heads, key_dim), dtype=torch.float32, device=q.device)
    k_pre = torch.empty_like(q_pre)
    gb_pre = torch.empty((tokens, value_heads, 2), dtype=torch.float32, device=q.device)
    _gdn_tree_pre_kernel[(tokens, value_heads)](
        A_log,
        a,
        dt_bias,
        1.0,
        20.0,
        q,
        k,
        b,
        q_pre,
        k_pre,
        gb_pre,
        scale,
        a.stride(-2),
        q.stride(1),
        k.stride(1),
        b.stride(-2),
        H=heads,
        HV=value_heads,
        K=key_dim,
        BK=key_dim,
        num_warps=num_warps,
    )
    return q_pre, k_pre, gb_pre


def gdn_tree_chunk_update(
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    *,
    state,
    state_indices,
    query_start_loc,
    parent,
    records,
    record_indices,
    width,
    use_flydsl=True,
):
    """Write outputs and delta/key/decay records without per-node state snapshots.

    State remains in fp32 inside the chunk, including with a bf16 persistent
    pool. Accepted-path commit must therefore use round_state=False.
    """
    _, tokens, heads, key_dim = k.shape
    value_heads, value_dim = v.shape[2:]
    batch = query_start_loc.numel() - 1
    scale = key_dim**-0.5
    out = torch.empty_like(v)
    if use_flydsl and width <= 16:
        try:
            from eagle_harness.kernels import gdn_tree_chunk_flydsl as fly
        except ImportError:
            fly = None
        if fly is not None and fly.supported(q, k, v, state, records, 1.0, 20.0):
            return fly.gdn_tree_chunk_flydsl(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                out,
                state,
                state_indices,
                query_start_loc,
                parent,
                records,
                record_indices,
                scale,
            )
    q_pre, k_pre, gb_pre = _gdn_tree_pre(A_log, dt_bias, q, k, a, b, scale, num_warps=4)
    padded_width = max(16, triton.next_power_of_2(width))
    # Match AMD's chunk layout; split value columns at small batch sizes.
    value_split = 4 if batch * value_heads <= 192 else (2 if batch * value_heads <= 768 else 1)
    value_block = min(32, value_dim // value_split)
    _gdn_tree_chunk_kernel[(batch, value_heads, value_split)](
        q_pre,
        k_pre,
        gb_pre,
        v,
        out,
        state,
        state_indices,
        state.stride(0),
        query_start_loc,
        parent,
        parent.stride(0),
        records,
        record_indices,
        records.stride(0),
        v.stride(1),
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BV=value_block,
        NP=padded_width,
        REC=records.shape[-1],
        LOG2N=padded_width.bit_length() - 1,
        VSPLIT=value_split,
        num_warps=4,
    )
    return out
