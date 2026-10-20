# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# Derived from the recurrence of gdn_tree_recurrence.py (sglang, Apache-2.0, after flash-linear-attention, MIT), rewritten in
# Gluon by Eagle AI Corp. Notices and license texts: THIRD_PARTY_NOTICES.md and LICENSES/.
"""GDN tree verify recurrence in Gluon (explicit-layout Triton).

Bitwise the serial walk of ``gdn_tree_recurrence`` (``_walk_kernel`` over 4-wide V tiles) on sm_103, in both slot
layouts (sglang: a snapshot scratch row per request; vLLM: one bf16 or fp32 state slot per verify row). Both engines'
tree verify asks for it through ``gdn_tree_recurrent_update(..., gluon=True)`` -> :func:`gluon_dispatch`, which runs it
for the launches it was validated for and returns None for every other launch, which keeps the Triton walk.

Why it is faster: the Triton walk runs one-warp programs over four-column V tiles (per-rank v-heads x 32: 1536 at
TP1, 384 at TP4), and every program redoes the per-token work (q/k L2 norm, gate, beta, 1-D -> tile layout shuffles)
inside its serial loop over the tree nodes. Here a program of 4 warps covers 16 V columns of one head (384 programs at
TP1, 96 at TP4; the same warps) and computes the per-token work, and every node's v tile, once for all T tokens into
shared memory (warps split the tokens), so the serial loop keeps only the state math. The cold initial-state load is
issued first, before griddepcontrol.wait under PDL, and the dependent launch is released right after the wait (as the
Triton walk's PDL path does). Inside the loop:
  * node s - 1's output reduction (h.q, off the state chain) runs in iteration s next to node s's h.k reduction, so
    the two 16-lane shuffle chains overlap; at TP4 every SM sub-partition runs a single warp, which otherwise waits out
    both;
  * node s + 1's parent reload is issued at the end of iteration s, after node s's snapshot store it may read back,
    behind one uniform branch instead of a masked load and a select of the whole state.

Why it is bit-identical (tested bitwise in ``tests/test_gdn_tree_gluon_gpu.py``):
  * the state tile keeps the Triton walk's per-warp layout, which Triton derives from the state's 16-byte vectors
    along K: BlockedLayout([8,1],[16,2]) for a bf16 state, ([4,1],[32,1]) for fp32 (per V column: the same K elements
    per thread on the same lanes), so h.k and h.q reduce in the same in-thread order and xor butterfly;
  * the per-token q/k norms reduce rows of a [TP, K] tile in BlockedLayout([1,4],[1,32]): per token the same
    4-per-thread + xor-16..1 order as the walk's 1-D [4]/[32] layout, with the same div.full, sqrt.approx, ex2.approx
    and log sequences; gate and beta are elementwise, and beta is rounded to bf16 as there;
  * both kernels compile without FMA contraction and write the same FMAs (the state update h + k v^T, and the h.k
    reduction with adjacent K rows paired, ``_dot_k``), so no compiler choice of where to fuse can split them;
  * every step starts from the snapshot of its parent (the same reload; with an fp32 state a chain child keeps the
    registers, which equal the exact fp32 reload) and stores the same snapshot and bf16 output.
A change of either kernel's expressions or layouts needs the same change in the other and a re-run of the bitwise
tests. At run time, the first launch of every signature is checked bitwise against the Triton walk
(:func:`gluon_dispatch`); a mismatch or a compile / launch error switches the process back to the Triton walk.
"""

import logging
import random
from typing import Callable, Optional

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from eagle_harness.kernels import arch
from eagle_harness.kernels.gdn_tree import PAD_SLOT_ID
from eagle_harness.kernels.gdn_tree_recurrence import gdn_tree_state_args

logger = logging.getLogger(__name__)

NUM_WARPS = 4  # 4 x 4 V columns per program (1/2/8 warps slower; 8 columns per warp not exact)
# Widths 1-2: the Triton walk is as fast or faster. Width 32 (vLLM's widest verify) needs ~38 KB of shared memory per
# program for the precompute.
MIN_WIDTH, MAX_WIDTH = 3, 32
# Head shapes (H, HV, K, V) per rank the kernel takes: 128-wide keys and values (the register layouts are written
# for them) and whole GQA groups (HV a multiple of H). Measured faster than the Triton walk at every TP shape of
# Qwen3.8-27B, TP1 (16, 48) to TP8 (2, 6): e.g. 12.0 / 8.9 / 6.5 / 6.2 us vs 16.2 / 13.0 / 10.8 / 10.7 us at width
# 16 on B300 (tools/bench_kernels.py). The GPUs are arch.GLUON_GDN_SM (arch.gluon_gdn). Bit-identity rests on
# matching layouts and on the private Gluon API _reinterpret, so it is not assumed from a version or a shape:
# gluon_dispatch checks each launch signature bitwise against the Triton walk before serving it.
HEAD_DIM = 128


def admits_shape(h: int, hv: int, k: int, v: int) -> bool:
    """Whether the kernel's layouts cover the per-rank head shape."""
    return k == HEAD_DIM and v == HEAD_DIM and h > 0 and hv % h == 0


@gluon.jit
def _dot_k(h, k, layout: gl.constexpr):
    """The Triton walk's _dot_k (sum over K of h * k, adjacent K rows folded into one FMA), in `layout`."""
    kk = gl.broadcast(gl.expand_dims(k, 1), h)[0]
    h_even, h_odd = gl.split(gl.permute(gl.reshape(h, [h.shape[0] // 2, 2, h.shape[1]]), (0, 2, 1)))
    k_even, k_odd = gl.split(gl.permute(gl.reshape(kk, [h.shape[0] // 2, 2, h.shape[1]]), (0, 2, 1)))
    return gl.convert_layout(gl.sum(gl.fma(h_even, k_even, h_odd * k_odd), axis=0), layout, assert_trivial=True)


@gluon.jit
def _recurrence_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    o,
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
    retrieve_parent_token_ptr,
    stride_rpt_seq,
    stride_rpt_tok,
    publish,
    stride_publish,
    num_slot_cols,
    softplus_beta,
    softplus_threshold,
    scale,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    stride_o,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    NW: gl.constexpr,
    TP: gl.constexpr,
    SP: gl.constexpr,
    ROW_SLOTS: gl.constexpr,
    HAS_NUM_ACCEPTED: gl.constexpr,
    PUBLISH: gl.constexpr,
    PAD_SLOT_ID: gl.constexpr,
    USE_PDL: gl.constexpr,
    SKIP_CHAIN_RELOAD: gl.constexpr,
    STATE_VEC: gl.constexpr,
):
    """The tree recurrence of one (request, v-head) over BVC V columns. Slots as the Triton walk addresses them
    (ROW_SLOTS: one snapshot slot per row, else one per request)."""
    BVC: gl.constexpr = 4 * NW
    # the walk's state tile: STATE_VEC (16 bytes of state) consecutive K per thread, the warp's lanes over the rest of K
    LH: gl.constexpr = gl.BlockedLayout([STATE_VEC, 1], [K // STATE_VEC, 32 * STATE_VEC // K], [1, NW], [0, 1])
    LKV: gl.constexpr = gl.SliceLayout(1, LH)
    LVV: gl.constexpr = gl.SliceLayout(0, LH)
    LQK: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [NW, 1], [1, 0])
    LQT: gl.constexpr = gl.SliceLayout(1, LQK)
    LQC: gl.constexpr = gl.SliceLayout(0, LQK)
    LG: gl.constexpr = gl.BlockedLayout([1, 1], [32 // BVC, BVC], [NW, 1], [1, 0])
    LGT: gl.constexpr = gl.SliceLayout(1, LG)
    LGC: gl.constexpr = gl.SliceLayout(0, LG)
    SMEM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    SMEM1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    i_vc = gl.program_id(0)
    i_nh = gl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    bos = gl.load(cu_seqlens + i_n).to(gl.int64)
    eos = gl.load(cu_seqlens + i_n + 1).to(gl.int64)
    T = eos - bos

    # cold initial state first (its DRAM latency overlaps the precompute, and with PDL the tail of the producer kernel)
    kv = gl.arange(0, K, layout=LKV)
    vv = i_vc * BVC + gl.arange(0, BVC, layout=LVV)
    o_state = i_hv * K * V + gl.expand_dims(vv, 0) * K + gl.expand_dims(kv, 1)
    p_idx = h0_indices + i_n * stride_h0_indices_seq
    if HAS_NUM_ACCEPTED:
        p_idx += (gl.load(num_accepted + i_n).to(gl.int64) - 1) * stride_h0_indices_col
    idx = gl.load(p_idx).to(gl.int64)
    h0_raw = gl.load(
        h0_source + idx * stride_h0_source + o_state,
        mask=(gl.expand_dims(vv, 0) < V) & (idx > PAD_SLOT_ID),
        other=0,
    )
    if USE_PDL:
        # PDL invariant: the reads above (cu_seqlens, the state slot, num_accepted and the pool state) may run while
        # the preceding kernels are still running, so no kernel of the verify forward may write them. The metadata
        # copies before a graph replay write cu_seqlens and the slot, and the state commit after the previous verify
        # writes the pool. The direct predecessor here is the PDL tree conv (which waits for its own predecessor,
        # the input projection, before it releases this launch), or a non-PDL kernel; neither writes them. q/k/v/a/b,
        # the parents and every other input are read below the wait only.
        gl.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r", [], dtype=gl.int32, is_pure=False, pack=1)
        # Release the dependent launch at once, as the Triton walk's PDL path does: the gated RMSNorm after it waits
        # for this grid's completion (its own griddepcontrol.wait) before reading o.
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; // dummy $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )

    p_slots = intermediate_state_indices + i_n * stride_snap_indices_seq
    if PUBLISH:
        if (i_vc == 0) & (i_hv == 0):
            LS: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
            o_s = gl.arange(0, SP, layout=LS)
            m_s = o_s < num_slot_cols
            gl.store(
                publish + i_n * stride_publish + o_s,
                gl.load(p_slots + o_s * stride_snap_indices_row, mask=m_s),
                mask=m_s,
            )

    # per-token precompute, once per program (warps split the tokens)
    tq = gl.arange(0, TP, layout=LQT)
    kq = gl.arange(0, K, layout=LQC)
    row_ok = gl.expand_dims(tq < T, 1)
    row_off = gl.expand_dims(bos + tq, 1)
    b_qr = gl.load(q + row_off * stride_q + i_h * K + gl.expand_dims(kq, 0), mask=row_ok, other=0).to(gl.float32)
    b_kr = gl.load(k + row_off * stride_k + i_h * K + gl.expand_dims(kq, 0), mask=row_ok, other=0).to(gl.float32)
    b_qn = b_qr / gl.expand_dims(gl.sqrt(gl.sum(b_qr * b_qr, axis=1) + 1e-6), 1)
    b_kn = b_kr / gl.expand_dims(gl.sqrt(gl.sum(b_kr * b_kr, axis=1) + 1e-6), 1)
    b_qn = b_qn * scale
    s_q = gl.allocate_shared_memory(gl.float32, [TP, K], SMEM)
    s_k = gl.allocate_shared_memory(gl.float32, [TP, K], SMEM)
    s_q.store(b_qn)
    s_k.store(b_kn)

    tg = gl.arange(0, TP, layout=LGT)
    cg = gl.arange(0, BVC, layout=LGC)
    g_ok = gl.expand_dims(tg < T, 1) & gl.expand_dims(cg < BVC, 0)
    g_row = gl.expand_dims(bos + tg, 1) + gl.expand_dims(cg * 0, 0)
    b_a = gl.load(a + g_row * stride_a + i_hv, mask=g_ok, other=0).to(gl.float32)
    b_b = gl.load(b + g_row * stride_b + i_hv, mask=g_ok, other=0).to(gl.float32)
    b_A_log = gl.load(A_log + i_hv).to(gl.float32)
    b_dt_bias = gl.load(dt_bias + i_hv).to(gl.float32)
    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = gl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
        x,
    )
    b_g = -gl.exp(b_A_log) * softplus_x
    b_eg = gl.exp(b_g)
    b_beta = 1.0 / (1.0 + gl.exp(-b_b))
    b_beta = b_beta.to(b.dtype.element_ty).to(gl.float32)
    s_eg = gl.allocate_shared_memory(gl.float32, [TP, BVC], SMEM)
    s_beta = gl.allocate_shared_memory(gl.float32, [TP, BVC], SMEM)
    s_eg.store(b_eg)
    s_beta.store(b_beta)
    p_vt = v + g_row * stride_v + i_hv * V + i_vc * BVC + gl.expand_dims(cg, 0)
    b_vt = gl.load(p_vt, mask=g_ok, other=0).to(gl.float32)
    s_v = gl.allocate_shared_memory(gl.float32, [TP, BVC], SMEM)
    s_v.store(b_vt)
    gl.barrier()
    # same bytes viewed as [TP] rows (the leading dim becomes indexable)
    r_q = s_q._reinterpret(gl.float32, [TP, K], SMEM1)
    r_k = s_k._reinterpret(gl.float32, [TP, K], SMEM1)
    r_eg = s_eg._reinterpret(gl.float32, [TP, BVC], SMEM1)
    r_beta = s_beta._reinterpret(gl.float32, [TP, BVC], SMEM1)
    r_v = s_v._reinterpret(gl.float32, [TP, BVC], SMEM1)

    # state: zeros + h0 (as the Triton walk)
    b_h = gl.zeros([K, BVC], gl.float32, LH)
    if idx > PAD_SLOT_ID:
        b_h += h0_raw.to(gl.float32)

    # the request's snapshot slot (ROW_SLOTS: row 0's); snapshot of row r in slot s at p_snap + s * stride_snap_slot +
    # r * stride_snap_row
    cache_idx = gl.load(p_slots).to(gl.int64)
    p_snap = intermediate_states_buffer + o_state
    p_par = retrieve_parent_token_ptr + i_n * stride_rpt_seq
    p_o = o + bos * stride_o + i_hv * V + vv
    col_ok = gl.expand_dims(vv, 0) < V

    # The parent reload is unmasked: the grid tiles V in whole BVC-column tiles, so col_ok holds everywhere. Without a
    # snapshot slot nothing is reloaded and the state carries in registers. T <= TP: the launcher only runs requests
    # of at most `width` tokens.
    b_hp = gl.zeros([K, BVC], gl.float32, LH)
    b_qp = gl.zeros([K], gl.float32, LKV)
    for s in range(0, T):
        n_parent = gl.load(p_par + (s + 1) * stride_rpt_tok, mask=s + 1 < T, other=0)
        slot = cache_idx
        if ROW_SLOTS:
            slot = gl.load(p_slots + s * stride_snap_indices_row).to(gl.int64)
        s32 = s.to(gl.int32)
        b_q = r_q.index(s32).load(LKV)
        b_k = r_k.index(s32).load(LKV)
        b_egs = r_eg.index(s32).load(LVV)
        b_bts = r_beta.index(s32).load(LVV)
        b_v = r_v.index(s32).load(LVV)

        # node s - 1's output, next to node s's h.k reduction
        b_op = gl.sum(b_hp * gl.expand_dims(b_qp, 1), axis=0)
        gl.store(p_o + (s - 1) * stride_o, b_op.to(o.dtype.element_ty), mask=(vv < V) & (s > 0))
        b_h *= gl.expand_dims(b_egs, 0)
        b_v -= _dot_k(b_h, b_k, LVV)
        n_parent_slot = cache_idx
        if ROW_SLOTS:
            # issued here, where n_parent has landed
            n_parent_slot = gl.load(p_slots + n_parent * stride_snap_indices_row, mask=s + 1 < T, other=PAD_SLOT_ID).to(
                gl.int64
            )
        b_v *= b_bts
        b_h = gl.fma(gl.expand_dims(b_k, 1), gl.expand_dims(b_v, 0), b_h)
        gl.store(
            p_snap + slot * stride_snap_slot + s * stride_snap_row,
            b_h.to(intermediate_states_buffer.dtype.element_ty),
            mask=col_ok & (slot > PAD_SLOT_ID),
        )
        b_hp = b_h
        b_qp = b_q
        # node s + 1's parent, after node s's snapshot store it may read back (same thread, program order); an fp32
        # snapshot reloads exactly what the registers hold, so a chain child keeps them (as the walk does)
        if (s + 1 < T) & (n_parent_slot > PAD_SLOT_ID) & ((not SKIP_CHAIN_RELOAD) | (n_parent != s)):
            b_h = gl.load(p_snap + n_parent_slot * stride_snap_slot + n_parent * stride_snap_row).to(gl.float32)
    b_op = gl.sum(b_hp * gl.expand_dims(b_qp, 1), axis=0)
    gl.store(p_o + (T - 1) * stride_o, b_op.to(o.dtype.element_ty), mask=(vv < V) & (T > 0))


def gluon_recurrent_update(
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    *,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    intermediate_states_buffer,
    intermediate_state_indices,
    retrieve_parent_token,
    num_accepted_tokens=None,
    pad_slot_id: int = PAD_SLOT_ID,
    out=None,
    publish=None,
    scale=None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_pdl: bool = False,
) -> torch.Tensor:
    """Launch the kernel. Same contract as the Triton walk of ``gdn_tree_recurrent_update``. Every request must have
    at most retrieve_parent_token.shape[1] tokens (the precompute holds max(8, next_pow2(width)) of them);
    :func:`gluon_admits` checks the launch."""
    _, _, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    BVC = 4 * NUM_WARPS
    # the precompute layouts cover 8 token rows
    TP = max(8, triton.next_power_of_2(retrieve_parent_token.shape[1]))
    if scale is None:
        scale = K**-0.5
    if out is None:
        out = q.new_empty(v.shape)
    extra = {"launch_pdl": True} if use_pdl else {}
    _recurrence_kernel[(V // BVC, N * HV)](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        o=out,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
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
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_rpt_seq=retrieve_parent_token.stride(0),
        stride_rpt_tok=retrieve_parent_token.stride(1),
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        stride_a=a.stride()[-2],
        stride_q=q.stride()[1],
        stride_k=k.stride()[1],
        stride_v=v.stride()[1],
        stride_b=b.stride()[-2],
        stride_o=out.stride(1),
        H=H,
        HV=HV,
        K=K,
        V=V,
        NW=NUM_WARPS,
        TP=TP,
        PAD_SLOT_ID=pad_slot_id,
        USE_PDL=use_pdl,
        SKIP_CHAIN_RELOAD=intermediate_states_buffer.dtype == torch.float32,
        STATE_VEC=16 // intermediate_states_buffer.element_size(),
        num_warps=NUM_WARPS,
        enable_fp_fusion=False,  # only the written FMAs, as in the Triton walk's launch
        **extra,
    )
    return out


def gluon_admits(
    *,
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    intermediate_states_buffer,
    intermediate_state_indices,
    retrieve_parent_token,
    num_accepted_tokens=None,
    out=None,
    publish=None,
) -> bool:
    """Whether a tree verify launch is one the kernel was validated for: requests of width 3..32 (at most `width`
    tokens each: the caller's contract, checked for one request), the validated head shapes, bf16 activations with
    fp32 A_log, a pool and snapshot buffer of one dtype (bf16 or fp32) in either slot layout, int32 tables, the serving layouts, a
    and b with a unit column stride, 16-byte aligned pointers for every other tensor (the self-check reproduces the
    live strides and the alignment of a and b, so its inputs specialise the same way), and q / k / v token strides
    divisible by 16, on a validated GPU."""
    i32 = torch.int32
    # common rejections first: an eager tree verify passes int64 tree links
    if not (
        retrieve_parent_token.dtype == i32
        and retrieve_parent_token.ndim == 2
        and cu_seqlens.dtype == i32
        and cu_seqlens.ndim == 1
        and MIN_WIDTH <= retrieve_parent_token.shape[1] <= MAX_WIDTH
    ):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or a.ndim != 2 or b.ndim != 2:
        return False
    if initial_state_source is None or initial_state_source.ndim != 4:
        return False
    row_slots = intermediate_state_indices.ndim == 2
    if intermediate_states_buffer.ndim != (4 if row_slots else 5):
        return False
    if initial_state_indices.ndim != (1 if num_accepted_tokens is None else 2):
        return False
    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    width = retrieve_parent_token.shape[1]
    bf16 = torch.bfloat16
    state_dtypes = (bf16, torch.float32)
    if row_slots:
        # one slot per row (vLLM): a [slots, HV, V, K] buffer with any slot stride
        snapshots_ok = (
            intermediate_states_buffer.shape[1:] == (HV, V, K)
            and intermediate_states_buffer.stride()[1:] == (V * K, K, 1)
            and intermediate_state_indices.shape[1] >= width
        )
    else:
        # one scratch row of `steps` snapshots per request (sglang)
        snapshots_ok = (
            intermediate_states_buffer.shape[2:] == (HV, V, K)
            and intermediate_states_buffer.is_contiguous()
            and width <= intermediate_states_buffer.shape[1]
        )
    optional = [t for t in (num_accepted_tokens, out, publish) if t is not None]
    # a and b are read with scalar per-head loads through stride_a / stride_b, so they only need a unit column stride
    # (a tree verify may pass them as column slices of the in_proj_ba output [b | a], a at a 24-byte offset at TP4);
    # every other tensor stays 16-byte aligned.
    tensors = [
        A_log,
        dt_bias,
        q,
        k,
        v,
        initial_state_source,
        initial_state_indices,
        cu_seqlens,
        intermediate_states_buffer,
        intermediate_state_indices,
        retrieve_parent_token,
        *optional,
    ]
    return (
        admits_shape(H, HV, K, V)
        and snapshots_ok
        and q.shape == k.shape
        and q.shape[0] == 1
        and v.shape[:2] == q.shape[:2]
        and q.stride(3) == 1
        and q.stride(2) == K
        and k.stride(3) == 1
        and k.stride(2) == K
        and v.stride(3) == 1
        and v.stride(2) == V
        # Triton vectorizes the walk's q / k rows (4 consecutive K per lane, the Gluon layout's reduction order) only
        # where it knows the rows are aligned; other strides reduce the norms in another order.
        and q.stride(1) % 16 == 0
        and k.stride(1) % 16 == 0
        and v.stride(1) % 16 == 0
        and a.shape == (T, HV)
        and b.shape == (T, HV)
        and a.stride(-1) == 1
        and b.stride(-1) == 1
        and q.dtype == bf16
        and k.dtype == bf16
        and v.dtype == bf16
        and a.dtype == bf16
        and b.dtype == bf16
        and dt_bias.dtype == bf16
        and A_log.dtype == torch.float32
        and initial_state_source.dtype in state_dtypes
        and initial_state_source.dtype == intermediate_states_buffer.dtype
        and initial_state_source.shape[1:] == (HV, V, K)
        and initial_state_source.stride()[1:] == (V * K, K, 1)
        and intermediate_states_buffer.dtype in state_dtypes
        and initial_state_indices.dtype == i32
        and intermediate_state_indices.dtype == i32
        and (num_accepted_tokens is None or num_accepted_tokens.dtype == i32)
        and (out is None or (out.dtype == bf16 and out.shape == v.shape and out.stride()[2:] == (V, 1)))
        and (publish is None or (publish.dtype == i32 and publish.ndim == 2 and publish.stride(1) == 1))
        and T <= N * width
        and all(t.data_ptr() % 16 == 0 for t in tensors)
        and q.is_cuda
        and arch.gluon_gdn(q.device)
    )


class _GluonState:
    """Process-wide dispatch state: signatures that passed the self-check, and a permanent switch back to the Triton
    walk after any failure."""

    def __init__(self):
        self.disabled = False
        self.checked = set()


_STATE = _GluonState()


def reset_gdn_tree_gluon_state() -> None:
    """Forget the self-check results and re-enable the kernel (tests)."""
    global _STATE
    _STATE = _GluonState()


def gdn_tree_gluon_disabled() -> bool:
    """Whether a failed check switched the Gluon kernel off for this process."""
    return _STATE.disabled


def _disable(reason: str) -> None:
    if not _STATE.disabled:
        logger.warning("GDN tree verify: %s; using the Triton walk for the rest of this process.", reason)
    _STATE.disabled = True


def _signature(live: dict, use_pdl: bool) -> tuple:
    # everything that can change the compiled variants of either kernel for a launch gluon_admits admits:
    # dtypes, strides, which optional tensors are given, the slot layout, the width (TP), and the 16-byte alignment of
    # every pointer (Triton specialises a pointer on it); table and pool sizes along the slot axis do not matter
    def desc(name):
        t = live[name]
        if t is None:
            return None
        sizes = tuple(t.shape) if name in ("q", "k", "v", "a", "b", "out", "retrieve_parent_token") else t.shape[1:]
        return t.dtype, tuple(sizes), t.stride(), t.data_ptr() % 16

    return (live["q"].device.index, use_pdl, live["pad_slot_id"], *(desc(name) for name in sorted(_TENSORS)))


def _bitwise_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    int_dtype = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    view = int_dtype[x.element_size()]
    return torch.equal(x.contiguous().view(view), y.contiguous().view(view))


_TENSORS = (
    "A_log",
    "a",
    "dt_bias",
    "q",
    "k",
    "v",
    "b",
    "initial_state_source",
    "initial_state_indices",
    "num_accepted_tokens",
    "cu_seqlens",
    "intermediate_states_buffer",
    "intermediate_state_indices",
    "retrieve_parent_token",
    "out",
    "publish",
)


def _self_check(reference: Callable[..., torch.Tensor], live: dict, use_pdl: bool) -> bool:
    """Run one random tree of the live launch's width (one request) through the Triton walk (reference) and this
    kernel, on fresh tensors with the live
    dtypes, strides, slot layout and offsets from a 16-byte boundary (so both kernels compile the variants serving
    uses), and compare o and every state, slot and publish buffer bitwise. Uses its own generators (no global RNG
    draw)."""
    q = live["q"]
    dev = q.device
    parents_live = live["retrieve_parent_token"]
    width = parents_live.shape[1]
    T = min(q.shape[1], width)
    H = q.shape[2]
    HV, V, K = live["initial_state_source"].shape[1:]
    gen = torch.Generator(device=dev)
    gen.manual_seed(20260930 + width)
    rng = random.Random(20260930 + width)

    def like(t, values):
        # values in a tensor with t's strides, dtype and offset from a 16-byte boundary (a / b may be unaligned
        # column slices)
        pad = t.data_ptr() % 16 // t.element_size()
        shape = values.shape
        size = pad + (1 + sum((n - 1) * s for n, s in zip(shape, t.stride())) if values.numel() else 0)
        out = torch.empty(size, dtype=t.dtype, device=dev).as_strided(shape, t.stride(), pad)
        out.copy_(values)
        return out

    def randn(shape, scale=1.0):
        return torch.randn(shape, device=dev, generator=gen) * scale

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=dev)

    act = torch.nn.functional.silu
    args = dict(
        A_log=like(live["A_log"], torch.log(torch.rand(live["A_log"].shape, device=dev, generator=gen) * 15 + 1)),
        a=like(live["a"], randn((T, HV))),
        dt_bias=like(live["dt_bias"], randn(live["dt_bias"].shape, 0.5)),
        softplus_beta=live["softplus_beta"],
        softplus_threshold=live["softplus_threshold"],
        q=like(q, act(randn((1, T, H, K)))),
        k=like(live["k"], act(randn((1, T, H, K)))),
        v=like(live["v"], act(randn((1, T, HV, V)))),
        b=like(live["b"], randn((T, HV))),
        scale=live["scale"],
        cu_seqlens=like(live["cu_seqlens"], i32([0, T])),
        retrieve_parent_token=like(parents_live, i32([[0] + [rng.randrange(i) for i in range(1, width)]])),
        pad_slot_id=live["pad_slot_id"],
    )
    # slot tables (a 2-D table addresses slots 1..S of a pool of S + 1; vLLM passes one table for both uses) and the
    # state buffers, random so a stray or missing store shows up
    pool_live, scratch_live = live["initial_state_source"], live["intermediate_states_buffer"]
    snap_live, init_live = live["intermediate_state_indices"], live["initial_state_indices"]
    if snap_live.ndim == 2:
        slots = snap_live.shape[1]
        snap_idx = like(snap_live, torch.arange(1, slots + 1, dtype=torch.int32, device=dev)[None])
        scratch = like(scratch_live, randn((slots + 1, HV, V, K), 0.05))
    else:
        snap_idx = like(snap_live, i32([0]))
        scratch = like(scratch_live, randn((1, scratch_live.shape[1], HV, V, K)))
    if init_live is snap_live:
        init_idx = snap_idx
    elif init_live.ndim == 2:
        init_idx = like(init_live, torch.arange(1, init_live.shape[1] + 1, dtype=torch.int32, device=dev)[None])
    else:
        init_idx = like(init_live, i32([0]))
    states = dict(
        initial_state_source=scratch
        if pool_live is scratch_live
        else like(pool_live, randn((max(2, init_idx.shape[-1] + 1), HV, V, K), 0.05)),
        intermediate_states_buffer=scratch,
    )
    if live["out"] is not None:
        states["out"] = like(live["out"], torch.zeros(1, T, HV, V))
    if live["publish"] is not None:
        states["publish"] = like(live["publish"], torch.full((1, live["publish"].shape[1]), -7))
    args.update(initial_state_indices=init_idx, intermediate_state_indices=snap_idx)
    if live["num_accepted_tokens"] is not None:
        args["num_accepted_tokens"] = like(live["num_accepted_tokens"], i32([min(2, init_idx.shape[-1])]))

    def fresh_states():
        copies = {name: like(t, t) for name, t in states.items() if name != "initial_state_source"}
        pool = states["initial_state_source"]
        copies["initial_state_source"] = copies["intermediate_states_buffer"] if pool is scratch else like(pool, pool)
        return copies

    ref_states, gl_states = fresh_states(), fresh_states()
    o_ref = reference(**args, **ref_states)
    # The Triton walk is this launch's predecessor; it writes none of the pre-wait inputs (pool, slots,
    # num_accepted, cu_seqlens) the Gluon kernel reads before its griddepcontrol.wait.
    o_gl = gluon_recurrent_update(**args, **gl_states, use_pdl=use_pdl)
    return _bitwise_equal(o_ref, o_gl) and all(_bitwise_equal(ref_states[n], gl_states[n]) for n in ref_states)


def gluon_dispatch(reference: Callable[..., torch.Tensor], *, use_pdl: bool, **launch) -> Optional[torch.Tensor]:
    """Run the Gluon kernel for a validated launch and return o, or return None so the caller runs the Triton walk.
    ``launch`` holds gdn_tree_recurrent_update's arguments; reference is that launcher restricted to the Triton walk
    (the self-check's reference).

    The first launch of each signature runs _self_check first. It needs synchronisation and JIT compilation, so it
    never runs inside CUDA-graph capture: an unchecked signature met during capture is served by the Triton walk
    (serving warms every verify graph eagerly first, so its signatures are checked before capture). Any failure
    disables the kernel for the process. Under torch.compile tracing the Triton walk serves as well."""
    if _STATE.disabled or torch.compiler.is_compiling():
        return None
    live = {name: launch.get(name) for name in _TENSORS}
    if not gluon_admits(**live):
        return None
    live.update(
        pad_slot_id=launch["pad_slot_id"],
        softplus_beta=launch["softplus_beta"],
        softplus_threshold=launch["softplus_threshold"],
        scale=launch["scale"],
    )
    key = _signature(live, use_pdl)
    if key not in _STATE.checked:
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            ok = _self_check(reference, live, use_pdl)
        except Exception as e:  # compile / launch error
            _disable(f"Gluon kernel self-check raised {type(e).__name__}: {e}")
            return None
        if not ok:
            _disable("Gluon kernel is not bitwise equal to the Triton walk (self-check)")
            return None
        _STATE.checked.add(key)
        logger.info(
            "GDN tree verify: Gluon kernel self-check passed (width %d, pdl %s, %s slots).",
            launch["retrieve_parent_token"].shape[1],
            use_pdl,
            "per-row" if launch["intermediate_state_indices"].ndim == 2 else "per-request",
        )
    try:
        return gluon_recurrent_update(**launch, use_pdl=use_pdl)
    except Exception as e:  # launch error after a passing self-check
        _disable(f"Gluon kernel launch raised {type(e).__name__}: {e}")
        return None
