# Modified by Eagle AI Corp.
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The GDN tree chunk (gdn_tree_verify._gdn_tree_chunk_kernel) as one FlyDSL kernel (gfx950): from the raw q / k /
v / gates to o and the replay records, one wave per (request, v-head, V slice), trees of <= 16 nodes.

Per wave, on the 16 x 16 MFMA accumulator layout (a lane holds rows 4g..4g+3 of column n16):
  - per token: q / k L2 norms (q * scale), exp(g) and beta from A_log / a / dt_bias / b (the per-token pre-pass);
  - the strict-ancestor masks by pointer jumping on the parent links (4 rounds in LDS, any node order), the root-path
    log-decays logG, the decay ratios G_t / G_j;
  - K K^T and Q K^T on the 16x16x32 bf16 MFMA with hi + lo bf16 operands (~fp32);
  - Tinv = (I + L)^-1, L = diag(beta) (R o G_t/G_j o K K^T), by doubling on the 16x16x4 fp32 MFMA, and
    W = (R + I) o G_t/G_j o Q K^T;
  - per 16-column V tile: K S0 (hi + lo K) and Q S0 on the bf16 MFMA, S0 streamed once from the bf16 pool, then
      rhs = beta (V - G K S0),   D = Tinv rhs,   O = G Q S0 + W D
    in fp32 (the other rows of a column come from three xor shuffles, Tinv / W read at the matching permuted
    columns); O stored bf16, D into the records with the keys and decays.
K S0 feeds the committed state: K enters the MFMA as three bf16 terms (hi + lo + lo2: the deltas then match an fp64
reference to ~5e-7 relative, like the Triton chunk's fp32 dots; hi + lo alone left ~2e-6, and K rounded to bf16 alone
drifted the states enough to fail SPEED-Bench's lossless check); K K^T / Q K^T use all four hi / lo products; Q S0
only feeds the step's bf16 output, Q's hi alone. Tinv / W are read from LDS in the tile loop (128 more live
registers per lane cost ~30% at batch 16-32).

MI355X per layer, 16-node trees, vs the Triton pre-pass + chunk it replaces: TP1 1 / 4 / 16 / 32 requests 13.1 ->
10.5 / 17.4 -> 11.9 / 34.2 -> 23.8 / 58.9 -> 34.7 us, TP4 32 requests 24.0 -> 15.8 us (the earlier Triton prep +
FlyDSL apply: 30.3 / 45.9 / 21.2 us at TP1 16 / 32, TP4 32). The keys are stored early: keeping the fp32 k alive
to the end also cost ~30% (registers).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

NP = 16  # tree nodes per request (one MFMA row tile)
WAVE = 64
_FX = {torch.bfloat16: fx.BFloat16, torch.float32: fx.Float32, torch.int32: fx.Int32, torch.int64: fx.Int64}


def _mfma_f32(a, b, c):
    # 16x16x4 fp32 MFMA (FlyDSL 0.2.4 exposes only the raw ROCDL op for it)
    ops = rocdl._split_mfma_operands([a, b, c])
    return rocdl.mfma_f32_16x16x4f32(T.vec(4, T.f32), *ops).result


@functools.lru_cache(maxsize=None)
def build(H, HV, K, V, REC, vsplit, scale, dtypes):
    assert K % 32 == 0 and V % (16 * vsplit) == 0
    n_kb = K // 32
    tiles = V // 16 // vsplit
    gqa = HV // H
    dt = dict(dtypes)
    a_bf16, b_bf16, alog_bf16, dtb_bf16 = (dt[n] == torch.bfloat16 for n in ("a", "b", "A_log", "dt_bias"))

    @fx.struct
    class Shared:
        jump: fx.Array[fx.Int32, 16, 16]
        anc: fx.Array[fx.Int32, 16, 16]
        lg: fx.Array[fx.Float32, 16, 16]  # log exp(g) per node
        lgsum: fx.Array[fx.Float32, 16, 16]  # logG per node
        beta: fx.Array[fx.Float32, 16, 16]
        x: fx.Array[fx.Float32, 256, 16]  # fp32 MFMA operand staging
        y: fx.Array[fx.Float32, 256, 16]
        tinv: fx.Array[fx.Float32, 256, 16]
        w: fx.Array[fx.Float32, 256, 16]

    def f32(val, is_bf16):  # outside the kernel body: the Python branch stays a Python branch
        return fx.BFloat16(val).to(fx.Float32) if is_bf16 else fx.Float32(val)

    def round_beta(beta):
        # the serial walk rounds beta to b's dtype
        return beta.to(fx.BFloat16).to(fx.Float32) if b_bf16 else beta

    @flyc.kernel(name=f"gdn_tree_chunk_h{H}_hv{HV}_vs{vsplit}", known_block_size=[WAVE, 1, 1])
    def kernel(
        q: fx.Pointer,  # bf16 (1, tokens, H, K), token stride stride_q
        k: fx.Pointer,
        v: fx.Pointer,  # bf16 (1, tokens, HV, V), token stride stride_v
        a: fx.Pointer,  # (tokens, HV), token stride stride_a
        b: fx.Pointer,
        A_log: fx.Pointer,  # (HV,)
        dt_bias: fx.Pointer,
        o: fx.Pointer,  # bf16 [tokens, HV, V]
        pool: fx.Pointer,  # bf16 [slots, HV, V, K]
        h0_indices: fx.Pointer,
        cu_seqlens: fx.Pointer,
        parent: fx.Pointer,  # (batch, width) parent node of every node, node 0 the root
        records: fx.Pointer,  # f32 (rows, steps, HV, REC): delta [0, V), key [V, V + K), decay at V + K
        record_indices: fx.Pointer,
        stride_q: fx.Int32,
        stride_k: fx.Int32,
        stride_v: fx.Int32,
        stride_a: fx.Int32,
        stride_b: fx.Int32,
        stride_parent: fx.Int32,
        width_parent: fx.Int32,
        stride_pool: fx.Int32,
        stride_records_row: fx.Int32,
    ):
        lane = fx.thread_idx.x
        n16, g = lane % 16, lane // 16
        i_n, i_hv, i_vs = fx.block_idx.x, fx.block_idx.y, fx.block_idx.z
        i_h = i_hv // gqa
        bos = fx.Int32(fx.ptr_load(cu_seqlens + i_n))
        T_ = fx.Int32(fx.ptr_load(cu_seqlens + i_n + 1)) - bos
        idx = fx.Int32(fx.ptr_load(h0_indices + i_n))
        row = fx.Int32(fx.ptr_load(record_indices + i_n))
        sh = fx.SharedAllocator().allocate(Shared).peek()
        s_jump, s_anc, s_lg, s_lgsum, s_beta = sh.jump.ptr, sh.anc.ptr, sh.lg.ptr, sh.lgsum.ptr, sh.beta.ptr
        s_x, s_y, s_t, s_w = sh.x.ptr, sh.y.ptr, sh.tinv.ptr, sh.w.ptr
        # column node n16 (rows past T are zeroed)
        tok_ok = n16 < T_
        tok = bos + arith.select(tok_ok, n16, fx.Int32(0))

        def load_unit(src, stride, mul):
            # token n16's head i_h, dims kb * 32 + g * 8 .. + 8 (the MFMA A fragment), L2-normalized over the 128 dims
            # of four lanes, times mul
            xs = []
            ss = fx.Float32(0.0)
            for kb in range_constexpr(n_kb):
                x = fx.Vector(fx.ptr_load(src + tok * stride + i_h * K + kb * 32 + g * 8, result_type=T.vec(8, T.bf16)))
                vals = [
                    fx.Float32(arith.select(tok_ok, fx.BFloat16(x[i]).to(fx.Float32), fx.Float32(0.0)))
                    for i in range_constexpr(8)
                ]
                for vv in vals:
                    ss = ss + vv * vv
                xs.append(vals)
            ss = ss + ss.shuffle_xor(16, WAVE)
            ss = ss + ss.shuffle_xor(32, WAVE)
            d = fx.Float32(fmath.sqrt(ss + 1e-6))
            return [[xi / d * mul for xi in xb] for xb in xs]

        def split(xs, with_lo):
            hi, lo = [], []
            for xb in xs:
                h = [xi.to(fx.BFloat16) for xi in xb]
                hi.append(fx.Vector.from_elements(h, fx.BFloat16))
                if with_lo:
                    lo.append(
                        fx.Vector.from_elements(
                            [(xb[i] - h[i].to(fx.Float32)).to(fx.BFloat16) for i in range_constexpr(8)], fx.BFloat16
                        )
                    )
            return hi, lo

        def split3(xs):
            # the third bf16 term of x = hi + lo + lo2
            out = []
            for xb in xs:
                r = []
                for i in range_constexpr(8):
                    h = xb[i].to(fx.BFloat16).to(fx.Float32)
                    l = (xb[i] - h).to(fx.BFloat16).to(fx.Float32)  # noqa: E741 - preserve AMD arithmetic notation
                    r.append((xb[i] - h - l).to(fx.BFloat16))
                out.append(fx.Vector.from_elements(r, fx.BFloat16))
            return out

        kf = load_unit(k, stride_k, 1.0)
        qf = load_unit(q, stride_q, scale)
        k_hi, k_lo = split(kf, True)
        k_lo2 = split3(kf)
        q_hi, q_lo = split(qf, True)

        # gates of the column token: exp(g) and beta (the serial walk's expressions; softplus beta 1, threshold 20)
        a_t = f32(fx.ptr_load(a + tok * stride_a + i_hv), a_bf16)
        b_t = f32(fx.ptr_load(b + tok * stride_b + i_hv), b_bf16)
        one, zero = fx.Float32(1.0), fx.Float32(0.0)
        xg = a_t + f32(fx.ptr_load(dt_bias + i_hv), dtb_bf16)
        sp = fx.Float32(arith.select(xg <= 20.0, fx.Float32(fmath.log(one + fx.Float32(fmath.exp(xg)))), xg))
        neg_a = zero - fx.Float32(fmath.exp(f32(fx.ptr_load(A_log + i_hv), alog_bf16)))
        eg = fx.Float32(fmath.exp(neg_a * sp))
        eg = fx.Float32(arith.select(tok_ok, eg, fx.Float32(1.0)))
        beta_t = round_beta(one / (one + fx.Float32(fmath.exp(zero - b_t))))
        beta_t = fx.Float32(arith.select(tok_ok, beta_t, fx.Float32(0.0)))
        lg = fx.Float32(fmath.log(fx.Float32(arith.select(eg > 1e-30, eg, fx.Float32(1e-30)))))
        # the records' keys (normalized k) and decays, once per (request, v-head); early, so kf's registers free up
        if tok_ok & (row >= 0) & (i_vs == 0):
            rec = records + row * stride_records_row + (n16 * HV + i_hv) * REC
            for kb in range_constexpr(n_kb):
                fx.ptr_store(fx.Vector.from_elements(kf[kb], fx.Float32), rec + V + kb * 32 + g * 8)
            if g == 0:
                fx.ptr_store(eg, rec + V + K)

        # strict-ancestor bit masks by pointer jumping: node n16 holds (jump, anc); 4 rounds cover depth 15
        has_par = tok_ok & (n16 > 0)
        par = fx.Int32(fx.ptr_load(parent + i_n * stride_parent + arith.select(has_par, n16, fx.Int32(0))))
        has_par = has_par & (par >= 0) & (par < width_parent)
        jmp = arith.select(has_par, par, n16)
        anc = arith.select(has_par, fx.Int32(1) << arith.select(has_par, par, fx.Int32(0)), fx.Int32(0))
        if g == 0:
            fx.ptr_store(jmp, s_jump + n16)
            fx.ptr_store(anc, s_anc + n16)
            fx.ptr_store(lg, s_lg + n16)
            fx.ptr_store(beta_t, s_beta + n16)
        fx.barrier()
        for _ in range_constexpr(4):
            a2 = fx.Int32(fx.ptr_load(s_anc + jmp))
            j2 = fx.Int32(fx.ptr_load(s_jump + jmp))
            fx.barrier()
            anc = anc | a2
            jmp = j2
            if g == 0:
                fx.ptr_store(jmp, s_jump + n16)
                fx.ptr_store(anc, s_anc + n16)
            fx.barrier()
        # logG of node n16: its own log-decay and its ancestors'
        lgs = lg
        for j in range_constexpr(NP):
            lgs = lgs + fx.Float32(
                arith.select(((anc >> j) & 1) == 1, fx.Float32(fx.ptr_load(s_lg + j)), fx.Float32(0.0))
            )
        if g == 0:
            fx.ptr_store(lgs, s_lgsum + n16)
        fx.barrier()

        # this lane's rows t = 4g + e
        rows_t = [g * 4 + e for e in range_constexpr(4)]
        row_ok = [rows_t[e] < T_ for e in range_constexpr(4)]
        lgs_r = [fx.Float32(fx.ptr_load(s_lgsum + rows_t[e])) for e in range_constexpr(4)]
        G = [fx.Float32(fmath.exp(lgs_r[e])) for e in range_constexpr(4)]
        beta = [fx.Float32(fx.ptr_load(s_beta + rows_t[e])) for e in range_constexpr(4)]
        R = []
        ratio = []
        for e in range_constexpr(4):
            anc_r = fx.Int32(fx.ptr_load(s_anc + rows_t[e]))
            strict = ((anc_r >> n16) & 1) == 1
            in_m = strict | (rows_t[e] == n16)
            R.append(fx.Float32(arith.select(strict, fx.Float32(1.0), fx.Float32(0.0))))
            ratio.append(fx.Float32(arith.select(in_m, fx.Float32(fmath.exp(lgs_r[e] - lgs)), fx.Float32(0.0))))

        def mfma(a_, b_, c):
            return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a_, b_, c]))

        # K K^T and Q K^T: the B fragment of K^T is K's own A fragment; all four hi / lo products (~fp32, as the
        # Triton chunk's fp32 dots: K K^T feeds Tinv and so the committed deltas)
        kk = fx.Vector.filled(4, 0.0, fx.Float32)
        qk = fx.Vector.filled(4, 0.0, fx.Float32)
        for kb in range_constexpr(n_kb):
            kk = mfma(k_lo[kb], k_lo[kb], kk)
            kk = mfma(k_hi[kb], k_lo[kb], kk)
            kk = mfma(k_lo[kb], k_hi[kb], kk)
            kk = mfma(k_hi[kb], k_hi[kb], kk)
            qk = mfma(q_lo[kb], k_lo[kb], qk)
            qk = mfma(q_hi[kb], k_lo[kb], qk)
            qk = mfma(q_lo[kb], k_hi[kb], qk)
            qk = mfma(q_hi[kb], k_hi[kb], qk)

        def mm(xm, ym):
            # 16 x 16 fp32 product on the 16x16x4 fp32 MFMA; inputs and output in the accumulator layout, the
            # operands re-read through LDS
            for e in range_constexpr(4):
                fx.ptr_store(xm[e], s_x + rows_t[e] * NP + n16)
                fx.ptr_store(ym[e], s_y + rows_t[e] * NP + n16)
            fx.barrier()
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kb in range_constexpr(4):
                av = fx.Float32(fx.ptr_load(s_x + n16 * NP + kb * 4 + g))
                bv = fx.Float32(fx.ptr_load(s_y + (kb * 4 + g) * NP + n16))
                acc = fx.Vector(_mfma_f32(av, bv, acc))
            fx.barrier()
            return [fx.Float32(acc[e]) for e in range_constexpr(4)]

        eye = [fx.Float32(arith.select(rows_t[e] == n16, fx.Float32(1.0), fx.Float32(0.0))) for e in range_constexpr(4)]
        lmat = [beta[e] * R[e] * ratio[e] * fx.Float32(kk[e]) for e in range_constexpr(4)]
        # (I + L)^-1 = (I - L)(I + L^2)(I + L^4)(I + L^8): L is nilpotent (strict ancestors only)
        tinv_m = [eye[e] - lmat[e] for e in range_constexpr(4)]
        p_m = mm(lmat, lmat)
        tinv_m = mm(tinv_m, [eye[e] + p_m[e] for e in range_constexpr(4)])
        p_m = mm(p_m, p_m)
        tinv_m = mm(tinv_m, [eye[e] + p_m[e] for e in range_constexpr(4)])
        p_m = mm(p_m, p_m)
        tinv_m = mm(tinv_m, [eye[e] + p_m[e] for e in range_constexpr(4)])
        for e in range_constexpr(4):
            fx.ptr_store(tinv_m[e], s_t + rows_t[e] * NP + n16)
            fx.ptr_store(ratio[e] * fx.Float32(qk[e]), s_w + rows_t[e] * NP + n16)
        fx.barrier()

        def xor_g(x):
            # g ^ x for g, x in 0..3 without a runtime xor: bit 0 / bit 1 flipped where x has them
            r = g
            if x & 1:
                r = r + 1 - 2 * (r % 2)
            if x & 2:
                r = r + 2 - 4 * ((r // 2) % 2)
            return r

        # Tinv / W at the columns 4 (g ^ x) + e' that the xor-x shuffle brings in
        # read from LDS inside the tile loop (keeping all 128 values in registers costs occupancy)
        t_off = [[rows_t[e] * NP + xor_g(x) * 4 for x in range_constexpr(4)] for e in range_constexpr(4)]
        has_state = idx >= 0
        s0_base = pool + arith.select(has_state, idx, fx.Int32(0)) * stride_pool + i_hv * V * K

        def gather(vals):
            # vals_x[e'] = the value of row 4 (g ^ x) + e' of this column
            out = [vals]
            for x in range_constexpr(1, 4):
                out.append([fx.Float32(vals[e2]).shuffle_xor(16 * x, 64) for e2 in range_constexpr(4)])
            return out

        for vt in range_constexpr(tiles):
            col = (i_vs * tiles + vt) * 16 + n16
            ks = fx.Vector.filled(4, 0.0, fx.Float32)
            qs = fx.Vector.filled(4, 0.0, fx.Float32)
            for kb in range_constexpr(n_kb):
                s0 = fx.Vector(fx.ptr_load(s0_base + col * K + kb * 32 + g * 8, result_type=T.vec(8, T.bf16)))
                s0 = fx.Vector.from_elements(
                    [fx.BFloat16(arith.select(has_state, s0[i], fx.BFloat16(0.0))) for i in range_constexpr(8)],
                    fx.BFloat16,
                )
                ks = mfma(k_hi[kb], s0, ks)
                ks = mfma(k_lo2[kb], s0, ks)
                ks = mfma(k_lo[kb], s0, ks)
                qs = mfma(q_hi[kb], s0, qs)
            rhs = []
            for e in range_constexpr(4):
                t = arith.select(row_ok[e], rows_t[e], fx.Int32(0))
                vv = fx.BFloat16(fx.ptr_load(v + (bos + t) * stride_v + i_hv * V + col)).to(fx.Float32)
                vv = fx.Float32(arith.select(row_ok[e], vv, fx.Float32(0.0)))
                rhs.append(beta[e] * (vv - G[e] * fx.Float32(ks[e])))
            rg = gather(rhs)
            d = []
            for e in range_constexpr(4):
                acc = fx.Float32(0.0)
                for x in range_constexpr(4):
                    for e2 in range_constexpr(4):
                        acc = acc + fx.Float32(fx.ptr_load(s_t + t_off[e][x] + e2)) * rg[x][e2]
                d.append(acc)
            dg = gather(d)
            for e in range_constexpr(4):
                acc = G[e] * fx.Float32(qs[e])
                for x in range_constexpr(4):
                    for e2 in range_constexpr(4):
                        acc = acc + fx.Float32(fx.ptr_load(s_w + t_off[e][x] + e2)) * dg[x][e2]
                if row_ok[e]:
                    fx.ptr_store(acc.to(fx.BFloat16), o + ((bos + rows_t[e]) * HV + i_hv) * V + col)
                    if row >= 0:
                        fx.ptr_store(d[e], records + row * stride_records_row + (rows_t[e] * HV + i_hv) * REC + col)

    @flyc.jit
    def launch(
        q: fx.Pointer,
        k: fx.Pointer,
        v: fx.Pointer,
        a: fx.Pointer,
        b: fx.Pointer,
        A_log: fx.Pointer,
        dt_bias: fx.Pointer,
        o: fx.Pointer,
        pool: fx.Pointer,
        h0_indices: fx.Pointer,
        cu_seqlens: fx.Pointer,
        parent: fx.Pointer,
        records: fx.Pointer,
        record_indices: fx.Pointer,
        stride_q: fx.Int32,
        stride_k: fx.Int32,
        stride_v: fx.Int32,
        stride_a: fx.Int32,
        stride_b: fx.Int32,
        stride_parent: fx.Int32,
        width_parent: fx.Int32,
        stride_pool: fx.Int32,
        stride_records_row: fx.Int32,
        n_req: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            o,
            pool,
            h0_indices,
            cu_seqlens,
            parent,
            records,
            record_indices,
            stride_q,
            stride_k,
            stride_v,
            stride_a,
            stride_b,
            stride_parent,
            width_parent,
            stride_pool,
            stride_records_row,
        ).launch(grid=(n_req, HV, vsplit), block=(WAVE, 1, 1), stream=stream)

    return launch


def _ptr(t):
    return flyc.from_c_void_p(_FX[t.dtype], t.data_ptr(), assumed_align=16 if t.data_ptr() % 16 == 0 else 2)


def supported(q, k, v, pool, records, softplus_beta, softplus_threshold) -> bool:
    return (
        k.shape[-1] == 128
        and v.shape[-1] % 64 == 0
        and pool.dtype == torch.bfloat16
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and q.stride(2) == q.shape[-1]
        and k.stride(2) == k.shape[-1]
        and v.stride(-1) == 1
        and v.stride(2) == v.shape[-1]
        and records.dtype == torch.float32
        and softplus_beta == 1.0
        and softplus_threshold == 20.0
    )


def gdn_tree_chunk_flydsl(
    q, k, v, a, b, A_log, dt_bias, o, pool, h0_indices, cu_seqlens, parent, records, record_indices, scale, vsplit=None
):
    """The tree verify of every request (<= 16 nodes each) in one launch: writes o and the replay records."""
    n = cu_seqlens.shape[0] - 1
    _, _, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    if vsplit is None:
        # few (request, v-head) waves: split V wider
        vsplit = 4 if n * HV <= 192 else (2 if n * HV <= 768 else 1)
    dtypes = (("a", a.dtype), ("b", b.dtype), ("A_log", A_log.dtype), ("dt_bias", dt_bias.dtype))
    launch = build(H, HV, K, V, records.shape[3], vsplit, float(scale), tuple(sorted(dtypes)))
    launch(
        _ptr(q),
        _ptr(k),
        _ptr(v),
        _ptr(a),
        _ptr(b),
        _ptr(A_log),
        _ptr(dt_bias),
        _ptr(o),
        _ptr(pool),
        _ptr(h0_indices),
        _ptr(cu_seqlens),
        _ptr(parent),
        _ptr(records),
        _ptr(record_indices),
        q.stride(1),
        k.stride(1),
        v.stride(1),
        a.stride(-2),
        b.stride(-2),
        parent.stride(0),
        parent.shape[1],
        pool.stride(0),
        records.stride(0),
        n,
        fx.Stream(torch.cuda.current_stream().cuda_stream),
    )
    return o
