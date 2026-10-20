// Copyright 2026 Eagle AI Corp.
// SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

// EAGLE-4 candidate head: CUDA versions of the Triton kernels in candidate_triton.py (shard top-k
// chunk + merge, candidate tail) with the same results bit for bit. Torch-free: a C ABI over raw
// pointers and a stream (one binary per CUDA major serves every torch and Python), called through
// ctypes by candidate.py; the shard top-k needs sm_90+ clusters, the tail runs from sm_80.
//
// Selection semantics (torch.topk(sorted=False), as the Triton kernels): keys are torch's
// order-preserving radix keys (NaN largest); the K-th largest key `kth` is found exactly, then
// the output holds the keys above kth in ascending position order followed by the kth ties in
// ascending position order until K are taken.
//
// Shard top-k (one launch instead of chunk + merge): one cluster of 8 CTAs per row, each CTA a
// contiguous 1/8 of the row, each thread E contiguous positions held as keys in registers.
// Two 8-bit histogram passes (high byte, then the low byte of the keys in the K-th key's bin)
// find kth; every CTA pushes its 256-bin histogram into every CTA's shared memory and one
// cluster barrier per pass publishes them, so each CTA also knows how many selected keys the
// earlier CTAs hold and writes its own selected positions straight to their output slots.
// After the second barrier no CTA touches another's shared memory, so none waits at exit.
//
// Candidate tail: one CTA per row, one gathered candidate (rank r, column j: candidate order
// r * NCL + j) per thread. The NC-th largest key by a bitwise radix select (one
// __syncthreads_count per key bit), output slots by warp ballots; then warp 0 runs torch's
// softmax_warp_forward arithmetic (lane l holds slots l, l + 32, ...; ternary max and
// xor-butterfly reductions; expf; IEEE division) and fast_topk: the top_k largest
// probabilities by repeated warp argmax (largest key, then lowest slot: torch.max for k = 1, and
// torch.topk's selection), output in descending order when their keys are distinct; with ties
// among them, torch.topk(sorted=True)'s order comes from replaying its gatherTopK placement and
// the 32-wide SmallBitonicSort network on the selected entries.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace cg = cooperative_groups;

namespace {

constexpr int kCluster = 8;
constexpr uint32_t kNegInf = 0xff800000u, kQuietNaN = 0x7fc00000u;  // fp32 bit patterns

template <typename T>
__device__ __forceinline__ float to_float(T v);
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_float<float>(float v) {
  return v;
}

// torch.topk's order-preserving key (TopKTypeConfig::convert); NaN maps to the maximum.
template <typename T>
__device__ __forceinline__ uint32_t radix_key(T v);
template <>
__device__ __forceinline__ uint32_t radix_key<__nv_bfloat16>(__nv_bfloat16 v) {
  const uint32_t b = __bfloat16_as_ushort(v);
  const uint32_t key = (b & 0x8000u) ? (b ^ 0xFFFFu) : (b ^ 0x8000u);
  const float f = __bfloat162float(v);
  return (f != f) ? 0xFFFFu : key;
}
template <>
__device__ __forceinline__ uint32_t radix_key<__half>(__half v) {
  const uint32_t b = __half_as_ushort(v);
  const uint32_t key = (b & 0x8000u) ? (b ^ 0xFFFFu) : (b ^ 0x8000u);
  const float f = __half2float(v);
  return (f != f) ? 0xFFFFu : key;
}
template <>
__device__ __forceinline__ uint32_t radix_key<float>(float v) {
  const uint32_t b = __float_as_uint(v);
  const uint32_t key = (b >> 31) ? ~b : (b | 0x80000000u);
  return (v != v) ? 0xFFFFFFFFu : key;
}

__device__ __forceinline__ float radix_key_to_float(uint32_t key) {
  const uint32_t b = (key >> 31) ? (key ^ 0x80000000u) : ~key;
  return __uint_as_float(b);
}

template <typename T>
__device__ __forceinline__ T zero_value();
template <>
__device__ __forceinline__ __nv_bfloat16 zero_value<__nv_bfloat16>() {
  return __ushort_as_bfloat16(0);
}
template <>
__device__ __forceinline__ __half zero_value<__half>() {
  return __ushort_as_half(0);
}
template <>
__device__ __forceinline__ float zero_value<float>() {
  return 0.0f;
}

template <typename T>
struct KeyBits {
  static constexpr int value = 16;
};
template <>
struct KeyBits<float> {
  static constexpr int value = 32;
};

// ------------------------------------------------------------------------------------------
// Shard top-k: one cluster of kCluster CTAs per row.

// Exclusive prefix sum over the threads of the block (thread order), unsigned (two packed
// 16-bit fields). All threads call it, once (s_warp is not reusable afterwards).
template <int THREADS>
__device__ __forceinline__ uint32_t block_exclusive_scan_once(uint32_t v, uint32_t* s_warp) {
  constexpr int WARPS = THREADS / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t incl = v;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const uint32_t u = __shfl_up_sync(0xffffffffu, incl, off);
    if (lane >= off) incl += u;
  }
  if (lane == 31) s_warp[warp] = incl;
  __syncthreads();
  uint32_t before = 0;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) before += (w < warp) ? s_warp[w] : 0u;
  return before + incl - v;
}

// One warp, after the cluster barrier: hist[c][bin] holds CTA c's histogram. Finds the bin b of
// the cluster histogram g = sum_c hist[c] (keys in bin b are larger than those in b - 1) holding
// the target-th largest key, S(b) >= target > S(b + 1) with S(b) = sum_{b' >= b} g[b'], and
// writes b, S(b + 1), the same suffix sum over the CTAs before `rank` and their count in bin b.
__device__ __forceinline__ void find_bin_warp(const int (*hist)[256], int rank, int target, int* out) {
  const int lane = threadIdx.x & 31;
  int gv[8], pv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) gv[i] = pv[i] = 0;
#pragma unroll
  for (int c = 0; c < kCluster; ++c) {
    const int4 lo = reinterpret_cast<const int4*>(&hist[c][lane * 8])[0];
    const int4 hi = reinterpret_cast<const int4*>(&hist[c][lane * 8])[1];
    const int v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      gv[i] += v[i];
      pv[i] += c < rank ? v[i] : 0;
    }
  }
  int tg = 0, tp = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    tg += gv[i];
    tp += pv[i];
  }
  // sums over the bins of the higher lanes
  int sg = tg, sp = tp;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int ug = __shfl_down_sync(0xffffffffu, sg, off);
    const int up = __shfl_down_sync(0xffffffffu, sp, off);
    if (lane + off < 32) {
      sg += ug;
      sp += up;
    }
  }
  int above_g = sg - tg, above_p = sp - tp;  // bins > lane * 8 + 7
#pragma unroll
  for (int i = 7; i >= 0; --i) {
    if (above_g < target && above_g + gv[i] >= target) {
      out[0] = lane * 8 + i;
      out[1] = above_g;
      out[2] = above_p;
      out[3] = pv[i];
    }
    above_g += gv[i];
    above_p += pv[i];
  }
}

template <typename T, int THREADS, int E>
__global__ void __launch_bounds__(THREADS) eagle4_shard_topk_cluster_kernel(
        const T* __restrict__ scores,
        int64_t stride_sm,
        int N,
        int K,
        int64_t* __restrict__ local_ids,
        int64_t stride_lm,
        int64_t* __restrict__ global_ids,
        int64_t stride_gm,
        int64_t vocab_start) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 900
  __trap();  // clusters need sm_90; arch.clusters() keeps the launcher away from older GPUs
#else
  constexpr int WARPS = THREADS / 32;
  constexpr int HISTS = WARPS < 16 ? WARPS : 16;
  static_assert(THREADS >= 256 && THREADS % 32 == 0, "needs >= 256 threads");
  __shared__ int s_wh[HISTS][256];                        // per-warp(-pair) histograms
  __shared__ __align__(16) int s_hist[2][kCluster][256];  // [pass][CTA]: pushed by every CTA
  __shared__ uint32_t s_warp[WARPS];
  __shared__ int s_val[8];

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int m = blockIdx.y;
  const int t = threadIdx.x, warp = t >> 5;
  const T* row = scores + static_cast<int64_t>(m) * stride_sm;
  const int pos0 = (rank * THREADS + t) * E;
  // Distributed shared memory may only be written once its CTA runs: arrive now, wait before
  // the first push (the other CTAs started long before then; the wait is short).
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");

  int keys[E];  // radix keys, -1 = no element
#pragma unroll
  for (int i = 0; i < E; ++i) keys[i] = -1;
  const bool full = pos0 + E <= N;
  if constexpr (sizeof(T) == 2 && E % 8 == 0) {
    if (full && (reinterpret_cast<uintptr_t>(row + pos0) & 15u) == 0) {
#pragma unroll
      for (int v = 0; v < E / 8; ++v) {
        const uint4 q = *reinterpret_cast<const uint4*>(row + pos0 + v * 8);
        const T* h = reinterpret_cast<const T*>(&q);
#pragma unroll
        for (int i = 0; i < 8; ++i) keys[v * 8 + i] = static_cast<int>(radix_key<T>(h[i]));
      }
    } else {
#pragma unroll
      for (int i = 0; i < E; ++i)
        if (pos0 + i < N) keys[i] = static_cast<int>(radix_key<T>(row[pos0 + i]));
    }
  } else {
#pragma unroll
    for (int i = 0; i < E; ++i)
      if (pos0 + i < N) keys[i] = static_cast<int>(radix_key<T>(row[pos0 + i]));
  }
  for (int i = t; i < HISTS * 256; i += THREADS) (&s_wh[0][0])[i] = 0;
  __syncthreads();

  // Pass 1: histogram of the high byte, pushed into every CTA of the cluster.
#pragma unroll
  for (int i = 0; i < E; ++i)
    if (keys[i] >= 0) atomicAdd(&s_wh[warp % HISTS][keys[i] >> 8], 1);
  __syncthreads();
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
  if (t < 256) {
    int h = 0;
#pragma unroll
    for (int w = 0; w < HISTS; ++w) {
      h += s_wh[w][t];
      s_wh[w][t] = 0;
    }
#pragma unroll
    for (int c = 0; c < kCluster; ++c) cluster.map_shared_rank(&s_hist[0][rank][t], c)[0] = h;
  }
  cluster.sync();
  if (warp == 0) find_bin_warp(s_hist[0], rank, K, s_val);
  __syncthreads();
  const int b1 = s_val[0];         // high byte of the K-th largest key
  const int above1 = s_val[1];     // keys with a larger high byte
  const int pre_above = s_val[2];  // ... of them in the earlier CTAs

  // Pass 2: histogram of the low byte of the keys in bin b1.
#pragma unroll
  for (int i = 0; i < E; ++i)
    if (keys[i] >= 0 && (keys[i] >> 8) == b1) atomicAdd(&s_wh[warp % HISTS][keys[i] & 255], 1);
  __syncthreads();
  if (t < 256) {
    int h = 0;
#pragma unroll
    for (int w = 0; w < HISTS; ++w) h += s_wh[w][t];
#pragma unroll
    for (int c = 0; c < kCluster; ++c) cluster.map_shared_rank(&s_hist[1][rank][t], c)[0] = h;
  }
  // After this barrier no CTA touches another's shared memory, so none has to wait at exit.
  cluster.sync();
  if (warp == 0) find_bin_warp(s_hist[1], rank, K - above1, s_val + 4);
  __syncthreads();
  const int kth = (b1 << 8) | s_val[4];
  const int n_gt = above1 + s_val[5];        // keys above kth
  const int base_gt = pre_above + s_val[6];  // ... of them in the earlier CTAs
  const int base_eq = s_val[7];              // kth ties in the earlier CTAs
  const int remaining = K - n_gt;            // kth ties to take, in position order

  // Pass 3: place this chunk's selected positions.
  int n_gt_t = 0, n_eq_t = 0;
#pragma unroll
  for (int i = 0; i < E; ++i) {
    n_gt_t += keys[i] > kth;
    n_eq_t += keys[i] == kth;
  }
  const uint32_t packed = block_exclusive_scan_once<THREADS>(
      static_cast<uint32_t>(n_gt_t) | (static_cast<uint32_t>(n_eq_t) << 16), s_warp);
  int slot_gt = base_gt + static_cast<int>(packed & 0xFFFFu);
  int rank_eq = base_eq + static_cast<int>(packed >> 16);
  int64_t* lrow = local_ids + static_cast<int64_t>(m) * stride_lm;
  int64_t* grow = global_ids + static_cast<int64_t>(m) * stride_gm;
  if (n_gt_t + n_eq_t > 0) {
#pragma unroll
    for (int i = 0; i < E; ++i) {
      int slot = -1;
      if (keys[i] > kth) {
        slot = slot_gt++;
      } else if (keys[i] == kth) {
        if (rank_eq < remaining) slot = n_gt + rank_eq;
        ++rank_eq;
      }
      if (slot >= 0) {
        const int64_t pos = pos0 + i;
        lrow[slot] = pos;
        grow[slot] = pos + vocab_start;
      }
    }
  }
#endif
}

// ------------------------------------------------------------------------------------------
// Candidate tail: one CTA per row.

// One bitonicSwap round of torch's SmallBitonicSort<32> (descending, invalid entries last) on
// packed entries (rank code << 6 | selection index << 1 | valid), lane = position; the codes
// order the keys like the keys themselves (equal codes for equal keys).
__device__ __forceinline__ uint32_t bitonic_round(uint32_t g, int lane, int stride, int log2_stride, int half) {
  const uint32_t other = __shfl_xor_sync(0xffffffffu, g, stride);
  const bool is_a = (lane & stride) == 0;
  const uint32_t a = is_a ? g : other;
  const uint32_t b = is_a ? other : g;
  const int pos = is_a ? lane : (lane ^ stride);
  const int tx = ((pos >> (log2_stride + 1)) << log2_stride) | (pos & (stride - 1));
  const bool flag = half != 0 && (tx & half) != 0;
  const bool swap = (((a >> 6) > (b >> 6)) && (a & 1u)) || !(b & 1u);
  const bool exchange = swap == flag;
  return is_a ? (exchange ? b : a) : (exchange ? a : b);
}

template <typename LT, typename TT, int ITER>
__global__ void __launch_bounds__(1024) eagle4_candidate_tail_cuda_kernel(
    const LT* __restrict__ cand_logits,
    int64_t stride_cl_r,
    int64_t stride_cl_m,
    const TT* __restrict__ cand_tokens,
    int64_t stride_ct_r,
    int64_t stride_ct_m,
    float* __restrict__ logits_out,
    int64_t stride_lo_m,
    int64_t* __restrict__ tokens_out,
    int64_t stride_to_m,
    float* __restrict__ topk_probs,
    int64_t stride_tp_m,
    int64_t* __restrict__ topk_tokens,
    int64_t stride_tt_m,
    int R,
    int NCL,
    int NC,
    int topk) {
  constexpr int KEY_BITS = KeyBits<LT>::value;
  constexpr int P = ITER * 32;  // next power of two >= NC (NC is a power of two >= 32)
  __shared__ float s_logit[P];
  __shared__ int64_t s_tok[P];
  __shared__ uint16_t s_code[P];
  __shared__ int s_wg[32];
  __shared__ int s_we[32];
  __shared__ uint32_t s_pack[32];

  const int64_t m = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  if (R > 1) {
    const int W = R * NCL;
    const bool valid = t < W;
    const int r = valid ? t / NCL : 0;
    const int j = valid ? t - r * NCL : 0;
    LT x = zero_value<LT>();
    int64_t tok = 0;
    if (valid) {
      x = cand_logits[r * stride_cl_r + m * stride_cl_m + j];
      tok = cand_tokens[r * stride_ct_r + m * stride_ct_m + j];
    }
    const uint32_t key = valid ? radix_key<LT>(x) : 0u;
    // The NC-th largest key, bit by bit from the top (as torch's radix select, one bit per
    // block-wide count).
    uint32_t prefix = 0u;
    int remaining = NC;
#pragma unroll
    for (int bit = KEY_BITS - 1; bit >= 0; --bit) {
      const uint32_t cand = prefix | (1u << bit);
      const uint32_t hi = ~((1u << bit) - 1u);
      const int c = __syncthreads_count(valid && (key & hi) == cand);
      if (c >= remaining) {
        prefix = cand;
      } else {
        remaining -= c;
      }
    }
    const uint32_t kth = prefix;
    const int n_gt = NC - remaining;
    const bool gt = valid && key > kth;
    const bool eq = valid && key == kth;
    const unsigned bg = __ballot_sync(0xffffffffu, gt);
    const unsigned be = __ballot_sync(0xffffffffu, eq);
    if (lane == 0) {
      s_wg[warp] = __popc(bg);
      s_we[warp] = __popc(be);
    }
    __syncthreads();
    int pg = 0, pe = 0;
    for (int w = 0; w < warp; ++w) {
      pg += s_wg[w];
      pe += s_we[w];
    }
    const unsigned lt_mask = (1u << lane) - 1u;
    const int rg = pg + __popc(bg & lt_mask);
    const int re = pe + __popc(be & lt_mask);
    if (gt || (eq && re < remaining)) {
      const int dest = gt ? rg : n_gt + re;
      const float f = to_float<LT>(x);
      logits_out[m * stride_lo_m + dest] = f;
      tokens_out[m * stride_to_m + dest] = tok;
      s_logit[dest] = f;
      s_tok[dest] = tok;
    }
  } else {
    for (int i = t; i < NC; i += blockDim.x) {
      const float f = to_float<LT>(cand_logits[m * stride_cl_m + i]);
      logits_out[m * stride_lo_m + i] = f;
      s_logit[i] = f;
      if (topk > 0) s_tok[i] = cand_tokens[m * stride_ct_m + i];
    }
  }
  if (topk == 0) return;
  __syncthreads();
  if (warp != 0) return;

  // torch softmax_warp_forward (fp32, WARP_SIZE 32, WARP_ITERATIONS = P / 32), one row.
  float e[ITER];
#pragma unroll
  for (int it = 0; it < ITER; ++it) {
    const int idx = lane + it * 32;
    e[it] = idx < NC ? s_logit[idx] : __uint_as_float(kNegInf);
  }
  float mx = e[0];
#pragma unroll
  for (int it = 0; it < ITER; ++it) mx = mx > e[it] ? mx : e[it];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float b = __shfl_xor_sync(0xffffffffu, mx, off);
    mx = mx < b ? b : mx;
  }
  float sum = 0.0f;
#pragma unroll
  for (int it = 0; it < ITER; ++it) {
    e[it] = expf(e[it] - mx);
    sum += e[it];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum = sum + __shfl_xor_sync(0xffffffffu, sum, off);
  uint32_t pk[ITER];
#pragma unroll
  for (int it = 0; it < ITER; ++it) {
    const float p = (sum == 0.0f) ? __uint_as_float(kQuietNaN) : e[it] / sum;
    pk[it] = radix_key<float>(p);
  }

  // fast_topk(probs, topk) over the slots lane + 32 * it (valid below NC): the topk largest
  // (key, then lowest slot) by repeated warp argmax; lane i keeps the i-th.
  unsigned live = 0;
#pragma unroll
  for (int it = 0; it < ITER; ++it) live |= (lane + it * 32 < NC) ? (1u << it) : 0u;
  uint32_t sel_key = 0, prev = 0;
  int sel_slot = 0;
  bool distinct = true;
  for (int i = 0; i < topk; ++i) {
    uint32_t lk = 0;  // valid keys are >= radix_key(+0.0f) = 0x80000000
#pragma unroll
    for (int it = 0; it < ITER; ++it) lk = ((live >> it) & 1u) && pk[it] > lk ? pk[it] : lk;
    const uint32_t gk = __reduce_max_sync(0xffffffffu, lk);
    uint32_t ls = 0;
#pragma unroll
    for (int it = 0; it < ITER; ++it) {
      const uint32_t c = static_cast<uint32_t>(P - (lane + it * 32));
      ls = ((live >> it) & 1u) && pk[it] == gk && c > ls ? c : ls;
    }
    const int slot = P - static_cast<int>(__reduce_max_sync(0xffffffffu, ls));
    if ((slot & 31) == lane) live &= ~(1u << (slot >> 5));
    if (lane == i) {
      sel_key = gk;
      sel_slot = slot;
    }
    distinct = distinct && (i == 0 || gk != prev);
    prev = gk;
  }
  if (distinct) {  // torch.max (k = 1) / any sort of distinct keys: descending key order
    if (lane < topk) {
      topk_probs[m * stride_tp_m + lane] = radix_key_to_float(sel_key);
      topk_tokens[m * stride_tt_m + lane] = s_tok[sel_slot];
    }
    return;
  }
  // Ties among the selected keys: torch.topk(sorted=True)'s order, i.e. the gatherTopK
  // placement (keys above the k-th largest in slot order, then its ties in slot order) ...
  const uint32_t kth = prev;
  const unsigned lt_mask = (1u << lane) - 1u;
  const int n_gt = __popc(__ballot_sync(0xffffffffu, lane < topk && sel_key > kth));
  {
    // code: 32 - (index of the key among the distinct selected keys, largest first)
    const uint32_t up = __shfl_up_sync(0xffffffffu, sel_key, 1);
    const unsigned starts = __ballot_sync(0xffffffffu, lane < topk && (lane == 0 || sel_key != up));
    const int distinct_index = __popc(starts & ((2u << lane) - 1u)) - 1;
    if (lane < topk) s_code[sel_slot] = static_cast<uint16_t>(((32 - distinct_index) << 5) | lane);
  }
  s_pack[lane] = 0;
  __syncwarp();
  int seen_gt = 0, seen_eq = 0;
#pragma unroll
  for (int it = 0; it < ITER; ++it) {
    const int slot = lane + it * 32;
    const bool valid = slot < NC;
    const bool gt = valid && pk[it] > kth;
    const bool eq = valid && pk[it] == kth;
    const unsigned bg = __ballot_sync(0xffffffffu, gt);
    const unsigned be = __ballot_sync(0xffffffffu, eq);
    const int rg = seen_gt + __popc(bg & lt_mask);
    const int re = seen_eq + __popc(be & lt_mask);
    if (gt || (eq && re < topk - n_gt)) {
      const int dest = gt ? rg : n_gt + re;
      s_pack[dest] = (static_cast<uint32_t>(s_code[slot]) << 1) | 1u;
    }
    seen_gt += __popc(bg);
    seen_eq += __popc(be);
  }
  __syncwarp();
  // ... then the SmallBitonicSort<32> network (sizes 2..16, then the final merge).
  uint32_t g = s_pack[lane];
  g = bitonic_round(g, lane, 1, 0, 1);
  g = bitonic_round(g, lane, 2, 1, 2);
  g = bitonic_round(g, lane, 1, 0, 2);
  g = bitonic_round(g, lane, 4, 2, 4);
  g = bitonic_round(g, lane, 2, 1, 4);
  g = bitonic_round(g, lane, 1, 0, 4);
  g = bitonic_round(g, lane, 8, 3, 8);
  g = bitonic_round(g, lane, 4, 2, 8);
  g = bitonic_round(g, lane, 2, 1, 8);
  g = bitonic_round(g, lane, 1, 0, 8);
  g = bitonic_round(g, lane, 16, 4, 0);
  g = bitonic_round(g, lane, 8, 3, 0);
  g = bitonic_round(g, lane, 4, 2, 0);
  g = bitonic_round(g, lane, 2, 1, 0);
  g = bitonic_round(g, lane, 1, 0, 0);
  const int idx = static_cast<int>((g >> 1) & 31u);  // the entry's selection index
  const uint32_t out_key = __shfl_sync(0xffffffffu, sel_key, idx);
  const int out_slot = __shfl_sync(0xffffffffu, sel_slot, idx);
  if (lane < topk) {
    topk_probs[m * stride_tp_m + lane] = radix_key_to_float(out_key);
    topk_tokens[m * stride_tt_m + lane] = s_tok[out_slot];
  }
}

// ------------------------------------------------------------------------------------------
// C ABI. Every entry returns 0 (launched), kNotCovered (nothing launched: the caller runs its
// fallback) or a CUDA error code. dtype: 0 bfloat16, 1 float16, 2 float32.

constexpr int kNotCovered = -1;

template <typename T, int THREADS, int E>
int launch_shard_topk(const void* scores, int64_t stride_sm, int M, int N, int K, int64_t* local_ids,
                      int64_t stride_lm, int64_t* global_ids, int64_t stride_gm, int64_t vocab_start,
                      cudaStream_t stream) {
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {kCluster, 1, 1};
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(kCluster, M);
  config.blockDim = dim3(THREADS);
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, eagle4_shard_topk_cluster_kernel<T, THREADS, E>, static_cast<const T*>(scores),
                            stride_sm, N, K, local_ids, stride_lm, global_ids, stride_gm, vocab_start);
}

template <typename T>
int dispatch_shard_topk(const void* scores, int64_t stride_sm, int M, int N, int K, int64_t* local_ids,
                        int64_t stride_lm, int64_t* global_ids, int64_t stride_gm, int64_t vocab_start,
                        cudaStream_t stream) {
#define EAGLE4_TOPK_CASE(THREADS, E)                                                                      \
  return launch_shard_topk<T, THREADS, E>(scores, stride_sm, M, N, K, local_ids, stride_lm, global_ids,  \
                                          stride_gm, vocab_start, stream)
  if (N <= kCluster * 512 * 8) EAGLE4_TOPK_CASE(512, 8);    // TP8 shards of 248320
  if (N <= kCluster * 1024 * 8) EAGLE4_TOPK_CASE(1024, 8);  // TP4
  if (N <= kCluster * 1024 * 16) EAGLE4_TOPK_CASE(1024, 16);  // TP2
  if (N <= kCluster * 1024 * 32) EAGLE4_TOPK_CASE(1024, 32);  // TP1
#undef EAGLE4_TOPK_CASE
  return kNotCovered;
}

template <typename LT, typename TT, int ITER>
int launch_tail(const void* cl, int64_t cl_r, int64_t cl_m, const void* ct, int64_t ct_r, int64_t ct_m,
                float* logits_out, int64_t lo_m, int64_t* tokens_out, int64_t to_m, float* topk_probs,
                int64_t tp_m, int64_t* topk_tokens, int64_t tt_m, int R, int M, int NCL, int NC, int topk,
                cudaStream_t stream) {
  const int threads = R > 1 ? ((R * NCL + 31) / 32) * 32 : (NC < 1024 ? NC : 1024);
  eagle4_candidate_tail_cuda_kernel<LT, TT, ITER><<<static_cast<unsigned>(M), threads, 0, stream>>>(
      static_cast<const LT*>(cl), cl_r, cl_m, static_cast<const TT*>(ct), ct_r, ct_m, logits_out, lo_m, tokens_out,
      to_m, topk_probs, tp_m, topk_tokens, tt_m, R, NCL, NC, topk);
  return cudaGetLastError();
}

template <typename LT, typename TT>
int dispatch_tail(const void* cl, int64_t cl_r, int64_t cl_m, const void* ct, int64_t ct_r, int64_t ct_m,
                  float* logits_out, int64_t lo_m, int64_t* tokens_out, int64_t to_m, float* topk_probs,
                  int64_t tp_m, int64_t* topk_tokens, int64_t tt_m, int R, int M, int NCL, int NC, int topk,
                  cudaStream_t stream) {
  switch (NC) {
#define EAGLE4_TAIL_CASE(NCV, ITERV)                                                                         \
  case NCV:                                                                                                 \
    return launch_tail<LT, TT, ITERV>(cl, cl_r, cl_m, ct, ct_r, ct_m, logits_out, lo_m, tokens_out, to_m,  \
                                      topk_probs, tp_m, topk_tokens, tt_m, R, M, NCL, NC, topk, stream);
    EAGLE4_TAIL_CASE(32, 1)
    EAGLE4_TAIL_CASE(64, 2)
    EAGLE4_TAIL_CASE(128, 4)
    EAGLE4_TAIL_CASE(256, 8)
#undef EAGLE4_TAIL_CASE
    default:
      return kNotCovered;
  }
}

}  // namespace

extern "C" {

// scores [M, N] (unit column stride) -> local_ids / global_ids [M, K] int64, torch.topk(sorted=False) order.
int eagle4_shard_topk(int device, int dtype, const void* scores, int64_t stride_sm, int M, int N, int K,
                      int64_t* local_ids, int64_t stride_lm, int64_t* global_ids, int64_t stride_gm,
                      int64_t vocab_start, cudaStream_t stream) {
  if (M == 0) return 0;
  if (K < 1 || K > N || K > 32767 || M > 65535) return kNotCovered;
  if (cudaError_t e = cudaSetDevice(device)) return e;
  if (dtype == 0)
    return dispatch_shard_topk<__nv_bfloat16>(scores, stride_sm, M, N, K, local_ids, stride_lm, global_ids,
                                              stride_gm, vocab_start, stream);
  if (dtype == 1)
    return dispatch_shard_topk<__half>(scores, stride_sm, M, N, K, local_ids, stride_lm, global_ids, stride_gm,
                                       vocab_start, stream);
  return kNotCovered;
}

// cand_logits / cand_tokens: [R, M, NCL] views (any rank / row strides, unit column stride); tokens are int64,
// or int32 with fp32 logits (the packed TP gather). tokens_out is unused for R == 1; topk_probs / topk_tokens
// for topk == 0.
int eagle4_candidate_tail(int device, int dtype, int token_bytes, const void* cand_logits, int64_t cl_r,
                          int64_t cl_m, const void* cand_tokens, int64_t ct_r, int64_t ct_m, float* logits_out,
                          int64_t lo_m, int64_t* tokens_out, int64_t to_m, float* topk_probs, int64_t tp_m,
                          int64_t* topk_tokens, int64_t tt_m, int R, int M, int NCL, int NC, int topk,
                          cudaStream_t stream) {
  if (M == 0) return 0;
  if (NCL != NC || R < 1 || R * NCL > 1024 || topk < 0 || topk > 32) return kNotCovered;
  if (cudaError_t e = cudaSetDevice(device)) return e;
#define EAGLE4_TAIL_CALL(LT, TT)                                                                           \
  return dispatch_tail<LT, TT>(cand_logits, cl_r, cl_m, cand_tokens, ct_r, ct_m, logits_out, lo_m,       \
                               tokens_out, to_m, topk_probs, tp_m, topk_tokens, tt_m, R, M, NCL, NC,      \
                               topk, stream)
  if (token_bytes == 8) {
    if (dtype == 0) EAGLE4_TAIL_CALL(__nv_bfloat16, int64_t);
    if (dtype == 1) EAGLE4_TAIL_CALL(__half, int64_t);
    if (dtype == 2) EAGLE4_TAIL_CALL(float, int64_t);
  } else if (token_bytes == 4 && dtype == 2) {
    EAGLE4_TAIL_CALL(float, int32_t);
  }
#undef EAGLE4_TAIL_CALL
  return kNotCovered;
}

}  // extern "C"
