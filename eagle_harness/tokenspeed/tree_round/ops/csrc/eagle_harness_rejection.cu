// Copyright 2026 Eagle AI Corp.
// SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

// Recursive tree rejection sampling and its residual bonus token.

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

constexpr int MAX_TREE_NODES = 64;

// Verify ranking order, torch.argmax's: NaN first, then larger value, then lower index.
__device__ __forceinline__ bool argmax_before(float v, int i, float best_v, int best_i) {
  const bool an = isnan(v);
  const bool bn = isnan(best_v);
  if (an != bn) return an;
  if (!an && v != best_v) return v > best_v;
  return i < best_i;
}

// Tree rejection sampling, one CTA per request, one warp per node; greedy rows use the argmax only.
__device__ __forceinline__ int next_child(const int32_t* s_parent, int node, int after,
                                          int num_nodes) {
  for (int j = after + 1; j < num_nodes; ++j) {
    if (s_parent[j] == node) return j;
  }
  return -1;
}

__global__ void tree_reject_sample_kernel(
    float* __restrict__ probas, int64_t probas_stride, const int32_t* __restrict__ tokens,
    const int32_t* __restrict__ parents, const int64_t* __restrict__ q_rows,
    const int32_t* __restrict__ q_slot, const int32_t* __restrict__ q_pos,
    const float* __restrict__ q_store, const int32_t* __restrict__ qid_store,
    const float* __restrict__ coins, int64_t coins_stride, const int32_t* __restrict__ top_ks,
    const int64_t* __restrict__ argmax, int32_t* __restrict__ path,
    int32_t* __restrict__ predicts, int32_t* __restrict__ accept_length,
    int64_t* __restrict__ last_row, int num_nodes, int num_slots, int nc, int vocab) {
  const int b = blockIdx.x;
  __shared__ int32_t s_parent[MAX_TREE_NODES];
  __shared__ int s_accept_child[MAX_TREE_NODES];
  for (int i = threadIdx.x; i < num_nodes; i += blockDim.x) {
    s_parent[i] = parents[(int64_t)b * num_nodes + i];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int num_warps = blockDim.x >> 5;
  const int64_t q_row = q_rows != nullptr ? q_rows[b] : -1;
  for (int node = threadIdx.x >> 5; node < num_nodes; node += num_warps) {
    const int64_t row = (int64_t)b * num_nodes + node;
    float* p = probas + row * probas_stride;
    int child = next_child(s_parent, node, node, num_nodes);
    int accepted = -1;
    const bool greedy = top_ks[row] == 1;
    const int32_t slot =
        (!greedy && q_row >= 0 && child >= 0) ? q_slot[q_row * num_nodes + child] : -1;
    if (slot >= 0) {
      const int64_t store_row = (q_row * num_slots + slot) * nc;
      const float* q_row_ptr = q_store + store_row;
      const int32_t* id_row = qid_store + store_row;
      constexpr int MAX_SAMPLED_CHILDREN = 16;
      int rejected_pos[MAX_SAMPLED_CHILDREN];
      int rejected_count = 0;
      float q_left = 0.0f, p_cand = 0.0f;
      // Argmax before the updates, for a vanishing residual; each touched entry is folded in first.
      float best_v = -INFINITY;
      int best_i = INT_MAX;
      for (int j = lane; j < nc; j += 32) {
        const float q_j = q_row_ptr[j];
        if (q_j > 0.0f) {
          q_left += q_j;
          const int32_t token = id_row[j];
          const float p_j = p[token];
          p_cand += p_j;
          if (static_cast<unsigned>(token) < static_cast<unsigned>(vocab) &&
              argmax_before(p_j, token, best_v, best_i)) {
            best_v = p_j;
            best_i = token;
          }
        }
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        q_left += __shfl_xor_sync(0xffffffff, q_left, offset);
        p_cand += __shfl_xor_sync(0xffffffff, p_cand, offset);
      }
      const float z_out = fmaxf(1.0f - p_cand, 0.0f);
      float z = 1.0f;
      while (child >= 0) {
        const int pos = q_pos[q_row * num_nodes + child];
        const int64_t child_token = tokens[(int64_t)b * num_nodes + child];
        const float q_child = q_row_ptr[pos];
        if (q_child > 0.0f && q_left > 0.0f && z > 0.0f) {
          const float q_i = q_child / q_left;
          const float r_child = p[child_token];
          if (static_cast<uint64_t>(child_token) < static_cast<uint64_t>(vocab) &&
              argmax_before(r_child, static_cast<int>(child_token), best_v, best_i)) {
            best_v = r_child;
            best_i = static_cast<int>(child_token);
          }
          if (coins[(int64_t)b * coins_stride + child] * q_i * z < r_child) {
            accepted = child;
            break;
          }
          const float scale = z / q_left;
          float r_sum = 0.0f, q_sum = 0.0f;
          for (int j = lane; j < nc; j += 32) {
            const float q_j = q_row_ptr[j];
            if (q_j <= 0.0f || j == pos) continue;
            const int32_t token = id_row[j];
            bool gone = false;
            for (int r = 0; r < rejected_count; ++r) gone |= (rejected_pos[r] == j);
            if (gone) {
              r_sum += p[token];
              continue;
            }
            const float r_new = fmaxf(p[token] - scale * q_j, 0.0f);
            p[token] = r_new;
            r_sum += r_new;
            q_sum += q_j;
          }
          for (int offset = 16; offset > 0; offset >>= 1) {
            r_sum += __shfl_xor_sync(0xffffffff, r_sum, offset);
            q_sum += __shfl_xor_sync(0xffffffff, q_sum, offset);
          }
          const float r_y = fmaxf(r_child - scale * q_child, 0.0f);
          if (lane == 0) p[child_token] = r_y;
          __syncwarp();
          const float z_new = z_out + r_sum + r_y;
          if (!(z_new > 1e-6f * z)) {
            if (r_child > 0.0f) {
              accepted = child;
            } else {
              // Bonus = target's argmax: untouched entries keep target values (rare, so scanned).
              for (int j = lane; j < vocab; j += 32) {
                const float v = p[j];
                if (argmax_before(v, j, best_v, best_i)) {
                  best_v = v;
                  best_i = j;
                }
              }
              for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_v = __shfl_xor_sync(0xffffffff, best_v, offset);
                const int other_i = __shfl_xor_sync(0xffffffff, best_i, offset);
                if (argmax_before(other_v, other_i, best_v, best_i)) {
                  best_v = other_v;
                  best_i = other_i;
                }
              }
              if (lane == 0) p[best_i] = 1.0f;
              __syncwarp();
            }
            break;
          }
          z = z_new;
          q_left = q_sum;
          if (rejected_count < MAX_SAMPLED_CHILDREN) rejected_pos[rejected_count++] = pos;
        }
        child = next_child(s_parent, node, child, num_nodes);
      }
    } else if (lane == 0 && greedy) {
      const int64_t best = argmax[row];
      while (child >= 0 && tokens[(int64_t)b * num_nodes + child] != best) {
        child = next_child(s_parent, node, child, num_nodes);
      }
      accepted = child;
    } else if (lane == 0) {
      float total = 1.0f;
      while (child >= 0) {
        const int64_t child_token = tokens[(int64_t)b * num_nodes + child];
        if (static_cast<uint64_t>(child_token) < static_cast<uint64_t>(vocab)) {
          const float child_proba = p[child_token];
          // Accepts with probability p'(child) over the mass its rejected siblings left.
          if (child_proba > 0.0f &&
              coins[(int64_t)b * coins_stride + child] * total < child_proba) {
            accepted = child;
            break;
          }
          total = fmaxf(total - child_proba, 0.0f);
          p[child_token] = 0.0f;
        }
        child = next_child(s_parent, node, child, num_nodes);
      }
    }
    if (lane == 0) s_accept_child[node] = accepted;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int32_t* path_b = path + (int64_t)b * num_nodes;
    int32_t* predicts_b = predicts + (int64_t)b * num_nodes;
    int length = 1;
    int node = 0;
    path_b[0] = 0;
    for (int c = s_accept_child[0]; c >= 0; c = s_accept_child[node]) {
      predicts_b[length - 1] = tokens[(int64_t)b * num_nodes + c];
      path_b[length++] = c;
      node = c;
    }
    for (int i = length; i < num_nodes; ++i) path_b[i] = -1;
    accept_length[b] = length;
    last_row[b] = (int64_t)b * num_nodes + node;
  }
}

// Bonus token: inverse-CDF draw from the last accepted node's unnormalized row; greedy: its argmax.
constexpr int RESIDUAL_SAMPLE_THREADS = 1024;
// Runs of 16 staged in shared memory coalesce the loads; each lane still sums its segment in order.
constexpr int RESIDUAL_SAMPLE_RUN = 16;
static_assert(32 % RESIDUAL_SAMPLE_RUN == 0, "a warp load stages whole runs");
constexpr int RESIDUAL_SAMPLE_SMEM_BYTES =
    RESIDUAL_SAMPLE_THREADS * (RESIDUAL_SAMPLE_RUN + 1) * sizeof(float);

__global__ void __launch_bounds__(RESIDUAL_SAMPLE_THREADS)
    residual_sample_kernel(const float* __restrict__ probas, int64_t probas_stride,
                           const int64_t* __restrict__ last_row, const float* __restrict__ coins,
                           const int32_t* __restrict__ top_ks,
                           const int64_t* __restrict__ argmax,
                           const int32_t* __restrict__ accept_length,
                           int32_t* __restrict__ predicts, int num_nodes, int vocab) {
  const int b = blockIdx.x;
  const int64_t row = last_row[b];
  if (top_ks[row] == 1) {
    if (threadIdx.x == 0) {
      predicts[(int64_t)b * num_nodes + accept_length[b] - 1] = static_cast<int32_t>(argmax[row]);
    }
    return;
  }
  const float* p = probas + row * probas_stride;
  const int seg = (vocab + RESIDUAL_SAMPLE_THREADS - 1) / RESIDUAL_SAMPLE_THREADS;
  const int start = min(threadIdx.x * seg, vocab);
  const int end = min(start + seg, vocab);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  extern __shared__ float s_runs[];
  // Row r of this warp's [32, RUN + 1] tile holds a run of lane r's segment.
  float* runs = s_runs + warp * 32 * (RESIDUAL_SAMPLE_RUN + 1);
  // A warp load stages 32 / RUN runs: lane l fills column l % RUN of run l / RUN.
  const int col = lane % RESIDUAL_SAMPLE_RUN;
  float local = 0.0f;
  int last_positive = -1;
  for (int c = 0; c < seg; c += RESIDUAL_SAMPLE_RUN) {
#pragma unroll
    for (int r0 = 0; r0 < 32; r0 += 32 / RESIDUAL_SAMPLE_RUN) {
      const int r = r0 + lane / RESIDUAL_SAMPLE_RUN;
      const int r_start = min((warp * 32 + r) * seg, vocab);
      const int i = r_start + c + col;
      runs[r * (RESIDUAL_SAMPLE_RUN + 1) + col] =
          i < min(r_start + seg, vocab) ? p[i] : 0.0f;
    }
    __syncwarp();
    const int n = min(RESIDUAL_SAMPLE_RUN, end - start - c);
    for (int j = 0; j < n; ++j) {
      const float v = runs[lane * (RESIDUAL_SAMPLE_RUN + 1) + j];
      local += v;
      if (v > 0.0f) last_positive = start + c + j;
    }
    __syncwarp();
  }
  // Inclusive scan of the segment sums over the block.
  __shared__ float s_warp[RESIDUAL_SAMPLE_THREADS / 32];
  __shared__ int s_last[RESIDUAL_SAMPLE_THREADS / 32];
  __shared__ int s_token;
  float incl = local;
  int last = last_positive;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const float u = __shfl_up_sync(0xffffffff, incl, offset);
    if (lane >= offset) incl += u;
  }
  for (int offset = 16; offset > 0; offset >>= 1) last = max(last, __shfl_xor_sync(0xffffffff, last, offset));
  if (lane == 31) s_warp[warp] = incl;
  if (lane == 0) s_last[warp] = last;
  if (threadIdx.x == 0) s_token = INT_MAX;
  __syncthreads();
  float before = 0.0f, total = 0.0f;
  int any_last = -1;
  for (int w = 0; w < RESIDUAL_SAMPLE_THREADS / 32; ++w) {
    if (w < warp) before += s_warp[w];
    total += s_warp[w];
    any_last = max(any_last, s_last[w]);
  }
  const float exclusive = before + incl - local;
  const float target = coins[b] * total;
  // A claiming lane rescans its segment, staged run by run through its warp's tile.
  const bool claim = local > 0.0f && exclusive <= target && target < exclusive + local;
  for (unsigned claims = __ballot_sync(0xffffffff, claim); claims != 0; claims &= claims - 1) {
    const int r = __ffs(claims) - 1;
    const int r_start = __shfl_sync(0xffffffff, start, r);
    const int r_end = __shfl_sync(0xffffffff, end, r);
    float acc = exclusive;
    for (int c = r_start; c < r_end; c += RESIDUAL_SAMPLE_RUN) {
      if (lane < RESIDUAL_SAMPLE_RUN) runs[lane] = c + lane < r_end ? p[c + lane] : 0.0f;
      __syncwarp();
      int token = -1;
      if (lane == r) {
        const int n = min(RESIDUAL_SAMPLE_RUN, r_end - c);
        for (int j = 0; j < n; ++j) {
          acc += runs[j];
          if (acc > target && runs[j] > 0.0f) {
            token = c + j;
            break;
          }
        }
      }
      token = __shfl_sync(0xffffffff, token, r);
      __syncwarp();
      if (token >= 0) {
        // Rounding can make neighbouring segments overlap: the lowest claim wins.
        if (lane == 0) atomicMin(&s_token, token);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // Rounding can also leave the target in a gap, or at the very end of the mass.
    const int token = s_token != INT_MAX ? s_token : max(any_last, 0);
    predicts[(int64_t)b * num_nodes + accept_length[b] - 1] = token;
  }
}

void tree_reject_sample(TensorView probas, TensorView tokens, TensorView parents,
                                      Optional<TensorView> q_rows, TensorView q_slot,
                                      TensorView q_pos, TensorView q_store, TensorView qid_store,
                                      TensorView coins, TensorView top_ks, TensorView argmax,
                                      TensorView path, TensorView predicts,
                                      TensorView accept_length, TensorView last_row) {
  TVM_FFI_ICHECK(probas.stride(1) == 1 && coins.stride(1) == 1);
  const int batch_size = static_cast<int>(tokens.size(0));
  const int num_nodes = static_cast<int>(tokens.size(1));
  TVM_FFI_ICHECK_LE(num_nodes, MAX_TREE_NODES);
  // The kernel indexes the trees and their q tables with the trees' row width.
  TVM_FFI_ICHECK(tokens.stride(0) == num_nodes && parents.size(1) == num_nodes &&
                 parents.stride(0) == num_nodes && coins.size(1) >= num_nodes &&
                 probas.size(0) >= static_cast<int64_t>(batch_size) * num_nodes)
      << "tree_reject_sample: row widths";
  TVM_FFI_ICHECK(top_ks.ndim() == 1 && top_ks.stride(0) == 1 && argmax.ndim() == 1 &&
                 argmax.stride(0) == 1 &&
                 top_ks.size(0) >= static_cast<int64_t>(batch_size) * num_nodes &&
                 argmax.size(0) >= static_cast<int64_t>(batch_size) * num_nodes)
      << "tree_reject_sample: one top-k and argmax per row";
  if (q_rows.has_value()) {
    TVM_FFI_ICHECK(q_slot.size(1) == num_nodes && q_pos.size(1) == num_nodes &&
                   q_slot.stride(0) == num_nodes && q_pos.stride(0) == num_nodes)
        << "tree_reject_sample: q table row widths";
  }
  if (batch_size == 0) return;
  const bool sampled = q_rows.has_value();
  const int threads = std::min(1024, 32 * num_nodes);
  tree_reject_sample_kernel<<<batch_size, threads, 0, get_stream(probas.device())>>>(
      static_cast<float*>(probas.data_ptr()), probas.stride(0),
      static_cast<const int32_t*>(tokens.data_ptr()), static_cast<const int32_t*>(parents.data_ptr()),
      sampled ? static_cast<const int64_t*>(q_rows.value().data_ptr()) : nullptr,
      static_cast<const int32_t*>(q_slot.data_ptr()), static_cast<const int32_t*>(q_pos.data_ptr()),
      static_cast<const float*>(q_store.data_ptr()),
      static_cast<const int32_t*>(qid_store.data_ptr()), static_cast<const float*>(coins.data_ptr()),
      coins.stride(0), static_cast<const int32_t*>(top_ks.data_ptr()),
      static_cast<const int64_t*>(argmax.data_ptr()), static_cast<int32_t*>(path.data_ptr()),
      static_cast<int32_t*>(predicts.data_ptr()), static_cast<int32_t*>(accept_length.data_ptr()),
      static_cast<int64_t*>(last_row.data_ptr()), num_nodes, static_cast<int>(q_store.size(1)),
      static_cast<int>(q_store.size(2)), static_cast<int>(probas.size(1)));
}

void residual_sample(TensorView probas, TensorView last_row, TensorView coins,
                                   TensorView top_ks, TensorView argmax, TensorView accept_length,
                                   TensorView predicts, int64_t num_nodes) {
  TVM_FFI_ICHECK_EQ(probas.stride(1), 1);
  const int batch_size = static_cast<int>(last_row.size(0));
  TVM_FFI_ICHECK(predicts.size(0) >= batch_size * num_nodes && accept_length.size(0) >= batch_size)
      << "residual_sample: output rows";
  // last_row indexes probas rows, and so the per-row top-k and argmax.
  TVM_FFI_ICHECK(top_ks.ndim() == 1 && top_ks.stride(0) == 1 && argmax.ndim() == 1 &&
                 argmax.stride(0) == 1 && top_ks.size(0) >= probas.size(0) &&
                 argmax.size(0) >= probas.size(0))
      << "residual_sample: one top-k and argmax per row";
  if (batch_size == 0) return;
  // The opt-in to over 48 KB of dynamic shared memory is per device: once each.
  static std::atomic<uint64_t> opted_in{0};
  const int device = probas.device().device_id;
  TVM_FFI_ICHECK(device < 64) << "residual_sample: device " << device;
  if (!((opted_in.load(std::memory_order_relaxed) >> device) & 1)) {
    const cudaError_t smem_status = cudaFuncSetAttribute(
        residual_sample_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        RESIDUAL_SAMPLE_SMEM_BYTES);
    TVM_FFI_ICHECK(smem_status == cudaSuccess)
        << "residual_sample: " << cudaGetErrorString(smem_status);
    opted_in.fetch_or(uint64_t{1} << device, std::memory_order_relaxed);
  }
  residual_sample_kernel<<<batch_size, RESIDUAL_SAMPLE_THREADS, RESIDUAL_SAMPLE_SMEM_BYTES,
                           get_stream(probas.device())>>>(
      static_cast<const float*>(probas.data_ptr()), probas.stride(0),
      static_cast<const int64_t*>(last_row.data_ptr()), static_cast<const float*>(coins.data_ptr()),
      static_cast<const int32_t*>(top_ks.data_ptr()), static_cast<const int64_t*>(argmax.data_ptr()),
      static_cast<const int32_t*>(accept_length.data_ptr()),
      static_cast<int32_t*>(predicts.data_ptr()), static_cast<int>(num_nodes),
      static_cast<int>(probas.size(1)));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tree_reject_sample, tree_reject_sample);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(residual_sample, residual_sample);
