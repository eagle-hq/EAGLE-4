# SPDX-FileCopyrightText: Copyright 2025-2026 NVIDIA
# SPDX-FileCopyrightText: Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
# The attention variant below is derived from FlashInfer (Apache-2.0) include/flashinfer/attention/variants.cuh
# (DefaultAttention). Modified by Eagle AI Corp: the mask reads one tail word per query row. Notices and license
# texts: THIRD_PARTY_NOTICES.md and LICENSES/.
"""FlashInfer's FA2 paged prefill with a tree-tail mask, for the tree verify.

A tree verify attends ``width`` query rows to ``kv_len`` keys: every key before
the tree is visible to every row, the last ``width`` keys (the tree) to the
row's ancestors and itself. FlashInfer's custom-mask path loads one mask bit per
(query, key) element on every KV tile and needs ``width * kv_len`` bits written
and packed each round. This module compiles the same kernels (FlashInfer's FA2
templates and JIT flags, ``MaskMode::kCustom``) with an attention variant whose
mask is the stock mask evaluated from the tail alone: keys before the tail are
visible without a load, tail key ``kv_len - width + j`` is bit ``j`` of one
uint64 per query row, stored at the row's ``qo_indptr`` position in the custom
mask buffer. Outputs are the stock kernel's, bit for bit, as long as the prefix
is all-visible.

:func:`jit_args` are the arguments of FlashInfer's public
``gen_customize_batch_prefill_module``: a wrapper built with them
(``BatchPrefillWithPagedKVCacheWrapper(jit_args=..., variant_owns_mask=True)``)
runs this variant, and FlashInfer passes it the stock parameters by name.
"""

from __future__ import annotations

import functools

import torch

MAX_WIDTH = 64  # the tail word is a uint64 per query row

VARIANT_NAME = "TreeTailMaskAttention"

# Same members and arithmetic as flashinfer::DefaultAttention<true, false, false, false> except the mask lookup.
VARIANT_DECL = r"""
#include <flashinfer/attention/variant_helper.cuh>

struct TreeTailMaskAttention : flashinfer::AttentionVariantBase {
  static constexpr bool use_softmax = true;

  const uint64_t* tail_bits;
  uint32_t qo_len, kv_len, tail_start;
  uint32_t window_left;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ TreeTailMaskAttention(const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    sm_scale_log2 = params.sm_scale * flashinfer::math::log2e;
    tail_bits = reinterpret_cast<const uint64_t*>(params.maybe_custom_mask) + params.q_indptr[batch_idx];
    tail_start = kv_len > qo_len ? kv_len - qo_len : 0;
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
  }

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    bool mask = true;
    if (qo_idx >= qo_len || kv_idx >= kv_len) {
      mask = false;
    } else if (kv_idx >= tail_start) {
      mask = (tail_bits[qo_idx] >> (kv_idx - tail_start)) & 1;
    }
    return mask;
  })
};
"""

# The stock FA2 module's additional parameters (gen_batch_prefill_module); FlashInfer fills them by name.
_TENSOR_NAMES = [
    "maybe_custom_mask",
    "maybe_mask_indptr",
    "maybe_alibi_slopes",
    "maybe_prefix_len_ptr",
    "maybe_token_pos_in_items_ptr",
    "maybe_max_item_len_ptr",
    "maybe_k_cache_sf",
    "maybe_v_cache_sf",
]
_TENSOR_DTYPES = ["uint8_t", "int32_t", "float", "uint32_t", "uint16_t", "uint16_t", "uint8_t", "uint8_t"]
_SCALAR_NAMES = ["logits_soft_cap", "sm_scale", "rope_rcp_scale", "rope_rcp_theta", "token_pos_in_items_len"]
_SCALAR_DTYPES = ["double", "double", "double", "double", "int64_t"]


def supported(q_dtype: torch.dtype, kv_dtype: torch.dtype) -> bool:
    """Whether the tail-mask variant covers a wrapper's dtypes: 16-bit q and kv of one dtype."""
    return q_dtype == kv_dtype and q_dtype in (torch.float16, torch.bfloat16)


@functools.cache
def jit_args(dtype: torch.dtype, head_dim: int) -> tuple:
    """Arguments of FlashInfer's ``gen_customize_batch_prefill_module`` for the tail-mask FA2 variant with q,
    kv and output of ``dtype`` and int32 indices (the ``jit_args`` of a prefill wrapper)."""
    from flashinfer.jit.attention import get_batch_prefill_uri

    uri = "tree_tail_mask_" + get_batch_prefill_uri(
        "fa2", dtype, dtype, dtype, torch.int32, head_dim, head_dim, 0, False, False, False
    )
    return (
        uri,
        dtype,
        dtype,
        dtype,
        torch.int32,
        head_dim,
        head_dim,
        _TENSOR_NAMES,
        _TENSOR_DTYPES,
        _SCALAR_NAMES,
        _SCALAR_DTYPES,
        VARIANT_NAME,
        VARIANT_DECL,
    )
