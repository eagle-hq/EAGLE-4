# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The online-softmax step the engines' row attention kernels share (``@triton.jit``, inlined into each kernel)."""

import triton
import triton.language as tl


@triton.jit
def online_block(qv, k, v, ok, peak, denom, acc, scale_log2):
    """Online softmax of every row over one block of keys (``ok`` ``[rows, keys]`` masks the keys a row does not see).

    ``peak``, ``denom`` and ``acc`` are the running row maximum (log2 domain), normaliser and weighted value sum;
    returns them updated. A row that has seen no key yet keeps ``peak = -inf`` and adds nothing."""
    s = tl.dot(qv, tl.trans(k)) * scale_log2
    s = tl.where(ok, s, float("-inf"))
    new_peak = tl.maximum(peak, tl.max(s, 1))
    safe = tl.where(new_peak == float("-inf"), 0.0, new_peak)
    alpha = tl.exp2(peak - safe)
    p = tl.exp2(s - safe[:, None])
    denom = denom * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return new_peak, denom, acc
