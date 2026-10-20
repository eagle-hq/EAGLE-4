# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""L2 prefetch of a byte range.

Warms the L2 cache with data a later kernel reads (a GEMM's weights) while the
GPU's HBM would otherwise idle: a few CTAs issue ``prefetch.global.L2`` for
every 128-byte line of the range, on a forked side stream. The kernel has no
stores, so it never changes a result.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

LINE_BYTES = 128  # the kernel's line stride is the literal 128 (a global would have to be a constexpr)
BLOCK_LINES = 128  # one line per thread of a 4-warp CTA per step


@triton.jit
def _prefetch_lines_l2(ptrs, mask):
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.ne.s32 %p0, $2, 0;
            @%p0 prefetch.global.L2 [$1];
            mov.u32 $0, 0;
        }
        """,
        "=r,l,r",
        args=[ptrs, mask.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _l2_prefetch_kernel(base_ptr, num_lines, BLOCK: tl.constexpr):
    # CTA p prefetches the BLOCK-line blocks p, p + grid, p + 2 * grid, ...
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    offsets = tl.arange(0, BLOCK)
    for start in range(pid * BLOCK, num_lines, step):
        lines = start + offsets
        _prefetch_lines_l2(base_ptr + lines.to(tl.int64) * 128, lines < num_lines)


def l2_prefetch(tensor: torch.Tensor, start: int, nbytes: int, num_ctas: int) -> None:
    """Prefetch bytes ``[start, start + nbytes)`` of the contiguous CUDA ``tensor`` into L2
    (the 128-byte lines they touch) from ``num_ctas`` CTAs on the current stream."""
    if not (tensor.is_contiguous() and tensor.is_cuda):
        raise ValueError("l2_prefetch needs a contiguous CUDA tensor")
    total = tensor.numel() * tensor.element_size()
    start = max(0, min(start, total))
    nbytes = max(0, min(nbytes, total - start))
    if nbytes == 0 or num_ctas <= 0:
        return
    window = tensor.view(-1).view(torch.uint8)[start : start + nbytes]
    _l2_prefetch_kernel[(num_ctas,)](window, triton.cdiv(nbytes, LINE_BYTES), BLOCK=BLOCK_LINES, num_warps=4)
