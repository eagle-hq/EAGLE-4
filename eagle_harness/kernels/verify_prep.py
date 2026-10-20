# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""One launch that writes a tree verify's tail-mask words (batch size 1).

The tail mask is the part of a tree verify's mask that is not "everything
before the tree is visible": one 64-bit word per query row, bit ``j`` set when
the row sees tree key ``j``. The words come from the verify input's flattened
``[width, kv_len]`` bool mask. The committed length is read on the device, so
the launch can sit inside the CUDA graph that builds the verify input.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_TAIL_WIDTH = 64  # one uint64 per row


@triton.jit
def _tail_words_kernel(mask_ptr, committed_len_ptr, tail_bits_ptr, width, WIDTH_P2: tl.constexpr):
    kv_len = tl.load(committed_len_ptr).to(tl.int64) + width
    rows = tl.arange(0, WIDTH_P2)
    cols = tl.arange(0, WIDTH_P2)
    visible = tl.load(
        mask_ptr + rows[:, None] * kv_len + (kv_len - width) + cols[None, :],
        mask=(rows[:, None] < width) & (cols[None, :] < width),
        other=0,
    )
    bits = tl.sum(visible.to(tl.int64) << cols[None, :].to(tl.int64), axis=1)
    tl.store(tail_bits_ptr + rows, bits, mask=rows < width)


def tree_tail_words(mask: torch.Tensor, committed_len: torch.Tensor, tail_bits: torch.Tensor, width: int) -> None:
    """Write ``tail_bits[:width]`` (int64), the tail-mask words of one request's verify of ``width`` tokens
    after ``committed_len`` KV slots, read from ``mask`` (its flattened ``[width, kv_len]`` mask, bool or
    uint8). ``committed_len`` is a one-element device tensor, so the launch can be captured in a CUDA graph;
    ``width`` is a host value."""
    if not 0 < width <= MAX_TAIL_WIDTH:
        raise ValueError(f"verify width {width} outside 1..{MAX_TAIL_WIDTH}")
    _tail_words_kernel[(1,)](
        mask.view(torch.uint8), committed_len, tail_bits, width, WIDTH_P2=triton.next_power_of_2(width)
    )
