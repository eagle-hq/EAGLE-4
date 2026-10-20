# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The tree-tail mask attention of the batch-size-1 tree verify (FlashInfer variant).

sglang plans each target-verify replay from host-known lengths, but its tree
mask is a packed ``[width x kv_len]`` bit array that the kernel loads on every
KV tile. The tail-mask variant (:mod:`.tail_mask`) needs only one 64-bit word
per query row. A :class:`TreeTailMask` gives sglang that variant for the
``(1, width)`` verify graphs (``SpecInput.flashinfer_verify_variant``, through
FlashInfer's public ``jit_args``) and owns the mask buffer the variant reads,
which the draft's verify-input graph fills each round (:meth:`TreeTailMask.write`).
"""

from __future__ import annotations

import logging
from typing import Optional, Set

import torch

from sglang.srt.speculative.spec_info import FlashInferVerifyVariant

from eagle_harness.kernels.verify_prep import MAX_TAIL_WIDTH, tree_tail_words
from eagle_harness.sglang import tail_mask

logger = logging.getLogger(__name__)


class TreeTailMask:
    """The tail-mask variant of the batch-size-1 verify graphs and the words it reads."""

    def __init__(self, device: torch.device):
        # Words of the rows at qo_indptr 0..width-1 (one request), 8-byte aligned.
        self.tail_bits = torch.zeros(MAX_TAIL_WIDTH, dtype=torch.int64, device=device)
        self.widths: Set[int] = set()  # verify widths whose graph runs the variant

    def variant(
        self, bs: int, num_tokens: int, q_dtype: torch.dtype, kv_dtype: torch.dtype, head_dim: int
    ) -> Optional[FlashInferVerifyVariant]:
        """The variant of a ``(bs, num_tokens)`` verify graph under capture: batch size 1, a tree of at
        most :data:`MAX_TAIL_WIDTH` tokens, 16-bit q and kv; else None (sglang's mask kernel)."""
        if bs != 1 or not 1 < num_tokens <= MAX_TAIL_WIDTH or not tail_mask.supported(q_dtype, kv_dtype):
            return None
        self.widths.add(num_tokens)
        logger.info("Tree verify at batch size 1, width %d: tree-tail mask attention.", num_tokens)
        return FlashInferVerifyVariant(
            jit_args=tail_mask.jit_args(q_dtype, head_dim), mask_buf=self.tail_bits.view(torch.uint8)
        )

    def write(self, mask: torch.Tensor, committed_len: torch.Tensor, width: int) -> None:
        """Write the words of one request's verify of ``width`` tokens from its flattened ``[width, kv_len]``
        mask, when its graph runs the variant (one launch; device inputs, so it can be graph-captured)."""
        if width in self.widths:
            tree_tail_words(mask, committed_len, self.tail_bits, width)
