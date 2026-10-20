# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Tree attention for the target's full-attention layers.

A tree verify forward holds, per request, ``W`` new tokens (the verify rows). Every row attends to the
request's committed context, as in a chain, and to its own ancestors among the new tokens only (the ancestor
bit masks of :class:`~eagle_harness.vllm.verify_layout.VerifyStage`). :func:`~eagle_harness.vllm.verify_attention.
verify_attention` computes both, split over the context so a small batch uses the whole GPU.

Everything but the scheduler's choice of batch shape is on the device, so the forward replays in CUDA
graphs. Batches the stage is not active for (prefill, or no speculator) run FlashAttention unchanged, and
so do the draft model's layers.

Select it with ``--attention-backend CUSTOM`` (the plugin binds that name to :class:`TreeFlashAttentionBackend`).
"""

from __future__ import annotations

import torch

from vllm.v1.attention.backends.fa_utils import is_quantized_kv_cache
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl, FlashAttentionMetadata

from eagle_harness.vllm.verify_attention import verify_attention
from eagle_harness.vllm.verify_layout import get_stage


class TreeFlashAttentionImpl(FlashAttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        stage = get_stage()
        # A query of one token per request has no tree; a CUDA graph captured for it keeps the plain path.
        if (
            stage is None
            or not stage.verifying
            or attn_metadata is None
            or attn_metadata.max_query_len <= 1
            or getattr(layer, "is_draft_layer", False)
        ):
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale
            )
        return self._forward_tree(query, key, value, kv_cache, attn_metadata, output, stage.masks, stage.context_lens)

    def _forward_tree(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        masks: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """``context_lens``: the committed tokens of each request (``seq_lens - query lens``), staged once per forward
        by :class:`~eagle_harness.vllm.verify_layout.VerifyStage` (computing it here cost two kernels per layer)."""
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("tree verify needs an unquantized KV cache")
        tokens = attn_metadata.num_actual_tokens
        num_reqs = attn_metadata.query_start_loc.shape[0] - 1
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        verify_attention(
            query[:tokens],
            key[:tokens],
            value[:tokens],
            key_cache,
            value_cache,
            attn_metadata.block_table,
            context_lens[:num_reqs],
            attn_metadata.query_start_loc,
            masks,
            self.scale,
            output,
            max_query_len=attn_metadata.max_query_len,
        )
        return output


class TreeFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[TreeFlashAttentionImpl]:
        return TreeFlashAttentionImpl
