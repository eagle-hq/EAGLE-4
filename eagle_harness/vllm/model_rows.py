# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Row forwards shared by native EAGLE3 and candidate-head drafts."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import torch
from vllm import ir
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.platforms import current_platform

from eagle_harness.vllm.draft_rows import stack_layer_input

Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class DraftRowsMixin:
    def _init_row_forward(self, max_rows: int) -> None:
        self.tp_size = get_tensor_model_parallel_world_size()
        self._max_rows = max_rows
        self._full_embed = None
        self._input_norm_weights = None
        self._norm_before_residual = bool(getattr(self.config, "norm_before_residual", False))

    # ---- the draft layer on rows with caller-supplied attention
    def forward_rows(
        self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor, attend: Attend
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``forward`` with ``attend(q, k, v)`` in place of the layer's attention.

        Returns ``(normed, hidden)``: the final-norm output (the candidate head's input) and the hidden
        state the next draft forward takes as input. The rows run outside torch.compile, where vLLM's
        custom ops would fall back to their unfused torch code: they use the CUDA kernels instead.
        """
        model = self.model
        layer = model.layers[0]
        attn, mlp = layer.self_attn, layer.mlp
        with _kernel_ops():
            x, residual = self._layer_input(input_ids, hidden_states)
            if getattr(layer, "eh_proj", None) is not None:
                residual = layer.eh_proj(x)
                x = layer.pre_attention_layernorm(residual)
            qkv, _ = attn.qkv_proj(x)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            q, k = _kernel(attn.rotary_emb)(positions, q, k)
            x, _ = attn.o_proj(attend(q, k, v))
            x, residual = layer.post_attention_layernorm(x, residual)
            if getattr(self, "latent_moe", False):
                x = self._moe_rows(mlp, x)
            else:
                gate_up, _ = mlp.gate_up_proj(x)
                x, _ = mlp.down_proj(_kernel(mlp.act_fn)(gate_up))
            normed, summed = model.norm(x, residual)
        return normed, normed if model.norm_output else summed

    def _layer_input(self, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cat([input_layernorm(embed(ids)), hidden_norm(hidden)], -1), residual)`` of the draft layer.

        On CUDA, with a vocab whose shards are plain slices, it is two launches instead of five: the token rows come
        from a full copy of the embedding table (no TP all-reduce of the shards' rows, which only adds zeros) and land
        beside the hidden rows (:func:`stack_layer_input`), then one RMSNorm call with a per-row weight normalises all
        ``2 * rows`` rows into the concatenated layout. The rows go through the same norm kernel as before."""
        layer = self.model.layers[0]
        if not self._fused_layer_input(layer, hidden_states):
            embeds = layer.input_layernorm(self.model.embed_input_ids(input_ids))
            x = layer.hidden_norm(hidden_states)
            residual = x if self._norm_before_residual else hidden_states
            return torch.cat([embeds, x], dim=-1), residual
        rows, hidden_size = hidden_states.shape
        stacked = stack_layer_input(self._full_embed, input_ids, hidden_states).view(2 * rows, hidden_size)
        normed = ir.ops.rms_norm(stacked, self._input_norm_weights[: 2 * rows], layer.input_layernorm.variance_epsilon)
        return normed.view(rows, 2 * hidden_size), hidden_states

    def _fused_layer_input(self, layer: Any, hidden_states: torch.Tensor) -> bool:
        """Whether :meth:`_layer_input` takes the two-launch path (builds its tables on the first eager call)."""
        if self._input_norm_weights is not None:
            return hidden_states.shape[0] <= self._max_rows
        if not current_platform.is_cuda() or torch.cuda.is_current_stream_capturing():
            return False
        norms = (layer.input_layernorm, layer.hidden_norm)
        if (
            self._norm_before_residual
            or not hidden_states.is_contiguous()
            or norms[0].variance_epsilon != norms[1].variance_epsilon
            or not all(norm.has_weight for norm in norms)
        ):
            return False
        embed = self.model.embed_tokens
        shard = embed.shard_indices
        per_rank = embed.weight.shape[0]
        if (
            shard.num_added_elements != 0
            or shard.num_org_vocab_padding != 0
            or shard.org_vocab_start_index != get_tensor_model_parallel_rank() * per_rank
        ):
            return False
        weight = embed.weight.detach()
        self._full_embed = tensor_model_parallel_all_gather(weight, dim=0) if self.tp_size > 1 else weight
        self._input_norm_weights = (
            torch.stack([norms[0].weight.detach(), norms[1].weight.detach()]).repeat(self._max_rows, 1).contiguous()
        )
        return hidden_states.shape[0] <= self._max_rows


def _kernel(op: Any) -> Callable[..., Any]:
    """The CUDA-kernel forward of a vLLM custom op where the platform has one, else the op's own dispatch."""
    return op.forward_cuda if current_platform.is_cuda_alike() else op


@contextlib.contextmanager
def _kernel_ops():
    """Dispatch vLLM's RMSNorm IR ops to their compiled kernels (an inductor-compiled model selects the native torch code,
    which is one launch per elementwise step outside a compiled graph)."""
    with contextlib.ExitStack() as stack:
        if current_platform.is_cuda_alike():
            for op in (ir.ops.rms_norm, ir.ops.fused_add_rms_norm):
                stack.enter_context(op.set_priority(["vllm_c", "native"]))
        yield
