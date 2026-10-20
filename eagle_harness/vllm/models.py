# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The EAGLE-4 draft model for vLLM (registered as ``LlamaForCausalLMEagle4``).

The single-layer EAGLE-4 draft is vLLM's EAGLE-3 Llama draft (``fc`` over three normed aux hidden
states, ``concat(input_layernorm(embed), hidden_norm(fc_out))`` into the qkv projection, ``norm_output``,
the sliding window of ``layer_types``) plus its own ``lm_head`` and a low-rank candidate head. Three structures of
the Nemotron drafts are built on top of it (:class:`_Eagle4DecoderLayer`): no rotary positions (``use_rope: false``),
``attn_norm`` (``eh_proj`` maps the concatenated input to the hidden size, that projection is the attention residual
and its RMSNorm feeds q/k/v) and the latent ReLU^2 MoE MLP (``moe_architecture: nemotron_latent_relu2``), which is
vLLM's own ``NemotronHMoE``, the target's block. The candidate head:

1. ``scores = (hidden @ down_proj.T) @ efficient_lm_head.T`` (rank-r estimate over this shard's vocab);
2. the top ``C`` scores of the shard are the candidates;
3. their logits are re-scored with the full ``lm_head`` rows (:func:`gather_dot`);
4. the candidates of every TP rank are merged.

:meth:`compute_logits` returns ``[rows, vocab]`` fp32 logits that are ``-inf`` outside the candidates, so
vLLM's stock draft sampling works on them; :meth:`candidate_topk` is the top-``k`` the draft tree reads.
:meth:`forward_rows` runs the layer on rows whose attention the caller supplies (the draft tree steps).
"""

from __future__ import annotations

import contextlib
import copy
import logging
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import is_forward_context_available, set_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.llama_eagle3 import LlamaDecoderLayer as Eagle3LlamaDecoderLayer
from vllm.model_executor.models.utils import AutoWeightsLoader

from eagle_harness.kernels.candidate_ops import candidate_tail, shard_topk
from eagle_harness.kernels.candidate_ops import candidate_topk as fused_candidate_topk
from eagle_harness.kernels.gather_dot import gather_dot, gather_dot_packed
from eagle_harness.models.eagle4_config import UNPORTED, check_ported
from eagle_harness.models.eagle4_structure import (
    LATENT_MOE_ARCHITECTURE as LATENT_MOE,
    base_layer_overrides,
    draft_moe_architecture as _moe_architecture,
    draft_uses_rope as uses_rope,
    latent_moe_config,
)
from eagle_harness.models.low_rank_head import low_rank_scores
from eagle_harness.vllm.candidate_gather import CandidateGather
from eagle_harness.vllm.model_rows import DraftRowsMixin, _kernel_ops

logger = logging.getLogger(__name__)

# Attributes vLLM adds to draft configs, which the shared validator does not recognize.
_VLLM_KEYS = ("eagle_aux_hidden_state_layer_ids", "target_layer_count")

# The structures of UNPORTED this model builds: NoPE, attn_norm and the latent MoE (checked below).
_BUILT = ("enable_moe", "attn_norm", "use_rope", "position_embedding_type", "partial_rotary_factor")
_VLLM_UNPORTED = {key: value for key, value in UNPORTED.items() if key not in _BUILT}


def check_supported(config: Any) -> None:
    """Validate the config and refuse draft structures this model does not build."""
    checked = copy.copy(config)
    for key in _VLLM_KEYS:
        checked.__dict__.pop(key, None)
    check_ported(checked, _VLLM_UNPORTED)
    if _moe_architecture(config) not in (None, LATENT_MOE):
        raise NotImplementedError(
            f"EAGLE-4 MoE drafts are ported for {LATENT_MOE} only, got {_moe_architecture(config)}."
        )


@contextlib.contextmanager
def _base_layer_config(config: Any):
    """Let vLLM's EAGLE-3 layer build on a config whose MLP and rotary parts this model then replaces: its dense MLP
    takes SiLU only (the latent MoE draft says relu2), and a NoPE draft needs no full-length rotary table (vLLM keeps
    every table in a process-wide cache). The config is vLLM's draft config object, restored on exit."""
    hidden_act = config.hidden_act
    max_position_embeddings = config.max_position_embeddings
    try:
        overrides = base_layer_overrides(config)
        config.hidden_act = overrides.get("hidden_act", hidden_act)
        config.max_position_embeddings = overrides.get("max_position_embeddings", max_position_embeddings)
        yield
    finally:
        config.hidden_act = hidden_act
        config.max_position_embeddings = max_position_embeddings


class _NoRotary(nn.Module):
    """Positions do not rotate q and k (``use_rope: false``); they still address the KV cache."""

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return query, key

    forward_cuda = forward


def _latent_moe(config: Any, vllm_config: VllmConfig, prefix: str) -> nn.Module:
    """The draft's latent ReLU^2 MoE, built as the target's block (vLLM's ``NemotronHMoE``): sigmoid router over
    bias-corrected scores, experts in the latent space between ``fc1_latent_proj`` / ``fc2_latent_proj``,
    ``routed_scaling_factor`` on the routed sum, plus the ReLU^2 shared expert; reduced once over TP."""
    from vllm.model_executor.models.nemotron_h import NemotronHMoE

    moe_config = latent_moe_config(config)
    return NemotronHMoE(moe_config, quant_config=None, parallel_config=vllm_config.parallel_config, prefix=prefix)


class _Eagle4DecoderLayer(Eagle3LlamaDecoderLayer):
    """vLLM's EAGLE-3 layer 0 with ``attn_norm``: ``residual = eh_proj(cat(embeds, hidden))`` and its RMSNorm feeds
    q/k/v (``eh_proj`` and ``pre_attention_layernorm`` are None without it). Built by
    :meth:`LlamaForCausalLMEagle4._build_structures`."""

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.input_layernorm(embeds)
        hidden_states, residual = self._residual_norm(hidden_states=hidden_states)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        if self.eh_proj is not None:
            residual = self.eh_proj(hidden_states)
            hidden_states = self.pre_attention_layernorm(residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class LlamaForCausalLMEagle4(DraftRowsMixin, Eagle3LlamaForCausalLM):
    # The draft trains its own lm_head; the embedding is the target's (vLLM shares it).
    has_own_lm_head = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        draft_config = vllm_config.speculative_config.draft_model_config.hf_config
        check_supported(draft_config)
        with _base_layer_config(draft_config):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        self._build_structures(vllm_config)
        head = self.config.efficient_lm_head_config
        self.num_candidates = int(head["num_candidates"])
        rank, hidden, vocab = int(head["rank"]), self.config.hidden_size, self.config.vocab_size
        self.efficient_lm_head = ParallelLMHead(vocab, rank, prefix=f"{prefix}.efficient_lm_head")
        self.efficient_lm_head_down_proj = nn.Linear(hidden, rank, bias=False)
        self.tp_size = get_tensor_model_parallel_world_size()
        # A draft row's candidates: up to 8 rows (node_top_k) per request.
        max_rows = 8 * vllm_config.scheduler_config.max_num_seqs
        self.candidate_gather = CandidateGather(get_tp_group(), max_rows) if self.tp_size > 1 else None
        self._init_row_forward(max_rows)

    def _build_structures(self, vllm_config: VllmConfig) -> None:
        """Turn vLLM's EAGLE-3 layer 0 into the draft's: NoPE, attn_norm and the latent MoE where the config says so."""
        config, layer = self.config, self.model.layers[0]
        layer.__class__ = _Eagle4DecoderLayer
        # Instance attributes, not class defaults: a class attribute would shadow the submodules nn.Module registers.
        layer.eh_proj = layer.pre_attention_layernorm = None
        layer_prefix = layer.self_attn.attn.layer_name.removesuffix(".self_attn.attn")
        hidden = config.hidden_size
        if not uses_rope(config):
            layer.self_attn.rotary_emb = _NoRotary()
        if getattr(config, "attn_norm", False):
            attn = layer.self_attn
            attn.qkv_proj = QKVParallelLinear(
                hidden,
                attn.head_dim,
                attn.total_num_heads,
                attn.total_num_kv_heads,
                bias=False,
                quant_config=None,
                prefix=f"{layer_prefix}.self_attn.qkv_proj",
            )
            layer.eh_proj = ReplicatedLinear(
                2 * hidden, hidden, bias=False, quant_config=None, prefix=f"{layer_prefix}.eh_proj", return_bias=False
            )
            layer.pre_attention_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.latent_moe = _moe_architecture(config) == LATENT_MOE
        if self.latent_moe:
            layer.mlp = _latent_moe(config, vllm_config, f"{layer_prefix}.mlp")
            # Auxiliary-stream event joins cannot cross conditional graph captures.
            shared = layer.mlp.experts.shared_experts
            if shared is not None:
                disable = getattr(shared, "disable_aux_stream", None)
                if disable is None:
                    raise NotImplementedError("Nemotron drafts require vLLM SharedExperts.disable_aux_stream")
                disable()

    def load_weights(self, weights: Any) -> Any:
        """vLLM's EAGLE-3 loading, except the latent MoE's tensors (expert-per-index ``up_proj`` / ``down_proj``, the
        router, the latent projections, the shared expert): they go to the MoE block's own loaders, as the target's
        do, because the EAGLE-3 mapping would stack every ``up_proj`` into a ``gate_up_proj``."""
        if not self.latent_moe:
            return super().load_weights(weights)
        moe: list[tuple[str, torch.Tensor]] = []

        def dense(items: Any) -> Any:
            for name, tensor in items:
                head, sep, rest = name.partition(".mlp.")
                if sep and head.removeprefix("model.").removeprefix("midlayer").removeprefix("layers.0") == "":
                    moe.append((rest, tensor))
                else:
                    yield name, tensor

        loaded = super().load_weights(dense(weights))
        AutoWeightsLoader(self.model.layers[0].mlp).load_weights(moe)
        if not moe:
            raise ValueError("the latent MoE draft checkpoint has no layer-0 .mlp. tensors")
        return loaded

    # ---- candidate head
    def _scores(self, hidden: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Steps 1-2: ``(low-rank scores [rows, shard vocab], first vocab id of the shard)``."""
        shard = self.lm_head.shard_indices
        scores = low_rank_scores(
            hidden,
            self.efficient_lm_head_down_proj.weight,
            self.efficient_lm_head.weight,
            shard.org_vocab_end_index - shard.org_vocab_start_index,
        )
        return scores, shard.org_vocab_start_index

    def candidate_topk(self, hidden: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-``k`` next tokens of each row: ``([rows, k]`` fp32 probabilities, ``[rows, k]`` ids).

        The softmax runs over the ``C`` candidates only, which equals the softmax over the full-vocab
        row of :meth:`compute_logits`.
        """
        scores, vocab_start = self._scores(hidden)
        return fused_candidate_topk(
            scores,
            self.lm_head.weight,
            hidden,
            num_candidates=self.num_candidates,
            vocab_start=vocab_start,
            top_k=k,
            all_gather=self._gather_candidates if self.candidate_gather is not None else None,
        )

    def _gather_candidates(self, packed: torch.Tensor) -> torch.Tensor:
        """``[rows, n]`` -> ``[tp, rows, n]``: a view of the dim-0 gather."""
        return self.candidate_gather(packed).view(self.tp_size, *packed.shape)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores, vocab_start = self._scores(hidden_states)
        local_ids, global_ids = shard_topk(scores, self.num_candidates, vocab_start)
        if self.tp_size == 1:
            logits, ids = gather_dot(self.lm_head.weight, local_ids, hidden_states), global_ids
        else:
            packed = gather_dot_packed(self.lm_head.weight, local_ids, hidden_states, vocab_start)
            packed = tensor_model_parallel_all_gather(packed, dim=0)
            logits, ids = packed[:, : self.num_candidates].view(torch.float32), packed[:, self.num_candidates :]
        logits, ids, _ = candidate_tail(logits, ids, hidden_states.shape[0], 0)
        full = logits.new_full((logits.shape[0], self.config.vocab_size), float("-inf"))
        return full.scatter_(1, ids, logits)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """vLLM's per-chunk RMSNorms and ``fc`` over the aux hidden states, with the norms on vLLM's CUDA kernel: the draft
        prefill calls this eagerly, where the native norm is ~7 launches per chunk (~45 us each at TP4)."""
        with _kernel_ops():
            return super().combine_hidden_states(hidden_states)

    def _moe_rows(self, mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """The MoE block on draft rows. vLLM's fused-MoE op finds its layer through the forward context, which the
        draft tree's steps (and their CUDA graph captures) may run without: give them the engine's."""
        if is_forward_context_available():
            return mlp(x)
        with set_forward_context(None, self.vllm_config):
            return mlp(x)
