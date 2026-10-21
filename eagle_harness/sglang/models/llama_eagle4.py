# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright 2025 SafeAI Lab (SAIL)
# Derived from sglang (Apache-2.0) python/sglang/srt/models/llama_eagle3.py, itself adapted from SafeAILab EAGLE
# (Apache-2.0) eagle/model/cnets.py. Modified by Eagle AI Corp: the EAGLE-4 draft (fc over three aux hidden states,
# sliding window, low-rank candidate head). Notices and license texts: THIRD_PARTY_NOTICES.md and LICENSES/.
"""EAGLE-4 draft model on sglang layers (``LlamaForCausalLMEagle4``).

Registered with sglang's ``ModelRegistry`` by the plugin entry point. Covers the
dense single-layer EAGLE-4 drafts (the README Qwen3.8-27B draft):

- fc input: three target aux hidden states, each RMS-normed (``fc_norm``), then
  one row-parallel ``fc`` (3H -> H);
- layer 0: ``concat(input_layernorm(embed), hidden_norm(fc_out)) -> qkv`` (2H in),
  or with ``attn_norm`` through ``eh_proj`` (2H -> H, the attention residual) and
  ``pre_attention_layernorm`` first; Llama attention with RoPE or none (NoPE) and an
  optional sliding window; a Llama MLP or the latent ReLU^2 MoE of a
  ``nemotron_latent_relu2`` draft (sglang's ``NemotronHMoE``, the Nemotron-H block);
- output: ``norm(residual + mlp)`` feeds the LM head and, with ``norm_output``,
  the next draft step (else the un-normed sum does);
- LM head: the low-rank candidate head (:mod:`eagle_harness.sglang.candidate_logits`).

SwiGLU-MoE and post-MLP-norm drafts are refused with a clear error rather than
served partly.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sglang.kernels.ops.embeddings.vocab_parallel_embedding import vocab_parallel_embedding
from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import MultimemAllGatherer, recommended_max_tokens
from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

from eagle_harness.models.eagle4_config import check_ported, resolve_draft_sliding_window
from eagle_harness.models.eagle4_structure import (
    LATENT_MOE_ARCHITECTURE,
    base_layer_overrides,
    draft_moe_architecture,
    draft_uses_rope,
    latent_moe_config,
)
from eagle_harness.models.low_rank_head import low_rank_scores
from eagle_harness.models.low_rank_head import low_rank_factors
from eagle_harness.kernels.candidate_ops import candidate_head
from eagle_harness.sglang.algorithm import harness_config
from eagle_harness.sglang.candidate_logits import (
    Candidates,
    build_candidate_logits_processor,
    merge_candidates,
    shard_candidates,
)

logger = logging.getLogger(__name__)

NUM_AUX_HIDDEN_STATES = 3


def replicate_embedding(
    shard: torch.Tensor, vocab_size: int, all_gather: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """The full ``[vocab_size, H]`` embedding table from this rank's contiguous vocab shard.

    ``all_gather`` stacks every rank's shard along dim 0 in rank order; rows past ``vocab_size`` are padding.
    A draft step then looks its tokens up locally instead of all-reducing the shards' partial lookups.
    """
    return all_gather(shard)[:vocab_size].contiguous()


# sglang's EAGLE draft worker always shares the target embedding.
_UNSUPPORTED = {"enable_post_mlp_norm": True, "share_draft_embedding_from_target": False}


def check_supported(config) -> None:
    """Validate the config and refuse draft structures this adapter does not build."""
    check_ported(config, _UNSUPPORTED)
    moe = draft_moe_architecture(config)
    if moe not in (None, LATENT_MOE_ARCHITECTURE):
        raise NotImplementedError(f"EAGLE-4 drafts with the {moe!r} MoE are not ported to eagle-harness on sglang.")


def served_sliding_window(config) -> Optional[int]:
    """The draft's attention window as served, or None."""
    return resolve_draft_sliding_window(config)


class _NoRotary(nn.Module):
    """RoPE of a draft trained without positions (NoPE): q and k pass unchanged."""

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, *args, **kwargs):
        return query, key


def _base_layer_config(config):
    """Config of the Llama base layer, whose SiLU MLP and RoPE table a draft may replace right after.

    The base refuses any activation but SiLU, and a NoPE draft's full-length RoPE table would stay alive
    in ``get_rope``'s process-wide cache.
    """
    base = copy.copy(config)
    vars(base).update(base_layer_overrides(config))
    return base


def build_latent_moe(config, prefix: str) -> nn.Module:
    """The latent ReLU^2 MoE of a ``nemotron_latent_relu2`` draft: sglang's ``NemotronHMoE``, the target's block.

    Sigmoid router over bias-corrected scores, routed experts in the latent space between ``fc1_latent_proj``
    and ``fc2_latent_proj``, ``routed_scaling_factor`` on the routed sum, plus a ReLU^2 shared expert. Experts and
    the shared expert split over TP and the block returns this rank's partial sum, reduced by the layer.
    """
    parallel = get_parallel()
    if parallel.moe_ep_size != 1 or parallel.moe_tp_size != parallel.tp_size or not get_moe_a2a_backend().is_none():
        raise ValueError(
            "The EAGLE-4 latent MoE draft keeps every expert on each rank and splits it over TP: serve it with "
            f"moe EP 1 (got {parallel.moe_ep_size}), moe TP = TP (got {parallel.moe_tp_size} vs "
            f"{parallel.tp_size}) and --speculative-moe-a2a-backend none (got {get_moe_a2a_backend()})."
        )
    # Imported here: sglang's model registry turns an import error into a warning and would drop the draft.
    from sglang.srt.models.nemotron_h import NemotronHMoE

    return NemotronHMoE(latent_moe_config(config), layer_idx=0, quant_config=None, prefix=prefix)


class Eagle4DecoderLayer(LlamaDecoderLayer):
    """Llama layer whose qkv reads ``concat(normed embed, normed hidden)`` (through ``eh_proj`` with ``attn_norm``)."""

    def __init__(self, config, quant_config: Optional[QuantizationConfig], prefix: str, window: Optional[int]):
        super().__init__(
            _base_layer_config(config), 0, quant_config=quant_config, prefix=prefix, stage_boundaries=False
        )
        attn = self.self_attn
        self.attn_norm = bool(getattr(config, "attn_norm", False))
        if self.attn_norm:
            # MTP-style input fusion: eh_proj maps the concat back to H; that projection is the attention residual
            # and its norm feeds q/k/v. Unquantized, like fc.
            self.eh_proj = ReplicatedLinear(
                2 * config.hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=None,
                prefix=add_prefix("eh_proj", prefix),
            )
            self.pre_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if not draft_uses_rope(config):
            attn.rotary_emb = _NoRotary()
        self.reduce_mlp = False
        if draft_moe_architecture(config) == LATENT_MOE_ARCHITECTURE:
            if quant_config is not None:
                raise ValueError(
                    "The EAGLE-4 latent MoE draft is trained in BF16: serve it with "
                    "--speculative-draft-model-quantization unquant."
                )
            self.mlp = build_latent_moe(config, add_prefix("mlp", prefix))
            self.reduce_mlp = get_parallel().tp_size > 1
        self.self_attn.qkv_proj = QKVParallelLinear(
            config.hidden_size if self.attn_norm else 2 * config.hidden_size,
            attn.head_dim,
            attn.total_num_heads,
            attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("self_attn.qkv_proj", prefix),
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if window is not None:
            # As trained: W keys ending at the query (FlashInfer window_left = W - 1).
            attn.attn.sliding_window_size = window - 1

    def forward(
        self, positions: torch.Tensor, embeds: torch.Tensor, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The layer: norm the embeds and the hidden separately, attend over their concat, then the MLP."""
        residual = hidden_states
        x = torch.cat([self.input_layernorm(embeds), self.hidden_norm(hidden_states)], dim=-1)
        if self.attn_norm:
            residual, _ = self.eh_proj(x)
            x = self.pre_attention_layernorm(residual)
        x = self.self_attn(positions=positions, hidden_states=x, forward_batch=forward_batch)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        if self.reduce_mlp:
            x = tensor_model_parallel_all_reduce(x)
        return x, residual


class Eagle4Model(nn.Module):
    """EAGLE-4 draft body: fc over three aux hidden states, one layer, final norm."""

    def __init__(self, config, quant_config: Optional[QuantizationConfig], prefix: str, window: Optional[int]):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.norm_output = bool(getattr(config, "norm_output", False))
        target_hidden = getattr(config, "target_hidden_size", config.hidden_size)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=add_prefix("embed_tokens", prefix)
        )
        self.embed_full: Optional[torch.Tensor] = None  # the replicated table (LlamaForCausalLMEagle4.set_embed)
        self.triton_embed = harness_config().draft_model.embed_triton  # CUDA lookups on sglang's Triton embedding
        self.fc_norm = (
            nn.ModuleList(RMSNorm(target_hidden, eps=config.rms_norm_eps) for _ in range(NUM_AUX_HIDDEN_STATES))
            if getattr(config, "fc_norm", False) or getattr(config, "enable_fc_norm", False)
            else None
        )
        self.fc = RowParallelLinear(
            target_hidden * NUM_AUX_HIDDEN_STATES,
            config.hidden_size,
            bias=False,
            quant_config=None,
            input_is_parallel=False,
            prefix=add_prefix("fc", prefix),
        )
        with layer_stack():
            self.layers = nn.ModuleList(
                [Eagle4DecoderLayer(config, quant_config, add_prefix("layers.0", prefix), window)]
            )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The rows' token embeddings. From a table of the whole vocab on this rank (the replicated table, or
        the only shard at TP 1) a CUDA lookup is sglang's Triton embedding kernel (``draft_model.embed_triton``),
        a plain row copy that is faster than ``F.embedding``'s index_select for a draft step's few rows."""
        table = self.embed_full
        if table is None and self.embed_tokens.tp_size == 1:
            table = self.embed_tokens.weight
        if table is None:
            return self.embed_tokens(input_ids)
        if not (self.triton_embed and input_ids.is_cuda):
            return F.embedding(input_ids, table)
        rows = table.shape[0]
        return vocab_parallel_embedding(
            input_ids.contiguous(),
            table,
            org_vocab_start_index=0,
            org_vocab_end_index=rows,
            num_org_vocab_padding=0,
            added_vocab_start_index=rows,  # no added vocab
            added_vocab_end_index=rows,
        )

    def _project_target_features(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.fc_norm is not None:
            chunks = hidden.chunk(NUM_AUX_HIDDEN_STATES, dim=-1)
            hidden = torch.cat([norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)], dim=-1)
        out, _ = self.fc(hidden)
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """Embed ``input_ids``, run the layer on the draft's hidden states; returns the normed output and a list."""
        embeds = input_embeds if input_embeds is not None else self._embed(input_ids)
        hidden = forward_batch.spec_info.hidden_states
        if hidden.shape[-1] != self.hidden_size:
            hidden = self._project_target_features(hidden)
        if hidden.shape[0] == 0:
            return hidden, [hidden]
        mlp_out, residual = self.layers[0](positions, embeds, hidden, forward_batch)
        summed = residual + mlp_out
        normed = self.norm(summed)
        return normed, [normed if self.norm_output else summed]


class LlamaForCausalLMEagle4(LlamaForCausalLM):
    """EAGLE-4 draft (checkpoint class ``LlamaForCausalLMEagle4``) with the candidate LM head."""

    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None, prefix: str = "") -> None:
        nn.Module.__init__(self)
        check_supported(config)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.window = served_sliding_window(config)
        self.model = Eagle4Model(config, quant_config, add_prefix("model", prefix), self.window)

        head = config.efficient_lm_head_config
        self.num_candidates = int(head["num_candidates"])
        self.low_rank_type = head["type"]
        self.smooth_factor = head.get("smooth_factor", 10)
        rank = int(head["rank"])
        self.load_lm_head_from_target = False
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=add_prefix("lm_head", prefix))
        self.efficient_lm_head = ParallelLMHead(config.vocab_size, rank, prefix=add_prefix("efficient_lm_head", prefix))
        self.efficient_lm_head_down_proj = nn.Linear(config.hidden_size, rank, bias=False)
        self.logits_processor = build_candidate_logits_processor(config, self._candidates)
        self.capture_aux_hidden_states = True
        self.hot_token_id = None
        self._tp_size = get_parallel().tp_size
        self._candidate_gatherer: Optional[MultimemAllGatherer] = None
        self._cuda_candidate_kernels = harness_config().draft_model.cuda_candidate_kernels
        # A measured top-k crossover for this exact GPU/toolchain/shape (tools/calibrate_candidate_topk.py), on
        # every vendor; without one the dispatcher's built-in limit stands.
        import triton

        from eagle_harness.core.candidate_calibration import load_candidate_crossover

        self._candidate_crossover = load_candidate_crossover(
            harness_config().data_path,
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "dtype": str(self.lm_head.weight.dtype),
                "vocab_size": self.lm_head.weight.shape[0],
                "num_candidates": self.num_candidates,
            },
        )
        logger.info(
            "eagle-harness: candidate top-k crossover %s",
            "built-in" if self._candidate_crossover is None else f"measured ({self._candidate_crossover} scores)",
        )

    def set_embed(self, embed: torch.Tensor) -> None:
        """Share the target's embedding shard, and with ``draft_model.embed_full`` replicate the table on every rank."""
        super().set_embed(embed)
        if self._tp_size > 1 and harness_config().draft_model.embed_full:
            self.model.embed_full = replicate_embedding(
                embed.data, self.config.vocab_size, lambda shard: tensor_model_parallel_all_gather(shard, dim=0)
            )

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        """Share only the target's embedding: the draft keeps its own LM head (``load_lm_head_from_target`` is
        False), which sglang's EAGLE draft setup would otherwise replace with the target's."""
        self.set_embed(embed)

    # ---- candidate head
    def _candidates(self, hidden: torch.Tensor) -> Candidates:
        shard = self.lm_head.shard_indices
        local = shard_candidates(
            hidden,
            self.efficient_lm_head_down_proj.weight,
            self.efficient_lm_head.weight,
            self.lm_head.weight,
            num_candidates=self.num_candidates,
            vocab_start=shard.org_vocab_start_index,
            vocab_rows=shard.org_vocab_end_index - shard.org_vocab_start_index,
        )
        gather = tensor_model_parallel_all_gather if self._tp_size > 1 else None
        return merge_candidates(local, self.num_candidates, gather)

    def proposal_candidates(self, hidden: torch.Tensor) -> Candidates:
        """The complete sparse distribution the sampled draft draws from (``[rows, C]`` fp32 logits, int64 ids); on
        CUDA the greedy path's candidate pipeline without its top-k, so a sampled step costs the greedy step's
        collectives."""
        if not hidden.is_cuda:
            return self._candidates(hidden)
        logits, tokens, _ = self._candidate_head(hidden, 0)
        return Candidates(logits, tokens)

    def candidate_topk(self, hidden: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-``k`` next tokens of each row: ``([rows, k]`` fp32 probabilities, ``[rows, k]`` ids).

        The softmax runs over the ``C`` candidates only, which equals the
        full-vocab softmax of :class:`CandidateLogitsProcessor`'s rows.
        """
        if hidden.is_cuda:
            return self._candidate_head(hidden, k)[2]
        candidates = self._candidates(hidden)
        probs, pick = torch.topk(torch.softmax(candidates.logits, dim=-1), k, dim=-1)
        return probs, torch.gather(candidates.token_ids, 1, pick)

    def _candidate_head(self, hidden: torch.Tensor, k: int):
        """``candidate_ops.candidate_head`` on this head's shard (bit-identical to torch): logits, ids, top-``k``."""
        shard = self.lm_head.shard_indices
        scores = low_rank_scores(
            hidden,
            self.efficient_lm_head_down_proj.weight,
            self.efficient_lm_head.weight,
            shard.org_vocab_end_index - shard.org_vocab_start_index,
        )
        return candidate_head(
            scores,
            self.lm_head.weight,
            hidden,
            num_candidates=self.num_candidates,
            vocab_start=shard.org_vocab_start_index,
            top_k=k,
            all_gather=self._gather_packed_candidates if self._tp_size > 1 else None,
            cuda_kernels=self._cuda_candidate_kernels,
            max_triton_elements=self._candidate_crossover,
        )

    def _gather_packed_candidates(self, packed: torch.Tensor) -> torch.Tensor:
        """``[rows, n]`` int32 -> ``[tp, rows, n]`` (a view of the last-dim gather), on sglang's multimem all-gather
        (bf16 bit copies; NCCL along the last dim where it does not apply)."""
        if self._candidate_gatherer is None:
            # Without the entry barrier: consecutive candidate gathers are separated by this draft's own all-reduces
            # (every draft forward reduces its o_proj and down_proj before the next candidate_topk), so no rank can
            # write into another's symmetric buffer before that rank has consumed the previous gather. The same
            # argument sglang's LogitsProcessor makes for its logits gatherer.
            self._candidate_gatherer = MultimemAllGatherer(
                max_tokens=recommended_max_tokens(include_prefill=False, floor=128), skip_entry_sync=True
            )
        gathered = self._candidate_gatherer(packed.view(torch.bfloat16)).view(torch.int32)
        return gathered.view(packed.shape[0], self._tp_size, packed.shape[1]).transpose(0, 1)

    def get_hot_token_id(self) -> None:
        """No hot-token vocab subset."""
        return None

    def get_attention_sliding_window_size(self) -> Optional[int]:
        """The draft's sliding window as sglang counts it (one less than the config's), or None."""
        return None if self.window is None else self.window - 1

    # ---- weights
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load the checkpoint, fusing q/k/v and gate/up and reading the low-rank head factors."""
        params = dict(self.named_parameters())
        stacked = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        loaded = set()
        latent_moe = draft_moe_architecture(self.config) == LATENT_MOE_ARCHITECTURE
        expert_tensors = set()
        for name, weight in weights:
            name = _checkpoint_name(name)
            if latent_moe and ".mlp." in name:
                loaded.add(self._load_latent_moe_weight(params, name, weight))
                if ".mlp.experts." in name:
                    expert_tensors.add(name)
                continue
            for param_suffix, ckpt_suffix, shard_id in stacked:
                if ckpt_suffix in name:
                    target = name.replace(ckpt_suffix, param_suffix)
                    param = params[target]
                    param.weight_loader(param, weight, shard_id)
                    loaded.add(target)
                    break
            else:
                if name not in params:
                    raise KeyError(f"EAGLE-4 checkpoint tensor {name!r} has no parameter in the model.")
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(name)
        low_rank = {"efficient_lm_head.weight", "efficient_lm_head_down_proj.weight"}
        missing = set(params) - loaded - low_rank - {"model.embed_tokens.weight"}
        if latent_moe:
            expected_experts = {
                f"model.layers.0.mlp.experts.{expert}.{proj}_proj.weight"
                for expert in range(int(self.config.n_routed_experts))
                for proj in ("up", "down")
            }
            missing.update(expected_experts - expert_tensors)
        if missing:
            raise RuntimeError(f"EAGLE-4 checkpoint is missing {sorted(missing)}.")
        if not low_rank <= loaded:
            self._low_rank_head_from_svd()

    @staticmethod
    def _load_latent_moe_weight(params: dict, name: str, weight: torch.Tensor) -> str:
        """Load one latent-MoE tensor; returns its parameter name. The checkpoint keeps one ``up_proj`` /
        ``down_proj`` per routed expert, which go into the ungated experts' ``w13`` (``w1`` alone) and ``w2``."""
        expert = re.fullmatch(r"(.*\.mlp\.experts)\.(\d+)\.(up_proj|down_proj)\.weight", name)
        if expert is None:
            if name not in params:
                raise KeyError(f"EAGLE-4 checkpoint tensor {name!r} has no parameter in the model.")
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
            return name
        prefix, expert_id, proj = expert.groups()
        shard_id, param_name = ("w1", f"{prefix}.w13_weight") if proj == "up_proj" else ("w2", f"{prefix}.w2_weight")
        param = params[param_name]
        param.weight_loader(param, weight, param_name, shard_id=shard_id, expert_id=int(expert_id))
        return param_name

    def _low_rank_head_from_svd(self) -> None:
        """Checkpoints without the low-rank factors: SVD of this rank's LM-head shard."""
        frequency = None
        if self.low_rank_type == "weighted_low_rank":
            from eagle_harness.models.low_rank_head import load_token_frequency

            frequency = load_token_frequency(self.config._name_or_path, self.config.vocab_size)
        shard = self.lm_head.shard_indices
        factors = low_rank_factors(
            self.lm_head.weight,
            self.efficient_lm_head_down_proj.out_features,
            token_frequency=frequency,
            smooth_factor=self.smooth_factor,
            shard=(shard.org_vocab_start_index, shard.org_vocab_end_index),
        )
        with torch.no_grad():
            self.efficient_lm_head_down_proj.weight.copy_(factors.down_proj)
            self.efficient_lm_head.weight.copy_(factors.scores)
        logger.info("EAGLE-4 low-rank head from SVD: energy retention %.4f", factors.energy_retention)


_TOP_LEVEL = ("lm_head.", "efficient_lm_head.", "efficient_lm_head_down_proj.")


def _checkpoint_name(name: str) -> str:
    """Checkpoint tensor name -> parameter name (``layers.0.*`` -> ``model.layers.0.*``)."""
    name = name.removeprefix("model.")
    if name.startswith("midlayer."):
        name = "layers.0." + name[len("midlayer.") :]
    if name == "fc.linear.weight":
        name = "fc.weight"
    return name if name.startswith(_TOP_LEVEL) else "model." + name


EntryClass = [LlamaForCausalLMEagle4]
