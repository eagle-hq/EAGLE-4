# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The EAGLE-4 draft model for TokenSpeed (registered as ``LlamaForCausalLMEagle4``).

TokenSpeed owns the single-layer draft, including dense or latent-MoE blocks and normalized or concatenated attention
inputs. This adapter adds its own ``lm_head`` and a low-rank candidate head in place of full-vocabulary logits:

1. ``scores = (hidden @ down_proj.T) @ efficient_lm_head.T`` (rank-r estimate over this rank's vocab shard);
2. the top ``C`` scores of the shard are the candidates;
3. their logits are re-scored with the full ``lm_head`` rows (:func:`~eagle_harness.kernels.gather_dot.gather_dot`);
4. the candidates of every TP rank are merged.

The model's exit (:meth:`exit_logits`) selects the forward's sampled rows, as TokenSpeed's logits processor does, and
returns them without any logits (:class:`CandidateRows`): the drafter asks :meth:`candidate_topk` for the top-``k``
it needs. The whole-vocab LM head GEMM of every draft step is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed.runtime.configs.model_config import configure_mha_attention
from tokenspeed.runtime.configs.model_profile import ModelProfile
from tokenspeed.runtime.distributed.comm_ops import all_gather
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.models.llama_eagle3 import Eagle3DecoderLayer, LlamaForCausalLMEagle3
from torch import nn

from eagle_harness.kernels.candidate_ops import candidate_head as fused_candidate_head
from eagle_harness.models.eagle4_config import check_ported
from eagle_harness.models.low_rank_head import low_rank_scores


@dataclass
class CandidateRows(LogitsProcessorOutput):
    """A draft forward's exit: the sampled rows, no logits.

    ``normed``: the final-norm output of each sampled row, the candidate head's input; ``hidden_states``: what the next
    draft forward takes as input (``norm_output`` drafts: the same rows); ``next_token_logits`` is empty (``[rows, 0]``).
    """

    normed: torch.Tensor | None = None


class LlamaForCausalLMEagle4(LlamaForCausalLMEagle3):
    def __init__(self, config: Any, mapping: Any, quant_config: Any = None, prefix: str = ""):
        # The engine owns dense and latent-MoE draft layers, including attn_norm and NoPE.
        check_ported(config, unported={"enable_post_mlp_norm": True})
        extended_draft = (
            getattr(config, "enable_moe", False)
            or getattr(config, "attn_norm", False)
            or not getattr(config, "use_rope", True)
        )
        if extended_draft and not hasattr(Eagle3DecoderLayer, "prepare_attention_input"):
            raise RuntimeError("This draft requires TokenSpeed's latent-MoE/attn_norm/NoPE model support.")
        if (getattr(config, "draft_vocab_size", None) or config.vocab_size) != config.vocab_size:
            raise NotImplementedError("EAGLE-4 drafts with a reduced vocab (d2t) are not ported.")
        super().__init__(config, mapping, quant_config, prefix)
        head = config.efficient_lm_head_config
        self.num_candidates = int(head["num_candidates"])
        rank = int(head["rank"])
        lm_head = mapping.lm_head
        self.efficient_lm_head = ParallelLMHead(
            config.vocab_size,
            rank,
            tp_rank=lm_head.tp_rank,
            tp_size=lm_head.tp_size,
            tp_group=lm_head.tp_group,
            prefix=f"{prefix}.efficient_lm_head" if prefix else "efficient_lm_head",
        )
        self.efficient_lm_head_down_proj = nn.Linear(config.hidden_size, rank, bias=False)
        self.tp_size = lm_head.tp_size
        self.tp_group = lm_head.tp_group
        self.cuda_candidate_kernels = True  # HarnessConfig.draft_model.cuda_candidate_kernels, set by the drafter

    @classmethod
    def model_profile(cls, hf_config: Any) -> ModelProfile:
        """The draft's cache facts: one ordinary multi-head attention layer, as TokenSpeed's EAGLE-3 draft."""
        del hf_config
        return ModelProfile(
            configure_attention=configure_mha_attention,
            cache_family="mha",
            linear_attention=None,
            default_attention_backend=None,
            default_prefix_granularity=None,
            request_token_history=False,
            is_multimodal=False,
            tokenizer_kwargs={},
            attention_instances_per_layer=1,
            numerics_envelopes=frozenset({"auto"}),
        )

    # ---- the model exit: sampled rows, no logits
    def exit_logits(self, input_ids: torch.Tensor, hidden_states: torch.Tensor, aux_hidden_states: Any, ctx: Any):
        """The forward's sampled rows (``ctx.gather_ids``) as :class:`CandidateRows`; no LM head."""
        if ctx.input_logprob_rows is not None or ctx.query_shard is not None:
            raise NotImplementedError("the EAGLE-4 draft scores no prompt logprobs and runs no query shards")
        (aux,) = aux_hidden_states
        gather = ctx.gather_ids
        # Rows already one per request (the first draft step narrows its attention to the live rows) stay as they are.
        if gather is not None and not (ctx.logits_rows_selected or gather.shape[0] == hidden_states.shape[0]):
            hidden_states, aux = hidden_states[gather], aux[gather]
        empty = hidden_states.new_empty(hidden_states.shape[0], 0)
        return CandidateRows(next_token_logits=empty, hidden_states=aux, normed=hidden_states)

    # ---- candidate head
    def candidate_topk(self, normed: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-``k`` next tokens of each row: ``([rows, k]`` fp32 probabilities, ``[rows, k]`` int64 ids).

        The softmax runs over the ``C`` candidates of every rank together, which equals the softmax over the full
        vocabulary with everything outside the candidates at ``-inf``.
        """
        return self.candidate_head(normed, k)[2]

    def candidate_head(self, normed: torch.Tensor, k: int):
        """Candidate logits, target ids and top-k for recorded-q proposals."""
        shard = self.lm_head.shard_indices
        scores = low_rank_scores(
            normed,
            self.efficient_lm_head_down_proj.weight,
            self.efficient_lm_head.weight,
            shard.org_vocab_end_index - shard.org_vocab_start_index,
        )
        return fused_candidate_head(
            scores,
            self.lm_head.weight,
            normed,
            num_candidates=self.num_candidates,
            vocab_start=shard.org_vocab_start_index,
            top_k=k,
            all_gather=self._gather_candidates if self.tp_size > 1 else None,
            cuda_kernels=self.cuda_candidate_kernels,
        )

    def _gather_candidates(self, packed: torch.Tensor) -> torch.Tensor:
        """``[rows, n]`` int32 -> ``[tp, rows, n]`` in rank order.

        Gathered along the rows as 16-bit words: that is the shape the engine serves with its barrier-free Lamport
        exchange on the all-reduce workspace (as its own EAGLE-4 head does), so the draft's round graphs hold no NCCL
        kernel and stay device-launchable.
        """
        rows, n = packed.shape
        words = all_gather(packed.view(torch.int16), self.tp_group, dim=-1)
        return words.view(rows, self.tp_size, 2 * n).transpose(0, 1).view(torch.int32)
