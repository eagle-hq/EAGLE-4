# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-4 candidate LM head (low-rank scores, exact re-scoring of the top candidates).

Per TP vocab shard:

1. ``scores = (hidden @ down_proj.T) @ score_head_shard.T``   (rank-r estimate)
2. ``local = topk(scores, C, sorted=False)``                   (this shard's candidates)
3. ``logits = lm_head_shard[local] . hidden``                  (exact, fp32 accumulate)
4. all-gather the ``C`` logits and global ids of every rank, keep the top ``C``.

sglang's EAGLE workers sample and take the top-k over a full-vocab logits row,
so :class:`CandidateLogitsProcessor` returns ``[rows, vocab]`` fp32 logits that
are ``-inf`` outside the candidates; a softmax over that row equals the softmax
over the ``C`` candidates.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import torch

from eagle_harness.kernels.gather_dot import gather_dot
from eagle_harness.models.low_rank_head import low_rank_scores

AllGather = Callable[[torch.Tensor], torch.Tensor]  # [rows, n] -> [rows, tp * n], rank-major


class Candidates(NamedTuple):
    """Per-row candidate logits and their global vocab ids."""

    logits: torch.Tensor  # [rows, C] fp32
    token_ids: torch.Tensor  # [rows, C] int64, global vocab ids


def shard_candidates(
    hidden: torch.Tensor,
    down_proj: torch.Tensor,
    score_head: torch.Tensor,
    lm_head: torch.Tensor,
    *,
    num_candidates: int,
    vocab_start: int,
    vocab_rows: int,
) -> Candidates:
    """Steps 1-3 on one vocab shard. ``vocab_rows``: real (unpadded) rows of the shard."""
    scores = low_rank_scores(hidden, down_proj, score_head, vocab_rows)
    _, local_ids = torch.topk(scores, num_candidates, dim=-1, sorted=False)
    logits = gather_dot(lm_head, local_ids, hidden)
    return Candidates(logits=logits, token_ids=local_ids + vocab_start)


def merge_candidates(shard: Candidates, num_candidates: int, all_gather: Optional[AllGather]) -> Candidates:
    """Step 4: the top ``num_candidates`` over every rank's candidates (fp32 out)."""
    if all_gather is None:
        return Candidates(shard.logits.float(), shard.token_ids)
    logits = all_gather(shard.logits)
    token_ids = all_gather(shard.token_ids)
    logits, pick = torch.topk(logits, num_candidates, dim=-1, sorted=False)
    return Candidates(logits.float(), torch.gather(token_ids, -1, pick))


def scatter_to_vocab(candidates: Candidates, vocab_size: int) -> torch.Tensor:
    """``[rows, vocab_size]`` fp32 logits, ``-inf`` outside the candidates."""
    rows = candidates.logits.shape[0]
    full = candidates.logits.new_full((rows, vocab_size), float("-inf"))
    full.scatter_(1, candidates.token_ids, candidates.logits)
    return full


def build_candidate_logits_processor(config, candidate_fn: Callable[[torch.Tensor], Candidates]):
    """An sglang ``LogitsProcessor`` whose LM head is the candidate head.

    ``candidate_fn(hidden) -> Candidates`` runs steps 1-4 (it owns the weights).
    Built lazily so this module imports without sglang.
    """
    from sglang.srt.layers.logits_processor import LogitsProcessor

    class CandidateLogitsProcessor(LogitsProcessor):
        """``LogitsProcessor`` that scatters the candidate head's logits into a full-vocab row."""

        def __init__(self) -> None:
            super().__init__(config)
            if self.do_tensor_parallel_all_gather_dp_attn or self.use_attn_tp_group:
                raise NotImplementedError("The EAGLE-4 candidate head supports plain TP only (no DP attention).")
            if self.logit_scale is not None or self.final_logit_softcapping:
                raise NotImplementedError("The EAGLE-4 candidate head has no logit scale or softcapping.")
            self._candidate_fn = candidate_fn

        def _get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias=None, use_logits_buffer=True):
            if embedding_bias is not None:
                raise NotImplementedError("The EAGLE-4 candidate head takes no embedding bias.")
            logits = scatter_to_vocab(self._candidate_fn(hidden_states), self.vocab_size)
            return self._copy_logits_to_buffer(logits, logits_metadata, use_buffer=use_logits_buffer)

    return CandidateLogitsProcessor()
