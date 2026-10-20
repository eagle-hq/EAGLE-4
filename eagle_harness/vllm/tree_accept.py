# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Verification of a draft tree, greedy or sampled.

Every verify row first gets the target's token: what vLLM's sampler would draw from that row's processed logits
(temperature, top-k/top-p, min-p, logit bias), with vLLM's per-request Gumbel-max noise keyed by the request's
seed and the row's own position (its depth). A greedy request's token is the argmax. The walk then goes from the
root to the child whose draft token equals the current node's token, until a node has no such child. The
accepted rows are the path (root first); the sampled tokens are the tokens of the path rows, the last of which is
the bonus token.

This is lossless: the token emitted at each position is a draw from the target's distribution given the accepted
prefix (the row there has exactly that prefix), with noise that no other emitted token shares (the path's rows
are at distinct positions). It is also the draw vLLM's sampler makes without speculation for the same seed and
position, so a seeded request decodes the same text up to the logits' numerics. The draft tree must be fixed
before the noise is drawn, which holds: the harness drafts deterministic top-k trees.

The tree is the batch's :class:`~eagle_harness.vllm.verify_layout.VerifyStage` (a chain when the forward
verified chains, so the walk is then vLLM's chain rule). The accepted rows of each request are kept in
:class:`~eagle_harness.vllm.verify_layout.TreeStore` for the rest of the round (KV compaction, the draft's
re-seeding and the GDN state commit read them).
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from eagle_harness.vllm.verify_layout import TreeStore, VerifyStage


@triton.jit
def _tree_accept_kernel(
    target_ptr,  # [num_logits] int64: the target's token at every row (argmax, or a draw from its distribution)
    drafted_ptr,  # [num_logits] int64: the token every row holds (row 0: the last sampled token)
    cu_num_logits_ptr,  # [num_reqs + 1]
    idx_mapping_ptr,  # [num_reqs]
    parents_ptr,  # [slots, W] int32: the staged tree of each request state slot
    stride_parents,
    sampled_ptr,  # [num_reqs, S] int64
    stride_sampled,
    num_sampled_ptr,  # [num_reqs] int32
    path_ptr,  # [slots, W] int32: the accepted rows, root first
    stride_path,
    path_len_ptr,  # [slots] int32
    W: tl.constexpr,
    WP: tl.constexpr,
):
    b = tl.program_id(0)
    slot = tl.load(idx_mapping_ptr + b)
    start = tl.load(cu_num_logits_ptr + b)
    num_rows = tl.load(cu_num_logits_ptr + b + 1) - start
    rows = tl.arange(0, WP)
    parents = tl.load(parents_ptr + slot * stride_parents + rows, mask=rows < W, other=-1)
    drafted = tl.load(drafted_ptr + start + rows, mask=rows < num_rows, other=-1)
    target = tl.load(target_ptr + start + rows, mask=rows < num_rows, other=-1)

    node = 0
    alive = num_rows > 0
    count = 0
    for step in range(0, W):
        token = tl.sum(tl.where(rows == node, target, 0))
        tl.store(sampled_ptr + b * stride_sampled + step, token, mask=alive)
        tl.store(path_ptr + slot * stride_path + step, node, mask=alive)
        count += alive.to(tl.int32)
        children = (parents == node) & (drafted == token) & (rows > 0) & (rows < num_rows)
        child = tl.min(tl.where(children, rows, WP))
        alive = alive & (child < WP)
        node = tl.where(alive, child, node)
    tl.store(num_sampled_ptr + b, count)
    tl.store(path_len_ptr + slot, count)


def accept_tree(
    target: torch.Tensor,
    drafted: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    parents: torch.Tensor,
    store: TreeStore,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The walk for every request: ``(sampled [num_reqs, W], num_sampled [num_reqs])``; the path goes to ``store``."""
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = drafted.new_empty(num_reqs, store.width)
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=target.device)
    _tree_accept_kernel[(num_reqs,)](
        target,
        drafted,
        cu_num_logits,
        idx_mapping,
        parents,
        parents.stride(0),
        sampled,
        sampled.stride(0),
        num_sampled,
        store.path,
        store.path.stride(0),
        store.path_len,
        W=store.width,
        WP=triton.next_power_of_2(store.width),
        num_warps=1,
    )
    return sampled, num_sampled


def accepted_rows_first(
    expanded_idx_mapping: torch.Tensor, expanded_local_pos: torch.Tensor, store: TreeStore
) -> torch.Tensor:
    """For every logits row, the row it should hold: row ``k`` of a request takes its ``k``-th accepted row (path
    order) while ``k`` is below the path length, and keeps its own place after it."""
    rows = torch.arange(expanded_local_pos.shape[0], device=expanded_local_pos.device)
    slots, k = expanded_idx_mapping.long(), expanded_local_pos.long()
    accepted = rows - k + store.path[slots, k].long()
    return torch.where(k < store.path_len[slots], accepted, rows)


class TreeRejectionSampler(RejectionSampler):
    """vLLM's rejection sampler with the chain rule replaced by the tree walk, for greedy and sampled requests."""

    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
        store: TreeStore,
        stage: VerifyStage,
    ):
        super().__init__(sampler, spec_config, device)
        self.store = store
        self.stage = stage

    def reads_draft_prefix(self, idx_mapping_np: np.ndarray) -> bool:
        """Whether a request of the batch has logits processing that reads the tokens before each row: penalties,
        bad words, a thinking budget, or a custom logits processor (assumed to).

        vLLM hands a processor the rows before a row of the request as that row's prefix; in a tree those are not the
        row's ancestors, so the speculator verifies such a batch as chains. Logit bias (also allowed tokens and
        min_tokens) reads only the row's position, which :meth:`_verify` passes per row.
        """
        sampler = self.sampler
        if not np.any(sampler.needs_logits_processing[idx_mapping_np]):
            return False
        for processor in sampler.logits_processors:
            if isinstance(processor, PenaltiesState):
                uses = processor.use_penalty[idx_mapping_np]
            elif isinstance(processor, BadWordsState):
                uses = processor.num_bad_words.np[idx_mapping_np] > 0
            elif isinstance(processor, LogitBiasState):
                continue
            else:
                return True
            if np.any(uses):
                return True
        thinking = sampler.thinking_budget_state  # its per-request flags exist only with a reasoning config
        return thinking.enabled and bool(np.any(thinking.use_thinking_budget[idx_mapping_np]))

    def __call__(self, logits, input_batch, draft_logits=None, invalid_drafts=None):
        # The forward ran on depth positions; the sampler and everything after it use chain positions.
        self.stage.restore_positions(input_batch.positions, input_batch.num_tokens)
        if input_batch.has_structured_output_reqs:
            # The host masks a flattened chain, not each node's ancestry. Only the root is safe;
            # keep the whole mixed batch root-only until the host can supply tree-aware masks.
            non_root = input_batch.expanded_local_pos > 0
            invalid_drafts = non_root if invalid_drafts is None else invalid_drafts | non_root
        return super().__call__(logits, input_batch, draft_logits, invalid_drafts)

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        seq_lens_upper_bound_np: np.ndarray,
        verify_draft_sampled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # `pos` holds chain positions; a tree row's own position is the root's plus its depth.
        depth = self.stage.slot_depths[expanded_idx_mapping.long(), expanded_local_pos.long()]
        row_pos = pos - expanded_local_pos + depth
        processed = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            row_pos,
            draft_sampled,
            expanded_local_pos,
            seq_lens_upper_bound_np,
        )
        states = self.sampler.sampling_states
        target = gumbel_sample(
            processed,
            expanded_idx_mapping,
            states.temperature.gpu,
            states.seeds.gpu,
            row_pos,
            apply_temperature=False,  # apply_sampling_params did
            is_drafting=False,
            use_fp64=self.sampler.use_fp64_gumbel,
        )
        sampled, num_sampled = accept_tree(
            target, verify_draft_sampled, cu_num_logits, idx_mapping, self.stage.slot_parents, self.store
        )
        if self.sampler.return_sampling_mask or (
            self.sampler.sampling_states.max_num_logprobs(idx_mapping_np) != NO_LOGPROBS
        ):
            # vLLM reads the logprobs and the sampling mask of sampled token k from logits row k of the request (raw
            # rows from `logits`, processed ones from what this returns); a tree's accepted rows are not a prefix.
            rows = accepted_rows_first(expanded_idx_mapping, expanded_local_pos, self.store)
            if processed is not logits:
                processed = processed[rows]
            logits.copy_(logits[rows])
        return processed, sampled, num_sampled
