# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
"""Recorded proposals indexed by the existing draft tree's expansion slots."""

import torch

from eagle_harness.kernels.sampled_draw import sample_children


class SampledTree:
    """Static proposal storage; node ids remain the ordinary DraftTree ids.

    Expansion slot 0 is the root, then K frontier rows per draft step. Scores
    are q's order statistics assigned in draw order, so pruning preserves
    sibling prefixes without conditioning on the proposed token itself.

    Draws come from counter-based Philox keyed by the TP-synchronized seed, a
    device round counter advanced by the root draw's graph and the draft step,
    so every TP rank draws the same tree with no collective. One Triton launch
    per draft step records q and draws the children.
    """

    def __init__(self, tree, candidates: int, seed: int = 0, *, req_index=None, rows: int = 0):
        self.tree = tree
        shape = tree.shape
        device = tree.tokens.device
        slots = 1 + (shape.max_depth - 1) * shape.top_k
        # Stores are indexed by the request's pool slot (``req_index`` = the drafter's staged req_pool_indices), so a
        # recorded root proposal follows its request across batch changes; without a map, by batch position.
        rows = max(rows, tree.max_bs)
        self.req_index = req_index if req_index is not None else torch.arange(tree.max_bs, dtype=torch.int64, device=device)
        self.q = torch.zeros((rows, slots, candidates), device=device)
        self.ids = torch.zeros_like(self.q, dtype=torch.int64)
        self.positions = torch.zeros((rows, tree.tokens.shape[1]), dtype=tree.tokens.dtype, device=device)
        self.root_valid = torch.zeros(rows, dtype=torch.bool, device=device)
        self.seed = int(seed)
        self.round = torch.zeros(1, dtype=torch.int32, device=device)
        self.active = False

    def draw(self, model, hidden, bs, step):
        """Record the exact candidate distribution and draw the children, in place, in one launch."""
        candidates = model.proposal_candidates(hidden)
        k = self.tree.shape.top_k
        rows = 1 if step == 0 else k
        slot = 0 if step == 0 else 1 + (step - 1) * k
        if step == 0:
            self.round += 1  # a round's draws start with the root; replayed identically on every rank
            self.root_valid.index_fill_(0, self.req_index[:bs], True)
        return sample_children(
            candidates.logits,
            candidates.token_ids,
            k,
            seed=self.seed,
            launch=step,
            counter=self.round,
            q=self.q,
            q_ids=self.ids,
            positions=self.positions,
            slot=slot,
            start=self.tree.shape.layer_start(step + 1),
            rows_per_req=rows,
            req_index=self.req_index[:bs],
        )

    def invalidate(self, req_pool_indices: torch.Tensor) -> None:
        """Requests whose root tokens come from a prefill (no recorded q): their first round emits one target token."""
        self.root_valid.index_fill_(0, req_pool_indices.to(device=self.root_valid.device, dtype=torch.int64), False)

    def sample(self, verify_input, target_probs, uniforms, final_uniforms):
        """Probability-only host seam, after target temperature and filtering."""
        from eagle_harness.kernels.sampled_tree import sample_tree

        bs = target_probs.shape[0]
        return sample_tree(
            target_probs,
            self.q,
            self.ids,
            self.positions,
            self.root_valid,
            verify_input.proposal_nodes,
            verify_input.draft_token,
            verify_input.retrieve_index,
            verify_input.retrieve_next_token,
            verify_input.retrieve_next_sibling,
            uniforms,
            final_uniforms,
            top_k=self.tree.shape.top_k,
            max_depth=verify_input.max_tree_depth,
            req_index=self.req_index[:bs],
        )
