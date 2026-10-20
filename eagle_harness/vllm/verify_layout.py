# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The tree of the current verify forward, in buffers that CUDA graphs replay.

``propose`` stores the parent row of every verify row of a request (:class:`TreeStore`, by request state
slot). Before each target forward :meth:`VerifyStage.stage` gathers the batch's rows into
:class:`VerifyStage` (by batch row) and rewrites the forward's positions from chain positions to depths.
The target's tree attention (:mod:`.tree_attention`), its GDN layers (:mod:`.gdn_tree_verify`) and the
tree rejection sampler (:mod:`.tree_accept`) read the stage; it is one per process, like the forward
context, because the attention and GDN layers do not take the speculator as an argument.

A verify row ``r`` of a request holds a draft node; row 0 is the root (the last sampled token). Rows are
ordered parent first, so any prefix of the rows is a tree too: when the scheduler grants fewer drafts than
the tree has rows, the first ``1 + drafts`` rows are verified.
"""

from __future__ import annotations


import torch
import triton
import triton.language as tl

# The conv window of the root reads the request's conv state: codes -1, -2, -3 are the last, second to last
# and third to last entry of the window before the root.
WINDOW_CODES = (-1, -2, -3)


@triton.jit
def _stage_tree_kernel(
    parents_ptr,  # [slots, W] int32: parent row of each verify row (row 0: -1)
    stride_parents,
    idx_mapping_ptr,  # [num_reqs] request state slot of each batch row
    query_start_loc_ptr,  # [num_reqs + 1] int32
    cu_num_logits_ptr,  # [num_reqs + 1] int32
    positions_ptr,  # [num_tokens] chain positions in, depth positions out
    stage_parents_ptr,  # [max_reqs, W] int32
    stage_anc_ptr,  # [max_reqs, 3, W] int32: codes of the 3 window ancestors of each row
    stage_mask_ptr,  # [max_reqs, W] int64: bit j set iff row j is an ancestor of the row or the row itself
    context_lens_ptr,  # [max_reqs] int32: committed tokens of each request (its first query token's position)
    slot_parents_ptr,  # [slots, W] int32: the staged parents again, by request state slot
    slot_depths_ptr,  # [slots, W] int32: the depth of each row, by request state slot
    W: tl.constexpr,
    WP: tl.constexpr,
    TREE: tl.constexpr,  # False: every request is staged as a chain (the forward verifies a chain)
):
    b = tl.program_id(0)
    slot = tl.load(idx_mapping_ptr + b)
    start = tl.load(query_start_loc_ptr + b)
    width = tl.load(query_start_loc_ptr + b + 1) - start
    tl.store(context_lens_ptr + b, tl.load(positions_ptr + start).to(tl.int32))
    # A request verifies a tree when its query is all drafts (a prefill chunk has one logits row).
    is_tree = TREE & (width > 1) & (width == tl.load(cu_num_logits_ptr + b + 1) - tl.load(cu_num_logits_ptr + b))
    rows = tl.arange(0, WP)
    in_tree = rows < tl.minimum(width, W)
    parents = tl.load(parents_ptr + slot * stride_parents + rows, mask=rows < W, other=-1)
    # Chain layout for anything that is not a tree, so the stage is valid for every request.
    parents = tl.where(is_tree, parents, rows - 1)

    depth = tl.zeros([WP], dtype=tl.int32)
    anc1 = tl.full([WP], -1, dtype=tl.int32)
    anc2 = tl.full([WP], -2, dtype=tl.int32)
    anc3 = tl.full([WP], -3, dtype=tl.int32)
    mask = tl.full([WP], 1, dtype=tl.int64)
    for r in range(1, W):
        parent = tl.sum(tl.where(rows == r, parents, 0))
        at_parent = rows == parent
        depth = tl.where(rows == r, tl.sum(tl.where(at_parent, depth, 0)) + 1, depth)
        anc1 = tl.where(rows == r, parent, anc1)
        anc2 = tl.where(rows == r, tl.sum(tl.where(at_parent, anc1, 0)), anc2)
        anc3 = tl.where(rows == r, tl.sum(tl.where(at_parent, anc2, 0)), anc3)
        own_bit = tl.cast(1, tl.int64) << r
        mask = tl.where(rows == r, tl.sum(tl.where(at_parent, mask, 0)) | own_bit, mask)

    out = b * W + rows
    tl.store(stage_parents_ptr + out, parents, mask=rows < W)
    tl.store(stage_anc_ptr + b * 3 * W + rows, anc1, mask=rows < W)
    tl.store(stage_anc_ptr + b * 3 * W + W + rows, anc2, mask=rows < W)
    tl.store(stage_anc_ptr + b * 3 * W + 2 * W + rows, anc3, mask=rows < W)
    tl.store(stage_mask_ptr + out, mask, mask=rows < W)
    tl.store(slot_parents_ptr + slot * W + rows, parents, mask=rows < W)
    tl.store(slot_depths_ptr + slot * W + rows, depth, mask=rows < W)
    # chain position = context + row; tree position = context + depth
    positions = tl.load(positions_ptr + start + rows, mask=in_tree, other=0)
    tl.store(positions_ptr + start + rows, positions - rows + depth, mask=in_tree & is_tree)


class TreeStore:
    """The tree of each request by request state slot: written by ``propose`` (``parents``) and by the
    tree rejection sampler (``path``: the accepted rows, root first; ``path_len`` of them)."""

    def __init__(self, max_reqs: int, width: int, device: torch.device):
        self.width = width
        self.parents = torch.full((max_reqs, width), -1, dtype=torch.int32, device=device)
        self.path = torch.zeros(max_reqs, width, dtype=torch.int32, device=device)
        self.path_len = torch.ones(max_reqs, dtype=torch.int32, device=device)


class VerifyStage:
    """The batch's trees, by batch row.

    ``active``: the forward being built verifies trees (every request is a decode with drafts); otherwise
    it verifies chains and the stage holds chain layouts, so the tree rejection sampler, which reads the
    stage, walks a chain. The layers ask :attr:`verifying`, which is also true while a CUDA graph is captured.
    """

    def __init__(self, max_reqs: int, max_tokens: int, width: int, device: torch.device):
        if width > 64:
            raise ValueError(f"a verify tree has at most 64 rows, got {width}")
        self.width = width
        self.parents = torch.zeros(max_reqs, width, dtype=torch.int32, device=device)
        self.ancestors = torch.zeros(max_reqs, 3, width, dtype=torch.int32, device=device)
        self.masks = torch.ones(max_reqs, width, dtype=torch.int64, device=device)
        # Committed tokens of each request (the tree attention's context), once per forward instead of per layer.
        self.context_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        # The staged parents and the rows' depths by request state slot: the rejection sampler splits a large batch
        # into chunks of requests, which it addresses by slot.
        self.slot_parents = torch.zeros(max_reqs, width, dtype=torch.int32, device=device)
        self.slot_depths = torch.zeros(max_reqs, width, dtype=torch.int32, device=device)
        self.chain_positions = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        # The GDN state slots of each verify row of the batch, per linear-attention KV cache group: what the forward's
        # linear layers addressed. One layer of each group (its key) copies them in while the forward runs, so a CUDA
        # graph replay refreshes them too.
        self.max_reqs, self.device = max_reqs, device
        self.state_slots: dict[str, torch.Tensor] = {}  # by the publishing layer's name
        self.active = True

    @property
    def verifying(self) -> bool:
        """Whether the target's layers take the tree path now: a tree is staged, or a CUDA graph is being captured.

        vLLM captures the target's verify graphs outside ``prepare_verify_inputs`` (no batch is staged then, and
        ``active`` is whatever the last dummy run left), and a graph only replays the kernels it captured; so a
        capture always takes the tree path. Only verify-shaped batches (several query tokens per request) reach
        these layers in a capture; decode graphs of one token per request never do.
        """
        return self.active or torch.cuda.is_current_stream_capturing()

    def publish_slots_of(self, layer_name: str) -> torch.Tensor:
        """The buffer a linear-attention layer (one per KV cache group) publishes its group's slots to."""
        buf = self.state_slots[layer_name] = torch.zeros(
            self.max_reqs, self.width, dtype=torch.int32, device=self.device
        )
        return buf

    def stage(
        self,
        store: TreeStore,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        cu_num_logits: torch.Tensor,
        positions: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
        *,
        tree: bool,
        model_positions: torch.Tensor | None = None,
    ) -> None:
        """Gather the batch's trees from ``store`` and turn ``positions`` from chain to depth positions
        (``tree=False``: stage chains and leave the positions). ``model_positions``: the positions the model
        takes when that is another tensor than ``positions`` (``[dims, tokens]`` for M-RoPE models), moved by
        the same amount."""
        self.chain_positions[:num_tokens].copy_(positions[:num_tokens])
        _stage_tree_kernel[(num_reqs,)](
            store.parents,
            store.parents.stride(0),
            idx_mapping,
            query_start_loc,
            cu_num_logits,
            positions,
            self.parents,
            self.ancestors,
            self.masks,
            self.context_lens,
            self.slot_parents,
            self.slot_depths,
            W=self.width,
            WP=triton.next_power_of_2(self.width),
            TREE=tree,
            num_warps=1,
        )
        self.active = tree
        if model_positions is not None:
            model_positions[:, :num_tokens] += positions[:num_tokens] - self.chain_positions[:num_tokens]

    def restore_positions(self, positions: torch.Tensor, num_tokens: int) -> None:
        """Put the chain positions back after the verify forward (the sampler reads them)."""
        if self.active:
            positions[:num_tokens].copy_(self.chain_positions[:num_tokens])


_STAGE: VerifyStage | None = None


def set_stage(stage: VerifyStage | None) -> None:
    global _STAGE
    _STAGE = stage


def get_stage() -> VerifyStage | None:
    """The process's verify stage; ``None`` while no harness speculator is running."""
    return _STAGE
