# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao
# Derived from sglang (Apache-2.0): the tree conv from python/sglang/kernels/ops/mamba/causal_conv1d_triton.py (itself
# adapted from vLLM and causal-conv1d, BSD-3-Clause) and the recurrence from
# python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py (sglang's fla kernels adapt
# flash-linear-attention, MIT). Modified by Eagle AI Corp: ancestor tables, slot tables for both engines, path replay.
# Notices and license texts: THIRD_PARTY_NOTICES.md and LICENSES/.
"""GDN tree verify kernels.

A tree verify of a gated delta net layer runs, per node, the width-4 causal conv1d and the gated delta rule
recurrence, each node continuing from its parent. The kernels are drop-in equivalents of the tree paths of
``causal_conv1d_update`` and ``fused_sigmoid_gating_delta_rule_update`` (tree links, per-node conv windows and SSM
snapshots, no state write-back), in three modules:

- this module derives the tree once per forward from the first-child / next-sibling links
  (:func:`gdn_tree_ancestors`): the parent table the recurrence reads, the three window ancestors of every node for the
  conv and, for the path replay, the first-child chains;
- ``gdn_tree_conv``: the conv gathers each node's window through the ancestor table instead of chasing the links in
  every layer and channel block;
- ``gdn_tree_recurrence``: the recurrence walks the nodes in order with the next step's parent index and inputs
  prefetched, or replays each chain's root path from the request state (few rows, fp32 state); its Gluon twin
  (``gdn_tree_gluon``) runs the walk bitwise faster on B300.

Conv output (16-bit inputs), conv windows, conv state and the parent table are bit-identical to the upstream kernels.
The recurrence computes the upstream kernel's math; its snapshots can differ from the upstream kernel's in the last
bit: the walk compiles without FMA contraction and writes its multiply-adds as FMAs. States are addressed through slot
tables, so the kernels serve sglang's layout (a scratch row of snapshots per request) and vLLM's (one state slot per
verify row) alike.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1  # a padded request row of a batch (sglang's causal_conv1d PAD_SLOT_ID)


@triton.jit
def _gdn_tree_paths(
    parents_in,  # (NP2_SEQLEN,) parent links of one request
    idx_tokens,
    num_tokens,  # verify tokens of the request, <= seqlen
    paths_ptr,  # this request's (seqlen, 2 + seqlen) rows
    stride_paths_path,
    seqlen: tl.constexpr,
):
    # Row p is [length, write_from, root path...] of the request's p-th leaf in node order; the path's nodes at depth
    # >= write_from are the leaf's chain (its run of first children). Every node is on exactly one chain; rows past
    # the leaf count have length 0. Links must be ancestor-first (parent < node), as the recurrence walk requires;
    # other parents are replaced by node - 1 so the table stays in bounds.
    valid = idx_tokens < num_tokens
    parents = tl.where(
        (idx_tokens > 0) & (parents_in >= 0) & (parents_in < idx_tokens),
        parents_in,
        tl.maximum(idx_tokens - 1, 0),
    )
    # one pass in node order; per node, packed: depth | chain-start depth << 8 | has-a-child << 16
    packed = tl.zeros(idx_tokens.shape, dtype=tl.int32)
    for idx_token in range(1, seqlen):
        parent = tl.sum(tl.where(idx_tokens == idx_token, parents, 0))
        packed_parent = tl.sum(tl.where(idx_tokens == parent, packed, 0))
        depth = (packed_parent & 255) + 1
        # a first child continues its parent's chain, a later sibling starts one
        chain_depth = tl.where(packed_parent >= 65536, depth, (packed_parent >> 8) & 255)
        is_node = idx_token < num_tokens
        packed = tl.where((idx_tokens == idx_token) & is_node, depth + (chain_depth << 8), packed)
        packed = tl.where((idx_tokens == parent) & is_node, packed | 65536, packed)
    depths = packed & 255
    is_leaf = valid & (packed < 65536)
    leaf_rank = tl.cumsum(is_leaf.to(tl.int32), 0) - 1
    num_leaves = tl.sum(is_leaf.to(tl.int32))
    row = paths_ptr + leaf_rank * stride_paths_path
    tl.store(row, depths + 1, mask=is_leaf)
    tl.store(row + 1, (packed >> 8) & 255, mask=is_leaf)
    tl.store(
        paths_ptr + idx_tokens * stride_paths_path,
        tl.zeros(idx_tokens.shape, dtype=tl.int32),
        mask=(idx_tokens >= num_leaves) & (idx_tokens < seqlen),
    )
    # root paths, written leaf first
    node = idx_tokens
    for step in range(0, tl.max(tl.where(is_leaf, depths, 0)) + 1):
        tl.store(row + 2 + depths - step, node, mask=is_leaf & (step <= depths))
        node = tl.sum(tl.where(idx_tokens[None, :] == node[:, None], parents[None, :], 0), 1)


@triton.jit
def _gdn_tree_ancestors_kernel(
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    retrieve_parent_token_ptr,
    conv_state_indices_ptr,
    ancestors_ptr,  # (batch, 3, seqlen) int32
    query_start_loc_ptr,
    paths_ptr,  # (batch, seqlen, 2 + seqlen) int32
    stride_next_seq,
    stride_next_token,
    stride_sibling_seq,
    stride_sibling_token,
    stride_parent_seq,
    stride_parent_token,
    stride_state_indices,
    stride_paths_seq,
    stride_paths_path,
    pad_slot_id: tl.constexpr,
    seqlen: tl.constexpr,
    NP2_SEQLEN: tl.constexpr,
    BUILD_PATHS: tl.constexpr,
):
    # Replays the parent derivation of the upstream tree conv in its link order, so the parent table is identical,
    # and records the three window ancestors that conv visits for each node: code v >= 0 is node v, codes -1/-2/-3
    # are conv_state columns 2/1/0.
    idx_seq = tl.program_id(0)
    idx_tokens = tl.arange(0, NP2_SEQLEN)
    mask_retrieve = idx_tokens < seqlen
    conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices)
    if BUILD_PATHS:
        num_tokens = tl.minimum(
            tl.maximum(tl.load(query_start_loc_ptr + idx_seq + 1) - tl.load(query_start_loc_ptr + idx_seq), 0),
            seqlen,
        )
    if conv_state_batch_coord == pad_slot_id:
        # padded rows keep their parent links; their paths follow them
        if BUILD_PATHS:
            parents = tl.load(
                retrieve_parent_token_ptr + idx_seq * stride_parent_seq + idx_tokens * stride_parent_token,
                mask=mask_retrieve,
                other=0,
            ).to(tl.int32)
            _gdn_tree_paths(
                parents, idx_tokens, num_tokens, paths_ptr + idx_seq * stride_paths_seq, stride_paths_path, seqlen
            )
        return

    retrieve_next_tokens = tl.load(
        retrieve_next_token_ptr + idx_seq * stride_next_seq + idx_tokens * stride_next_token,
        mask=mask_retrieve,
        other=-1,
    )
    retrieve_next_siblings = tl.load(
        retrieve_next_sibling_ptr + idx_seq * stride_sibling_seq + idx_tokens * stride_sibling_token,
        mask=mask_retrieve,
        other=-1,
    )
    parent_idx_tokens = tl.zeros((NP2_SEQLEN,), dtype=tl.int32)
    # node 0 reads conv_state columns 2, 1, 0
    anc1 = tl.full((NP2_SEQLEN,), -1, dtype=tl.int32)
    anc2 = tl.full((NP2_SEQLEN,), -2, dtype=tl.int32)
    anc3 = tl.full((NP2_SEQLEN,), -3, dtype=tl.int32)
    for idx_token in range(seqlen):
        next_idx = tl.sum(tl.where(idx_tokens == idx_token, retrieve_next_tokens, 0))
        parent_idx_tokens = tl.where((idx_tokens == next_idx) & (next_idx != -1), idx_token, parent_idx_tokens)
        sibling_idx = tl.sum(tl.where(idx_tokens == idx_token, retrieve_next_siblings, 0))
        # the sibling update below cannot change this node's own parent
        parent_idx_token = tl.sum(tl.where(idx_tokens == idx_token, parent_idx_tokens, 0))
        parent_idx_tokens = tl.where(
            (idx_tokens == sibling_idx) & (sibling_idx != -1), parent_idx_token, parent_idx_tokens
        )
        # ancestors under the links derived so far, chased as the upstream conv does
        grand_idx_token = tl.sum(tl.where(idx_tokens == parent_idx_token, parent_idx_tokens, 0))
        grand_idx_token = tl.where(parent_idx_token > 0, grand_idx_token, parent_idx_token - 1)
        great_idx_token = tl.sum(tl.where(idx_tokens == grand_idx_token, parent_idx_tokens, 0))
        great_idx_token = tl.where(grand_idx_token > 0, great_idx_token, grand_idx_token - 1)
        is_token = (idx_tokens == idx_token) & (idx_token > 0)
        anc1 = tl.where(is_token, parent_idx_token, anc1)
        anc2 = tl.where(is_token, grand_idx_token, anc2)
        anc3 = tl.where(is_token, great_idx_token, anc3)

    tl.store(
        retrieve_parent_token_ptr + idx_seq * stride_parent_seq + idx_tokens * stride_parent_token,
        parent_idx_tokens,
        mask=mask_retrieve,
    )
    anc_base = ancestors_ptr + idx_seq * 3 * seqlen + idx_tokens
    tl.store(anc_base, anc1, mask=mask_retrieve)
    tl.store(anc_base + seqlen, anc2, mask=mask_retrieve)
    tl.store(anc_base + 2 * seqlen, anc3, mask=mask_retrieve)
    if BUILD_PATHS:
        _gdn_tree_paths(
            parent_idx_tokens, idx_tokens, num_tokens, paths_ptr + idx_seq * stride_paths_seq, stride_paths_path, seqlen
        )


def gdn_tree_ancestors(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    retrieve_parent_token: torch.Tensor,
    conv_state_indices: torch.Tensor,
    width: int,
    ancestors: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    paths: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
) -> None:
    """Derive the tree of every request from the first ``width`` columns of the links.

    Writes ``retrieve_parent_token[:, :width]`` and ``ancestors`` (batch, 3, width), and with ``paths`` (batch, width,
    2 + width) the path table of the path replay (each request's node count from ``query_start_loc``). Rows whose
    ``conv_state_indices`` entry is ``pad_slot_id`` keep their parents and ancestors; their paths follow their current
    parents.
    """
    batch = ancestors.shape[0]
    assert tuple(ancestors.shape) == (batch, 3, width) and ancestors.dtype == torch.int32
    assert ancestors.is_contiguous()
    build_paths = paths is not None
    if build_paths:
        assert query_start_loc is not None and width < 256
        assert tuple(paths.shape) == (batch, width, 2 + width) and paths.dtype == torch.int32
        assert paths.stride(2) == 1
    _gdn_tree_ancestors_kernel[(batch,)](
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        conv_state_indices,
        ancestors,
        query_start_loc if build_paths else ancestors,
        paths if build_paths else ancestors,
        retrieve_next_token.stride(0),
        retrieve_next_token.stride(1),
        retrieve_next_sibling.stride(0),
        retrieve_next_sibling.stride(1),
        retrieve_parent_token.stride(0),
        retrieve_parent_token.stride(1),
        conv_state_indices.stride(0),
        paths.stride(0) if build_paths else 0,
        paths.stride(1) if build_paths else 0,
        pad_slot_id=pad_slot_id,
        seqlen=width,
        NP2_SEQLEN=triton.next_power_of_2(width),
        BUILD_PATHS=build_paths,
        num_warps=1,
    )


def gdn_tree_paths_buffer(batch: int, width: int, device) -> torch.Tensor:
    """A path table for ``gdn_tree_ancestors(paths=...)``."""
    return torch.zeros((batch, width, 2 + width), dtype=torch.int32, device=device)
