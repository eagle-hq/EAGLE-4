# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Mamba2 tree verify on the target: the conv1d through ancestor tables.

sglang's Mamba2 target verify follows a tree's links inside ``causal_conv1d_update_triton`` (which also writes the
parent table) in every layer and channel block. This extension derives the tree once per verify forward
(:func:`gdn_tree_ancestors`: the parent table and the three conv-window ancestors of each node) and runs the tree
conv on it (``eagle_harness.kernels.gdn_tree_conv``; the window walk does not depend on the mixer). sglang's
selective state update then runs on the parent table built here, and the accept / commit path stays sglang's: the conv
windows have the backend kernel's layout.

A captured verify graph records the table build (a static table shared by every batch size and width); eager forwards
build a fresh one.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.attention.linear.verify_extension import (
    LinearAttnStateLayout,
    LinearAttnVerifyExtension,
    LinearAttnVerifyLayout,
)

from eagle_harness.kernels.gdn_tree import gdn_tree_ancestors
from eagle_harness.kernels.gdn_tree_conv import gdn_tree_conv1d_update

logger = logging.getLogger(__name__)


class Mamba2TreeVerify(LinearAttnVerifyExtension):
    """Tree verify table of one Mamba2 backend and the tree conv that reads it."""

    def __init__(self, device: torch.device):
        self.device = device
        self._graph_ancestors: Optional[torch.Tensor] = None
        # Table of the forward being built or captured; None: not a tree verify.
        self._ancestors: Optional[torch.Tensor] = None

    @classmethod
    def create(cls, state: LinearAttnStateLayout) -> Optional["Mamba2TreeVerify"]:
        """The extension for a Mamba2 backend with ``state``, or None where the tree conv does not apply
        (no per-step SSM snapshots, a conv width other than 4, a 32-bit conv state)."""
        if not state.has_state_snapshots or state.conv_state_len != 3 or state.conv_dtype.itemsize != 2:
            logger.info("eagle-harness: Mamba2 tree verify off (this target's state layout is not covered)")
            return None
        return cls(state.device)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        """Allocate the static tree table captured graphs write and read."""
        self._graph_ancestors = torch.zeros(3 * max_num_tokens, dtype=torch.int32, device=self.device)

    def begin_forward(self, layout: Optional[LinearAttnVerifyLayout], *, in_graph: bool) -> None:
        """Build this forward's tree table and parent links (a target verify with tree links); otherwise clear it."""
        self._ancestors = None
        if layout is None:
            return
        bs, width = layout.next_token.shape[0], layout.width
        if in_graph:
            ancestors = self._graph_ancestors[: bs * 3 * width].view(bs, 3, width)
        else:
            ancestors = torch.empty((bs, 3, width), dtype=torch.int32, device=self.device)
        # Also writes layout.parent_token, which sglang's selective state update reads.
        gdn_tree_ancestors(
            layout.next_token, layout.next_sibling, layout.parent_token, layout.cache_indices, width, ancestors
        )
        self._ancestors = ancestors

    def conv_update(
        self,
        layer,
        mixed_qkv,
        *,
        conv_states,
        cache_indices,
        intermediate_conv_window,
        intermediate_state_indices,
    ) -> Optional[torch.Tensor]:
        """The tree conv1d over the verify tokens' x / B / C projection (None: sglang's own kernel runs)."""
        if self._ancestors is None:
            return None
        conv = layer.conv1d
        return gdn_tree_conv1d_update(
            mixed_qkv,
            conv_states,
            conv.weight.view(conv.weight.shape[0], conv.weight.shape[-1]),
            conv.bias,
            layer.activation,
            conv_state_indices=cache_indices,
            intermediate_conv_window=intermediate_conv_window,
            intermediate_state_indices=intermediate_state_indices,
            tree_ancestors=self._ancestors,
        )
