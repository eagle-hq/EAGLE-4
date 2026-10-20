# Modified by Eagle AI Corp.
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN tree verify on the target (hook H10): conv1d and recurrence through ancestor tables.

sglang's verify kernels follow a tree's links in every layer and channel block.
This extension derives the tree once per verify forward
(:func:`gdn_tree_ancestors`: parent table, the three conv-window ancestors of
each node, first-child chains for the path replay) and runs the tree kernels
on it (``eagle_harness.kernels.gdn_tree_conv`` and ``gdn_tree_recurrence``,
with the Gluon twin of the recurrence walk where it is validated). Output, conv windows and SSM
snapshots have the backend kernels' layout, so the accept / commit path is
sglang's. Both kernels take token-strided inputs, so the GDN layers hand them
views of their input projections (``reads_strided_inputs``) instead of running
the split kernels.

A captured verify graph records the table build (static tables shared by every
batch size and width); eager forwards build fresh tables.
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

from eagle_harness.kernels import arch
from eagle_harness.kernels.gdn_tree import gdn_tree_ancestors, gdn_tree_paths_buffer
from eagle_harness.kernels.gdn_tree_conv import gdn_tree_conv1d_update
from eagle_harness.kernels.gdn_tree_recurrence import gdn_tree_recurrent_update

logger = logging.getLogger(__name__)

# The path replay runs while the batch has few (request, v-head) pairs and the tree is wide enough to branch.
PATH_REPLAY_MAX_ROWS = 24
PATH_REPLAY_MIN_WIDTH = 5


# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
# Modified by Eagle AI Corp.
class GDNTreeVerify(LinearAttnVerifyExtension):
    """Tree verify tables of one GDN backend and the kernels they select."""

    def __init__(self, device: torch.device, num_v_heads: int, fp32_state: bool):
        self.device = device
        self.num_v_heads = num_v_heads
        self.fp32_state = fp32_state
        self.records: torch.Tensor | None = None
        self._graph_ancestors: Optional[torch.Tensor] = None
        self._graph_paths: Optional[torch.Tensor] = None
        # Tables of the forward being built or captured; None: not a tree verify.
        self._ancestors: Optional[torch.Tensor] = None
        self._paths: Optional[torch.Tensor] = None
        self._parent_token: Optional[torch.Tensor] = None

    @classmethod
    def create(cls, state: LinearAttnStateLayout) -> Optional["GDNTreeVerify"]:
        """The extension for a backend with ``state``, or None where the tree kernels do not apply
        (no per-step SSM snapshots, a conv width other than 4, a 32-bit conv state)."""
        if not state.has_state_snapshots or state.conv_state_len != 3 or state.conv_dtype.itemsize != 2:
            logger.info("eagle-harness: GDN tree verify off (this target's state layout is not covered)")
            return None
        fp32 = state.ssm_dtype == torch.float32
        result = cls(state.device, num_v_heads=state.num_v_heads, fp32_state=fp32)
        shape = getattr(state, "snapshot_shape", ())
        # Replay records need the engine's state-commit seam (no tracked/overlapped commits) and the snapshot layout.
        if arch.gdn_records_replay(state.device) and getattr(state, "can_commit_state", False) and len(shape) == 6:
            layers, rows, steps, heads, value_dim, key_dim = shape
            if (
                steps <= 64
                and key_dim >= 16
                and key_dim & (key_dim - 1) == 0
                and value_dim >= 64
                and value_dim % 64 == 0
            ):
                result.records = torch.zeros(
                    (layers, rows, steps, heads, value_dim + key_dim + 4),
                    dtype=torch.float32,
                    device=state.device,
                )

        logger.info(
            "eagle-harness: GDN tree verify on, %s",
            "replay records (accepted path replayed into the state pool)"
            if result.records is not None
            else "per-node state snapshots",
        )
        return result

    def _uses_path_replay(self, bs: int, width: int) -> bool:
        if self.records is not None:
            return not self.fp32_state
        return self.fp32_state and width >= PATH_REPLAY_MIN_WIDTH and bs * self.num_v_heads <= PATH_REPLAY_MAX_ROWS

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        """Allocate the static tree tables captured graphs write and read."""
        max_width = max_num_tokens // max_bs
        self._graph_ancestors = torch.zeros(max_bs * 3 * max_width, dtype=torch.int32, device=self.device)
        path_bs = max((bs for bs in range(1, max_bs + 1) if self._uses_path_replay(bs, max_width)), default=0)
        if path_bs:
            self._graph_paths = torch.zeros(
                path_bs * max_width * (2 + max_width), dtype=torch.int32, device=self.device
            )

    def begin_forward(self, layout: Optional[LinearAttnVerifyLayout], *, in_graph: bool) -> None:
        """Build this forward's tree tables (a target verify with tree links); otherwise clear them."""
        self._ancestors = self._paths = self._parent_token = None
        if layout is None:
            return
        bs, width = layout.next_token.shape[0], layout.width
        path_replay = self._uses_path_replay(bs, width)
        if in_graph:
            ancestors = self._graph_ancestors[: bs * 3 * width].view(bs, 3, width)
            paths = self._graph_paths[: bs * width * (2 + width)].view(bs, width, 2 + width) if path_replay else None
        else:
            ancestors = torch.empty((bs, 3, width), dtype=torch.int32, device=self.device)
            paths = gdn_tree_paths_buffer(bs, width, self.device) if path_replay else None
        gdn_tree_ancestors(
            layout.next_token,
            layout.next_sibling,
            layout.parent_token,
            layout.cache_indices,
            width,
            ancestors,
            query_start_loc=layout.query_start_loc,
            paths=paths,
        )
        self._ancestors, self._paths, self._parent_token = ancestors, paths, layout.parent_token

    def commit_state(self, state, state_indices, record_indices, commit_steps, parent) -> bool:
        """Replay accepted paths using live metadata, including on graph replay."""
        if self.records is None:
            return False
        from eagle_harness.kernels.gdn_tree_records import gdn_tree_records_commit

        gdn_tree_records_commit(
            state,
            self.records,
            state_indices,
            record_indices,
            commit_steps,
            parent,
            not self.fp32_state,
        )
        return True

    def reads_strided_inputs(self) -> bool:
        """H10: the tree kernels take token-strided inputs, so the GDN layers pass views of their projections
        (no split kernels, and the conv does not wait for the b / a projection)."""
        return self._ancestors is not None

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
        """The tree conv1d over the verify tokens (None: sglang's own kernel runs)."""
        if self._ancestors is None:
            return None
        return gdn_tree_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
            intermediate_conv_window=intermediate_conv_window,
            intermediate_state_indices=intermediate_state_indices,
            tree_ancestors=self._ancestors,
        )

    def recurrent_update(
        self,
        layer,
        *,
        query,
        key,
        value,
        a,
        b,
        ssm_states,
        cache_indices,
        query_start_loc,
        intermediate_states,
        intermediate_state_indices,
    ) -> Optional[torch.Tensor]:
        """The tree recurrence over the verify tokens (None: sglang's own kernel runs)."""
        if self._ancestors is None:
            return None
        if self.records is not None:
            from eagle_harness.kernels.gdn_tree_records import gdn_tree_records_update

            assert intermediate_states.is_contiguous()
            layer_index, remainder = divmod(intermediate_states.storage_offset(), intermediate_states.numel())
            assert remainder == 0, "state snapshot layers must be contiguous slices"
            return gdn_tree_records_update(
                layer.A_log,
                layer.dt_bias,
                query,
                key,
                value,
                a,
                b,
                state=ssm_states,
                state_indices=cache_indices,
                query_start_loc=query_start_loc,
                parent=self._parent_token,
                tree_paths=self._paths,
                records=self.records[layer_index],
                record_indices=intermediate_state_indices,
                width=self._ancestors.shape[-1],
            )
        return gdn_tree_recurrent_update(
            layer.A_log,
            layer.dt_bias,
            query,
            key,
            value,
            a,
            b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            intermediate_states_buffer=intermediate_states,
            intermediate_state_indices=intermediate_state_indices,
            retrieve_parent_token=self._parent_token,
            tree_paths=self._paths,
            gluon=True,
        )
