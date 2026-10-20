# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The core draft tree in TokenSpeed's terms: lane visibility masks and the verify window.

TokenSpeed drafts a tree in ``K`` lanes per request (``TreeDraftInputs``): draft step ``s >= 1`` forwards ``K`` rows
per request, lane ``r`` writing its draft KV to window slot ``(s - 1) * K + r`` past the accepted frontier, and a lane
row sees the committed prefix plus the window slots whose bits its 64-bit ``lane_mask`` sets (its ancestors and
itself). The core :class:`~eagle_harness.core.draft_tree.DraftTree` keeps the scores, the frontier and the tree; this
module carries its choices over to the lanes: after step ``s`` the frontier node ``j`` is a child of lane
``parent_lane[j]``, so it sees that lane's slots plus its own next slot ``s * K + j``.

TokenSpeed verifies the window ``[bs, N]`` (node 0 the root, the last verified token) with ``parent[0] = -1`` and
``parent[i] < i`` (:meth:`DraftTree.verify_rows` gives exactly that shape). Pure torch: CPU-testable, graph-capturable.
"""

from __future__ import annotations

import torch

from eagle_harness.core.draft_tree import DraftTree


def lane_bits(top_k: int, num_steps: int, device: torch.device) -> torch.Tensor:
    """``[num_steps, K]`` int64: bit of window slot ``s * K + j`` (bit 63 as the int64 sign bit)."""
    if (num_steps - 1) * top_k > 64:
        raise ValueError(f"a lane window of {(num_steps - 1) * top_k} slots exceeds the 64-bit lane mask")
    bits = [[_int64_bit(step * top_k + j) for j in range(top_k)] for step in range(num_steps)]
    return torch.tensor(bits, dtype=torch.int64, device=device)


def _int64_bit(position: int) -> int:
    value = 1 << position if position < 64 else 0  # slots past the window are never written
    return value - (1 << 64) if value >> 63 else value


def seed_lane_masks(lane_mask: torch.Tensor, bs: int, bits: torch.Tensor) -> None:
    """Depth 1: lane ``r`` sees its own step-1 slot ``r``."""
    k = bits.shape[1]
    lane_mask[: bs * k].view(bs, k).copy_(bits[0].expand(bs, k))


def parent_lanes(tree: DraftTree, bs: int, step: int) -> torch.Tensor:
    """``[bs, K]`` int64: the lane (frontier row of depth ``step``) each new frontier node of depth ``step + 1`` grew
    from, read from the tree's frontier list after :meth:`DraftTree.step`."""
    shape = tree.shape
    first = shape.num_parents(step + 1)
    nodes = tree.parents[:bs, first : first + shape.top_k]
    return torch.div(nodes - shape.layer_start(step + 1), shape.top_k, rounding_mode="floor")


def advance_lane_masks(tree: DraftTree, lane_mask: torch.Tensor, bs: int, step: int, bits: torch.Tensor) -> None:
    """After draft step ``step``: each new frontier node sees its parent lane's slots and its own step-``step + 1``
    slot (in place on ``lane_mask``, ``[>= bs * K]`` int64)."""
    k = tree.shape.top_k
    masks = lane_mask[: bs * k].view(bs, k)
    masks.copy_(torch.gather(masks, 1, parent_lanes(tree, bs, step)) | bits[step])


def verify_window(
    tree: DraftTree, bs: int, depth: int, root_tokens: torch.Tensor, window: torch.Tensor, parents: torch.Tensor
) -> None:
    """Write the next verify window of a ``depth``-level tree: ``window`` ``[bs, N]`` int32 tokens (column 0 the root,
    ``root_tokens`` ``[bs]``) and ``parents`` ``[bs, N]`` int32 (``-1`` for the root)."""
    width = window.shape[1]
    tokens, parent_rows = tree.verify_rows(bs, depth, width - 1)
    window[:, 0] = root_tokens
    window[:, 1:] = tokens
    parents.copy_(parent_rows)
