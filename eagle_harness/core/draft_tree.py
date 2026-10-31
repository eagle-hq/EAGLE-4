# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Draft tree of one decode round: node scores, frontier, verify selection.

Pure torch on fixed-shape tensors, so every per-depth update can be captured
in a CUDA graph. Tree layers, cumulative draft state and work deltas use a fixed
expansion width ``K``, which is the width the cost model always chooses at batch size 1:

- The root (the last accepted token) has score ``ROOT_SCORE``; its ``K``
  children come from the draft extend's top-k and form depth 1.
- A draft step expands the ``K`` frontier nodes of depth ``d``: each yields its
  top-``K`` children, scored ``parent score * child probability``. All ``K*K``
  children join the tree; the best ``K`` of them are the frontier of depth
  ``d + 1``.
- The best-score set (undecayed, root included) and the newest layer's top
  ``K`` give the work deltas the cost model's stop rule reads, for every
  verify budget at once.
- The verify set is the best nodes by depth-decayed score (``DEPTH_DECAY**d``
  for ``d >= 2``), which also orders a parent before an equally scored child.

Node numbering is sglang's EAGLE tree layout (``organize_draft_results``,
``build_tree_kernel_efficient``): depth 1 at ``[0, K)``, the children of depth
``d`` at ``K + (d - 1) * K * K``, parents as ``arange(-1, K)`` followed by the
global index of each later frontier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, NamedTuple, Optional, Tuple

import torch

ROOT_SCORE = 1.001
DEPTH_DECAY = 0.9999


def _next_power_of_2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


@dataclass(frozen=True)
class TreeShape:
    """Static tree dimensions.

    ``top_k``: children per expanded node and expanded nodes per depth.
    ``max_depth``: deepest node depth (``--speculative-num-steps``); a round
    runs at most ``max_depth - 1`` draft steps.
    ``work_cap``: width of the best-score set (>= any verify budget the cost
    model asks the deltas for).
    """

    top_k: int
    max_depth: int
    work_cap: int = 64

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.max_depth < 2:
            raise ValueError(f"tree needs top_k >= 1 and max_depth >= 2, got {self}")

    def layer_start(self, depth: int) -> int:
        """First node index of ``depth`` (1-based)."""
        if depth == 1:
            return 0
        return self.top_k + (depth - 2) * self.top_k * self.top_k

    def num_nodes(self, depth: int) -> int:
        """Nodes of depths ``1..depth``."""
        return self.layer_start(depth + 1)

    def num_parents(self, depth: int) -> int:
        """Length of sglang's ``parent_list`` for a tree of ``depth`` levels."""
        return 0 if depth < 2 else (self.top_k + 1) + (depth - 2) * self.top_k

    def count_before_step(self, step: int) -> int:
        """Tree size (root included) before step ``step`` adds its layer."""
        return 1 + self.top_k + (step - 1) * self.top_k * self.top_k

    @property
    def capacity(self) -> int:
        """Node slots of the deepest tree."""
        return self.num_nodes(self.max_depth)

    @property
    def parent_capacity(self) -> int:
        """Parent slots of the deepest tree."""
        return (self.top_k + 1) + (self.max_depth - 1) * self.top_k

    def decay(self, device: torch.device) -> torch.Tensor:
        """Per-node depth decay, ``[capacity]`` fp32."""
        out = torch.ones(self.capacity, dtype=torch.float32)
        for depth in range(2, self.max_depth + 1):
            out[self.layer_start(depth) : self.layer_start(depth + 1)] = DEPTH_DECAY**depth
        return out.to(device)


class VerifySelection(NamedTuple):
    """sglang ``build_tree_kernel_efficient`` inputs for ``u - 1`` verified nodes."""

    parent_list: torch.Tensor  # [bs, num_parents(depth)] int64
    top_scores_index: torch.Tensor  # [bs, u - 1] int64, ascending
    draft_tokens: torch.Tensor  # [bs, u - 1] int64


class DraftTree:
    """Per-round tree state for up to ``max_bs`` requests (all buffers static).

    ``root_probs`` / ``root_tokens`` / ``root_hidden`` hold the round's input
    (the draft extend's top-``K`` and hidden state); :meth:`begin` seeds
    depth 1 from them. After :meth:`begin` (slot 0) and after step ``d``
    (slot ``d``), for every slot in ``delta_slots`` (default: all),
    ``deltas[slot]`` holds the work deltas for every verify budget
    ``1..max_verify`` (row ``v - 1``), so the cost model's input is computed
    inside the step's graph. ``max_verify`` must cover every budget the cost
    model asks for (:attr:`CostModel.max_history_verify_num`), which can
    exceed the verify width.
    """

    def __init__(
        self,
        shape: TreeShape,
        max_bs: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        max_verify: int,
        delta_slots: Optional[Iterable[int]] = None,
    ):
        k = shape.top_k
        if not 1 <= max_verify <= shape.work_cap:
            raise ValueError(f"max_verify={max_verify} must be in 1..work_cap={shape.work_cap}")
        slots = frozenset(range(shape.max_depth) if delta_slots is None else delta_slots)
        if not slots <= set(range(shape.max_depth)):
            raise ValueError(f"delta_slots {sorted(slots)} outside 0..{shape.max_depth - 1}")
        self.delta_slots = slots
        self.shape = shape
        self.max_bs = max_bs
        self.max_verify = max_verify
        f32, i64 = torch.float32, torch.int64
        self.root_probs = torch.zeros(max_bs, k, dtype=f32, device=device)
        self.root_tokens = torch.zeros(max_bs, k, dtype=i64, device=device)
        self.root_hidden = torch.zeros(max_bs, hidden_size, dtype=dtype, device=device)
        self.scores = torch.zeros(max_bs, shape.capacity, dtype=f32, device=device)
        self.tokens = torch.zeros(max_bs, shape.capacity, dtype=i64, device=device)
        self.parents = torch.zeros(max_bs, shape.parent_capacity, dtype=i64, device=device)
        self.parents[:, : k + 1] = torch.arange(-1, k, dtype=i64, device=device)
        self.front_scores = torch.zeros(max_bs, k, dtype=f32, device=device)
        self.front_tokens = torch.zeros(max_bs * k, dtype=i64, device=device)
        self.front_hidden = torch.zeros(max_bs * k, hidden_size, dtype=dtype, device=device)
        # Draft KV slot of each frontier node's ancestors (depth d at column d - 1).
        self.ancestors = torch.zeros(max_bs, k, shape.max_depth, dtype=torch.int32, device=device)
        self.best = torch.zeros(max_bs, shape.work_cap, dtype=f32, device=device)
        self.best_prev = torch.zeros_like(self.best)
        self.layer_top = torch.zeros(max_bs, k, dtype=f32, device=device)
        self.decay = shape.decay(device)
        self.deltas = torch.zeros(shape.max_depth, max_verify, k + 1, dtype=f32, device=device)
        self.delta_rows = torch.zeros(max_bs, max_verify, k + 1, dtype=f32, device=device)  # per-row deltas (kernel)
        self._extra_mask = self._delta_mask(device)
        # verify_rows: the select kernel's per-node row scratch, and the torch path's layer tables
        self._row_of = torch.empty(max_bs * _next_power_of_2(shape.capacity), dtype=torch.int32, device=device)
        depths = range(1, shape.max_depth + 1)
        self._layer_bounds = torch.tensor([shape.layer_start(d + 1) for d in depths], dtype=i64, device=device)
        self._layer_starts = torch.tensor([shape.layer_start(d) for d in depths], dtype=i64, device=device)
        # column of depth d's frontier list in ``parents`` (entry 0 is the root)
        self._frontier_base = torch.tensor(
            [shape.num_parents(d) if d > 1 else 1 for d in depths], dtype=i64, device=device
        )

    # ------------------------------------------------------------------ round
    def stage_root(self, bs: int, probs: torch.Tensor, tokens: torch.Tensor, hidden: torch.Tensor) -> None:
        """Copy the round's input into the root buffers unless it already lives there."""
        for dst, src in ((self.root_probs, probs), (self.root_tokens, tokens), (self.root_hidden, hidden)):
            if src.data_ptr() != dst.data_ptr():
                dst[:bs].copy_(src)

    def begin(self, bs: int) -> None:
        """Seed depth 1 from the root buffers (``[bs, K]`` probabilities / ids, ``[bs, H]`` hidden)
        (one Triton launch on CUDA, :meth:`begin_reference` elsewhere)."""
        if self.root_probs.is_cuda:
            from eagle_harness.kernels.tree_step import tree_begin

            tree_begin(self, bs, ROOT_SCORE)
        else:
            self.begin_reference(bs)
        if 0 in self.delta_slots:
            self._store_deltas(bs, 0)

    def begin_reference(self, bs: int) -> None:
        """:meth:`begin` without the work deltas, in torch (CPU, and the reference of the CUDA kernel)."""
        k = self.shape.top_k
        probs, order = torch.sort(self.root_probs[:bs], dim=-1, descending=True, stable=True)  # ties: lower node
        tokens = torch.gather(self.root_tokens[:bs], 1, order)
        self.scores[:bs, :k] = probs
        self.tokens[:bs, :k] = tokens
        self.front_scores[:bs] = probs
        self.front_tokens[: bs * k] = tokens.reshape(-1)
        self.front_hidden[: bs * k] = self.root_hidden[:bs].repeat_interleave(k, dim=0)
        self.best_prev[:bs].zero_()
        self.best_prev[:bs, 0] = ROOT_SCORE
        self.best[:bs].zero_()
        self.best[:bs, 0] = ROOT_SCORE
        self.best[:bs, 1 : 1 + k] = probs
        self.layer_top[:bs] = probs

    def set_step_slots(self, bs: int, step: int, slots: torch.Tensor) -> None:
        """Record the draft KV slots ``[bs, K]`` the depth-``step`` frontier writes."""
        self.ancestors[:bs, :, step - 1] = slots.to(torch.int32)

    def keep_rows(self, rows: torch.Tensor) -> None:
        """Keep the requests at batch positions ``rows`` (in that order) as rows ``0..len(rows)-1``: a drafted tree
        then serves the batch that remains once some of its requests left. Rows are independent, so the kept rows
        are the tree those requests alone would have grown."""
        n, k = rows.numel(), self.shape.top_k
        per_request = (self.root_probs, self.root_tokens, self.root_hidden, self.scores, self.tokens, self.parents,
                       self.front_scores, self.ancestors, self.best, self.best_prev, self.layer_top, self.delta_rows)
        per_node = (self.front_tokens.view(self.max_bs, k), self.front_hidden.view(self.max_bs, k, -1))
        for buf in per_request + per_node:
            buf[:n] = buf.index_select(0, rows)

    def step(self, bs: int, step: int, probs: torch.Tensor, tokens: torch.Tensor, hidden: torch.Tensor) -> None:
        """Add the children of the depth-``step`` frontier (``1 <= step < max_depth``).

        ``probs`` / ``tokens``: ``[bs * K, K]`` top-k of each frontier row;
        ``hidden``: ``[bs * K, H]`` the rows' draft hidden states.
        """
        if not 1 <= step < self.shape.max_depth:
            raise ValueError(f"step {step} outside 1..{self.shape.max_depth - 1}")
        if probs.is_cuda:
            from eagle_harness.kernels.tree_step import tree_step

            tree_step(self, bs, step, probs, tokens, hidden, deltas=step in self.delta_slots)
        else:
            self.step_reference(bs, step, probs, tokens, hidden)
            if step in self.delta_slots:
                self._store_deltas(bs, step)

    def step_reference(
        self, bs: int, step: int, probs: torch.Tensor, tokens: torch.Tensor, hidden: torch.Tensor
    ) -> None:
        """:meth:`step` without the work deltas, in torch (CPU, and the reference of the CUDA kernel)."""
        shape = self.shape
        k = shape.top_k
        layer = (self.front_scores[:bs].unsqueeze(-1) * probs.float().view(bs, k, k)).view(bs, k * k)
        start = shape.layer_start(step + 1)
        self.scores[:bs, start : start + k * k] = layer
        layer_tokens = tokens.long().view(bs, k * k)
        self.tokens[:bs, start : start + k * k] = layer_tokens
        top, pick = torch.topk(layer, k, dim=-1)
        pstart = shape.num_parents(step + 1)
        self.parents[:bs, pstart : pstart + k] = pick + start
        lane = torch.div(pick, k, rounding_mode="floor")
        rows = (lane + torch.arange(bs, device=lane.device).unsqueeze(-1) * k).view(-1)
        self.front_scores[:bs] = top
        self.front_tokens[: bs * k] = torch.gather(layer_tokens, 1, pick).view(-1)
        self.front_hidden[: bs * k] = hidden.view(bs * k, -1).index_select(0, rows)
        self.ancestors[:bs] = self.ancestors[:bs].view(bs * k, -1).index_select(0, rows).view(bs, k, -1)
        self.best_prev[:bs] = self.best[:bs]
        merged = torch.cat([self.best[:bs], layer], dim=-1)
        self.best[:bs] = torch.topk(merged, shape.work_cap, dim=-1).values
        self.layer_top[:bs] = top

    # ----------------------------------------------------------- cost model
    def _delta_mask(self, device: torch.device) -> torch.Tensor:
        """``[K + 1, K]``: row ``j`` keeps the layer's best ``j`` scores."""
        k = self.shape.top_k
        return (torch.arange(k).view(1, -1) < torch.arange(k + 1).view(-1, 1)).float().to(device)

    def _store_deltas(self, bs: int, slot: int) -> None:
        """Compute work deltas on the newest layer, for every verify budget ``v`` (one Triton
        launch on CUDA, :meth:`_store_deltas_reference` elsewhere)."""
        if self.deltas.is_cuda:
            from eagle_harness.kernels.tree_step import work_deltas

            work_deltas(self, bs, slot)
        else:
            self._store_deltas_reference(bs, slot)

    def _store_deltas_reference(self, bs: int, slot: int) -> None:
        """The torch version of :meth:`_store_deltas`.

        Sum the top ``min(n0 + j, v)`` of the merge of the best set
        before the layer (``n0 = min(v, count)`` nodes, ``count``: tree size
        before it) and the layer's best ``j``. The best set is sorted, so for
        ``v <= count`` that is the top ``v`` of the merge with the whole best
        set, and for ``v > count`` the entries past ``count`` are zeros (scores
        are non-negative). Either way ``deltas[slot, v - 1, j]`` is the batch
        mean of the ``v``-th prefix sum of ``sort(best_prev ++ layer_top[:j])``,
        one sort for every budget.
        """
        k = self.shape.top_k
        extra = self.layer_top[:bs, None, :] * self._extra_mask.view(1, k + 1, k)  # [bs, K + 1, K]
        best = self.best_prev[:bs, None, :].expand(bs, k + 1, self.shape.work_cap)
        merged = torch.sort(torch.cat([best, extra], dim=-1), dim=-1, descending=True).values
        sums = merged[..., : self.max_verify].cumsum(dim=-1)  # [bs, K + 1, V]
        self.deltas[slot] = sums.mean(dim=0).transpose(0, 1)

    # --------------------------------------------------------------- verify
    def rank(self, bs: int, depth: int, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """The best ``count`` nodes of a ``depth``-level tree by decayed score, best first (ties: the lower
        node), ``[bs, count]``, and the cumulative verify scores of their prefixes, root first,
        ``[bs, count + 1]`` (the cost model's lagged input). One Triton launch on CUDA,
        :meth:`rank_reference` elsewhere."""
        count = min(count, self.shape.num_nodes(depth))
        if self.scores.is_cuda:
            from eagle_harness.kernels.tree_step import tree_rank

            return tree_rank(self, bs, depth, count, ROOT_SCORE)
        return self.rank_reference(bs, depth, count)

    def rank_reference(self, bs: int, depth: int, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """:meth:`rank` in torch (CPU, and the reference of the CUDA kernel)."""
        n = self.shape.num_nodes(depth)
        decayed = self.scores[:bs, :n] * self.decay[:n]
        top, ranked = torch.sort(decayed, dim=-1, descending=True, stable=True)
        root = top.new_full((bs, 1), ROOT_SCORE)
        return ranked[:, :count], torch.cumsum(torch.cat([root, top[:, :count]], dim=-1), dim=-1)

    def select(self, bs: int, depth: int, ranked: torch.Tensor, verify_tokens: int) -> VerifySelection:
        """The best ``verify_tokens - 1`` nodes (plus the root) in sglang's layout (the node sort and token
        gather in one Triton launch on CUDA)."""
        chosen = ranked[:, : verify_tokens - 1]
        if chosen.is_cuda:
            from eagle_harness.kernels.tree_step import tree_select

            index, tokens = tree_select(chosen, self.tokens)
        else:
            index = torch.sort(chosen, dim=-1).values
            tokens = torch.gather(self.tokens[:bs], 1, index)
        return VerifySelection(
            # sglang's tree kernel reads a dense [bs, parents] array; the column slice of a batch is strided
            parent_list=self.parents[:bs, : self.shape.num_parents(depth)].contiguous(),
            top_scores_index=index,
            draft_tokens=tokens,
        )

    def verify_rows(self, bs: int, depth: int, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """The best ``count`` nodes of a ``depth``-level tree as verify rows: ``(tokens [bs, count] int64,
        parent_rows [bs, count + 1] int32)``.

        Rows follow the node numbering, which puts a parent before its children, so any prefix of the rows is a tree.
        Row 0 of ``parent_rows`` is the root (-1); the node of row ``r`` has the parent row ``parent_rows[:, r]`` (0: the
        root). Ties at the last row go to the lowest node, as in :meth:`rank`. One launch on CUDA
        (:func:`~eagle_harness.kernels.tree_step.select_rows`), :meth:`verify_rows_reference` elsewhere.
        """
        n = self.shape.num_nodes(depth)
        if self.scores.is_cuda:
            from eagle_harness.kernels.tree_step import select_rows

            n_block = _next_power_of_2(n)
            return select_rows(
                self.scores[:bs, :n],
                self.decay[:n],
                self.tokens[:bs, :n],
                self.parents[:bs],
                count,
                self.shape.top_k,
                self._row_of[: bs * n_block].view(bs, n_block),
            )
        return self.verify_rows_reference(bs, depth, count)

    def verify_rows_reference(self, bs: int, depth: int, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """:meth:`verify_rows` in torch (CPU, and the reference of the kernel)."""
        ranked, _ = self.rank_reference(bs, depth, count)
        index = torch.sort(ranked, dim=-1).values
        tokens = torch.gather(self.tokens[:bs], 1, index)
        layer = torch.bucketize(index, self._layer_bounds, right=True)  # 0-based: depth 1 is 0
        local = index - self._layer_starts[layer]
        column = self._frontier_base[(layer - 1).clamp(min=0)] + local // self.shape.top_k
        parent_node = torch.where(layer == 0, -1, self.parents[:bs].gather(1, column))
        is_parent = index[:, None, :] == parent_node[:, :, None]  # [bs, child, parent]
        parent_row = torch.where(is_parent.any(-1), is_parent.int().argmax(-1) + 1, 0)
        root = parent_row.new_full((bs, 1), -1)
        return tokens, torch.cat([root, parent_row], dim=1).int()
