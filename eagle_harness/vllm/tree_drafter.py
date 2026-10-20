# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The draft tree of a round on vLLM: growth with the draft model's row forwards, and the verify rows.

After the draft prefill (which yields the depth-1 candidates and the root hidden state, see
:class:`~eagle_harness.vllm.speculator.HarnessSpeculator`), step ``s`` expands the ``K`` frontier nodes of
depth ``s``: each row is one decode token of the draft layer at position ``c + s - 1`` (``c``: the committed
length) that writes its draft KV to a scratch slot past ``c`` and attends to the committed draft KV inside
the sliding window plus its own ancestors (:mod:`~eagle_harness.vllm.draft_attention`). Scores, frontier and
the tree itself are the core's :class:`~eagle_harness.core.draft_tree.DraftTree`, shared with the sglang
adapter. With ``min_gain`` set, a captured growth runs each step after the first inside a CUDA graph conditional
node and skips it when :mod:`~eagle_harness.vllm.depth_stop` says the tree is done: a step must pay its measured cost
when ``cost_tables`` price it, else ``min_gain`` (0: the verify rows are exactly those of the full-depth tree).
:meth:`TreeDrafter.select` turns the best nodes into the verify rows vLLM schedules: their tokens in node order (a
parent before its children) and, for each row, the row of its parent.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping

import torch

from eagle_harness.core.cost_table import BatchCostTable
from eagle_harness.core.draft_tree import DEPTH_DECAY, DraftTree, TreeShape
from eagle_harness.vllm.depth_stop import keep_growing, round_costs
from eagle_harness.vllm.draft_attention import tree_step_attention
from eagle_harness.vllm.draft_rows import draft_step_slots, store_rows_kv


class TreeDrafter:
    def __init__(
        self,
        model: Any,
        attn: Any,
        *,
        top_k: int,
        depth: int,
        max_reqs: int,
        device: torch.device,
        min_gain: float | None = None,
        cost_tables: Mapping[int, BatchCostTable] | None = None,
    ):
        self.model = model
        self.top_k = top_k
        self.depth = depth
        # Early stop of a captured growth (None: every round runs ``depth - 1`` steps).
        self.min_gain = min_gain
        self.cost_tables = cost_tables  # what a step costs against its round (None: it costs min_gain)
        self._go = torch.zeros((), dtype=torch.bool, device=device)
        self.depth_stats = torch.zeros(2, dtype=torch.int64, device=device)  # rounds, draft steps they ran
        # The fp32 decay of each depth, as ``DraftTree.decay`` rounds it.
        self._decay_of = [float(torch.tensor(DEPTH_DECAY**d, dtype=torch.float32)) for d in range(depth + 2)]
        config = model.config
        self.tree = DraftTree(
            TreeShape(top_k=top_k, max_depth=depth),
            max_reqs,
            config.hidden_size,
            model.model.layers[0].self_attn.qkv_proj.weight.dtype,
            device,
            max_verify=1,
            delta_slots=(),  # the work deltas are the cost model's input; a fixed depth does not read them
        )
        self.scale = attn.impl.scale
        self.window = attn.sliding_window or 0  # keys a row sees, itself included; 0: the whole context
        self.num_heads = attn.num_heads
        if attn.num_kv_heads <= 0 or attn.num_heads % attn.num_kv_heads:
            raise ValueError("draft query heads must be divisible by KV heads")
        self.head_dim = attn.head_size
        self.kv_view: torch.Tensor | None = None
        self.table_block_size = 0
        # CUDA graphs of the whole growth and selection, one per (batch size, count): the steps are device-only (no host
        # reads), so a round's ``depth - 1`` draft forwards and the choice of verify rows replay as one launch. ``graphs``: switched on by the speculator when
        # the engine runs with graphs.
        self.graphs = False
        self.device = device
        self._committed = torch.zeros(max_reqs, dtype=torch.int64, device=device)
        self._graphs: dict[tuple[int, int], Any] = {}
        # Growth graphs run serially and their outputs are consumed before the next call.
        # Reuse transient forward/logits storage across shapes instead of retaining it per batch size.
        self._graph_pool = None
        self._capture_stream = None
        self._conditional_stream = None

    def bind_cache(self, kv_cache: torch.Tensor, table_block_size: int) -> None:
        """The draft layer's KV cache ``[blocks, heads, slots, 2 * head_dim]``, viewed ``[blocks, slots, heads, 2 * D]``;
        ``table_block_size``: tokens per block of the block table (a multiple of the cache's own blocks)."""
        self.kv_view = kv_cache.transpose(1, 2)
        self.table_block_size = table_block_size

    # ------------------------------------------------------------------ growth
    def seed(self, num_reqs: int, probs: torch.Tensor, tokens: torch.Tensor, hidden: torch.Tensor) -> None:
        """Depth 1 from the draft prefill's candidates of each request's last committed row."""
        tree = self.tree
        tree.root_probs[:num_reqs] = probs
        tree.root_tokens[:num_reqs] = tokens
        tree.root_hidden[:num_reqs] = hidden
        tree.begin(num_reqs)

    def grow_and_select(
        self, num_reqs: int, count: int, committed: torch.Tensor, block_table: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run draft steps ``1 .. depth - 1``, then :meth:`select` ``count`` verify rows.

        ``committed``: ``[num_reqs]`` int64; ``block_table``: the draft KV group's (the batch's rows of the
        persistent table: a graph keeps its address). With graphs, the whole growth and the selection replay as
        one launch per (batch size, count); the returned tensors are the graph's, valid until the next call.
        """
        if not self.graphs:
            return self._grow_and_select(num_reqs, count, committed, block_table)
        key = (num_reqs, count)
        entry = self._graphs.get(key)
        if entry is None:  # the first round of a shape runs eagerly: it compiles the kernels
            self._graphs[key] = False
            return self._grow_and_select(num_reqs, count, committed, block_table)
        self._committed[:num_reqs].copy_(committed)
        if entry is False:
            from vllm.distributed.parallel_state import graph_capture

            if self._graph_pool is None:
                self._graph_pool = torch.cuda.graph_pool_handle()
                self._capture_stream = torch.cuda.Stream(device=self.device)
                self._conditional_stream = torch.cuda.Stream(device=self.device)
            assert self._capture_stream is not None
            graph = torch.cuda.CUDAGraph()
            with graph_capture(device=self.device) as capture:
                # The allocator reuses a pool's blocks on the same capture stream.
                self._capture_stream.wait_stream(capture.stream)
                with torch.cuda.graph(graph, pool=self._graph_pool, stream=self._capture_stream):
                    stop = graph if self.stops_early(count) else None
                    out = self._grow_and_select(num_reqs, count, self._committed[:num_reqs], block_table, stop)
            entry = self._graphs[key] = (graph, block_table.data_ptr(), out)
        graph, address, out = entry
        if block_table.data_ptr() != address:  # not the table the graph captured
            return self._grow_and_select(num_reqs, count, committed, block_table)
        graph.replay()
        return out

    def _grow_and_select(
        self, num_reqs: int, count: int, committed: torch.Tensor, block_table: torch.Tensor, stop: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.grow(num_reqs, committed, block_table, count=count, stop=stop)
        return self.select(num_reqs, count)

    def stops_early(self, count: int) -> bool:
        """Whether a captured growth for ``count`` verify rows skips the steps it does not need (with ``min_gain``)."""
        return self.min_gain is not None and self._first_decision(count) is not None

    def _first_decision(self, count: int) -> int | None:
        """The first step after which the tree has ``count`` nodes to rank against, if a later step remains."""
        return next((s for s in range(1, self.depth - 1) if self.tree.shape.num_nodes(s + 1) >= count), None)

    def grow(
        self, num_reqs: int, committed: torch.Tensor, block_table: torch.Tensor, *, count: int = 0, stop: Any = None
    ) -> None:
        """Draft steps ``1 .. depth - 1`` (eager; :meth:`grow_and_select` is the serving path).

        ``stop``: the CUDA graph being captured; each step after the tree first holds ``count`` nodes is then captured
        into a conditional node that runs only while :func:`~eagle_harness.vllm.depth_stop.keep_growing` keeps
        ``count`` verify rows open.
        """
        steps, k = self.depth - 1, self.top_k
        if steps == 0:
            return
        first = self._first_decision(count) if stop is not None else None
        if first is None:
            stop = None
        else:  # the layers a skipped step leaves out must not rank with an older round's scores
            self.tree.scores[:num_reqs, self.tree.shape.layer_start(first + 2) :] = 0
        # The draft KV slots and positions of every step at once, one launch per round.
        slots, blocks, offsets, positions = draft_step_slots(
            block_table,
            committed,
            steps=steps,
            top_k=k,
            table_block=self.table_block_size,
            cache_block=self.kv_view.shape[1],
        )
        for step in range(1, self.depth):
            args = (slots[:, step - 1], blocks[step - 1], offsets[step - 1], positions[step - 1])
            if stop is None:
                self._step(num_reqs, step, committed, block_table, *args)
                continue
            with self._capture_if(stop if step > first else None):
                self._step(num_reqs, step, committed, block_table, *args)
                if first <= step < steps:
                    self._decide(num_reqs, step, count, first)

    @contextmanager
    def _capture_if(self, graph):
        if graph is None:
            yield
            return
        graph.begin_capture_to_if_node(self._go)
        child = torch.cuda.current_stream(self.device)
        # PyTorch creates a fresh child stream per node. Allocator blocks are stream-keyed,
        # so route serial conditional work through one stream to reuse the shared pool.
        assert self._conditional_stream is not None
        self._conditional_stream.wait_stream(child)
        try:
            with torch.cuda.stream(self._conditional_stream):
                yield
        finally:
            child.wait_stream(self._conditional_stream)
            graph.end_capture_to_conditional_node()

    def _decide(self, bs: int, step: int, count: int, first: int) -> None:
        """Whether step ``step + 1`` runs, into ``self._go`` (``first``: the first deciding step)."""
        tree = self.tree
        costs = round_costs(self.cost_tables, bs, self.top_k, count)
        keep_growing(
            self._go,
            self.depth_stats,
            tree.scores[:bs],
            tree.decay,
            tree.front_scores[:bs],
            count,
            self._decay_of[step + 2],
            self.min_gain,
            (tree.shape.layer_start(step), tree.shape.layer_start(step + 1)),
            steps_before=first if step == first else 0,
            step_share=costs.step_share(step) if costs else 0.0,
        )

    def mean_steps(self) -> float | None:
        """Draft steps per captured round with early stop so far (a host read; None before the first)."""
        rounds, steps = self.depth_stats.tolist()
        return steps / rounds if rounds else None

    def _step(
        self,
        bs: int,
        step: int,
        committed: torch.Tensor,
        block_table: torch.Tensor,
        slots: torch.Tensor,
        blocks: torch.Tensor,
        offsets: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Draft step ``step``: ``slots`` ``[bs, K]`` the draft KV slots its rows write (``blocks`` / ``offsets`` in the
        cache's own blocks, flat), ``positions`` ``[bs * K]``."""
        tree = self.tree
        rows = bs * self.top_k
        tree.set_step_slots(bs, step, slots)

        def attend(q: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            heads, dim = self.num_heads, self.head_dim
            store_rows_kv(self.kv_view, key, value, blocks, offsets)
            out = tree_step_attention(
                q.view(rows, heads, dim),
                self.kv_view,
                block_table,
                self.table_block_size,
                committed,
                tree.ancestors[:bs],
                self.scale,
                step=step,
                window=self.window,
            )
            return out.view(rows, heads * dim)

        normed, hidden = self.model.forward_rows(tree.front_tokens[:rows], positions, tree.front_hidden[:rows], attend)
        probs, tokens = self.model.candidate_topk(normed, self.top_k)
        tree.step(bs, step, probs, tokens, hidden)

    # ------------------------------------------------------------------ verify rows
    def select(self, num_reqs: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The best ``count`` nodes of the grown tree as verify rows: ``(tokens [bs, count], parent_rows [bs, count + 1])``
        (:meth:`~eagle_harness.core.draft_tree.DraftTree.verify_rows`: a parent's row before its children's, row 0 the
        root)."""
        return self.tree.verify_rows(num_reqs, self.depth, count)
