# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Draft steps of a round, replayed as one CUDA graph per (batch size, step).

Step 0 seeds the tree (:meth:`DraftTree.begin`). Step ``i`` runs the draft model on the depth-``i`` frontier (``K`` rows per
request) with tree attention, takes each row's top-``K`` candidates from the
EAGLE-4 candidate head (no full-vocab softmax), and adds the children to the
:class:`DraftTree`. Every input lives in a static buffer (the tree's frontier,
the batch's ``seq_lens`` / ``req_pool_indices``), so one captured graph per
step serves every round; the host only decides whether to run the next one.
The graphs of the cost model's launch depths also copy the step's work deltas
to page-locked host memory, which the cost model reads one step later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, TypeVar

import torch

from eagle_harness.sglang.draft_head import draft_topk

from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

from eagle_harness.core.draft_tree import DraftTree
from eagle_harness.sglang.graph_pool import graph_memory_pool
from eagle_harness.sglang.row_attention import DraftRowAttention

logger = logging.getLogger(__name__)

K = TypeVar("K")


@dataclass
class DraftRowsInput:
    """The ``spec_info`` the EAGLE-4 draft model reads: the rows' input hidden states."""

    hidden_states: torch.Tensor


def forward_draft_rows(
    model,
    attention: DraftRowAttention,
    *,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    hidden_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The EAGLE-4 draft layer on independent rows whose key lists ``attention`` holds.

    Returns ``(normed, hidden)``: the final-norm output (candidate-head input)
    and the hidden state the next draft forward takes as input.
    """
    rows = input_ids.shape[0]
    unread = attention.unread_rows[:rows]
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=rows,
        input_ids=input_ids,
        req_pool_indices=unread,
        seq_lens=unread,
        out_cache_loc=out_cache_loc.to(torch.int64),
        seq_lens_sum=0,
        positions=positions,
        spec_info=DraftRowsInput(hidden_states=hidden_states),
    )
    with forward_context(ForwardContext(attn_backend=attention)):
        normed, aux = model.model(input_ids, positions, forward_batch)
    return normed, aux[0]


def capture_graphs(keys: Sequence[K], body: Callable[[K], None]) -> Dict[K, torch.cuda.CUDAGraph]:
    """One CUDA graph of ``body(key)`` per key, in order, in sglang's global graph memory pool."""
    pool = graph_memory_pool()
    graphs: Dict[K, torch.cuda.CUDAGraph] = {}
    with graph_capture() as ctx:
        stream = ctx.stream
        for key in keys:
            with torch.cuda.stream(stream):
                for _ in range(2):  # warm up (Triton compiles, allocator settles)
                    body(key)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                body(key)
            graphs[key] = graph
    torch.cuda.synchronize()
    return graphs


class DraftSteps:
    """Owns the per-step graphs of the draft tree."""

    def __init__(self, draft_runner, tree: DraftTree, attention: DraftRowAttention, *, sampled_seed: Optional[int] = None):
        """``sampled_seed``: the Philox seed of the recorded-q sampled tree (the same on every TP rank); None builds
        no sampled tree (the engine's verify cannot sample it)."""
        self.runner = draft_runner
        self.model = draft_runner.model
        self.tree = tree
        self.attention = attention
        self.req_to_token = draft_runner.req_to_token_pool.req_to_token
        device = tree.scores.device
        self.seq_lens = torch.zeros(tree.max_bs, dtype=torch.int64, device=device)
        self.req_pool_indices = torch.zeros(tree.max_bs, dtype=torch.int64, device=device)
        rows = tree.max_bs * tree.shape.top_k
        self.slots = torch.zeros(rows, dtype=torch.int64, device=device)  # the step's draft KV slots, one per row
        self.positions = torch.zeros(rows, dtype=torch.int64, device=device)
        # Host mirror of tree.deltas, written by the graphs of the tree's delta slots (0: begin, d: step d).
        self.host_deltas = torch.zeros(tree.deltas.shape, dtype=torch.float32, pin_memory=True)
        self._graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.sampled = None
        if sampled_seed is not None and hasattr(self.model, "proposal_candidates"):
            from eagle_harness.sglang.sampled_tree import SampledTree

            self.sampled = SampledTree(
                tree, self.model.num_candidates, sampled_seed, req_index=self.req_pool_indices, rows=self.req_to_token.shape[0]
            )
        self._sampled_graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}

    @property
    def max_step(self) -> int:
        """Deepest draft step of the tree."""
        return self.tree.shape.max_depth - 1

    def load_batch(self, bs: int, seq_lens: torch.Tensor, req_pool_indices: torch.Tensor) -> None:
        """Stage the round's batch into the static buffers."""
        self.seq_lens[:bs].copy_(seq_lens, non_blocking=True)
        self.req_pool_indices[:bs].copy_(req_pool_indices, non_blocking=True)

    def begin(self, bs: int, probs: torch.Tensor, tokens: torch.Tensor, hidden: torch.Tensor) -> None:
        """Seed the round's tree from the draft extend's top-``K`` and hidden state."""
        self.tree.stage_root(bs, probs, tokens, hidden)
        self.run(bs, 0)

    def begin_body(self, bs: int) -> None:
        """:meth:`begin`'s device work on the root buffers (the tree's begin and its work deltas), for a
        caller's own graph."""
        self._body((bs, 0))

    def run(self, bs: int, step: int) -> None:
        """Replay draft step ``step`` (0: :meth:`DraftTree.begin`)."""
        graphs = self._sampled_graphs if self.sampled is not None and self.sampled.active else self._graphs
        graph = graphs.get((bs, step))
        if graph is not None:
            graph.replay()
        else:
            self._body((bs, step))

    def capture(self, batch_sizes: Sequence[int]) -> None:
        """Capture begin and every step for each batch size (largest first)."""
        keys = [(bs, step) for bs in sorted(batch_sizes, reverse=True) for step in range(self.max_step, -1, -1)]
        self._graphs = capture_graphs(keys, self._body)
        if self.sampled is not None:
            self.sampled.active = True
            try:
                self._sampled_graphs = capture_graphs(keys, self._body)
            finally:
                self.sampled.active = False
        logger.info("eagle-harness: captured %d draft-step graphs (bs %s)", len(self._graphs), sorted(batch_sizes))

    # ------------------------------------------------------------------ body
    def _body(self, key: Tuple[int, int]) -> None:
        bs, step = key
        if step == 0:
            self.tree.begin(bs)
        else:
            self._step(bs, step)
        if step in self.tree.delta_slots:
            self.host_deltas[step].copy_(self.tree.deltas[step], non_blocking=True)

    def _step(self, bs: int, step: int) -> None:
        tree = self.tree
        k = tree.shape.top_k
        rows = bs * k
        seq_lens = self.seq_lens[:bs]
        req = self.req_pool_indices[:bs]
        slots, positions = self.slots[:rows], self.positions[:rows]
        self.attention.build_tree_rows(tree.ancestors[:bs], seq_lens, req, slots, positions, bs=bs, step=step)
        normed, hidden = forward_draft_rows(
            self.model,
            self.attention,
            input_ids=tree.front_tokens[:rows],
            positions=positions,
            out_cache_loc=slots,
            hidden_states=tree.front_hidden[:rows],
        )
        if self.sampled is not None and self.sampled.active:
            probs, tokens = self.sampled.draw(self.model, normed, bs, step)
        else:
            probs, tokens = draft_topk(self.model, normed, k)
        tree.step(bs, step, probs, tokens, hidden)
