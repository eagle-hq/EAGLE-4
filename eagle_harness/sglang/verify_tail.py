# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The device work between a round's last draft step and its verify forward, as small CUDA graphs.

Once the cost model stops the tree, a round ranks the nodes, scores the verify
sets, selects the best ``width - 1`` and builds sglang's verify input (tree
mask, positions, retrieve tables): about 40 tiny kernels launched from Python,
a few hundred microseconds of host time while the GPU idles. Their inputs are
static buffers (the tree, the staged ``seq_lens`` and bonus tokens), so one
graph per ``(batch size, depth)`` for the ranking and one per ``(batch size,
depth, width)`` for the selection and the build replay them with a launch
each. When the width is known before the ranking (the cost model reads the
previous round's scores), :meth:`VerifyTail.ranked_verify_input` replays the
ranking, the hand-off of its scores and the build as one graph. A graph is
captured the first time its key occurs.
"""

from __future__ import annotations

from typing import Callable, Dict, NamedTuple, Tuple

import torch

from sglang.srt.speculative.eagle_info import EagleVerifyInput

from eagle_harness.core.draft_tree import DraftTree, VerifySelection
from eagle_harness.sglang.graph_pool import graph_memory_pool

# (batch size, tree selection, num_steps, num_draft_tokens) -> sglang's verify input.
BuildInput = Callable[[int, VerifySelection, int, int], EagleVerifyInput]


class Ranking(NamedTuple):
    """A tree's nodes ranked by score, with the cumulative verify scores."""

    ranked: torch.Tensor  # [bs, count] node indices, best first
    scores: torch.Tensor  # [count + 1] cumulative verify scores (root first), batch mean


def _capture(body: Callable[[], object]):
    """Run ``body`` once eagerly (Triton compiles), then capture it; returns ``(graph, outputs)``."""
    body()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=graph_memory_pool(), capture_error_mode="thread_local"):
        outputs = body()
    return graph, outputs


class VerifyTail:
    """Graph-replayed ranking and verify-input build of a drafted tree."""

    def __init__(self, tree: DraftTree, build_input: BuildInput):
        self.tree = tree
        self._build_input = build_input
        self._rankings: Dict[Tuple[int, int], Tuple[torch.cuda.CUDAGraph, Ranking]] = {}
        self._inputs: Dict[Tuple[int, int, int], Tuple[torch.cuda.CUDAGraph, EagleVerifyInput]] = {}
        self._ranked_inputs: Dict[Tuple[int, int, int, int], Tuple[torch.cuda.CUDAGraph, EagleVerifyInput]] = {}

    def rank(self, bs: int, depth: int, count: int) -> Ranking:
        """The best ``count`` nodes of the ``depth``-level tree and the cumulative scores of their prefixes."""
        key = (bs, depth)
        if key not in self._rankings:
            self._rankings[key] = _capture(lambda: self._rank(bs, depth, count))
        graph, ranking = self._rankings[key]
        graph.replay()
        return ranking

    def verify_input(self, bs: int, depth: int, width: int, ranking: Ranking) -> EagleVerifyInput:
        """sglang's verify input of the best ``width - 1`` ranked nodes plus the root."""
        key = (bs, depth, width)
        if key not in self._inputs:
            self._inputs[key] = _capture(lambda: self._select(bs, depth, width, ranking))
        graph, verify_input = self._inputs[key]
        graph.replay()
        return verify_input

    def ranked_verify_input(
        self, bs: int, depth: int, count: int, width: int, on_rank: Callable[[Ranking], None]
    ) -> EagleVerifyInput:
        """:meth:`rank`, ``on_rank(ranking)`` (device work, e.g. copying the scores out) and
        :meth:`verify_input` in one graph replay."""
        key = (bs, depth, count, width)
        if key not in self._ranked_inputs:

            def body() -> EagleVerifyInput:
                ranking = self._rank(bs, depth, count)
                on_rank(ranking)
                return self._select(bs, depth, width, ranking)

            self._ranked_inputs[key] = _capture(body)
        graph, verify_input = self._ranked_inputs[key]
        graph.replay()
        return verify_input

    def _rank(self, bs: int, depth: int, count: int) -> Ranking:
        ranked, scores = self.tree.rank(bs, depth, count)
        return Ranking(ranked, scores[0] if bs == 1 else scores.mean(dim=0))

    def _select(self, bs: int, depth: int, width: int, ranking: Ranking) -> EagleVerifyInput:
        # The verify set is ancestor-closed (a parent outranks its children), so its deepest node is at
        # most width - 1 deep; sglang sizes the accept path by it.
        steps = min(depth, width - 1)
        selection = self.tree.select(bs, steps, ranking.ranked, width)
        return self._build_input(bs, selection, steps, width)
