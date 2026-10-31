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
ranking, the hand-off of its scores and the build as one graph.

A key runs eagerly until it has occurred :data:`CAPTURE_AFTER` times, then is
captured. A capture records and instantiates the whole tail and keeps its
outputs in the graph pool: the host time of many eager tails, with the GPU
waiting on the scheduler. Capturing a key at first sight paid that for every
batch size a large batch passes through as it drains, most of which never
recur. Waiting until the eager runs have cost about one capture (rent or buy)
bounds the host time of a key at twice the better of the two.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, NamedTuple, Tuple

import torch

from sglang.srt.speculative.eagle_info import EagleVerifyInput

from eagle_harness.core.draft_tree import DraftTree, VerifySelection
from eagle_harness.sglang.graph_pool import capture

# (batch size, tree selection, num_steps, num_draft_tokens) -> sglang's verify input.
BuildInput = Callable[[int, VerifySelection, int, int], EagleVerifyInput]


class Ranking(NamedTuple):
    """A tree's nodes ranked by score, with the cumulative verify scores."""

    ranked: torch.Tensor  # [bs, count] node indices, best first
    scores: torch.Tensor  # [count + 1] cumulative verify scores (root first), batch mean


# Eager runs of a key before it is captured (see the module docstring).
CAPTURE_AFTER = 16


def _capture(body: Callable[[], object]):
    """Capture ``body`` (its kernels are compiled: it has run eagerly); returns ``(graph, outputs)``."""
    return capture(body)


class _Graphs:
    """``body()`` per key: eager for the key's first :data:`CAPTURE_AFTER` runs, then from its captured graph."""

    def __init__(self) -> None:
        self.graphs: Dict[Tuple, Tuple[torch.cuda.CUDAGraph, object]] = {}
        self._runs: Counter = Counter()

    def run(self, key: Tuple, body: Callable[[], object], static: bool = True):
        """``static=False``: ``body`` reads tensors other than the ones a graph would replay, so it runs eagerly."""
        if not static:
            return body()
        if key not in self.graphs:
            self._runs[key] += 1
            if self._runs[key] <= CAPTURE_AFTER:
                return body()
            self.graphs[key] = _capture(body)
        graph, outputs = self.graphs[key]
        graph.replay()
        return outputs


class VerifyTail:
    """Graph-replayed ranking and verify-input build of a drafted tree."""

    def __init__(self, tree: DraftTree, build_input: BuildInput):
        self.tree = tree
        self._build_input = build_input
        self._rankings = _Graphs()
        self._inputs = _Graphs()
        self._ranked_inputs = _Graphs()

    def rank(self, bs: int, depth: int, count: int) -> Ranking:
        """The best ``count`` nodes of the ``depth``-level tree and the cumulative scores of their prefixes."""
        return self._rankings.run((bs, depth), lambda: self._rank(bs, depth, count))

    def verify_input(self, bs: int, depth: int, width: int, ranking: Ranking) -> EagleVerifyInput:
        """sglang's verify input of the best ``width - 1`` ranked nodes plus the root."""
        # A graph reads the tensors it was captured with: only the ranking graph's own outputs stay current.
        ranked = self._rankings.graphs.get((bs, depth))
        return self._inputs.run(
            (bs, depth, width),
            lambda: self._select(bs, depth, width, ranking),
            static=ranked is not None and ranked[1] is ranking,
        )

    def ranked_verify_input(
        self, bs: int, depth: int, count: int, width: int, on_rank: Callable[[Ranking], None]
    ) -> EagleVerifyInput:
        """:meth:`rank`, ``on_rank(ranking)`` (device work, e.g. copying the scores out) and
        :meth:`verify_input` in one graph replay."""

        def body() -> EagleVerifyInput:
            ranking = self._rank(bs, depth, count)
            on_rank(ranking)
            return self._select(bs, depth, width, ranking)

        return self._ranked_inputs.run((bs, depth, count, width), body)

    def _rank(self, bs: int, depth: int, count: int) -> Ranking:
        ranked, scores = self.tree.rank(bs, depth, count)
        return Ranking(ranked, scores[0] if bs == 1 else scores.mean(dim=0))

    def _select(self, bs: int, depth: int, width: int, ranking: Ranking) -> EagleVerifyInput:
        # The verify set is ancestor-closed (a parent outranks its children), so its deepest node is at
        # most width - 1 deep; sglang sizes the accept path by it.
        steps = min(depth, width - 1)
        selection = self.tree.select(bs, steps, ranking.ranked, width)
        return self._build_input(bs, selection, steps, width)
