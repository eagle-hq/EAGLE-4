# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The graph-replayed ranking and selection of a drafted tree equal their eager versions (skipped without CUDA).

Run in a GPU job: ``pytest -q tests/test_verify_tail_gpu.py``.
"""

import pytest
import torch

pytest.importorskip("sglang")

from eagle_harness.core.draft_tree import DraftTree, TreeShape  # noqa: E402
from eagle_harness.sglang.verify_tail import VerifyTail  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

K, DEPTH, H = 4, 12, 64


def _drafted_tree(bs: int, steps: int, seed: int) -> DraftTree:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tree = DraftTree(TreeShape(K, DEPTH), bs, H, torch.bfloat16, torch.device("cuda"), max_verify=32)
    tree.stage_root(
        bs,
        torch.rand(bs, K, device="cuda", generator=gen).softmax(-1),
        torch.randint(0, 1000, (bs, K), device="cuda", generator=gen),
        torch.randn(bs, H, device="cuda", generator=gen).to(torch.bfloat16),
    )
    tree.begin(bs)
    for step in range(1, steps + 1):
        probs = torch.sort(
            torch.rand(bs * K, K, device="cuda", generator=gen).softmax(-1), dim=-1, descending=True
        ).values
        ids = torch.randint(0, 1000, (bs * K, K), device="cuda", generator=gen)
        hidden = torch.randn(bs * K, H, device="cuda", generator=gen).to(torch.bfloat16)
        tree.step(bs, step, probs, ids, hidden)
    return tree


def _build(bs, selection, steps, width):
    """Stands in for sglang's verify-input build: the selection's tensors, as new tensors."""
    return selection.top_scores_index.clone(), selection.draft_tokens + 1, selection.parent_list.clone(), steps, width


@pytest.mark.parametrize("bs,depth,width", [(1, 6, 16), (1, 4, 8), (3, 7, 16)])
def test_replayed_ranking_and_selection_follow_the_trees_buffers(bs, depth, width):
    tree = _drafted_tree(bs, depth, seed=1)
    tail = VerifyTail(tree, _build)
    count = min(tree.shape.num_nodes(depth), 15)
    for seed in (1, 2):  # the second round refills the same static buffers with another tree
        if seed == 2:
            other = _drafted_tree(bs, depth, seed=2)
            for name in ("scores", "tokens", "parents"):
                getattr(tree, name).copy_(getattr(other, name))
        ranking = tail.rank(bs, depth, count)
        want_ranked, want_scores = tree.rank(bs, depth, count)
        want_scores = want_scores[0] if bs == 1 else want_scores.mean(dim=0)
        assert torch.equal(ranking.ranked, want_ranked)
        torch.testing.assert_close(ranking.scores, want_scores)
        got_index, got_tokens, got_parents, steps, got_width = tail.verify_input(bs, depth, width, ranking)
        want = tree.select(bs, min(depth, width - 1), want_ranked, width)
        assert torch.equal(got_index, want.top_scores_index)
        assert torch.equal(got_tokens, want.draft_tokens + 1)
        assert torch.equal(got_parents, want.parent_list)
        assert (steps, got_width) == (min(depth, width - 1), width)
