# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""DraftTree.verify_rows on a grown draft tree (CPU): verify rows, their tokens and the parent row of each."""

import pytest
import torch

from tests.triton_probe import needs_triton  # first: it picks Triton's interpreter on a host without a GPU

from eagle_harness.core.draft_tree import DraftTree, TreeShape  # noqa: E402

K, DEPTH, H, BS = 4, 6, 8, 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"  # the select kernel's device (the reference tree grows on CPU)


def grown_tree(seed=0, device="cpu"):
    """A ``DEPTH``-level tree of ``BS`` requests grown with random probabilities, and each node's parent node."""
    tree = DraftTree(TreeShape(top_k=K, max_depth=DEPTH), BS, H, torch.float32, torch.device(device), max_verify=1)
    gen = torch.Generator().manual_seed(seed)
    tree.root_probs[:BS] = torch.rand(BS, K, generator=gen).softmax(-1)
    tree.root_tokens[:BS] = torch.randint(0, 1000, (BS, K), generator=gen)
    tree.root_hidden[:BS] = torch.randn(BS, H, generator=gen)
    tree.begin_reference(BS)
    frontier = torch.arange(K).repeat(BS, 1)  # global node ids of the frontier, per request
    parent_of = torch.full((BS, tree.shape.capacity), -2, dtype=torch.int64)
    parent_of[:, :K] = -1
    for step in range(1, DEPTH):
        probs = torch.rand(BS * K, K, generator=gen).softmax(-1)
        tokens = torch.randint(0, 1000, (BS * K, K), generator=gen)
        tree.step_reference(BS, step, *(t.to(device) for t in (probs, tokens, torch.randn(BS * K, H, generator=gen))))
        start = tree.shape.layer_start(step + 1)
        parent_of[:, start : start + K * K] = frontier.repeat_interleave(K, dim=1)
        # the next frontier: the nodes the step recorded
        pstart = tree.shape.num_parents(step + 1)
        frontier = tree.parents[:BS, pstart : pstart + K].cpu()
    return tree, parent_of


@pytest.mark.parametrize("count", [1, 5, 15, 31])
def test_verify_rows_put_a_parent_first_with_the_right_parent_rows(count):
    tree, parent_of = grown_tree()
    tokens, parent_rows = tree.verify_rows(BS, DEPTH, count)
    assert tokens.shape == (BS, count) and parent_rows.shape == (BS, count + 1)
    ranked, _ = tree.rank(BS, DEPTH, count)
    index = torch.sort(ranked, dim=-1).values
    assert torch.equal(tokens, torch.gather(tree.tokens[:BS], 1, index))
    for b in range(BS):
        assert parent_rows[b, 0] == -1
        for row in range(1, count + 1):
            node = int(index[b, row - 1])
            parent = int(parent_of[b, node])
            expected = 0 if parent == -1 else 1 + index[b].tolist().index(parent)
            assert int(parent_rows[b, row]) == expected
            assert expected < row  # a parent comes first, so any prefix of the rows is a tree


def test_verify_rows_of_a_shallower_tree_use_its_nodes_only():
    tree, _ = grown_tree()
    tokens, parent_rows = tree.verify_rows(BS, 2, 8)
    n = tree.shape.num_nodes(2)
    ranked, _ = tree.rank_reference(BS, 2, 8)
    assert int(ranked.max()) < n
    assert torch.equal(tokens, torch.gather(tree.tokens[:BS], 1, torch.sort(ranked, dim=-1).values))


@needs_triton
@pytest.mark.parametrize("count", [1, 5, 15, 31])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("depth", [3, DEPTH])
def test_select_kernel_matches_the_torch_selection(count, seed, depth):
    cpu, _ = grown_tree(seed)
    want = cpu.verify_rows_reference(BS, depth, count)
    gpu, _ = grown_tree(seed, DEV)
    got = gpu.verify_rows(BS, depth, count) if DEV == "cuda" else _kernel_rows(gpu, depth, count)
    assert torch.equal(got[0].cpu(), want[0]) and torch.equal(got[1].cpu(), want[1])


def _kernel_rows(tree, depth, count):
    """The select kernel on CPU tensors (Triton's interpreter), as verify_rows launches it on CUDA."""
    from eagle_harness.kernels.tree_step import select_rows

    n = tree.shape.num_nodes(depth)
    n_block = 1 << (n - 1).bit_length()
    row_of = torch.empty(BS, n_block, dtype=torch.int32)
    return select_rows(tree.scores[:BS, :n], tree.decay[:n], tree.tokens[:BS, :n], tree.parents[:BS], count, K, row_of)


@needs_triton
def test_select_kernel_breaks_a_tie_at_the_last_row_by_node_number():
    from eagle_harness.kernels.tree_step import select_rows

    tree, _ = grown_tree(3)
    n = tree.shape.num_nodes(DEPTH)
    scores = torch.rand(1, n) * 0.01
    scores[0, :K] = torch.tensor([0.3, 0.3, 0.5, 0.1])  # depth 1 (decay 1): nodes 0 and 1 tie for the second row
    row_of = torch.empty(1, 1 << (n - 1).bit_length(), dtype=torch.int32, device=DEV)
    args = (scores, tree.decay[:n], tree.tokens[:1, :n], tree.parents[:1])
    tokens, parent_rows = select_rows(*(t.to(DEV) for t in args), 2, K, row_of)
    assert tokens[0].tolist() == tree.tokens[0, [0, 2]].tolist()
    assert parent_rows[0].tolist() == [-1, 0, 0]
