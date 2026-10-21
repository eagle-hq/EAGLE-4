# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""GPU parity of recorded-q verification with the CPU probability recursion."""

import os

import pytest
import torch

from eagle_harness.core.sampled_tree import draw_children, reject_children

pytest.importorskip("triton")
from tests.triton_probe import needs_triton  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
pytestmark = needs_triton


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("root_valid", [True, False])
@pytest.mark.parametrize("layout_dtype", [torch.int32, torch.int64])
def test_gpu_recursive_rejection_matches_reference(k, root_valid, layout_dtype):
    from eagle_harness.kernels.sampled_tree import sample_tree

    n, width = (32 if INTERPRET else 30000), k + 1
    gen = torch.Generator().manual_seed(295)
    p = torch.tensor([0.1, 0.0, 0.2, 0.35, 0.25, 0.1])
    q = torch.tensor([0.7, 0.2, 0.08, 0.02])
    ids = torch.tensor([4, 1, 0, 5])
    draw = draw_children(q.log().expand(n, -1), ids.expand(n, -1), k, generator=gen)
    coins = torch.rand(n, width, generator=gen)
    final = torch.rand(n, generator=gen)
    reference = reject_children(p.expand(n, -1), draw, coins[:, 1:], final)
    target = p.expand(n, width, -1).contiguous()
    next_child = torch.full((n, width), -1, dtype=layout_dtype)
    next_child[:, 0] = 1
    siblings = torch.full_like(next_child, -1)
    siblings[:, 1:-1] = torch.arange(2, width, dtype=layout_dtype)
    tokens = torch.cat((torch.zeros(n, 1, dtype=torch.int64), draw.tokens), dim=1)
    args = [
        target,
        draw.probs[:, None, :].contiguous(),
        draw.token_ids[:, None, :].contiguous(),
        draw.positions,
        torch.full((n,), root_valid),
        torch.arange(k).expand(n, -1).contiguous(),
        tokens,
        torch.arange(n * width).view(n, width).to(layout_dtype),
        next_child,
        siblings,
        coins,
        final,
    ]
    gpu_args = [a.to("cpu" if INTERPRET else "cuda") for a in args]
    predict, lengths, path = sample_tree(*gpu_args, top_k=k, max_depth=2)
    if not INTERPRET:
        torch.cuda.synchronize()
    actual = predict.cpu().view(n, width)[:, 0]
    if root_valid:
        torch.testing.assert_close(actual.long(), reference.tokens, rtol=0, atol=0)
        torch.testing.assert_close(lengths.cpu(), 1 + (reference.accepted_position >= 0).int(), rtol=0, atol=0)
    else:
        expect = torch.searchsorted(p.cumsum(0), final, right=True)
        torch.testing.assert_close(actual.long(), expect, rtol=0, atol=0)
        assert (lengths == 1).all()
    torch.testing.assert_close(path[:, 0].cpu(), torch.arange(n, dtype=torch.int32) * width)
    if INTERPRET:
        return
    # Capture and replay the exact same probability-level implementation.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = sample_tree(*gpu_args, top_k=k, max_depth=2)
    graph.replay()
    for a, b in zip(replayed, (predict, lengths, path)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_walk_reads_recorded_proposals_through_the_request_slot_map():
    """Rows of the stores are addressed by req_index: a permuted store with the matching map gives the same result."""
    from eagle_harness.kernels.sampled_tree import sample_tree

    n, k, width = 24, 2, 3
    gen = torch.Generator().manual_seed(31)
    p = torch.softmax(torch.randn(n, 6, generator=gen), -1)
    q = torch.softmax(torch.randn(n, 4, generator=gen), -1)
    ids = torch.stack([torch.randperm(6, generator=gen)[:4] for _ in range(n)])
    draw = draw_children(q.log(), ids, k, generator=gen)
    coins = torch.rand(n, width, generator=gen)
    final = torch.rand(n, generator=gen)
    target = p[:, None, :].expand(n, width, -1).contiguous()
    next_child = torch.full((n, width), -1, dtype=torch.int32)
    next_child[:, 0] = 1
    siblings = torch.full_like(next_child, -1)
    siblings[:, 1:-1] = torch.arange(2, width, dtype=torch.int32)
    tokens = torch.cat((torch.zeros(n, 1, dtype=torch.int64), draw.tokens), dim=1)
    common = [target, torch.full((n,), True), torch.arange(k).expand(n, -1).contiguous(), tokens,
              torch.arange(n * width).view(n, width).to(torch.int32), next_child, siblings, coins, final]
    dev = "cpu" if INTERPRET else "cuda"

    def run(qs, qids, pos, valid, req):
        args = [common[0], qs, qids, pos, valid, *common[2:-2], *common[-2:]]
        args = [a.to(dev) for a in args]
        out = sample_tree(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9],
                          args[10], args[11], top_k=k, max_depth=2, req_index=None if req is None else req.to(dev))
        if not INTERPRET:
            torch.cuda.synchronize()
        return [o.cpu() for o in out]

    plain = run(draw.probs[:, None, :].contiguous(), draw.token_ids[:, None, :].contiguous(), draw.positions,
                torch.full((n,), True), None)
    perm = torch.randperm(2 * n, generator=gen)[:n]  # store rows, in a bigger store
    qs = torch.zeros(2 * n, 1, 4)
    qids = torch.zeros(2 * n, 1, 4, dtype=torch.int64)
    pos = torch.zeros(2 * n, k, dtype=torch.int64)
    valid = torch.zeros(2 * n, dtype=torch.bool)
    qs[perm, 0] = draw.probs
    qids[perm, 0] = draw.token_ids
    pos[perm] = draw.positions
    valid[perm] = True
    mapped = run(qs, qids, pos, valid, perm)
    for a, b in zip(plain, mapped):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_width_one_has_no_proposal_node_to_load():
    from eagle_harness.kernels.sampled_tree import sample_tree

    device = "cpu" if INTERPRET else "cuda"
    target = torch.tensor([[[0.2, 0.3, 0.5]], [[0.5, 0.3, 0.2]]], device=device)
    q = torch.ones(2, 1, 2, device=device) / 2
    ids = torch.tensor([[[0, 2]], [[0, 2]]], device=device)
    empty_nodes = torch.empty(2, 0, dtype=torch.long, device=device)
    tokens = torch.zeros(2, 1, dtype=torch.long, device=device)
    index = torch.arange(2, dtype=torch.int32, device=device).view(2, 1)
    no_child = torch.full_like(index, -1)
    result = sample_tree(
        target,
        q,
        ids,
        tokens,
        torch.ones(2, dtype=torch.bool, device=device),
        empty_nodes,
        tokens,
        index,
        no_child,
        no_child,
        torch.zeros(2, 1, device=device),
        torch.tensor([0.4, 0.9], device=device),
        top_k=2,
        max_depth=1,
    )
    assert result[0].tolist() == [1, 2]
    assert result[1].tolist() == [1, 1]
    assert result[2].tolist() == [[0], [1]]


def test_gpu_pruned_two_level_tree_keeps_node_and_proposal_mapping():
    from eagle_harness.core.sampled_tree import ChildProposals
    from eagle_harness.kernels.sampled_tree import sample_tree

    n, k, vocab, width = (12 if INTERPRET else 2048), 3, 7, 7
    gen = torch.Generator().manual_seed(981)
    draws = []
    for _ in range(3):
        logits = torch.randn(n, 4, generator=gen)
        ids = torch.stack([torch.randperm(vocab, generator=gen)[:4] for _ in range(n)])
        draws.append(draw_children(logits, ids, k, generator=gen))
    selected = torch.tensor([0, 1, 2, 3, 4, 6])
    all_tokens = torch.cat([d.tokens for d in draws], dim=1)
    tokens = torch.cat((torch.zeros(n, 1, dtype=torch.int64), all_tokens[:, selected]), dim=1)
    target = torch.softmax(torch.randn(n, width, vocab, generator=gen), dim=-1)
    coins, final = torch.rand(n, width, generator=gen), torch.rand(n, generator=gen)
    root = reject_children(target[:, 0], draws[0], coins[:, 1:4], final)
    expected = torch.zeros(n, width, dtype=torch.int32)
    expected[:, 0] = root.tokens.int()
    expected_path = torch.full((n, 3), -1, dtype=torch.int32)
    offsets = torch.arange(n) * width
    expected_path[:, 0] = offsets.int()
    expected_lengths = torch.ones(n, dtype=torch.int32)
    for row in range(n):
        parent = int(root.accepted_position[row])
        if parent < 0:
            continue
        current = parent + 1
        expected_path[row, 1] = offsets[row] + current
        expected_lengths[row] = 2
        if parent < 2:
            count, first_child = (2, 4) if parent == 0 else (1, 6)
            d = draws[parent + 1]
            proposal = ChildProposals(
                d.probs[row : row + 1],
                d.token_ids[row : row + 1],
                d.positions[row : row + 1, :count],
                d.tokens[row : row + 1, :count],
                d.scores[row : row + 1, :count],
            )
            result = reject_children(
                target[row : row + 1, current],
                proposal,
                coins[row : row + 1, first_child : first_child + count],
                final[row : row + 1],
            )
            expected[row, current] = result.tokens[0].int()
            if result.accepted_position[0] >= 0:
                current = first_child + int(result.accepted_position[0])
                expected_path[row, 2] = offsets[row] + current
                expected_lengths[row] = 3
            else:
                continue
        cdf = target[row, current].cumsum(0)
        expected[row, current] = torch.searchsorted(cdf, final[row] * cdf[-1], right=True).int()
    args = [
        target,
        torch.stack([d.probs for d in draws], dim=1),
        torch.stack([d.token_ids for d in draws], dim=1),
        torch.cat([d.positions for d in draws], dim=1),
        torch.ones(n, dtype=torch.bool),
        selected.expand(n, -1).contiguous(),
        tokens,
        torch.arange(n * width).view(n, width).int(),
        torch.tensor([1, 4, 6, -1, -1, -1, -1], dtype=torch.int32).expand(n, -1).contiguous(),
        torch.tensor([-1, 2, 3, -1, 5, -1, -1], dtype=torch.int32).expand(n, -1).contiguous(),
        coins,
        final,
    ]
    result = sample_tree(*[a.to("cpu" if INTERPRET else "cuda") for a in args], top_k=k, max_depth=3)
    for actual, reference in zip(result, (expected.flatten(), expected_lengths, expected_path)):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
