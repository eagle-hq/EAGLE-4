# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""DraftTree bookkeeping on synthetic tensors (CPU), against direct re-implementations of
sglang's beam layout (select_top_k_tokens / organize_draft_results) and a work-delta
calculation over sorted scores."""

import pytest
import torch

from eagle_harness.core.cost_model import CostModel, CostModelParams, SizeClass, WidthStep
from eagle_harness.core.draft_tree import DEPTH_DECAY, ROOT_SCORE, DraftTree, TreeShape
from eagle_harness.core.limits import ComputeIntensity

K, DEPTH, H = 4, 6, 8


def _random_round(bs, steps, seed=0):
    gen = torch.Generator().manual_seed(seed)
    root_p = torch.rand(bs, K, generator=gen).softmax(-1)
    root_ids = torch.randint(0, 1000, (bs, K), generator=gen)
    rounds = []
    for _ in range(steps):
        p = torch.rand(bs * K, K, generator=gen).softmax(-1)
        p, _ = torch.sort(p, dim=-1, descending=True)
        ids = torch.randint(0, 1000, (bs * K, K), generator=gen)
        hidden = torch.randn(bs * K, H, generator=gen)
        rounds.append((p, ids, hidden))
    return root_p, root_ids, torch.randn(bs, H, generator=gen), rounds


def _begin(tree, bs, root_p, root_ids, root_h):
    tree.stage_root(bs, root_p, root_ids, root_h)
    tree.begin(bs)


def _sglang_reference(root_p, root_ids, rounds):
    """sglang EAGLE beam: score_list / token_list / parents_list per step."""
    bs = root_p.shape[0]
    order = torch.argsort(root_p, dim=-1, descending=True)
    scores = torch.gather(root_p, 1, order)
    score_list, token_list, parents_list = (
        [scores],
        [torch.gather(root_ids, 1, order)],
        [torch.arange(-1, K).expand(bs, -1)],
    )
    frontier_scores = scores
    for i, (p, ids, _) in enumerate(rounds, start=1):
        expand = (frontier_scores[:, :, None] * p.view(bs, K, K)).flatten(1)
        top, pick = torch.topk(expand, K, dim=-1)
        score_list.append(expand)
        token_list.append(ids.view(bs, K * K))
        parents_list.append(pick + K * K * (i - 1) + K)
        frontier_scores = top
    return torch.cat(score_list, 1), torch.cat(token_list, 1), parents_list


def test_layout_matches_sglang_beam():
    bs, steps = 2, DEPTH - 1
    root_p, root_ids, root_h, rounds = _random_round(bs, steps)
    tree = DraftTree(TreeShape(K, DEPTH), bs, H, torch.float32, torch.device("cpu"), max_verify=32)
    _begin(tree, bs, root_p, root_ids, root_h)
    for step, (p, ids, hidden) in enumerate(rounds, start=1):
        tree.set_step_slots(bs, step, torch.full((bs, K), step))
        tree.step(bs, step, p, ids, hidden)
    ref_scores, ref_tokens, ref_parents = _sglang_reference(root_p, root_ids, rounds)
    n = tree.shape.num_nodes(DEPTH)
    assert torch.allclose(tree.scores[:, :n], ref_scores)
    assert torch.equal(tree.tokens[:, :n], ref_tokens)
    for depth in range(2, DEPTH + 1):
        expected = torch.cat(ref_parents[: depth - 1], dim=1)
        assert torch.equal(tree.parents[:, : tree.shape.num_parents(depth)], expected)


def test_frontier_hidden_and_ancestors_follow_parents():
    bs = 1
    root_p, root_ids, root_h, rounds = _random_round(bs, 2, seed=3)
    tree = DraftTree(TreeShape(K, DEPTH), bs, H, torch.float32, torch.device("cpu"), max_verify=32)
    _begin(tree, bs, root_p, root_ids, root_h)
    assert torch.equal(tree.front_hidden[:K], root_h.expand(K, -1))
    tree.set_step_slots(bs, 1, torch.arange(10, 10 + K).view(1, K))
    p, ids, hidden = rounds[0]
    tree.step(bs, 1, p, ids, hidden)
    expand = tree.scores[0, K : K + K * K]
    pick = torch.topk(expand, K).indices
    lanes = pick // K
    assert torch.equal(tree.front_hidden[:K], hidden[lanes])
    assert torch.equal(tree.ancestors[0, :, 0], (10 + lanes).to(torch.int32))
    assert torch.equal(tree.front_tokens[:K], ids.view(-1)[pick])


def _grow(bs, root_p, root_ids, root_h, rounds, max_bs=None):
    tree = DraftTree(TreeShape(K, DEPTH), max_bs or bs, H, torch.float32, torch.device("cpu"), max_verify=32)
    _begin(tree, bs, root_p, root_ids, root_h)
    for step, (p, ids, hidden) in enumerate(rounds, start=1):
        tree.set_step_slots(bs, step, torch.arange(bs * K).view(bs, K) + 100 * step)
        tree.step(bs, step, p, ids, hidden)
    return tree


def test_kept_rows_are_the_tree_of_the_kept_requests():
    """A tree grown for 4 requests, compacted to requests 3, 0 and 2, equals the tree grown for those 3 alone."""
    bs, keep = 4, torch.tensor([3, 0, 2])
    root_p, root_ids, root_h, rounds = _random_round(bs, 3, seed=5)
    tree = _grow(bs, root_p, root_ids, root_h, rounds)
    tree.keep_rows(keep)
    nodes = (keep[:, None] * K + torch.arange(K)).reshape(-1)  # the kept requests' frontier rows
    kept_rounds = [(p[nodes], ids[nodes], hidden[nodes]) for p, ids, hidden in rounds]
    alone = _grow(3, root_p[keep], root_ids[keep], root_h[keep], kept_rounds, max_bs=bs)
    for name in ("root_probs", "root_tokens", "root_hidden", "scores", "tokens", "parents", "front_scores",
                 "best", "best_prev", "layer_top", "delta_rows"):
        assert torch.equal(getattr(tree, name)[:3], getattr(alone, name)[:3]), name
    assert torch.equal(tree.front_tokens[: 3 * K], alone.front_tokens[: 3 * K])
    assert torch.equal(tree.front_hidden[: 3 * K], alone.front_hidden[: 3 * K])
    first = _grow(bs, root_p, root_ids, root_h, rounds)
    assert torch.equal(tree.ancestors[:3], first.ancestors[keep])


def _fork_work_deltas(work_scores, node_count, top_k, verify_num):
    """calculate_work_deltas_kernel, one row."""
    out = []
    for j in range(top_k + 1):
        k = min(node_count + j, verify_num)
        a = work_scores[:node_count]
        b = work_scores[node_count : node_count + j]
        pa = pb = 0
        total = 0.0
        for _ in range(k):
            va = a[pa] if pa < node_count else 0.0
            vb = b[pb] if pb < j else 0.0
            if va >= vb:
                total += va
                pa += 1
            else:
                total += vb
                pb += 1
        out.append(total)
    return out


def test_work_deltas_match_fork_kernel():
    bs = 1
    root_p, root_ids, root_h, rounds = _random_round(bs, 4, seed=7)
    shape = TreeShape(K, DEPTH)
    tree = DraftTree(shape, bs, H, torch.float32, torch.device("cpu"), max_verify=32)
    _begin(tree, bs, root_p, root_ids, root_h)
    seg = [ROOT_SCORE] + torch.sort(root_p[0], descending=True).values.tolist()
    for verify_num in (1, 2, 16, 32):
        ref = torch.tensor(_fork_work_deltas(seg, 1, K, verify_num))
        assert torch.allclose(tree.deltas[0, verify_num - 1], ref)
    for step, (p, ids, hidden) in enumerate(rounds, start=1):
        best_before = tree.best[0].clone()
        tree.set_step_slots(bs, step, torch.zeros(bs, K))
        tree.step(bs, step, p, ids, hidden)
        count = shape.count_before_step(step)
        for verify_num in (1, 5, 16, 32):
            n0 = min(verify_num, count)
            seg = best_before[:n0].tolist() + tree.layer_top[0].tolist()
            ref = torch.tensor(_fork_work_deltas(seg, n0, K, verify_num))
            assert torch.allclose(tree.deltas[step, verify_num - 1], ref, atol=1e-6), (step, verify_num)


def test_work_deltas_are_batch_means():
    bs = 3
    root_p, root_ids, root_h, rounds = _random_round(bs, 2, seed=21)
    shape = TreeShape(K, DEPTH)
    tree = DraftTree(shape, bs, H, torch.float32, torch.device("cpu"), max_verify=16)
    _begin(tree, bs, root_p, root_ids, root_h)
    p, ids, hidden = rounds[0]
    best_before = tree.best.clone()
    tree.set_step_slots(bs, 1, torch.zeros(bs, K))
    tree.step(bs, 1, p, ids, hidden)
    n0 = min(16, shape.count_before_step(1))
    rows = [_fork_work_deltas(best_before[b, :n0].tolist() + tree.layer_top[b].tolist(), n0, K, 16) for b in range(bs)]
    assert torch.allclose(tree.deltas[1, 15], torch.tensor(rows).mean(dim=0), atol=1e-6)


def test_best_set_is_top_of_all_scores():
    bs = 1
    root_p, root_ids, root_h, rounds = _random_round(bs, 3, seed=11)
    shape = TreeShape(K, DEPTH, work_cap=20)
    tree = DraftTree(shape, bs, H, torch.float32, torch.device("cpu"), max_verify=20)
    _begin(tree, bs, root_p, root_ids, root_h)
    for step, (p, ids, hidden) in enumerate(rounds, start=1):
        tree.set_step_slots(bs, step, torch.zeros(bs, K))
        tree.step(bs, step, p, ids, hidden)
        n = shape.num_nodes(step + 1)
        every = torch.cat([torch.tensor([ROOT_SCORE]), tree.scores[0, :n]])
        expected = torch.sort(every, descending=True).values[:20]
        assert torch.allclose(tree.best[0, : expected.numel()], expected)


def test_selection_ranks_by_decayed_score():
    bs = 1
    root_p, root_ids, root_h, rounds = _random_round(bs, 3, seed=5)
    shape = TreeShape(K, DEPTH)
    tree = DraftTree(shape, bs, H, torch.float32, torch.device("cpu"), max_verify=32)
    _begin(tree, bs, root_p, root_ids, root_h)
    for step, (p, ids, hidden) in enumerate(rounds, start=1):
        tree.set_step_slots(bs, step, torch.zeros(bs, K))
        tree.step(bs, step, p, ids, hidden)
    depth = 4
    n = shape.num_nodes(depth)
    decay = torch.ones(n)
    for d in range(2, depth + 1):
        decay[shape.layer_start(d) : shape.layer_start(d + 1)] = DEPTH_DECAY**d
    ranked, scores = tree.rank(bs, depth, 15)
    expected = torch.topk(tree.scores[0, :n] * decay, 15).indices
    assert torch.equal(ranked[0], expected)
    sel = tree.select(bs, depth, ranked, 8)
    assert torch.equal(sel.top_scores_index[0], torch.sort(expected[:7]).values)
    assert torch.equal(sel.draft_tokens[0], tree.tokens[0, sel.top_scores_index[0]])
    scores = scores[0]
    assert scores.shape == (16,)
    assert abs(float(scores[0]) - ROOT_SCORE) < 1e-6
    assert torch.allclose(scores[1:] - scores[:-1], (tree.scores[0, :n] * decay)[expected])


def test_selection_parent_list_is_contiguous_for_a_batch():
    # sglang's tree kernel reads parent_list as a dense [bs, parents] array (its JIT kernel checks the strides); a
    # column slice of the [max_bs, parent_capacity] buffer has the capacity as row stride once bs > 1.
    bs, depth = 3, 4
    root_p, root_ids, root_h, rounds = _random_round(bs, 3, seed=7)
    shape = TreeShape(K, DEPTH)
    tree = DraftTree(shape, bs + 1, H, torch.float32, torch.device("cpu"), max_verify=32)
    _begin(tree, bs, root_p, root_ids, root_h)
    for step, (p, ids, hidden) in enumerate(rounds, start=1):
        tree.set_step_slots(bs, step, torch.zeros(bs, K))
        tree.step(bs, step, p, ids, hidden)
    ranked, _ = tree.rank(bs, depth, 15)
    sel = tree.select(bs, depth, ranked, 8)
    assert sel.parent_list.shape == (bs, shape.num_parents(depth)) and sel.parent_list.is_contiguous()
    assert torch.equal(sel.parent_list, tree.parents[:bs, : shape.num_parents(depth)])


def _node_depth(node: int) -> int:
    return 1 if node < K else 2 + (node - K) // (K * K)


def _node_parent(tree: DraftTree, b: int, node: int) -> int:
    """Parent node index of ``node`` (-1: the root), from sglang's parent layout."""
    if node < K:
        return -1
    depth = _node_depth(node)
    lane = (node - tree.shape.layer_start(depth)) // K
    return int(tree.parents[b, tree.shape.num_parents(depth) - K + lane])


def test_verify_set_is_ancestor_closed():
    bs = 2
    root_p, root_ids, root_h, rounds = _random_round(bs, DEPTH - 1, seed=13)
    tree = DraftTree(TreeShape(K, DEPTH), bs, H, torch.float32, torch.device("cpu"), max_verify=32)
    _begin(tree, bs, root_p, root_ids, root_h)
    for step, (p, ids, hidden) in enumerate(rounds, start=1):
        tree.set_step_slots(bs, step, torch.zeros(bs, K))
        tree.step(bs, step, p, ids, hidden)
    for width in (2, 4, 8, 16):
        ranked, _ = tree.rank(bs, DEPTH, width - 1)
        for b in range(bs):
            chosen = set(ranked[b].tolist())
            for node in chosen:
                parent = _node_parent(tree, b, node)
                assert parent == -1 or parent in chosen, (width, node, parent)
            assert max(_node_depth(n) for n in chosen) <= width - 1


def _cost_model(intensity, verify_num=16):
    return CostModel(
        tables={},
        params=CostModelParams(),
        intensity=intensity,
        node_top_k=K,
        tree_depth=DEPTH,
        verify_num=verify_num,
    )


def test_history_budget_exceeds_verify_width_and_fits_the_deltas():
    """Regression: at batch size 1 the history budget is floored to 32 while the verify width is 16, and
    the work deltas were sized by the width (IndexError on the first decode round)."""
    for intensity in (ComputeIntensity.HIGH, ComputeIntensity.CHAIN):
        cost = _cost_model(intensity)
        seen = set()
        for tiny in (False, True):
            cost.tier.tiny = tiny
            for size in SizeClass:
                cost.tier.size = size
                for verify_num in (1, 4, 16):
                    for _ in range(16):
                        cost.record_verify_num(1, verify_num)
                    seen.add(cost.history_verify_num(1))
        assert max(seen) <= cost.max_history_verify_num, intensity
        if intensity is ComputeIntensity.HIGH:
            assert max(seen) == 32 > cost.verify_num
    cost = _cost_model(ComputeIntensity.HIGH)
    cost.tier.tiny = True
    past = cost.history_verify_num(1)
    tree = DraftTree(
        TreeShape(K, DEPTH),
        1,
        H,
        torch.float32,
        torch.device("cpu"),
        max_verify=cost.max_history_verify_num,
        delta_slots=cost.launch_depths,
    )
    root_p, root_ids, root_h, rounds = _random_round(1, 2, seed=3)
    _begin(tree, 1, root_p, root_ids, root_h)
    for step, (p, ids, hidden) in enumerate(rounds, start=1):
        best_before = tree.best[0].clone()
        tree.set_step_slots(1, step, torch.zeros(1, K))
        tree.step(1, step, p, ids, hidden)
    # Step 2 is a launch depth: its row for budget ``past`` is the reference output with verify_num=past.
    count = tree.shape.count_before_step(2)
    n0 = min(past, count)
    ref = _fork_work_deltas(best_before[:n0].tolist() + tree.layer_top[0].tolist(), n0, K, past)
    assert torch.allclose(tree.deltas[2, past - 1], torch.tensor(ref), atol=1e-6)


def test_deltas_only_at_launch_depths():
    cost = _cost_model(ComputeIntensity.HIGH)
    assert cost.launch_depths == tuple(d for d in range(DEPTH - 1) if d % 2 == 0)
    for depth in range(DEPTH):
        if cost.width_step(depth) is WidthStep.LAUNCH:
            assert depth in cost.launch_depths
    tree = DraftTree(
        TreeShape(K, DEPTH), 1, H, torch.float32, torch.device("cpu"), max_verify=32, delta_slots=cost.launch_depths
    )
    root_p, root_ids, root_h, rounds = _random_round(1, 1, seed=5)
    _begin(tree, 1, root_p, root_ids, root_h)
    tree.set_step_slots(1, 1, torch.zeros(1, K))
    tree.step(1, 1, *rounds[0])
    assert tree.deltas[1].abs().sum() == 0  # odd depth: never read, never computed
    assert tree.deltas[0].abs().sum() > 0


def test_delta_budget_must_fit_the_best_set():
    with pytest.raises(ValueError, match="work_cap"):
        DraftTree(TreeShape(K, DEPTH, work_cap=16), 1, H, torch.float32, torch.device("cpu"), max_verify=32)
