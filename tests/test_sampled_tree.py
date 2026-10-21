# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Distribution and layout contracts of without-replacement tree proposals."""

import pytest
import torch


def chi_square_statistic(samples, probabilities):
    counts = torch.bincount(samples.cpu(), minlength=len(probabilities)).double()
    expected = probabilities.double().cpu() * samples.numel()
    mask = expected > 0
    statistic = ((counts[mask] - expected[mask]).square() / expected[mask]).sum().item()
    return statistic, counts[~mask].sum().item()


def chi_square(samples, probabilities):
    statistic, outside = chi_square_statistic(samples, probabilities)
    assert outside == 0
    # df <= 7; 40 is beyond the 1e-6 upper tail (fixed seeds, no retries).
    assert statistic < 40, statistic
    return statistic


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))]
)
def test_draws_record_the_actual_distribution_and_order(device):
    from eagle_harness.core.sampled_tree import draw_children

    gen = torch.Generator(device=device).manual_seed(182)
    q = torch.tensor([0.6, 0.25, 0.1, 0.05], device=device)
    ids = torch.tensor([4, 0, 6, 2], device=device)
    n = 40000
    draw = draw_children(q.log().expand(n, -1), ids.expand(n, -1), 3, generator=gen)
    torch.testing.assert_close(draw.probs, q.expand(n, -1))
    torch.testing.assert_close(draw.token_ids, ids.expand(n, -1))
    torch.testing.assert_close(draw.tokens, draw.token_ids.gather(1, draw.positions))
    assert (draw.positions.sort(dim=1).values.diff(dim=1) > 0).all()
    chi_square(draw.positions[:, 0], q)
    second = torch.stack([sum(q[i] * q[j] / (1 - q[i]) for i in range(4) if i != j) for j in range(4)])
    chi_square(draw.positions[:, 1], second)
    # Policy scores must not reveal the sampled token (sorting samples biases q).
    torch.testing.assert_close(draw.scores, q[:3].expand(n, -1))


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", [28, 182, 917])
def test_recursive_rejection_emits_target_including_outside_proposals(width, seed):
    from eagle_harness.core.sampled_tree import draw_children, reject_children

    n = 30000
    gen = torch.Generator().manual_seed(seed)
    q = torch.tensor([0.7, 0.2, 0.08, 0.02])
    ids = torch.tensor([4, 1, 0, 5])
    p = torch.tensor([0.1, 0.0, 0.2, 0.35, 0.25, 0.1])
    draw = draw_children(q.log().expand(n, -1), ids.expand(n, -1), width, generator=gen)
    result = reject_children(p.expand(n, -1), draw, torch.rand(n, width, generator=gen), torch.rand(n, generator=gen))
    chi_square(result.tokens, p)
    assert (result.accepted_position >= -1).all()
    accepted = result.accepted_position >= 0
    assert accepted.any()
    assert torch.equal(
        result.tokens[accepted], draw.tokens[accepted].gather(1, result.accepted_position[accepted, None]).flatten()
    )


def test_a_verifier_that_subtracts_only_the_rejected_token_fails_the_distribution_check():
    """The gate has power: the classic mistake of removing only the rejected token's q mass from the residual
    (instead of the whole conditional q) biases the output, and the same check that passes the real recursion
    rejects it."""
    from eagle_harness.core.sampled_tree import draw_children

    n, width = 30000, 2
    gen = torch.Generator().manual_seed(28)
    q = torch.tensor([0.7, 0.2, 0.08, 0.02])
    ids = torch.tensor([4, 1, 0, 5])
    p = torch.tensor([0.1, 0.0, 0.2, 0.35, 0.25, 0.1])
    draw = draw_children(q.log().expand(n, -1), ids.expand(n, -1), width, generator=gen)
    coins, final = torch.rand(n, width, generator=gen), torch.rand(n, generator=gen)
    residual, remaining = p.double().expand(n, -1).clone(), draw.probs.double().clone()
    chosen = torch.full((n,), -1)
    for child in range(width):
        token = draw.tokens[:, child : child + 1]
        mass = residual.sum(dim=-1)
        q_y = (remaining / remaining.sum(dim=-1, keepdim=True)).gather(1, draw.positions[:, child : child + 1]).flatten()
        p_y = residual.gather(1, token).flatten()
        accept = (chosen == -1) & (coins[:, child] * mass * q_y < p_y)
        chosen = torch.where(accept, child, chosen)
        biased = residual.scatter(1, token, (p_y - mass * q_y).clamp_min(0)[:, None])  # the mutation
        residual = torch.where((chosen == -1)[:, None], biased, residual)
        remaining.scatter_(1, draw.positions[:, child : child + 1], 0)
    cdf = residual.cumsum(dim=-1)
    bonus = torch.searchsorted(cdf, (final.double() * cdf[:, -1])[:, None], right=True).flatten()
    tokens = torch.where(chosen == -1, bonus, draw.tokens.gather(1, chosen.clamp_min(0)[:, None]).flatten())
    statistic, outside = chi_square_statistic(tokens, p)
    assert statistic >= 40 or outside > 0, statistic


def test_equal_proposal_always_accepts_first_draw():
    from eagle_harness.core.sampled_tree import draw_children, reject_children

    gen = torch.Generator().manual_seed(9)
    p = torch.tensor([[0.6, 0.3, 0.1]]).expand(10000, -1)
    ids = torch.arange(3).expand(10000, -1)
    draw = draw_children(p.log(), ids, 3, generator=gen)
    out = reject_children(p, draw, torch.rand(10000, 3, generator=gen), torch.rand(10000, generator=gen))
    assert (out.accepted_position == 0).all()
    assert torch.equal(out.tokens, draw.tokens[:, 0])


def test_one_hot_and_exhausted_support_never_emit_padding():
    from eagle_harness.core.sampled_tree import draw_children, reject_children

    gen = torch.Generator().manual_seed(3)
    q = torch.tensor([[1.0, 0.0, 0.0]]).expand(1000, -1)
    ids = torch.tensor([[2, 0, 1]]).expand(1000, -1)
    draw = draw_children(q.log(), ids, 3, generator=gen)
    assert (draw.tokens[:, 0] == 2).all()
    assert (draw.scores[:, 1:] == 0).all()
    p = torch.tensor([[0.25, 0.75, 0.0]]).expand(1000, -1)
    out = reject_children(p, draw, torch.zeros(1000, 3), torch.rand(1000, generator=gen))
    assert (out.accepted_position == -1).all()
    chi_square(out.tokens, p[0])


@pytest.mark.parametrize("tied", [False, True])
def test_tree_pruning_preserves_proposal_prefixes_and_raw_node_mapping(tied):
    from eagle_harness.core.draft_tree import DraftTree, TreeShape
    from eagle_harness.core.sampled_tree import draw_children

    bs, k, depth, candidates = 4, 4, 4, 6
    generator = torch.Generator().manual_seed(932)
    shape = TreeShape(k, depth)
    tree = DraftTree(shape, bs, 2, torch.float32, torch.device("cpu"), max_verify=16)
    proposal_ids, proposal_positions = [], []
    for step in range(depth):
        rows = bs if step == 0 else bs * k
        logits = torch.randn(rows, candidates, generator=generator)
        if tied:
            logits.zero_()
        ids = torch.stack([torch.randperm(11, generator=generator)[:candidates] for _ in range(rows)])
        draw = draw_children(logits, ids, k, generator=generator)
        proposal_ids.append(ids.view(bs, -1, candidates))
        proposal_positions.append(draw.positions.view(bs, -1))
        hidden = torch.zeros(rows, 2)
        if step == 0:
            tree.stage_root(bs, draw.scores, draw.tokens, hidden)
            tree.begin(bs)
        else:
            tree.set_step_slots(bs, step, torch.zeros(bs, k, dtype=torch.long))
            tree.step(bs, step, draw.scores, draw.tokens, hidden)
        ranked, _ = tree.rank(bs, step + 1, 15)
        for width in range(1, min(16, shape.num_nodes(step + 1) + 1) + 1):
            selected = tree.select(bs, step + 1, ranked, width)
            q_ids = torch.cat(proposal_ids, dim=1)
            positions = torch.cat(proposal_positions, dim=1)
            for row, raw_nodes in enumerate(selected.top_scores_index.tolist()):
                retained = set(raw_nodes)
                for col, node in enumerate(raw_nodes):
                    slot, order = node // k, node % k
                    assert all(slot * k + earlier in retained for earlier in range(order))
                    if node >= k:
                        assert int(selected.parent_list[row, slot]) in retained
                    assert selected.draft_tokens[row, col] == q_ids[row, slot, positions[row, node]]


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6])
def test_token_dependent_descendants_do_not_bias_pruned_root_distribution(width):
    from eagle_harness.core.draft_tree import DraftTree, TreeShape
    from eagle_harness.core.sampled_tree import ChildProposals, draw_children, reject_children

    n, k = 30000, 2
    generator = torch.Generator().manual_seed(496)
    p = torch.tensor([0.1, 0.2, 0.3, 0.4])
    q = torch.tensor([0.6, 0.25, 0.1, 0.05])
    draw = draw_children(q.log().expand(n, -1), torch.arange(4).expand(n, -1), k, generator=generator)
    tree = DraftTree(TreeShape(k, 2), n, 1, torch.float32, torch.device("cpu"), max_verify=6)
    tree.stage_root(n, draw.scores, draw.tokens, torch.zeros(n, 1))
    tree.begin(n)
    # A sampled parent's identity controls how many descendants outrank its siblings.
    conditional = torch.tensor(
        [[0.97, 0.01, 0.01, 0.01], [0.25, 0.25, 0.25, 0.25], [0.6, 0.3, 0.05, 0.05], [0.4, 0.3, 0.2, 0.1]]
    )
    child = draw_children(
        conditional[tree.front_tokens].log(), torch.arange(4).expand(n * k, -1), k, generator=generator
    )
    tree.set_step_slots(n, 1, torch.zeros(n, k, dtype=torch.long))
    tree.step(n, 1, child.scores, child.tokens, torch.zeros(n * k, 1))
    ranked, _ = tree.rank(n, 2, 5)
    selected = tree.select(n, 2, ranked, width)
    retained = (selected.top_scores_index < k).sum(dim=1)
    coins, final = torch.rand(n, k, generator=generator), torch.rand(n, generator=generator)
    output = torch.empty(n, dtype=torch.long)
    for count in range(1, k + 1):
        mask = retained == count
        proposal = ChildProposals(
            draw.probs[mask],
            draw.token_ids[mask],
            draw.positions[mask, :count],
            draw.tokens[mask, :count],
            draw.scores[mask, :count],
        )
        output[mask] = reject_children(p.expand(int(mask.sum()), -1), proposal, coins[mask, :count], final[mask]).tokens
    assert (retained >= 1).all()
    chi_square(output, p)
