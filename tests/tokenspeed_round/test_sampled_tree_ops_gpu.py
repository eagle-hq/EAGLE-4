# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Reference sampled-tree proposal and rejection kernels, including their output law."""


import pytest
import torch

pytest.importorskip("tvm_ffi", reason="TokenSpeed round kernels require apache-tvm-ffi")

from eagle_harness.tokenspeed.tree_round.ops.tree import (
    eagle_harness_publish_draft_q,
    eagle_harness_residual_sample,
    eagle_harness_sample_children,
    eagle_harness_tree_reject_sample,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU is required"
)
_DEVICE = "cuda"


def _count(value: int) -> torch.Tensor:
    """A device int32 ``[1]`` count, as the drafter's round state holds them."""
    return torch.tensor([value], dtype=torch.int32, device=_DEVICE)


def test_sample_children_reads_strided_rows() -> None:
    """Each request's first R of K lane rows, read in place (and the head's
    row-strided candidate ids), draw exactly what packed copies draw."""
    reqs, k, expanded, nc, vocab = 3, 4, 2, 64, 300
    gen = torch.Generator().manual_seed(7)
    logits = (torch.randn((reqs * k, nc), generator=gen) * 2).to(_DEVICE)
    cand = torch.randint(0, vocab + 20, (reqs * k, nc + 3), generator=gen).to(_DEVICE)
    cand = cand[:, :nc]
    tokens = torch.randint(0, vocab, (reqs * k, k), generator=gen).to(_DEVICE)
    scores = torch.rand((reqs * k, k), generator=gen).to(_DEVICE)
    # A sampling request, a greedy one, and one cut to top-k / top-p.
    temps = torch.tensor([1.0, 0.0, 0.7], device=_DEVICE)
    top_ks = torch.tensor([1 << 30, 1 << 30, 5], dtype=torch.int32, device=_DEVICE)
    top_ps = torch.tensor([1.0, 1.0, 0.8], device=_DEVICE)

    def lanes(rows: torch.Tensor) -> torch.Tensor:
        return rows.view(reqs, k, -1)[:, :expanded]

    def draw(*rows: torch.Tensor, width: torch.Tensor | None) -> list[torch.Tensor]:
        tables = [
            torch.zeros((reqs, 1 + k, nc), device=_DEVICE),
            torch.zeros((reqs, 1 + k, nc), dtype=torch.int32, device=_DEVICE),
            torch.full((reqs, 64), -1, dtype=torch.int32, device=_DEVICE),
            torch.zeros((reqs, 64), dtype=torch.int32, device=_DEVICE),
        ]
        eagle_harness_sample_children(
            *rows, temps, top_ks, top_ps, *tables, width, 1, _count(5), vocab, 11,
            torch.tensor([3], device=_DEVICE), 1,
        )  # fmt: skip
        return tables

    def full(rows: torch.Tensor) -> torch.Tensor:
        return rows.view(reqs, k, -1)

    # All K lanes passed with the expanded width on the device draw what the
    # expanded lanes alone draw.
    packed = [lanes(rows).contiguous() for rows in (logits, cand, tokens, scores)]
    packed_tables = draw(*packed, width=None)
    in_place = tokens.clone(), scores.clone()
    tables = draw(
        full(logits), full(cand), *map(full, in_place), width=_count(expanded)
    )
    assert (packed_tables[2][0] >= 0).any() and (packed_tables[2][1] == -1).all()
    for actual, expected in zip(
        [*map(lanes, in_place), *tables], packed[2:] + packed_tables
    ):
        assert torch.equal(actual, expected)
    # The lanes past the expanded ones keep their children.
    for actual, before in zip(in_place, (tokens, scores)):
        assert torch.equal(
            actual.view(reqs, k, k)[:, expanded:], before.view(reqs, k, k)[:, expanded:]
        )


def test_sampled_proposals_advance_on_graph_replay():
    """A replay advances the device RNG counter; resetting it repeats the draw."""
    bs, nc, k = 8, 64, 4
    logits = torch.zeros((bs, 1, nc), device=_DEVICE)
    ids = torch.arange(nc, device=_DEVICE).view(1, 1, nc).repeat(bs, 1, 1)
    tokens = torch.zeros((bs, 1, k), dtype=torch.int64, device=_DEVICE)
    scores = torch.ones((bs, 1, k), device=_DEVICE)
    temps = torch.ones(bs, device=_DEVICE)
    top_ks, top_ps = _no_top_k(bs), torch.ones(bs, device=_DEVICE)
    q = torch.zeros((bs, 1, nc), device=_DEVICE)
    q_ids = torch.zeros_like(q, dtype=torch.int32)
    slot = torch.full((bs, k + 1), -1, dtype=torch.int32, device=_DEVICE)
    pos = torch.zeros_like(slot)
    counter, start = torch.zeros(1, dtype=torch.int64, device=_DEVICE), _count(1)

    def draw():
        counter.add_(1)
        eagle_harness_sample_children(
            logits, ids, tokens, scores, temps, top_ks, top_ps, q, q_ids,
            slot, pos, None, 0, start, nc, 31, counter, 0,
        )  # fmt: skip

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        draw()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        draw()
    counter.zero_()
    graph.replay()
    first = tokens.clone()
    graph.replay()
    second = tokens.clone()
    assert counter.item() == 2
    assert not torch.equal(first, second)
    counter.zero_()
    graph.replay()
    assert torch.equal(tokens, first)
    assert (slot[:, 0] == -1).all()
    assert (slot[:, 1:] == 0).all()
    torch.testing.assert_close(q, torch.full_like(q, 1 / nc), rtol=0, atol=0)


def _sampled_round(target_rows, draft_logits, cand, topk, generator):
    """One round per request at temperature 1: sample the root's K children
    (and each child's K grandchildren) from the draft, accept by rejection,
    draw the bonus; returns each request's emitted tokens (-1 past them).

    target_rows: ``[1 + K + K * K, V]`` the target's distribution at every node
    (root, children, grandchildren in pool order); draft_logits / cand:
    ``[1 + K, NC]`` the draft's logits over candidate ids at the root and at
    each child."""
    # The children kernel keys at most 1024 rows per launch: 256 requests x 3 children.
    reqs, nc = 256, cand.shape[1]
    k, vocab = topk, target_rows.shape[1]
    max_nodes, slots = 64, 1 + k
    nodes = 1 + k + k * k
    dev = "cuda"
    temps = torch.ones(reqs, device=dev)
    top_ks = _no_top_k(reqs)
    top_ps = torch.ones(reqs, device=dev)
    q_store = torch.zeros((reqs, slots, nc), device=dev)
    qid_store = torch.zeros((reqs, slots, nc), dtype=torch.int32, device=dev)
    node_slot = torch.full((reqs, max_nodes), -1, dtype=torch.int32, device=dev)
    node_pos = torch.zeros((reqs, max_nodes), dtype=torch.int32, device=dev)
    counter = torch.randint(0, 1 << 30, (1,), generator=generator).to(dev)
    seed = int(torch.randint(0, 1 << 40, (1,), generator=generator))
    # Depth 1: the root's children; depth 2: each child's children.
    root_tokens = cand[0, :k].repeat(reqs, 1).contiguous()
    eagle_harness_sample_children(
        draft_logits[:1].repeat(reqs, 1)[:, None], cand[:1].repeat(reqs, 1)[:, None],
        root_tokens[:, None], torch.zeros((reqs, 1, k), device=dev), temps, top_ks, top_ps,
        q_store, qid_store, node_slot, node_pos, None, 0, _count(1), vocab, seed, counter,
        0,
    )  # fmt: skip
    # A child's draft row is its own (indexed by the child's rank among the root's).
    child_logits = draft_logits[1:].repeat(reqs, 1).contiguous()
    child_cand = cand[1:].repeat(reqs, 1).contiguous()
    grand_tokens = child_cand[:, :k].contiguous()
    eagle_harness_sample_children(
        child_logits.view(reqs, k, -1), child_cand.view(reqs, k, -1),
        grand_tokens.view(reqs, k, k), torch.zeros((reqs, k, k), device=dev),
        temps, top_ks, top_ps, q_store, qid_store, node_slot, node_pos,
        None, 1, _count(1 + k), vocab, seed, counter, 1,
    )  # fmt: skip
    tokens = torch.cat(
        [torch.zeros((reqs, 1), dtype=torch.int64, device=dev), root_tokens,
         grand_tokens.view(reqs, k * k)], dim=1,
    ).to(torch.int32)  # fmt: skip
    parents = torch.tensor(
        [-1] + [0] * k + [1 + c for c in range(k) for _ in range(k)],
        dtype=torch.int32,
        device=dev,
    ).repeat(reqs, 1)
    rows = torch.arange(reqs, device=dev)
    probas = target_rows.repeat(reqs, 1).contiguous()
    argmax = probas.argmax(dim=-1)
    q_tab = torch.zeros_like(q_store)
    qid_tab = torch.zeros_like(qid_store)
    slot_tab = torch.zeros((reqs, nodes), dtype=torch.int32, device=dev)
    pos_tab = torch.zeros_like(slot_tab)
    ranked = torch.arange(nodes, device=dev).repeat(reqs, 1)
    eagle_harness_publish_draft_q(
        ranked, node_slot, node_pos, q_store, qid_store, rows, q_tab, qid_tab,
        slot_tab, pos_tab, _count(reqs), _count(nodes),
    )  # fmt: skip
    # Every sampled node's published (slot, position) names its own candidate.
    sampled = slot_tab >= 0
    req = torch.arange(reqs, device=dev)[:, None].expand(-1, nodes)
    named = qid_tab[req, slot_tab.clamp(min=0).long(), pos_tab.long()]
    assert torch.equal(named[sampled], tokens[sampled])
    coins = torch.rand((reqs, nodes), generator=generator).to(dev)
    final_coins = torch.rand(reqs, generator=generator).to(dev)
    path, predicts, accept_length, _ = _verify(
        probas, tokens, parents, rows, (slot_tab, pos_tab, q_tab, qid_tab), coins,
        final_coins, _no_top_k(reqs * nodes), argmax,
    )  # fmt: skip
    emitted = predicts[:, :3].cpu()
    length = accept_length.cpu()
    emitted[torch.arange(3)[None, :] >= length[:, None]] = -1
    return emitted, path.cpu()


def test_sampled_tree_emits_the_target_distribution() -> None:
    """Children drawn from the draft, recursive rejection and the residual bonus
    together emit exactly the target: the first token's law is the root's
    target, the second's (after an accepted child) that child's."""
    topk, vocab, nc = 3, 48, 32
    generator = torch.Generator().manual_seed(0)
    target = torch.softmax(
        torch.randn((1 + topk + topk * topk, vocab), generator=generator) * 1.5, dim=-1
    ).cuda()
    cand = torch.stack(
        [torch.randperm(vocab, generator=generator)[:nc] for _ in range(1 + topk)]
    ).cuda()
    draft_logits = (torch.randn((1 + topk, nc), generator=generator) * 1.5).cuda()
    first = torch.zeros(vocab)
    second = torch.zeros((topk, vocab))
    trials = 240
    for _ in range(trials):
        emitted, path = _sampled_round(target, draft_logits, cand, topk, generator)
        first += torch.bincount(emitted[:, 0], minlength=vocab).float()
        # Rows whose first token is an accepted child (node path[1]) and that emitted a second.
        child = path[:, 1].long() - 1
        has_second = (child >= 0) & (emitted[:, 1] >= 0)
        for c in range(topk):
            rows = has_second & (child == c)
            second[c] += torch.bincount(emitted[rows, 1], minlength=vocab).float()
    total = trials * 256
    torch.testing.assert_close(first / total, target[0].cpu(), rtol=0, atol=0.01)
    for c in range(topk):
        if second[c].sum() > 3000:
            torch.testing.assert_close(
                second[c] / second[c].sum(), target[1 + c].cpu(), rtol=0, atol=0.02
            )


def _no_top_k(rows: int) -> torch.Tensor:
    """Per-row top-k of sampled (never greedy) rows."""
    return torch.full((rows,), 1 << 30, dtype=torch.int32, device=_DEVICE)


def _two_node_tree(bs: int, child_token: int, q: torch.Tensor):
    """A root (token 7) and one child whose q (slot 0) is ``q`` over candidate
    ids 0..NC-1: tokens, parents and the q tables (slot, pos, q, ids)."""
    nc = q.numel()
    tokens = torch.tensor([7, child_token], dtype=torch.int32, device=_DEVICE).repeat(
        bs, 1
    )
    parents = torch.tensor([-1, 0], dtype=torch.int32, device=_DEVICE).repeat(bs, 1)
    q_tab = q.view(1, 1, nc).repeat(bs, 1, 1).contiguous()
    qid_tab = (
        torch.arange(nc, dtype=torch.int32, device=_DEVICE)
        .view(1, 1, nc)
        .repeat(bs, 1, 1)
    )
    slot_tab = torch.tensor([-1, 0], dtype=torch.int32, device=_DEVICE).repeat(bs, 1)
    pos_tab = torch.tensor([0, child_token], dtype=torch.int32, device=_DEVICE).repeat(
        bs, 1
    )
    return tokens, parents, (slot_tab, pos_tab, q_tab, qid_tab)


def _verify(
    probas, tokens, parents, q_rows, q_tables, coins, final_coins, top_ks, argmax
):
    """Rejection-sample each request's tree, then draw its bonus: path,
    predicts (``[bs, N]``), accept length and last row."""
    bs, n = tokens.shape
    path = torch.empty((bs, n), dtype=torch.int32, device=_DEVICE)
    predicts = torch.full((bs * n,), -1, dtype=torch.int32, device=_DEVICE)
    accept_length = torch.empty(bs, dtype=torch.int32, device=_DEVICE)
    last_row = torch.empty(bs, dtype=torch.int64, device=_DEVICE)
    eagle_harness_tree_reject_sample(
        probas, tokens, parents, q_rows, *q_tables, coins, top_ks, argmax, path,
        predicts, accept_length, last_row,
    )  # fmt: skip
    eagle_harness_residual_sample(
        probas, last_row, final_coins, top_ks, argmax, accept_length, predicts, n
    )
    return path, predicts.view(bs, n), accept_length, last_row


def _verify_sampled(probas, tokens, parents, q_tables, coins, final_coins):
    """``_verify`` of sampled rows, q tables at the requests' own rows; the
    argmax is the kernels' own, read from probas."""
    bs, n = tokens.shape
    unread = torch.full((bs * n,), -1, dtype=torch.int64, device=_DEVICE)
    _, predicts, accept_length, _ = _verify(
        probas, tokens, parents, torch.arange(bs, device=_DEVICE), q_tables, coins,
        final_coins, _no_top_k(bs * n), unread,
    )  # fmt: skip
    return predicts, accept_length


def _sample_root_children(logits, cand, tokens, *, top_k, top_p, vocab, seed):
    """Redraw each request's ``[bs, K]`` root children in place from its
    ``[bs, NC]`` draft logits over ``cand`` at temperature 1; returns the q
    tables (store, ids, node slots, node positions)."""
    bs, nc = logits.shape
    k = tokens.shape[1]
    tables = (
        torch.zeros((bs, 1, nc), device=_DEVICE),
        torch.zeros((bs, 1, nc), dtype=torch.int32, device=_DEVICE),
        torch.full((bs, 8), -1, dtype=torch.int32, device=_DEVICE),
        torch.zeros((bs, 8), dtype=torch.int32, device=_DEVICE),
    )
    eagle_harness_sample_children(
        logits[:, None], cand[:, None], tokens[:, None],
        torch.zeros((bs, 1, k), device=_DEVICE), torch.ones(bs, device=_DEVICE),
        torch.full((bs,), top_k, dtype=torch.int32, device=_DEVICE),
        torch.full((bs,), top_p, device=_DEVICE), *tables, None, 0, _count(1), vocab,
        seed, torch.zeros(1, dtype=torch.int64, device=_DEVICE), 0,
    )  # fmt: skip
    return tables


def test_exhausted_residual_never_emits_a_zero_probability_child() -> None:
    """q puts 3e-8 on the drawn child and the rest on the target's one-hot
    token: the child (p = 0) is always rejected and the bonus is the target's."""
    bs, vocab, nc = 64, 300, 16
    q = torch.zeros(nc, device=_DEVICE)
    q[5], q[9] = 1.0, 3.07e-8
    tree = _two_node_tree(bs, 9, q)
    probas = torch.zeros((bs * 2, vocab), device=_DEVICE)
    probas[:, 5] = 1.0
    coins = torch.rand((bs, 2), device=_DEVICE)
    predicts, accept_length = _verify_sampled(
        probas, *tree, coins, torch.rand(bs, device=_DEVICE)
    )
    assert (accept_length == 1).all() and (predicts[:, 0] == 5).all()


def test_exhausted_residual_accepts_the_target_argmax() -> None:
    """q one-hot on the target's one-hot token (a logit gap past fp32 exp): the
    drawn child is accepted for every coin, even one just below 1."""
    bs, vocab, nc, k = 64, 300, 16, 3
    logits = torch.full((bs, nc), -200.0, device=_DEVICE)
    logits[:, 5] = 0.0
    cand = torch.arange(nc, device=_DEVICE).repeat(bs, 1).contiguous()
    tokens = cand[:, :k].contiguous()
    q_store, qid_store, node_slot, node_pos = _sample_root_children(
        logits, cand, tokens, top_k=1 << 30, top_p=1.0, vocab=vocab, seed=3
    )
    assert (tokens[:, 0] == 5).all()
    tree = (
        torch.stack([torch.full((bs,), 7, device=_DEVICE), tokens[:, 0]], 1).to(torch.int32),
        torch.tensor([-1, 0], dtype=torch.int32, device=_DEVICE).repeat(bs, 1),
        (
            torch.stack([torch.full((bs,), -1, device=_DEVICE), node_slot[:, 1]], 1).to(torch.int32),
            torch.stack([torch.zeros(bs, device=_DEVICE), node_pos[:, 1]], 1).to(torch.int32),
            q_store, qid_store,
        ),
    )  # fmt: skip
    probas = torch.zeros((bs * 2, vocab), device=_DEVICE)
    probas[:, 5] = 1.0
    coins = torch.full((bs, 2), 1.0 - 2**-24, device=_DEVICE)
    predicts, accept_length = _verify_sampled(
        probas, *tree, coins, torch.rand(bs, device=_DEVICE)
    )
    assert (accept_length == 2).all() and (predicts[:, 0] == 5).all()


@pytest.mark.parametrize("top_k,top_p", [(3, 1.0), (1 << 30, 0.5)])
def test_sampled_children_respect_top_k_and_top_p(top_k: int, top_p: float) -> None:
    """A request draws only from the draft's ``top_k`` best candidates and,
    with top-p, from the smallest prefix (by q) holding that mass."""
    bs, vocab, nc, k = 256, 300, 16, 2
    gen = torch.Generator().manual_seed(4)
    logits = (
        (torch.randn((1, nc), generator=gen) * 2).to(_DEVICE).repeat(bs, 1).contiguous()
    )
    cand = (
        torch.randperm(vocab, generator=gen)[:nc].to(_DEVICE).repeat(bs, 1).contiguous()
    )
    probs = torch.softmax(logits[0], -1)
    order = torch.argsort(probs, descending=True)
    keep = min(top_k, int((torch.cumsum(probs[order], 0) < top_p).sum()) + 1)
    allowed = set(cand[0, order[:keep]].tolist())
    tokens = cand[:, :k].contiguous()
    _sample_root_children(
        logits, cand, tokens, top_k=top_k, top_p=top_p, vocab=vocab, seed=5
    )
    first = set(tokens[:, 0].tolist())
    assert first <= allowed and len(first) > 1


def test_publish_marks_slots_past_the_verify_width_deterministic() -> None:
    bs, nodes, nc = 4, 8, 16
    ranked = torch.arange(nodes, device=_DEVICE).repeat(bs, 1)
    node_slot = torch.zeros((bs, 64), dtype=torch.int32, device=_DEVICE)
    node_pos = torch.ones_like(node_slot)
    q_store = torch.rand((bs, 3, nc), device=_DEVICE)
    qid_store = torch.randint(0, 100, (bs, 3, nc), dtype=torch.int32, device=_DEVICE)
    slot_tab = torch.full((bs + 1, nodes), 7, dtype=torch.int32, device=_DEVICE)
    pos_tab = torch.zeros_like(slot_tab)
    q_tab = torch.zeros((bs + 1, 3, nc), device=_DEVICE)
    qid_tab = torch.zeros((bs + 1, 3, nc), dtype=torch.int32, device=_DEVICE)
    rows = torch.arange(1, bs + 1, device=_DEVICE)
    eagle_harness_publish_draft_q(
        ranked, node_slot, node_pos, q_store, qid_store, rows, q_tab, qid_tab,
        slot_tab, pos_tab, _count(bs), _count(5),
    )  # fmt: skip
    assert (slot_tab[1:, 0] == -1).all()  # the root
    assert (slot_tab[1:, 1:5] == 0).all()
    assert (slot_tab[1:, 5:] == -1).all()  # past the verify width
    torch.testing.assert_close(q_tab[1:], q_store, rtol=0, atol=0)


def test_publish_respects_request_permutation_and_padding():
    bs, nodes, nc = 4, 5, 8
    ranked = torch.tensor([0, 3, 1, 4, 2], device=_DEVICE).repeat(bs, 1)
    slot = torch.tensor([-1, 1, 1, 0, 0], dtype=torch.int32, device=_DEVICE).repeat(
        bs, 1
    )
    pos = torch.arange(nodes, dtype=torch.int32, device=_DEVICE).repeat(bs, 1)
    q = torch.arange(bs * 2 * nc, dtype=torch.float32, device=_DEVICE).view(bs, 2, nc)
    q_ids = q.to(torch.int32)
    rows = torch.tensor([3, 1, 4, 0], device=_DEVICE)
    q_tab = torch.full((6, 2, nc), -2.0, device=_DEVICE)
    qid_tab = torch.full((6, 2, nc), -2, dtype=torch.int32, device=_DEVICE)
    slot_tab = torch.full((6, nodes), -2, dtype=torch.int32, device=_DEVICE)
    pos_tab = torch.full_like(slot_tab, -2)
    eagle_harness_publish_draft_q(
        ranked, slot, pos, q, q_ids, rows, q_tab, qid_tab, slot_tab, pos_tab,
        _count(3), _count(4),
    )  # fmt: skip
    for batch, request in enumerate([3, 1, 4]):
        assert torch.equal(q_tab[request], q[batch])
        assert torch.equal(qid_tab[request], q_ids[batch])
        assert slot_tab[request].tolist() == [-1, 0, 1, 0, -1]
        assert pos_tab[request].tolist() == [0, 3, 1, 4, 0]
    for untouched in [0, 2, 5]:
        assert (q_tab[untouched] == -2).all()
        assert (slot_tab[untouched] == -2).all()


def test_vanished_residual_takes_the_rows_first_argmax() -> None:
    """A sampled row whose residual vanishes takes torch.argmax of its target
    row as the bonus, although the rejection already lowered that entry: the
    drawn child (p = 0) is rejected, and q equal to p on the other candidates
    leaves no residual. The target ties at tokens 50 and 20 (q position 0 and
    1): the lower token wins."""
    bs, vocab, nc = 64, 300, 16
    ids = torch.arange(nc, dtype=torch.int32, device=_DEVICE)
    ids[:4] = torch.tensor([50, 20, 80, 9], dtype=torch.int32)
    q = torch.zeros(nc, device=_DEVICE)
    q[:4] = torch.tensor([0.375, 0.375, 0.25 - 3.0e-8, 3.0e-8])
    tokens, parents, q_tables = _two_node_tree(bs, 9, q)
    _, pos_tab, _, qid_tab = q_tables
    qid_tab[:] = ids
    pos_tab[:, 1] = 3
    probas = torch.zeros((bs * 2, vocab), device=_DEVICE)
    probas[:, [50, 20, 80]] = torch.tensor([0.375, 0.375, 0.25], device=_DEVICE)
    coins = torch.rand((bs, 2), device=_DEVICE)
    # Final coins below 0.99: the bonus row's 3e-8 residual left on token 80 is never drawn.
    predicts, accept_length = _verify_sampled(
        probas, tokens, parents, q_tables, coins, torch.rand(bs, device=_DEVICE) * 0.99
    )
    assert (accept_length == 1).all() and (predicts[:, 0] == 20).all()
    # The root's row: its argmax set to 1 over the vanished residual.
    assert (probas[0::2, 20] == 1.0).all() and (probas[0::2, 50] < 1e-6).all()


@pytest.mark.parametrize("vocab", [151936, 40001])
def test_residual_bonus_inverts_the_cdf_of_a_wide_row(vocab: int) -> None:
    """Wide rows stage each thread's segment in several runs: the bonus is the
    token whose CDF interval holds the coin (aimed at interval middles, clear
    of rounding), over an unnormalized row."""
    bs, n, positive = 16, 4, 64
    gen = torch.Generator().manual_seed(vocab)
    probas = torch.zeros((bs * n, vocab))
    idx = torch.stack(
        [torch.randperm(vocab, generator=gen)[:positive] for _ in range(bs * n)]
    )
    probas.scatter_(1, idx, torch.rand((bs * n, positive), generator=gen) + 0.5)
    last_row = torch.arange(bs) * n + torch.randint(0, n, (bs,), generator=gen)
    expected = idx[last_row, torch.randint(0, positive, (bs,), generator=gen)]
    rows = probas[last_row].double()
    mass = rows.gather(1, expected[:, None]).squeeze(1)
    below = rows.cumsum(-1).gather(1, expected[:, None]).squeeze(1) - mass
    coins = ((below + 0.5 * mass) / rows.sum(-1)).float()
    predicts = torch.full((bs * n,), -1, dtype=torch.int32, device=_DEVICE)
    eagle_harness_residual_sample(
        probas.to(_DEVICE), last_row.to(_DEVICE), coins.to(_DEVICE),
        _no_top_k(bs * n), torch.full((bs * n,), -1, dtype=torch.int64, device=_DEVICE),
        torch.full((bs,), n, dtype=torch.int32, device=_DEVICE), predicts, n,
    )  # fmt: skip
    assert torch.equal(predicts.view(bs, n)[:, -1].cpu().long(), expected)
