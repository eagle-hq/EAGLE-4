# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available


"""Generic tree rejection and residual sampling against probability references."""

import pytest
import torch

pytest.importorskip("tokenspeed_kernel")
from tokenspeed_kernel.platform import current_platform
from eagle_harness.tokenspeed.tree_round.ops.rejection import (
    residual_sample,
    tree_reject_sample,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_nvidia,
    reason="tree rejection CUDA kernels are NVIDIA-only",
)

_DEVICE = "cuda"


def test_deterministic_children_emit_the_target_distribution() -> None:
    """Children without a draft q are accepted by p'(child) over the mass their
    rejected siblings left: with the residual bonus the first token's law is
    the root's target."""
    vocab, bs, trials = 16, 8192, 24
    generator = torch.Generator().manual_seed(1)
    target = torch.softmax(torch.randn(vocab, generator=generator) * 1.5, dim=-1)
    second, best = torch.topk(target, 2).indices.tolist()[::-1]
    tokens = torch.tensor([0, second, best], dtype=torch.int32, device=_DEVICE)
    tokens = tokens.repeat(bs, 1)
    parents = torch.tensor([-1, 0, 0], dtype=torch.int32, device=_DEVICE).repeat(bs, 1)
    q_tables = (
        torch.full((1, 3), -1, dtype=torch.int32, device=_DEVICE),
        torch.zeros((1, 3), dtype=torch.int32, device=_DEVICE),
        torch.zeros((1, 1, 1), device=_DEVICE),
        torch.zeros((1, 1, 1), dtype=torch.int32, device=_DEVICE),
    )
    unread = torch.full((bs * 3,), -1, dtype=torch.int64, device=_DEVICE)
    probas = target.to(_DEVICE).repeat(bs * 3, 1)
    first = torch.zeros(vocab)
    for _ in range(trials):
        _, predicts, _, _ = _verify(
            probas.clone(), tokens, parents, None, q_tables,
            torch.rand((bs, 3), device=_DEVICE), torch.rand(bs, device=_DEVICE),
            _no_top_k(bs * 3), unread,
        )  # fmt: skip
        first += torch.bincount(predicts[:, 0].long().cpu(), minlength=vocab).float()
    torch.testing.assert_close(first / (bs * trials), target, rtol=0, atol=0.005)


def _no_top_k(rows: int) -> torch.Tensor:
    """Per-row top-k of sampled (never greedy) rows."""
    return torch.full((rows,), 1 << 30, dtype=torch.int32, device=_DEVICE)


def _verify(probas, tokens, parents, q_rows, q_tables, coins, final_coins, top_ks, argmax):
    """Rejection-sample each request's tree, then draw its bonus: path,
    predicts (``[bs, N]``), accept length and last row."""
    bs, n = tokens.shape
    path = torch.empty((bs, n), dtype=torch.int32, device=_DEVICE)
    predicts = torch.full((bs * n,), -1, dtype=torch.int32, device=_DEVICE)
    accept_length = torch.empty(bs, dtype=torch.int32, device=_DEVICE)
    last_row = torch.empty(bs, dtype=torch.int64, device=_DEVICE)
    tree_reject_sample(
        probas, tokens, parents, q_rows, *q_tables, coins, top_ks, argmax, path,
        predicts, accept_length, last_row,
    )  # fmt: skip
    residual_sample(probas, last_row, final_coins, top_ks, argmax, accept_length, predicts, n)
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

    def rows(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=_DEVICE).repeat(bs, 1)

    # A root (token 7) and one child (token 9) drawn from q (slot 0, position 3).
    tokens, parents = rows([7, 9]), rows([-1, 0])
    q_tables = (rows([-1, 0]), rows([0, 3]), q.repeat(bs, 1, 1), ids.repeat(bs, 1, 1))
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
    idx = torch.stack([torch.randperm(vocab, generator=gen)[:positive] for _ in range(bs * n)])
    probas.scatter_(1, idx, torch.rand((bs * n, positive), generator=gen) + 0.5)
    last_row = torch.arange(bs) * n + torch.randint(0, n, (bs,), generator=gen)
    expected = idx[last_row, torch.randint(0, positive, (bs,), generator=gen)]
    rows = probas[last_row].double()
    mass = rows.gather(1, expected[:, None]).squeeze(1)
    below = rows.cumsum(-1).gather(1, expected[:, None]).squeeze(1) - mass
    coins = ((below + 0.5 * mass) / rows.sum(-1)).float()
    predicts = torch.full((bs * n,), -1, dtype=torch.int32, device=_DEVICE)
    residual_sample(
        probas.to(_DEVICE), last_row.to(_DEVICE), coins.to(_DEVICE),
        _no_top_k(bs * n), torch.full((bs * n,), -1, dtype=torch.int64, device=_DEVICE),
        torch.full((bs,), n, dtype=torch.int32, device=_DEVICE), predicts, n,
    )  # fmt: skip
    assert torch.equal(predicts.view(bs, n)[:, -1].cpu().long(), expected)
