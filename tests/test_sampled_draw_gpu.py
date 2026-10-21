# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The one-launch recorded-q draw equals the torch reference draw and its Philox stream is TP-rank independent."""

import os

import pytest
import torch

from eagle_harness.core.sampled_tree import draw_children

pytest.importorskip("triton")
from tests.triton_probe import needs_triton  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEV = "cpu" if INTERPRET else "cuda"
pytestmark = needs_triton


def _stores(bs, slots, nc, nodes):
    q = torch.zeros((bs, slots, nc), device=DEV)
    ids = torch.zeros((bs, slots, nc), dtype=torch.int64, device=DEV)
    positions = torch.zeros((bs, nodes), dtype=torch.int64, device=DEV)
    return q, ids, positions


@pytest.mark.parametrize("k, rows_per_req, step", [(4, 1, 0), (4, 4, 2), (2, 2, 1), (1, 1, 0)])
def test_fused_draw_matches_reference_on_shared_uniforms(k, rows_per_req, step):
    from eagle_harness.kernels.sampled_draw import sample_children

    bs, nc = 3, 128
    gen = torch.Generator().manual_seed(11 + k + step)
    rows = bs * rows_per_req
    logits = (torch.randn(rows, nc, generator=gen) * 2).to(DEV)
    token_ids = torch.stack([torch.randperm(5000, generator=gen)[:nc] for _ in range(rows)]).to(DEV)
    uniforms = torch.rand(rows, nc, generator=gen).to(DEV)
    reference = draw_children(logits, token_ids, k, uniforms=uniforms)
    slots, nodes = 1 + 3 * k, 1 + 4 * k * k
    slot = 0 if step == 0 else 1 + (step - 1) * k
    start = 1 + step * k * rows_per_req  # any layout offset: the kernel writes where it is told
    q, ids, positions = _stores(bs, slots, nc, nodes)
    scores, tokens = sample_children(
        logits, token_ids, k, seed=1, launch=step, counter=torch.ones(1, dtype=torch.int32, device=DEV),
        q=q, q_ids=ids, positions=positions, slot=slot, start=start, rows_per_req=rows_per_req, uniforms=uniforms,
    )
    if not INTERPRET:
        torch.cuda.synchronize()
    torch.testing.assert_close(tokens, reference.tokens, rtol=0, atol=0)
    torch.testing.assert_close(positions[:, start : start + rows_per_req * k].reshape(rows, k), reference.positions, rtol=0, atol=0)
    torch.testing.assert_close(scores, reference.scores, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(q[:, slot : slot + rows_per_req].reshape(rows, nc), reference.probs, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(ids[:, slot : slot + rows_per_req].reshape(rows, nc), reference.token_ids, rtol=0, atol=0)
    # nothing written outside the addressed slots / children
    untouched = torch.ones(slots, dtype=torch.bool)
    untouched[slot : slot + rows_per_req] = False
    assert q[:, untouched].abs().sum() == 0 and ids[:, untouched].abs().sum() == 0
    mask = torch.ones(nodes, dtype=torch.bool)
    mask[start : start + rows_per_req * k] = False
    assert positions[:, mask].abs().sum() == 0


def test_philox_draws_are_deterministic_per_round_and_launch_and_fresh_across_rounds():
    from eagle_harness.kernels.sampled_draw import sample_children

    nc, k = 64, 4
    gen = torch.Generator().manual_seed(5)
    logits = torch.randn(2, nc, generator=gen).to(DEV)
    token_ids = torch.arange(nc).expand(2, -1).contiguous().to(DEV)

    def draw(seed, rnd, launch):
        q, ids, positions = _stores(2, 1, nc, k)
        counter = torch.full((1,), rnd, dtype=torch.int32, device=DEV)
        _, tokens = sample_children(logits, token_ids, k, seed=seed, launch=launch, counter=counter, q=q, q_ids=ids,
                                    positions=positions, slot=0, start=0, rows_per_req=1)
        if not INTERPRET:
            torch.cuda.synchronize()
        return tokens.cpu()

    a, b = draw(721, 3, 1), draw(721, 3, 1)  # two ranks: same seed, same round, same launch -> same tree
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    others = [draw(721, 4, 1), draw(721, 3, 2), draw(722, 3, 1)]
    assert any(not torch.equal(a, o) for o in others)
    assert sum(not torch.equal(a, o) for o in others) >= 2


def test_philox_first_child_follows_q():
    """The first Gumbel-top-K draw is a sample from q: chi-square over many rows of one distribution."""
    from eagle_harness.kernels.sampled_draw import sample_children

    nc, k, rows = 8, 2, (2048 if INTERPRET else 65536)
    q_true = torch.tensor([0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.015, 0.005])
    logits = q_true.log().expand(rows, -1).contiguous().to(DEV)
    token_ids = torch.arange(nc).expand(rows, -1).contiguous().to(DEV)
    q, ids, positions = _stores(rows, 1, nc, k)
    _, tokens = sample_children(logits, token_ids, k, seed=9, launch=0, counter=torch.ones(1, dtype=torch.int32, device=DEV),
                                q=q, q_ids=ids, positions=positions, slot=0, start=0, rows_per_req=1)
    if not INTERPRET:
        torch.cuda.synchronize()
    counts = torch.bincount(tokens[:, 0].cpu(), minlength=nc).double()
    expected = q_true.double() * rows
    chi2 = ((counts - expected) ** 2 / expected).sum().item()
    assert chi2 < 30.0, chi2  # 7 dof: 30 is far beyond the 99.99% point (24.3)
    assert (tokens[:, 0] != tokens[:, 1]).all()  # without replacement


def test_stores_follow_the_request_slot_map():
    from eagle_harness.kernels.sampled_draw import sample_children

    nc, k = 32, 2
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(2, nc, generator=gen).to(DEV)
    token_ids = torch.arange(nc).expand(2, -1).contiguous().to(DEV)
    uniforms = torch.rand(2, nc, generator=gen).to(DEV)
    reference = draw_children(logits, token_ids, k, uniforms=uniforms)
    q, ids, positions = _stores(8, 3, nc, 5)
    req = torch.tensor([5, 2], device=DEV)
    sample_children(logits, token_ids, k, seed=1, launch=1, counter=torch.ones(1, dtype=torch.int32, device=DEV),
                    q=q, q_ids=ids, positions=positions, slot=1, start=3, rows_per_req=1, uniforms=uniforms, req_index=req)
    if not INTERPRET:
        torch.cuda.synchronize()
    torch.testing.assert_close(q[req, 1], reference.probs, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(positions[req, 3:5], reference.positions, rtol=0, atol=0)
    assert q[[0, 1, 3, 4, 6, 7]].abs().sum() == 0 and q[:, [0, 2]].abs().sum() == 0
