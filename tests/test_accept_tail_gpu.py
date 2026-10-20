# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The fused greedy accept tail against sglang's own kernels on random trees (GPU)."""

import random

import pytest
import torch

pytest.importorskip("sglang")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from sglang.kernels.ops.speculative.eagle import fill_accept_out_cache_loc_func, fill_bonus_tokens_func  # noqa: E402
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func  # noqa: E402
from sglang.srt.speculative.eagle_worker_common import _compact_accept_to_front  # noqa: E402
from sglang.srt.speculative.spec_utils import assign_extend_cache_locs  # noqa: E402

from eagle_harness.kernels.accept_tail import accept_tail, compact_rows  # noqa: E402


def _tree(width, rng):
    """Random tree tables (first child / next sibling), draft tokens and a target that often agrees."""
    parent = [-1] + [rng.randrange(0, i) if rng.random() < 0.5 else i - 1 for i in range(1, width)]
    children = [[j for j in range(width) if parent[j] == i] for i in range(width)]
    next_token = [c[0] if c else -1 for c in children]
    next_sibling = [-1] * width
    for c in children:
        for a, b in zip(c, c[1:]):
            next_sibling[a] = b
    candidates = [rng.randrange(0, 6) for _ in range(width)]
    target = [
        candidates[rng.choice(children[i])] if children[i] and rng.random() < 0.8 else rng.randrange(0, 6)
        for i in range(width)
    ]
    return candidates, next_token, next_sibling, target


def _reference(target, candidates, index, next_token, next_sibling, seq_lens, out_cache_loc, req_to_token, req, depth):
    width = target.numel()
    predict = torch.zeros(width, dtype=torch.int32, device="cuda")
    accept_index = torch.full((1, depth), -1, dtype=torch.int32, device="cuda")
    correct = torch.empty(1, dtype=torch.int32, device="cuda")
    predict, accept_index, correct = verify_tree_greedy_func(
        predicts=predict,
        accept_index=accept_index,
        accept_token_num=correct,
        candidates=candidates.reshape(1, width),
        retrieve_index=index,
        retrieve_next_token=next_token,
        retrieve_next_sibling=next_sibling,
        target_predict=target.reshape(1, width),
        topk=4,
    )
    lens = correct + 1
    bonus = torch.empty(1, dtype=torch.int32, device="cuda")
    fill_bonus_tokens_func(predict[accept_index], lens, bonus, depth, 1)
    committed = torch.zeros(depth, dtype=torch.int64, device="cuda")
    assign_extend_cache_locs[(1,)](req, req_to_token, seq_lens, seq_lens + lens, committed, req_to_token.shape[1], 1)
    accepted = torch.zeros(depth, dtype=torch.int64, device="cuda")
    fill_accept_out_cache_loc_func(accept_index, out_cache_loc, accepted, depth)
    compact = _compact_accept_to_front(predict, accept_index, 1, num_draft_tokens=width)
    return compact, accept_index, lens, seq_lens + lens, bonus, committed, accepted


@pytest.mark.parametrize("width,depth", [(16, 12), (9, 8), (4, 4), (31, 12)])
def test_matches_sglang(width, depth):
    rng = random.Random(width * 100 + depth)
    deepest = 0
    for _ in range(40):
        candidates, next_token, next_sibling, target = _tree(width, rng)
        seq_len = rng.randrange(1, 150)
        long = dict(dtype=torch.int64, device="cuda")
        tensors = dict(
            target=torch.tensor(target, **long),
            candidates=torch.tensor(candidates, **long),
            index=torch.arange(width, **long).reshape(1, width),
            next_token=torch.tensor(next_token, **long).reshape(1, width),
            next_sibling=torch.tensor(next_sibling, **long).reshape(1, width),
            seq_lens=torch.tensor([seq_len], **long),
            out_cache_loc=torch.randint(1, 1 << 20, (width,), **long),
            req_to_token=torch.randint(1, 1 << 20, (3, 200 + width), dtype=torch.int32, device="cuda"),
            req=torch.tensor([1], **long),
        )
        expected = _reference(**tensors, depth=depth)
        got = accept_tail(
            tensors["target"],
            tensors["candidates"],
            tensors["index"],
            tensors["next_token"],
            tensors["next_sibling"],
            tensors["seq_lens"],
            tensors["out_cache_loc"],
            tensors["req_to_token"],
            tensors["req"],
            depth,
        )
        names = (
            "predict",
            "accept_index",
            "accept_lens",
            "new_seq_lens",
            "bonus_tokens",
            "committed_slots",
            "accepted_slots",
        )
        for name, want, have in zip(names, expected, got):
            assert have.dtype == want.dtype, name
            assert torch.equal(have, want), (name, have.tolist(), want.tolist(), target, candidates, next_token)
        deepest = max(deepest, int(got.accept_lens))
    assert deepest > 2  # the random trees do accept more than the root


@pytest.mark.parametrize("cols", [7, 5000])
def test_compact_rows_matches_sglang(cols):
    rows = torch.randn(16, cols, device="cuda").to(torch.bfloat16)
    accept_index = torch.tensor([[0, 3, 7, 8, -1, -1, -1, -1]], dtype=torch.int32, device="cuda")
    expected = _compact_accept_to_front(rows, accept_index, 1, num_draft_tokens=16)
    assert torch.equal(compact_rows(rows, accept_index), expected)


def test_tail_graphs_run_a_non_idempotent_body_once_per_round():
    """First round eager, second captured and replayed, later ones replayed: each round runs the body once."""
    from eagle_harness.sglang.verify_step import AcceptTailGraphs

    graphs = AcceptTailGraphs()
    counter = torch.zeros(1, dtype=torch.int64, device="cuda")
    source = torch.arange(4, device="cuda")

    def body():
        counter.add_(1)
        return source * 2

    for _ in range(4):
        out = graphs.run(("key",), body)
        assert out.tolist() == [0, 2, 4, 6]
    torch.cuda.synchronize()
    assert counter.item() == 4
