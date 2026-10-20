# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-4 candidate LM head: shard top-k, exact re-scoring, TP merge, full-vocab logits (CPU)."""

from __future__ import annotations

import torch

from eagle_harness.kernels.gather_dot import gather_dot, gather_dot_packed, gather_dot_reference
from eagle_harness.sglang.candidate_logits import (
    Candidates,
    merge_candidates,
    scatter_to_vocab,
    shard_candidates,
)

VOCAB, HIDDEN, RANK, C = 512, 64, 8, 16


def _weights(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    lm_head = torch.randn(VOCAB, HIDDEN, generator=g)
    down = torch.randn(RANK, HIDDEN, generator=g)
    score = torch.randn(VOCAB, RANK, generator=g)
    hidden = torch.randn(3, HIDDEN, generator=g)
    return hidden, down, score, lm_head


def test_gather_dot_matches_rows_times_hidden():
    hidden, _, _, lm_head = _weights()
    ids = torch.tensor([[0, 5, 7], [511, 1, 2], [3, 3, 4]])
    expect = torch.stack([lm_head[ids[b]] @ hidden[b] for b in range(3)])
    torch.testing.assert_close(gather_dot(lm_head, ids, hidden), expect, rtol=1e-5, atol=1e-5)
    out = torch.empty(3, 3)
    assert gather_dot(lm_head, ids, hidden, out=out) is out
    torch.testing.assert_close(out, gather_dot_reference(lm_head, ids, hidden))


def test_single_shard_candidates_are_the_low_rank_top_c_rescored_exactly():
    hidden, down, score, lm_head = _weights()
    cand = shard_candidates(hidden, down, score, lm_head, num_candidates=C, vocab_start=0, vocab_rows=VOCAB)
    merged = merge_candidates(cand, C, None)
    approx = (hidden @ down.t()) @ score.t()
    for b in range(3):
        assert set(merged.token_ids[b].tolist()) == set(approx[b].topk(C).indices.tolist())
        exact = lm_head[merged.token_ids[b]] @ hidden[b]
        torch.testing.assert_close(merged.logits[b], exact, rtol=1e-5, atol=1e-5)
    assert merged.logits.dtype == torch.float32


def test_padding_rows_of_a_shard_are_never_candidates():
    hidden, down, score, lm_head = _weights()
    score = score.clone()
    score[VOCAB - 40 :] = 1e4  # padding rows with huge scores
    cand = shard_candidates(hidden, down, score, lm_head, num_candidates=C, vocab_start=0, vocab_rows=VOCAB - 40)
    assert int(cand.token_ids.max()) < VOCAB - 40


def test_tp_merge_keeps_the_best_exact_logits_over_every_rank():
    hidden, down, score, lm_head = _weights(1)
    tp, rows = 4, VOCAB // 4
    shards = [
        shard_candidates(
            hidden,
            down,
            score[r * rows : (r + 1) * rows],
            lm_head[r * rows : (r + 1) * rows],
            num_candidates=C,
            vocab_start=r * rows,
            vocab_rows=rows,
        )
        for r in range(tp)
    ]

    def all_gather(t: torch.Tensor) -> torch.Tensor:  # rank-major concat, as tensor_model_parallel_all_gather
        field = "logits" if t.is_floating_point() else "token_ids"
        return torch.cat([getattr(s, field) for s in shards], dim=-1)

    merged = merge_candidates(shards[0], C, all_gather)
    pool_ids = torch.cat([s.token_ids for s in shards], dim=-1)
    pool_logits = torch.cat([s.logits for s in shards], dim=-1)
    for b in range(3):
        best = pool_ids[b][pool_logits[b].topk(C).indices]
        assert set(merged.token_ids[b].tolist()) == set(best.tolist())
        torch.testing.assert_close(merged.logits[b], lm_head[merged.token_ids[b]] @ hidden[b], rtol=1e-5, atol=1e-5)


def test_full_vocab_row_softmax_equals_the_candidate_softmax():
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.0, 0.0, 3.0]])
    ids = torch.tensor([[10, 3, 7], [0, 11, 5]])
    full = scatter_to_vocab(Candidates(logits, ids), 12)
    assert full.shape == (2, 12) and torch.isinf(full).sum() == 2 * 9
    probs = torch.softmax(full, -1)
    torch.testing.assert_close(probs.gather(1, ids), torch.softmax(logits, -1))
    assert probs.argmax(-1).tolist() == [10, 5]


def test_packed_gather_dot_is_the_gathered_candidate_block():
    """[logits as fp32 bits | ids + vocab_start] int32: what the rank-major all-gather moves and the tail reads."""
    hidden, _, _, lm_head = _weights()
    hidden, lm_head = hidden.to(torch.bfloat16), lm_head.to(torch.bfloat16)
    ids = torch.tensor([[0, 5, 7, 9], [511, 1, 2, 3], [3, 3, 4, 8]])
    packed = gather_dot_packed(lm_head, ids, hidden, 1000)
    assert packed.dtype == torch.int32 and packed.shape == (3, 8)
    logits, tokens = packed[:, :4].view(torch.float32), packed[:, 4:]
    assert torch.equal(logits, gather_dot_reference(lm_head, ids, hidden).float())
    assert torch.equal(tokens, (ids + 1000).to(torch.int32))
    assert logits.stride(-1) == 1 and tokens.stride(-1) == 1
