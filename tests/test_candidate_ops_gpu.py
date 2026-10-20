# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""GPU checks of the candidate top-k launchers against their torch references (skipped without CUDA).

Run in a GPU job: ``pytest -q tests/test_candidate_ops_gpu.py``.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.mark.parametrize("cuda_kernels", [True, False])
@pytest.mark.parametrize("rows,vocab,k", [(1, 62080, 128), (16, 62080, 128), (64, 248320, 64)])
def test_shard_topk_matches_torch(cuda_kernels, rows, vocab, k):
    from eagle_harness.kernels.candidate_ops import shard_topk

    gen = torch.Generator(device="cuda").manual_seed(rows + vocab)
    scores = torch.randn(rows, vocab, device="cuda", dtype=torch.bfloat16, generator=gen)
    scores[:, 7] = scores[:, 9]  # ties keep torch's order
    local, global_ids = shard_topk(scores, k, 1000, cuda_kernels=cuda_kernels)
    ref = torch.topk(scores, k, dim=-1, sorted=False).indices
    assert torch.equal(global_ids, local + 1000)
    if torch.version.hip:
        # ROCm topk uses a different unsorted order and tie break than CUDA.
        values = torch.gather(scores, -1, local)
        assert torch.equal(values.sort().values, torch.gather(scores, -1, ref).sort().values)
        assert torch.all(local.sort().values[:, 1:] != local.sort().values[:, :-1])
    else:
        assert torch.equal(local, ref)


@pytest.mark.parametrize("cuda_kernels", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])  # fp32: the packed TP gather
@pytest.mark.parametrize("ranks,rows,nc,top_k", [(1, 4, 128, 4), (4, 1, 128, 4), (4, 16, 128, 8)])
def test_candidate_tail_matches_torch(cuda_kernels, dtype, ranks, rows, nc, top_k):
    from eagle_harness.kernels.candidate_ops import candidate_tail

    gen = torch.Generator(device="cuda").manual_seed(ranks * 100 + rows)
    logits = torch.randn(ranks * rows, nc, device="cuda", dtype=torch.bfloat16, generator=gen).to(dtype)
    tokens = torch.randperm(ranks * rows * nc, device="cuda", generator=gen).view(ranks * rows, nc)
    out_logits, out_tokens, (probs, topk_tokens) = candidate_tail(
        logits, tokens, rows, top_k, cuda_kernels=cuda_kernels
    )
    # The same candidates as [R, M, NC] views of a last-dim gather ([M, R * NC]) give the same result.
    by_row = (
        logits.view(ranks, rows, nc).transpose(0, 1).contiguous(),
        tokens.view(ranks, rows, nc).transpose(0, 1).contiguous(),
    )
    views = candidate_tail(by_row[0].transpose(0, 1), by_row[1].transpose(0, 1), rows, top_k, cuda_kernels=cuda_kernels)
    assert all(torch.equal(a, b) for a, b in zip((*views[:2], *views[2]), (out_logits, out_tokens, probs, topk_tokens)))
    gathered = logits.view(ranks, rows, nc).permute(1, 0, 2).reshape(rows, ranks * nc)
    gathered_tokens = tokens.view(ranks, rows, nc).permute(1, 0, 2).reshape(rows, ranks * nc)
    if ranks > 1:
        ref_logits, pos = torch.topk(gathered, nc, dim=-1, sorted=False)
        ref_tokens = torch.gather(gathered_tokens, -1, pos)
    else:
        ref_logits, ref_tokens = gathered, gathered_tokens
    ref_logits = ref_logits.float()
    if torch.version.hip:
        assert torch.equal(out_logits.sort().values, ref_logits.sort().values)
        # Check each selected token keeps its exact logit, independent of ties/order.
        source = torch.empty_like(logits.flatten()).scatter_(0, tokens.flatten(), logits.flatten())
        assert torch.equal(out_logits, source[out_tokens].float())
        for row in range(rows):
            assert torch.all(torch.isin(out_tokens[row], gathered_tokens[row]))
        ordered = out_tokens.sort().values
        assert torch.all(ordered[:, 1:] != ordered[:, :-1])
        ordered_topk = topk_tokens.sort().values
        assert torch.all(ordered_topk[:, 1:] != ordered_topk[:, :-1])
        exact = out_logits.double().softmax(-1)
        torch.testing.assert_close(probs.double(), exact.topk(top_k).values, rtol=1e-6, atol=1e-8)
        matches = out_tokens[:, :, None] == topk_tokens[:, None, :]
        assert torch.all(matches.sum(1) == 1)
        picked = (exact[:, :, None] * matches).sum(1)
        torch.testing.assert_close(probs.double(), picked, rtol=1e-6, atol=1e-8)
        return
    ref_probs, pick = torch.topk(torch.softmax(ref_logits, -1), top_k)
    assert torch.equal(out_logits, ref_logits)
    assert torch.equal(out_tokens, ref_tokens)
    assert torch.equal(probs, ref_probs)
    assert torch.equal(topk_tokens, torch.gather(ref_tokens, -1, pick))


def test_native_library_is_used_where_the_gpu_has_it():
    from eagle_harness.kernels import arch, native

    if not arch.native_candidate_tail("cuda"):
        pytest.skip("native kernels are NVIDIA sm_80+ only")
    assert native.library() is not None


@pytest.mark.parametrize("rows,k,h,vocab", [(1, 128, 5120, 62080), (4, 128, 5120, 62080), (64, 64, 640, 4096)])
def test_gather_dot_matches_reference_and_packed_layout(rows, k, h, vocab):
    from eagle_harness.kernels.gather_dot import gather_dot, gather_dot_packed, pack_reference

    gen = torch.Generator(device="cuda").manual_seed(rows + k)
    weight = torch.randn(vocab, h, device="cuda", dtype=torch.bfloat16, generator=gen)
    hidden = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16, generator=gen)
    ids = torch.randint(0, vocab, (rows, k), device="cuda", generator=gen)
    got = gather_dot(weight, ids, hidden)
    exact = torch.einsum("bkh,bh->bk", weight[ids].double(), hidden.double())
    torch.testing.assert_close(got.double(), exact, rtol=2**-7, atol=2**-6)  # one bf16 rounding of the exact sum
    packed = gather_dot_packed(weight, ids, hidden, 4096)
    assert torch.equal(packed, pack_reference(got, ids + 4096))


@pytest.mark.parametrize("cuda_kernels", [True, False])
@pytest.mark.parametrize("ranks,rows,nc,top_k", [(4, 1, 128, 4), (4, 16, 128, 8), (2, 4, 64, 4)])
def test_candidate_topk_packed_gather_matches_the_unfused_path(cuda_kernels, ranks, rows, nc, top_k):
    from eagle_harness.kernels.candidate_ops import candidate_tail, candidate_topk, shard_topk
    from eagle_harness.kernels.gather_dot import gather_dot, gather_dot_packed

    gen = torch.Generator(device="cuda").manual_seed(ranks * 31 + rows)
    shard, h = 4096, 256
    scores = [torch.randn(rows, shard, device="cuda", dtype=torch.bfloat16, generator=gen) for _ in range(ranks)]
    weights = [torch.randn(shard, h, device="cuda", dtype=torch.bfloat16, generator=gen) for _ in range(ranks)]
    hidden = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16, generator=gen)
    local = [shard_topk(scores[r], nc, r * shard, cuda_kernels=cuda_kernels) for r in range(ranks)]
    # Unfused: per-rank gather-dot, rank-major concatenation, the tail on fp32 logits and int64 ids.
    logits = torch.cat([gather_dot(weights[r], local[r][0], hidden) for r in range(ranks)]).float()
    ids = torch.cat([local[r][1] for r in range(ranks)])
    _, _, (want_probs, want_tokens) = candidate_tail(logits, ids, rows, top_k, cuda_kernels=cuda_kernels)
    # Fused: rank 0 calls candidate_topk; the all-gather delivers every rank's packed block as [R, M, n], here as the
    # views of a last-dim gather (sglang) and of a dim-0 gather (vLLM).
    blocks = [gather_dot_packed(weights[r], local[r][0], hidden, r * shard) for r in range(ranks)]
    gathers = {
        "last dim": lambda own: torch.cat([own, *blocks[1:]], dim=-1).view(rows, ranks, -1).transpose(0, 1),
        "dim 0": lambda own: torch.cat([own, *blocks[1:]]).view(ranks, rows, -1),
    }
    for name, gather in gathers.items():
        probs, tokens = candidate_topk(
            scores[0], weights[0], hidden, num_candidates=nc, vocab_start=0, top_k=top_k,
            all_gather=gather, cuda_kernels=cuda_kernels,
        )  # fmt: skip
        assert torch.equal(probs, want_probs) and torch.equal(tokens, want_tokens), name
