# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Tree verify attention (:func:`verify_attention`) against a dense torch reference: paged committed context plus each
row's ancestors among the new tokens, several requests of different widths, grouped query heads."""

import pytest
import torch

from tests.triton_probe import needs_triton

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# The Triton interpreter's tl.dot mishandles bf16 operands (garbage values); on CPU the kernel runs on fp32 data.
DTYPE = torch.bfloat16 if DEV == "cuda" else torch.float32


def _masks(parents: list[int], width: int) -> list[int]:
    masks = []
    for r, parent in enumerate(parents):
        masks.append((1 << r) | (masks[parent] if parent >= 0 else 0))
    return masks + [0] * (width - len(masks))


def _dense(q, k, v, scale):
    """Scores ``[Hq, rows, keys]`` of ``q`` over ``k``, and ``v`` repeated for each query head."""
    group = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
    return torch.einsum("qhd,khd->hqk", q, k) * scale, v


@needs_triton
@pytest.mark.parametrize(
    "contexts, parents, heads, kv_heads, d, page",
    [
        ([300], [[-1, 0, 0, 1, 1, 2, 3]], 6, 1, 64, 16),
        ([5, 700, 0], [[-1, 0, 1], [-1, 0, 0, 1, 2, 2, 3, 4, 5], [-1, 0]], 4, 2, 32, 32),
        ([129], [[-1] + [r - 1 for r in range(1, 24)]], 6, 1, 64, 16),  # 24 rows: a row tile with padding
    ],
)
def test_verify_attention_matches_dense(contexts, parents, heads, kv_heads, d, page):
    from eagle_harness.vllm.verify_attention import verify_attention

    gen = torch.Generator().manual_seed(len(contexts) * 100 + heads)
    num_reqs = len(contexts)
    widths = [len(p) for p in parents]
    width = max(widths)
    max_blocks = max(c // page + 1 for c in contexts) + 1
    num_blocks = num_reqs * max_blocks + 3
    kv = torch.randn(num_blocks, page, kv_heads, 2 * d, generator=gen).to(DTYPE)
    key_cache, value_cache = kv[..., :d], kv[..., d:]
    table = torch.randperm(num_blocks, generator=gen)[: num_reqs * max_blocks].view(num_reqs, max_blocks).int()
    tokens = sum(widths)
    q = torch.randn(tokens + 2, heads, d, generator=gen).to(DTYPE)  # two rows past the last request
    k_new = torch.randn(tokens + 2, kv_heads, d, generator=gen).to(DTYPE)
    v_new = torch.randn(tokens + 2, kv_heads, d, generator=gen).to(DTYPE)
    qsl = torch.tensor([0] + [sum(widths[: i + 1]) for i in range(num_reqs)], dtype=torch.int32)
    masks = torch.tensor([_masks(p, width) for p in parents], dtype=torch.int64)
    out = torch.full_like(q, 7.0)
    scale = d**-0.5

    to = lambda *ts: [t.to(DEV) for t in ts]  # noqa: E731
    q_d, kn_d, vn_d, kc_d, vc_d, out_d = to(q, k_new, v_new, key_cache, value_cache, out)
    verify_attention(
        q_d,
        kn_d,
        vn_d,
        kc_d,
        vc_d,
        table.to(DEV),
        torch.tensor(contexts, dtype=torch.int32, device=DEV),
        qsl.to(DEV),
        masks.to(DEV),
        scale,
        out_d,
        max_query_len=width,
    )
    out_d = out_d.cpu().float()

    for b in range(num_reqs):
        c, start, w = contexts[b], int(qsl[b]), widths[b]
        slots = [(int(table[b, p // page]), p % page) for p in range(c)]
        ctx_k = torch.stack([key_cache[blk, s] for blk, s in slots]) if slots else key_cache[:0, 0]
        ctx_v = torch.stack([value_cache[blk, s] for blk, s in slots]) if slots else value_cache[:0, 0]
        keys = torch.cat([ctx_k, k_new[start : start + w]]).float()
        values = torch.cat([ctx_v, v_new[start : start + w]]).float()
        scores, values = _dense(q[start : start + w].float(), keys, values, scale)
        tree_visible = torch.tensor([[bool(masks[b, r] >> j & 1) for j in range(w)] for r in range(w)])
        visible = torch.cat([torch.ones(w, c, dtype=torch.bool), tree_visible], 1)
        scores = scores.masked_fill(~visible, float("-inf"))
        want = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values)
        torch.testing.assert_close(out_d[start : start + w], want, atol=2e-2, rtol=2e-2)
    assert torch.equal(out_d[tokens:], torch.full_like(out_d[tokens:], 7.0))  # rows past the batch untouched
