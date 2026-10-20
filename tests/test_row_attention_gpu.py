# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The draft rows' attention kernels against a torch reference (skipped without CUDA).

Run in a GPU job: ``pytest -q tests/test_row_attention_gpu.py``.
"""

import pytest
import torch

pytest.importorskip("sglang", reason="the sglang kernels need sglang installed")

from eagle_harness.kernels.row_attention import rows_attention  # noqa: E402
from eagle_harness.sglang.row_attention import EXTEND_LAUNCH, MAX_SPLITS, STEP_LAUNCH, DraftRowAttention  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

K, DEPTH, HEADS, HEAD_DIM, WIDTH = 4, 12, 3, 128, 16


def _table(gen: torch.Generator) -> torch.Tensor:
    return torch.randperm(8 * 4096, device="cuda", generator=gen).to(torch.int32).view(8, 4096)


def _reference(q, k_buffer, v_buffer, keys, scale):
    """One row: ``q`` ``[heads, dim]`` over the KV slots ``keys``, softmax in fp32."""
    k = k_buffer[keys].float()  # [n, heads, dim]
    v = v_buffer[keys].float()
    scores = torch.einsum("hd,nhd->hn", q.float(), k) * scale
    return torch.einsum("hn,nhd->hd", scores.softmax(-1), v)


def _run(q, k_buffer, v_buffer, req_to_token, req, seq_lens, ancestors, *, rows_per_request, step, window):
    rows = q.shape[0]
    bs = rows // rows_per_request
    parts = bs * HEADS * (MAX_SPLITS + 1) * 16
    out = torch.empty_like(q)
    part_out = torch.empty(parts, HEAD_DIM, dtype=torch.float32, device="cuda")
    part_lse = torch.empty(parts, dtype=torch.float32, device="cuda")
    rows_attention(
        q,
        k_buffer,
        v_buffer,
        out,
        part_out,
        part_lse,
        req_to_token,
        seq_lens,
        ancestors,
        rows_per_request=rows_per_request,
        step=step,
        window=window,
        scale=HEAD_DIM**-0.5,
        launch=EXTEND_LAUNCH if step == 0 else STEP_LAUNCH,
        table_rows=req,
    )
    return out


def _inputs(gen, bs, rows_per_request, low, high):
    req_to_token = _table(gen)
    req = torch.tensor([5, 0, 3][:bs], dtype=torch.int64, device="cuda")
    seq_lens = torch.randint(low, high, (bs,), device="cuda", generator=gen)
    k_buffer = torch.randn(8 * 4096, HEADS, HEAD_DIM, device="cuda", generator=gen).bfloat16()
    v_buffer = torch.randn(8 * 4096, HEADS, HEAD_DIM, device="cuda", generator=gen).bfloat16()
    q = torch.randn(bs * rows_per_request, HEADS, HEAD_DIM, device="cuda", generator=gen).bfloat16()
    return req_to_token, req, seq_lens, k_buffer, v_buffer, q


@pytest.mark.parametrize("window", [0, 96])
@pytest.mark.parametrize("bs,step", [(1, 1), (1, 5), (3, 7), (2, 11)])
def test_tree_rows_allocate_the_step_slots(bs, step, window):
    gen = torch.Generator(device="cuda").manual_seed(bs * 100 + step)
    req_to_token = _table(gen)
    req = torch.tensor([5, 0, 3][:bs], dtype=torch.int64, device="cuda")
    seq_lens = torch.randint(40, 300, (bs,), device="cuda", generator=gen)
    ancestors = torch.randint(0, 10**6, (bs, K, DEPTH), dtype=torch.int32, device="cuda", generator=gen)
    attention = DraftRowAttention.__new__(DraftRowAttention)  # no draft runner: only the build kernel
    attention.req_to_token, attention.window = req_to_token, window
    slots = torch.zeros(bs * K, dtype=torch.int64, device="cuda")
    positions = torch.zeros(bs * K, dtype=torch.int64, device="cuda")
    before = ancestors.clone()

    attention.build_tree_rows(ancestors, seq_lens, req, slots, positions, bs=bs, step=step)

    for b in range(bs):
        length = int(seq_lens[b])
        for lane in range(K):
            row = b * K + lane
            own = int(req_to_token[req[b], length + (step - 1) * K + lane])
            assert int(slots[row]) == own and int(positions[row]) == length + step - 1
            assert int(ancestors[b, lane, step - 1]) == own
            assert torch.equal(ancestors[b, lane, : step - 1], before[b, lane, : step - 1])


@pytest.mark.parametrize("window,low,high", [(0, 40, 300), (96, 40, 300), (2048, 2500, 3900)])
@pytest.mark.parametrize("bs,step", [(1, 1), (1, 5), (3, 7), (2, 11)])
def test_tree_rows_attend_window_and_ancestors(bs, step, window, low, high):
    gen = torch.Generator(device="cuda").manual_seed(bs * 1000 + step + window)
    req_to_token, req, seq_lens, k_buffer, v_buffer, q = _inputs(gen, bs, K, low, high)
    ancestors = torch.randint(0, 8 * 4096, (bs, K, DEPTH), dtype=torch.int32, device="cuda", generator=gen)

    out = _run(
        q, k_buffer, v_buffer, req_to_token, req, seq_lens, ancestors, rows_per_request=K, step=step, window=window
    )

    for b in range(bs):
        length = int(seq_lens[b])
        start = max(length + step - 1 - window + 1, 0) if window > 0 else 0
        for lane in range(K):
            keys = torch.cat([req_to_token[req[b], start:length], ancestors[b, lane, :step]]).long()
            want = _reference(q[b * K + lane], k_buffer, v_buffer, keys, HEAD_DIM**-0.5)
            torch.testing.assert_close(out[b * K + lane].float(), want, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("width", [WIDTH, 8, 1])
@pytest.mark.parametrize("bs", [1, 3])
def test_extend_rows_take_the_block_table_after_the_committed_length(bs, width):
    gen = torch.Generator(device="cuda").manual_seed(bs * 31 + width)
    req_to_token = _table(gen)
    req = torch.tensor([5, 0, 3][:bs], dtype=torch.int64, device="cuda")
    seq_lens = torch.randint(40, 300, (bs,), device="cuda", generator=gen)
    accept_lens = torch.randint(1, width + 1, (bs,), device="cuda", generator=gen)
    attention = DraftRowAttention.__new__(DraftRowAttention)  # no draft runner: only the build kernel
    attention.req_to_token = req_to_token
    slots = torch.zeros(bs * width, dtype=torch.int64, device="cuda")
    positions = torch.zeros(bs * width, dtype=torch.int64, device="cuda")
    last = torch.zeros(bs, dtype=torch.int64, device="cuda")

    attention.build_extend_rows(seq_lens, req, accept_lens, slots, positions, last, bs=bs, width=width)

    want_positions = seq_lens[:, None] + torch.arange(width, device="cuda")
    assert torch.equal(positions, want_positions.reshape(-1))
    assert torch.equal(slots, req_to_token[req[:, None], want_positions].long().reshape(-1))
    assert torch.equal(last, torch.arange(bs, device="cuda") * width + accept_lens - 1)


@pytest.mark.parametrize("window,low,high", [(0, 40, 300), (96, 40, 300), (2048, 2500, 3900)])
@pytest.mark.parametrize("bs,width", [(1, WIDTH), (3, WIDTH), (1, 8), (2, 1)])
def test_extend_rows_attend_their_causal_window(bs, width, window, low, high):
    gen = torch.Generator(device="cuda").manual_seed(bs * 7 + window + width)
    req_to_token, req, seq_lens, k_buffer, v_buffer, q = _inputs(gen, bs, width, low, high)

    out = _run(q, k_buffer, v_buffer, req_to_token, req, seq_lens, None, rows_per_request=width, step=0, window=window)

    for b in range(bs):
        for j in range(width):
            pos = int(seq_lens[b]) + j
            start = max(pos - window + 1, 0) if window > 0 else 0
            keys = req_to_token[req[b], start : pos + 1].long()
            want = _reference(q[b * width + j], k_buffer, v_buffer, keys, HEAD_DIM**-0.5)
            torch.testing.assert_close(out[b * width + j].float(), want, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("heads", [8, 32])  # Nemotron draft at TP4 and TP1
@pytest.mark.parametrize("length", [4095, 4096, 4100, 8192])
@pytest.mark.parametrize("step", [0, 7])  # extend rows and a branching draft
def test_nemotron_window_boundary(heads, length, step):
    window, dim = 4096, 128
    rows = 4 if step else 16
    gen = torch.Generator(device="cuda").manual_seed(length + heads + step)
    slots = length + rows * max(step, 1)
    table = torch.randperm(slots, device="cuda", generator=gen).to(torch.int32).unsqueeze(0)
    req = torch.zeros(1, device="cuda", dtype=torch.int64)
    seq = torch.tensor([length], device="cuda", dtype=torch.int64)
    q = torch.randn(rows, heads, dim, device="cuda", generator=gen).bfloat16()
    k = torch.randn(slots, heads, dim, device="cuda", generator=gen).bfloat16()
    v = torch.randn(slots, heads, dim, device="cuda", generator=gen).bfloat16()
    ancestors = table[:, length:].view(1, rows, step) if step else None
    out = torch.empty_like(q)
    parts = heads * (MAX_SPLITS + 1) * 16
    partial = torch.empty(parts, dim, device="cuda", dtype=torch.float32)
    lse = torch.empty(parts, device="cuda", dtype=torch.float32)

    def run():
        rows_attention(
            q,
            k,
            v,
            out,
            partial,
            lse,
            table,
            req,
            seq,
            ancestors,
            rows_per_request=rows,
            step=step,
            window=window,
            scale=dim**-0.5,
        )
        return out.clone()

    got = run()
    for row in range(rows):
        pos = length + (step - 1 if step else row)
        start = max(pos - window + 1, 0)
        if step:
            assert ancestors is not None
            keys = torch.cat([table[0, start:length], ancestors[0, row]])
        else:
            keys = table[0, start : pos + 1]
        want = _reference(q[row], k, v, keys.long(), dim**-0.5)
        torch.testing.assert_close(got[row].float(), want, atol=2e-2, rtol=2e-2)

    # Every row excludes this prefix. Poisoning it must not affect the result.
    dropped = max(length + (step - 1 if step else 0) - window + 1, 0)
    k[table[0, :dropped].long()] = 100
    v[table[0, :dropped].long()] = -100
    assert torch.equal(got, run())
