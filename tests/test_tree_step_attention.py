# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""A draft step's shared-key attention (:func:`tree_step_attention`) against a dense torch reference."""

import pytest
import torch

from tests.triton_probe import needs_triton

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# The Triton interpreter's tl.dot mishandles bf16 operands (garbage values); on CPU the kernel runs on fp32 data.
DTYPE = torch.bfloat16 if DEV == "cuda" else torch.float32


def _case(bs, top_k, heads, d, step, committed, window, cache_block=16, per_table_block=4, depth=6, kv_heads=None):
    gen = torch.Generator().manual_seed(bs * 100 + step)
    table_block = cache_block * per_table_block
    max_blocks = 8
    num_cache_blocks = bs * max_blocks * per_table_block + 8
    raw = torch.randn(num_cache_blocks, kv_heads or heads, cache_block, 2 * d, generator=gen).to(DTYPE)
    kv = raw.to(DEV).transpose(1, 2)  # the drafter's view: [blocks, slots, heads, 2 * D]
    table = torch.randperm(bs * max_blocks, generator=gen).view(bs, max_blocks).int()
    committed_t = torch.tensor(committed, dtype=torch.int64)
    # ancestors: flat token slots past the committed length (the drafter's scratch slots)
    ancestors = torch.zeros(bs, top_k, depth, dtype=torch.int32)
    for b in range(bs):
        for lane in range(top_k):
            for s in range(step):
                p = committed[b] + s * top_k + lane
                ancestors[b, lane, s] = int(table[b, p // table_block]) * table_block + p % table_block
    q = torch.randn(bs * top_k, heads, d, generator=gen).to(DTYPE)
    return q.to(DEV), kv, table.to(DEV), table_block, committed_t.to(DEV), ancestors.to(DEV)


@needs_triton
@pytest.mark.parametrize(
    "bs, step, committed, window",
    [(1, 1, [70], 0), (2, 3, [5, 200], 0), (1, 4, [300], 100), (2, 2, [129, 64], 64)],
)
@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
def test_tree_step_attention_matches_dense_torch(bs, step, committed, window, group_size):
    from eagle_harness.vllm.draft_attention import tree_step_attention

    top_k, heads, d = 4, 8, 64
    kv_heads = heads // group_size
    q, kv, table, table_block, committed_t, ancestors = _case(
        bs, top_k, heads, d, step, committed, window, kv_heads=kv_heads
    )
    got = tree_step_attention(q, kv, table, table_block, committed_t, ancestors, d**-0.5, step=step, window=window)
    flat = kv.reshape(-1, kv_heads, 2 * d).repeat_interleave(group_size, dim=1).float()
    for b in range(bs):
        # each row sees the window frozen at the last committed position c - 1, then its own ancestors
        c = committed[b]
        start = max(0, c - window) if window else 0
        prefix = [int(table[b, p // table_block]) * table_block + p % table_block for p in range(start, c)]
        for lane in range(top_k):
            r = b * top_k + lane
            slots = torch.tensor(prefix + ancestors[b, lane, :step].tolist(), device=DEV)
            scores = torch.einsum("hd,khd->hk", q[r].float(), flat[slots, :, :d]) * d**-0.5
            dense = torch.einsum("hk,khd->hd", scores.softmax(-1), flat[slots, :, d:])
            torch.testing.assert_close(got[r].float(), dense, atol=2e-2, rtol=2e-2)


@needs_triton
@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("step,window", [(0, 0), (0, 7), (3, 0), (3, 2)])
def test_shared_rows_gqa_matches_torch_and_repeated_mha(group_size, packed, step, window):
    from eagle_harness.kernels.row_attention import RowsLaunch, rows_attention, rows_padded

    gen = torch.Generator().manual_seed(813 + group_size + step)
    bs, rows, heads, dim, slots = 2, 3, 8, 64, 256
    kv_heads = heads // group_size
    q = torch.randn(bs * rows, heads, dim, generator=gen).to(device=DEV, dtype=DTYPE)
    # Separate K/V have different strides; packed K/V preserves the one-pointer path.
    keys = torch.randn(slots, kv_heads, 2 * dim, generator=gen).to(device=DEV, dtype=DTYPE)
    values = (
        None if packed else torch.randn(kv_heads, slots, dim, generator=gen).to(device=DEV, dtype=DTYPE).transpose(0, 1)
    )
    if not packed:
        keys = keys[..., :dim]
    table = torch.randperm(slots, generator=gen).int().reshape(4, 64).to(DEV)
    table_rows = torch.tensor([3, 1], device=DEV)
    lengths = torch.tensor([0, 43], device=DEV)
    ancestors = torch.randint(slots, (bs, rows, 3), generator=gen).int().to(DEV)
    launch = RowsLaunch(splits=4, block_n=32)
    parts = torch.empty(bs, heads, launch.splits + 1, rows_padded(rows), dim, device=DEV)
    lse = torch.empty(parts.shape[:-1], device=DEV)

    def run(k, v):
        out = torch.empty_like(q)
        rows_attention(
            q,
            k,
            v,
            out,
            parts,
            lse,
            table,
            lengths,
            ancestors,
            rows_per_request=rows,
            step=step,
            window=window,
            scale=dim**-0.5,
            launch=launch,
            table_rows=table_rows,
        )
        return out

    got = run(keys, values)
    repeated = run(
        keys.repeat_interleave(group_size, dim=1), None if packed else values.repeat_interleave(group_size, dim=1)
    )
    assert torch.equal(got, repeated)
    k = keys[..., :dim].repeat_interleave(group_size, dim=1).float()
    v = (keys[..., dim:] if packed else values).repeat_interleave(group_size, dim=1).float()
    for b in range(bs):
        length = int(lengths[b])
        for row in range(rows):
            pos = length - 1 if step else length + row  # a draft step's window ends at the last committed position
            start = max(0, pos - window + 1) if window else 0
            end = length if step else pos + 1
            selected = table[table_rows[b], min(start, end) : end]
            if step:
                selected = torch.cat((selected, ancestors[b, row, :step]))
            selected = selected.long()
            scores = torch.einsum("hd,nhd->hn", q[b * rows + row].float(), k[selected]) * dim**-0.5
            want = torch.einsum("hn,nhd->hd", scores.softmax(-1), v[selected])
            torch.testing.assert_close(got[b * rows + row].float(), want, atol=2e-2, rtol=2e-2)


@needs_triton
@pytest.mark.parametrize("kv_heads", [0, 3, 16])
def test_rows_reject_nonintegral_query_groups(kv_heads):
    from eagle_harness.vllm.draft_attention import tree_step_attention

    q, cache, table, table_block, lengths, ancestors = _case(1, 4, 8, 64, 1, [70], 0)
    invalid = torch.empty(cache.shape[0], cache.shape[1], kv_heads, cache.shape[3], device=DEV)
    with pytest.raises(ValueError, match="must be divisible"):
        tree_step_attention(q, invalid, table, table_block, lengths, ancestors, 0.125, step=1, window=0)
