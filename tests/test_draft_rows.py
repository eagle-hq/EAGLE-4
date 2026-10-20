# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Draft-step index math and KV writes (:mod:`eagle_harness.vllm.draft_rows`) against the torch code they replace."""

import pytest
import torch

from tests.triton_probe import needs_triton

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@needs_triton
@pytest.mark.parametrize(
    "bs, steps, k, table_block, cache_block", [(1, 9, 4, 896, 16), (3, 5, 8, 32, 16), (2, 1, 2, 16, 16)]
)
def test_step_slots_match_the_torch_index_math(bs, steps, k, table_block, cache_block):
    from eagle_harness.vllm.draft_rows import draft_step_slots

    gen = torch.Generator().manual_seed(bs + steps)
    blocks_per_req = 8
    table = torch.randperm(100, generator=gen)[: bs * blocks_per_req].view(bs, blocks_per_req).int()
    committed = torch.randint(0, blocks_per_req * table_block - steps * k, (bs,), generator=gen)
    table, committed = table.to(DEV), committed.to(DEV)
    # the code this replaces (TreeDrafter.grow before)
    scratch = committed[:, None] + torch.arange(steps * k, device=DEV)
    slots = (table.gather(1, scratch // table_block).long() * table_block + scratch % table_block).view(bs, steps, k)
    rows = slots.transpose(0, 1).reshape(steps, bs * k)
    positions = (committed[None, :] + torch.arange(steps, device=DEV)[:, None]).repeat_interleave(k, dim=1)
    got = draft_step_slots(table, committed, steps=steps, top_k=k, table_block=table_block, cache_block=cache_block)
    want = (slots, rows // cache_block, rows % cache_block, positions)
    for g, w in zip(got, want):
        assert g.dtype == torch.int64 and torch.equal(g, w)


@needs_triton
@pytest.mark.parametrize("heads, dim", [(2, 64), (10, 128)])
def test_store_rows_kv_matches_the_indexed_store(heads, dim):
    from eagle_harness.vllm.draft_rows import store_rows_kv

    gen = torch.Generator().manual_seed(heads)
    rows, nblocks, block = 8, 6, 16
    # the draft layer's cache [blocks, heads, slots, 2D], viewed [blocks, slots, heads, 2D] (TreeDrafter.bind_cache)
    cache = torch.randn(nblocks, heads, block, 2 * dim, generator=gen).to(DEV, torch.bfloat16)
    view = cache.transpose(1, 2)
    qkv = torch.randn(rows, 3 * heads * dim, generator=gen).to(DEV, torch.bfloat16)
    _, key, value = qkv.split([heads * dim] * 3, dim=-1)  # strided rows, as the draft layer's split gives them
    flat = torch.randperm(nblocks * block, generator=gen)[:rows].to(DEV)
    blocks, offsets = flat // block, flat % block
    want = view.clone()
    want[blocks, offsets] = torch.cat((key.reshape(rows, heads, dim), value.reshape(rows, heads, dim)), -1)
    store_rows_kv(view, key, value, blocks, offsets)
    assert torch.equal(view, want)
