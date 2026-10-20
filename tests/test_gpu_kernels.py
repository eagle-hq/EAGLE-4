# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""GPU kernels of the vLLM adapter against dense torch references (needs a GPU and vLLM's GPU build)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
# a CPU-only vLLM build lacks the FlashAttention entry points the tree attention imports
pytest.importorskip("eagle_harness.vllm.tree_attention", reason="needs vLLM's GPU build", exc_type=ImportError)

from eagle_harness.vllm.draft_attention import build_tree_rows, row_attention  # noqa: E402
from eagle_harness.vllm.round_commit import KvCompactor  # noqa: E402
from eagle_harness.vllm.verify_layout import TreeStore, VerifyStage  # noqa: E402

DEV = torch.device("cuda")
PARENTS = [-1, 0, 0, 1, 1, 2, 3]  # width 7


def _stage(num_reqs: int, width: int, parents: list[int]):
    store = TreeStore(8, width, DEV)
    store.parents[:] = torch.tensor(parents + [-1] * (width - len(parents)), dtype=torch.int32, device=DEV)
    stage = VerifyStage(8, 8 * width, width, DEV)
    qsl = torch.arange(0, (num_reqs + 1) * width, width, dtype=torch.int32, device=DEV)
    positions = (100 + torch.arange(width, device=DEV)).repeat(num_reqs).to(torch.int64)
    stage.stage(
        store,
        torch.arange(num_reqs, dtype=torch.int32, device=DEV),
        qsl,
        qsl,
        positions,
        num_reqs,
        num_reqs * width,
        tree=True,
    )
    return stage, qsl, positions


def _random_parents(width: int, seed: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return [-1] + [int(torch.randint(0, r, (1,), generator=gen)) for r in range(1, width)]


@pytest.mark.parametrize(
    "parents", [PARENTS, _random_parents(16, 1), _random_parents(16, 2), _random_parents(64, 3), [-1, 0, 1, 2, 3]]
)
def test_stage_depths_masks_and_window_ancestors(parents):
    width = len(parents)
    stage, _, positions = _stage(2, width, parents)
    depth, window, mask = [0], [[-1, -2, -3]], [1]
    for row in range(1, width):
        p = parents[row]
        depth.append(depth[p] + 1)
        window.append([p, window[p][0], window[p][1]])
        mask.append(1 << row | mask[p])
    mask = [m - (1 << 64) if m >= 1 << 63 else m for m in mask]  # int64
    for b in range(2):
        assert (positions[b * width : (b + 1) * width] - 100).tolist() == depth
        assert stage.ancestors[b].t().tolist() == window
        assert stage.masks[b].tolist() == mask
        assert stage.parents[b].tolist() == parents
        assert stage.slot_parents[b].tolist() == parents  # request b sits in state slot b
        assert stage.slot_depths[b].tolist() == depth
        assert int(stage.context_lens[b]) == 100  # the first query token's chain position


def test_row_attention_matches_dense():
    torch.manual_seed(0)
    blocks, block_size, h, d, rows, n = 8, 16, 4, 128, 5, 100
    stored = torch.randn(blocks, h, block_size, 2 * d, device=DEV, dtype=torch.bfloat16)
    cache = stored.transpose(1, 2)  # the backend's view: (blocks, slots, heads, 2 * head_dim)
    q = torch.randn(rows, h, d, device=DEV, dtype=torch.bfloat16)
    keys = torch.randint(0, blocks * block_size, (rows, n), device=DEV, dtype=torch.int32)
    num_keys = torch.tensor([n, 1, 37, 64, 99], dtype=torch.int32, device=DEV)
    out = row_attention(q, cache, keys, num_keys, d**-0.5)
    flat = cache.reshape(-1, h, 2 * d)
    for r in range(rows):
        kv = flat[keys[r, : num_keys[r]].long()].float()
        scores = torch.einsum("hd,khd->hk", q[r].float(), kv[..., :d]) * d**-0.5
        want = torch.einsum("hk,khd->hd", scores.softmax(-1), kv[..., d:])
        torch.testing.assert_close(out[r].float(), want, atol=2e-2, rtol=2e-2)


def test_build_tree_rows_window():
    block_size = 16
    table = torch.arange(1, 9, dtype=torch.int32, device=DEV)[None].repeat(1, 1)
    committed = torch.tensor([50], dtype=torch.int32, device=DEV)
    ancestors = torch.tensor([[[900, 901], [900, 902]]], dtype=torch.int32, device=DEV)  # bs=1, K=2, depth=2
    keys = torch.zeros(2, 64, dtype=torch.int32, device=DEV)
    num_keys = torch.zeros(2, dtype=torch.int32, device=DEV)
    build_tree_rows(table, block_size, committed, ancestors, keys, num_keys, step=2, window=10)
    # the query at position 51 sees positions 42..51: 8 committed keys (42..49) and its 2 ancestors (50, 51)
    assert num_keys.tolist() == [10, 10]
    want = [int(table[0, p // block_size]) * block_size + p % block_size for p in range(42, 50)]
    assert keys[0, :10].tolist() == want + [900, 901]
    assert keys[1, :10].tolist() == want + [900, 902]


@pytest.mark.parametrize("slot_major", [False, True])  # the order of slots and heads in memory
@pytest.mark.parametrize(
    "blocks_per_table_block", [1, 4]
)  # the block table's blocks are made of that many cache blocks
def test_kv_compaction_moves_accepted_rows(slot_major, blocks_per_table_block):
    block_size, heads, d, num_layers = 16, 4, 8, 2
    table_block_size = block_size * blocks_per_table_block
    num_cache_blocks = 12 * blocks_per_table_block
    shape = (num_cache_blocks, block_size, heads, 2 * d) if slot_major else (num_cache_blocks, heads, block_size, 2 * d)
    raw = [torch.randn(shape, device=DEV, dtype=torch.bfloat16) for _ in range(num_layers)]
    # the backend's view of every cache: (blocks, heads, slots in a block, 2 * head_dim)
    caches = [r.transpose(1, 2) if slot_major else r for r in raw]
    before = [c.clone() for c in caches]
    table = torch.tensor([[3, 1, 7, 2, 5, 9]], dtype=torch.int32, device=DEV)  # shuffled blocks of the group
    store = TreeStore(2, 7, DEV)
    # the request lives in state slot 1 but is row 0 of the batch: the block table is the batch's, as the forward got it
    store.path[1, :4] = torch.tensor([0, 2, 5, 6], dtype=torch.int32, device=DEV)
    store.path_len[1] = 4
    context, width = 20 * blocks_per_table_block + 9, 7
    qsl = torch.tensor([0, width], dtype=torch.int32, device=DEV)
    compactor = KvCompactor(caches, [table] * num_layers, [table_block_size] * num_layers, DEV)
    compactor.compact(
        torch.ones(1, dtype=torch.int32, device=DEV),
        qsl,
        qsl,
        torch.tensor([context + width], dtype=torch.int32, device=DEV),
        store.path,
        store.path_len,
    )

    def where(token):  # (cache block, slot) of a token of the request
        block = (
            int(table[0, token // table_block_size]) * blocks_per_table_block + token % table_block_size // block_size
        )
        return block, token % block_size

    for layer in range(num_layers):
        for j, row in enumerate([0, 2, 5, 6]):
            (sb, ss), (db, ds) = where(context + row), where(context + j)
            torch.testing.assert_close(caches[layer][db, :, ds], before[layer][sb, :, ss])
        untouched = torch.ones(num_cache_blocks, block_size, dtype=torch.bool)
        for j in range(1, 4):
            untouched[where(context + j)] = False
        torch.testing.assert_close(caches[layer].transpose(1, 2)[untouched], before[layer].transpose(1, 2)[untouched])


def test_a_graph_capture_takes_the_tree_path_whatever_was_staged_last():
    stage = VerifyStage(2, 16, 8, DEV)
    stage.active = False  # what the last dummy or chain batch left
    assert not stage.verifying
    seen = []
    torch.zeros(1, device=DEV)  # initialize the context before capturing
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        seen.append(stage.verifying)
    assert seen == [True]
    stage.active = True
    assert stage.verifying
