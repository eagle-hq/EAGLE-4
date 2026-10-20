# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The tree round's small kernels against Python references (CPU through the Triton interpreter, or a GPU):
the tree walk, the draft's re-ordering of the accepted rows, and the GDN state commit."""

import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

pytest.importorskip("vllm")

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first  # noqa: E402

from eagle_harness.vllm.gdn_tree_verify import LinearStateCommit  # noqa: E402
from eagle_harness.vllm.round_commit import chainize  # noqa: E402
from eagle_harness.vllm.tree_accept import accept_tree  # noqa: E402
from eagle_harness.vllm.verify_layout import TreeStore  # noqa: E402
from tests.triton_probe import needs_triton  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = 8
PARENTS = [-1, 0, 0, 1, 1, 2, 3, 5]


def _walk(parents, drafted, target):
    node, path, tokens = 0, [], []
    while True:
        path.append(node)
        tokens.append(target[node])
        children = [r for r in range(1, len(parents)) if parents[r] == node and drafted[r] == target[node]]
        if not children:
            return path, tokens
        node = children[0]


@needs_triton
def test_walk_follows_the_matching_child_to_the_bonus_token():
    torch.manual_seed(0)
    store = TreeStore(4, W, DEV)
    drafted = torch.randint(0, 3, (2, W))  # few distinct tokens: siblings and matches are common
    target = torch.randint(0, 3, (2, W))
    parents = torch.tensor([PARENTS, PARENTS[:5] + [-1] * 3], dtype=torch.int32, device=DEV)
    cu = torch.tensor([0, W, W + 5], dtype=torch.int32, device=DEV)
    slots = torch.tensor([3, 1], dtype=torch.int32, device=DEV)  # request state slots
    slot_parents = torch.full_like(store.path, -1)
    slot_parents[slots.long()] = parents
    sampled, num_sampled = accept_tree(
        target.reshape(-1)[: W + 5].to(DEV), drafted.reshape(-1)[: W + 5].to(DEV), cu, slots, slot_parents, store
    )
    for b, (n, plist) in enumerate([(W, PARENTS), (5, PARENTS[:5])]):
        path, tokens = _walk(plist, drafted[b, :n].tolist(), target[b, :n].tolist())
        assert int(num_sampled[b]) == len(path)
        assert sampled[b, : len(path)].tolist() == tokens
        assert int(store.path_len[int(slots[b])]) == len(path)
        assert store.path[int(slots[b]), : len(path)].tolist() == path


@needs_triton
@pytest.mark.parametrize("batch_size", [33, 64, 128, 256, 512])
def test_large_batch_walk_keeps_request_paths_separate(batch_size):
    gen = torch.Generator().manual_seed(batch_size)
    store = TreeStore(batch_size, W, DEV)
    drafted = torch.randint(0, 3, (batch_size, W), generator=gen)
    target = torch.randint(0, 3, (batch_size, W), generator=gen)
    widths = [W if b % 2 == 0 else 5 for b in range(batch_size)]
    parents = torch.tensor([PARENTS[:n] + [-1] * (W - n) for n in widths], dtype=torch.int32, device=DEV)
    cu = torch.tensor([0] + widths, dtype=torch.int32, device=DEV).cumsum(0).to(torch.int32)
    slots = torch.randperm(batch_size, generator=gen).to(device=DEV, dtype=torch.int32)
    sampled, num_sampled = accept_tree(
        torch.cat([target[b, :n] for b, n in enumerate(widths)]).to(DEV),
        torch.cat([drafted[b, :n] for b, n in enumerate(widths)]).to(DEV),
        cu,
        slots,
        parents,
        store,
    )
    sampled, num_sampled = sampled.cpu(), num_sampled.cpu()
    paths, lengths, slots = store.path.cpu(), store.path_len.cpu(), slots.cpu()
    for b, n in enumerate(widths):
        path, tokens = _walk(PARENTS[:n], drafted[b, :n].tolist(), target[b, :n].tolist())
        assert int(num_sampled[b]) == len(path)
        assert sampled[b, : len(path)].tolist() == tokens
        assert int(lengths[int(slots[b])]) == len(path)
        assert paths[int(slots[b]), : len(path)].tolist() == path


def test_chainize_puts_the_accepted_rows_in_path_order():
    store = TreeStore(4, W, DEV)
    store.path[2, :4] = torch.tensor([0, 2, 5, 7], dtype=torch.int32, device=DEV)
    store.path_len[2] = 4
    context = 30
    qsl = torch.tensor([0, W, W + 1], dtype=torch.int32, device=DEV)
    cu = torch.tensor([0, W, W + 1], dtype=torch.int32, device=DEV)  # the second request is a plain decode
    seq_lens = torch.tensor([context + W, 50], dtype=torch.int32, device=DEV)
    input_ids = torch.arange(100, 100 + W + 1, dtype=torch.int64, device=DEV)
    positions = torch.tensor([context + d for d in (0, 1, 1, 2, 2, 2, 3, 3)] + [49], dtype=torch.int64, device=DEV)
    sampled = torch.tensor([[11, 12, 13, 14] + [0] * 4, [9] + [0] * 7], dtype=torch.int64, device=DEV)
    src = torch.arange(W + 1, dtype=torch.int64, device=DEV)
    chainize(
        input_ids,
        positions,
        src,
        idx_mapping=torch.tensor([2, 0], dtype=torch.int64, device=DEV),
        query_start_loc=qsl,
        cu_num_logits=cu,
        seq_lens=seq_lens,
        path=store.path,
        path_len=store.path_len,
        sampled=sampled,
        width=W,
    )
    assert src[:W].tolist() == [0, 2, 5, 7, 4, 5, 6, 7] and src[W] == W
    assert positions[:W].tolist() == [context + r for r in range(W)] and positions[W] == 49
    # row j holds the token the draft is given with it: the sampled token j - 1 (row 0 keeps the root)
    assert input_ids[:4].tolist() == [100, 11, 12, 13]
    assert input_ids[4:W].tolist() == [104, 105, 106, 107]


def test_commit_moves_the_accepted_rows_into_chain_order():
    layers = 3
    dim, state_len, slots_per_req, heads_state = 6, 3 + W, W + 1, 5

    dim_first = is_conv_state_dim_first()

    def window(raw):  # the (slots, dim, state_len) view the kernels see
        return raw if dim_first else raw.transpose(-1, -2)

    class Layer:
        def __init__(self):
            shape = (20, dim, state_len) if dim_first else (20, state_len, dim)
            self.kv_cache = (torch.randn(shape, device=DEV), torch.randn(20, heads_state, device=DEV))

    modules = [Layer() for _ in range(layers)]
    store = TreeStore(4, W, DEV)
    store.path[1, :3] = torch.tensor([0, 3, 6], dtype=torch.int32, device=DEV)
    store.path_len[1] = 3
    slots = torch.arange(2, 2 + slots_per_req, dtype=torch.int32, device=DEV)[None]
    before = [(window(c).clone(), s.clone()) for c, s in (m.kv_cache for m in modules)]
    LinearStateCommit(modules, [slots] * len(modules), DEV).commit(
        torch.tensor([1], dtype=torch.int64, device=DEV), store.path, store.path_len
    )
    for module, (conv0, ssm0) in zip(modules, before):
        conv, ssm = window(module.kv_cache[0]), module.kv_cache[1]
        at = int(slots[0, 0])
        torch.testing.assert_close(conv[at, :, 2], conv0[at, :, 2])  # row 0 stays
        torch.testing.assert_close(conv[at, :, 3], conv0[at, :, 2 + 3])  # path row 1 came from column 2 + 3
        torch.testing.assert_close(conv[at, :, 4], conv0[at, :, 2 + 6])
        torch.testing.assert_close(ssm[int(slots[0, 2])], ssm0[int(slots[0, 6])])  # the last accepted row's state
        torch.testing.assert_close(ssm[int(slots[0, 6])], ssm0[int(slots[0, 6])])
