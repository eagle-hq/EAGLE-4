# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The GDN tree verify on a pool that keeps replay records in place of state snapshots.

Under ``--enable-linear-replayssm-spec`` an engine with ``LinearAttnStateLayout.verify_records`` allocates the
records the plugin asks for (``linear_attn_verify_record_width``) and no per-node snapshots. Several verify rounds
through :class:`GDNTreeVerify` (two layers, random trees, random accepted nodes, ``commit_state``) against the
serial fp32 tree recurrence.
"""

import random
from types import SimpleNamespace

import pytest
import torch

H, HV, K, V = 4, 12, 128, 128  # Qwen3.8-27B GDN heads per TP4 rank
LAYERS, BATCH, WIDTH, ROUNDS = 2, 2, 16, 4


def _layout_types():
    extension = pytest.importorskip("sglang.srt.layers.attention.linear.verify_extension")
    if "verify_records" not in extension.LinearAttnStateLayout.__dataclass_fields__:
        pytest.skip("this sglang allocates no replay records in place of the snapshots")
    return extension.LinearAttnStateLayout, extension.LinearAttnVerifyLayout


def _tree(rng: random.Random, width: int, max_depth: int) -> list[int]:
    parent, depth = [-1], [0]
    for _ in range(1, width):
        node = rng.choice([i for i in range(len(parent)) if depth[i] < max_depth])
        parent.append(node)
        depth.append(depth[node] + 1)
    return parent


def _links(parent: list[int]) -> tuple[list[int], list[int]]:
    """First-child and next-sibling links of a tree in verify row order (-1 where absent)."""
    first, sibling, last = [-1] * len(parent), [-1] * len(parent), {}
    for node, up in enumerate(parent[1:], 1):
        if up in last:
            sibling[last[up]] = node
        else:
            first[up] = node
        last[up] = node
    return first, sibling


def _serial(state, q, k, v, a, b, A, bias, parent):
    """Outputs and per-node fp32 states of one request's tree, every node continuing from its parent's state."""
    states, out = [], torch.empty_like(v)
    for node, up in enumerate(parent):
        prior = state.float() if up < 0 else states[up]
        key = k[node].float()
        key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        query = q[node].float()
        query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt() / K**0.5
        key, query = key.repeat_interleave(HV // H, 0), query.repeat_interleave(HV // H, 0)
        decay = (-A.exp() * torch.nn.functional.softplus(a[node].float() + bias)).exp()
        beta = b[node].float().sigmoid().to(b.dtype).float()
        decayed = prior * decay[:, None, None]
        delta = (v[node].float() - (decayed * key[:, None, :]).sum(-1)) * beta[:, None]
        current = decayed + delta[:, :, None] * key[:, None, :]
        states.append(current)
        out[node] = (current * query[:, None, :]).sum(-1).to(v.dtype)
    return out, states


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("max_depth", [12, 4])
@pytest.mark.parametrize("seed", [3, 29])
def test_rounds_on_pool_records(max_depth, seed):
    from eagle_harness.kernels import arch
    from eagle_harness.sglang.gdn_tree_verify import GDNTreeVerify, record_width

    LinearAttnStateLayout, LinearAttnVerifyLayout = _layout_types()
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = random.Random(seed)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, device=device, dtype=dtype) * 0.2

    # The pool: fp32 states [layers, slots, HV, V, K] (slot 0 is padding), records [layers, rows, steps, HV, width].
    state = rand(LAYERS, BATCH + 2, HV, V, K, dtype=torch.float32)
    records = torch.full((LAYERS, BATCH + 1, WIDTH, HV, record_width(V, K)), -999.0, device=device)
    extension = GDNTreeVerify.create(
        LinearAttnStateLayout(
            conv_state_len=3,
            conv_dtype=torch.bfloat16,
            ssm_dtype=torch.float32,
            num_v_heads=HV,
            has_state_snapshots=False,
            device=torch.device(device),
            can_commit_state=True,
            verify_records=records,
        )
    )
    assert extension.pool_records and extension.records is records
    layers = [
        SimpleNamespace(A_log=rand(HV, dtype=torch.float32), dt_bias=rand(HV, dtype=torch.float32))
        for _ in range(LAYERS)
    ]
    slots = torch.tensor([2, 1], dtype=torch.int32, device=device)
    rows = torch.tensor([1, 0], dtype=torch.int32, device=device)
    qsl = torch.arange(BATCH + 1, dtype=torch.int32, device=device) * WIDTH
    reference = state.clone()

    for _ in range(ROUNDS):
        trees = [_tree(rng, WIDTH, max_depth) for _ in range(BATCH)]
        links = [_links(tree) for tree in trees]
        parent = torch.tensor(trees, dtype=torch.int32, device=device)
        layout = LinearAttnVerifyLayout(
            next_token=torch.tensor([first for first, _ in links], dtype=torch.int32, device=device),
            next_sibling=torch.tensor([sibling for _, sibling in links], dtype=torch.int32, device=device),
            parent_token=parent,
            cache_indices=slots,
            query_start_loc=qsl,
            width=WIDTH,
        )
        extension.begin_forward(layout, in_graph=False)
        # fp32 records take the path replay wherever the chunk is not the measured choice
        assert (extension._paths is not None) == (not arch.gdn_records_chunk(device))

        accepted = [rng.randrange(WIDTH) for _ in range(BATCH)]
        for layer_index, layer in enumerate(layers):
            q, k = rand(1, BATCH * WIDTH, H, K), rand(1, BATCH * WIDTH, H, K)
            v = rand(1, BATCH * WIDTH, HV, V)
            a, b = rand(BATCH * WIDTH, HV), rand(BATCH * WIDTH, HV)
            out = extension.recurrent_update(
                layer,
                query=q,
                key=k,
                value=v,
                a=a,
                b=b,
                ssm_states=state[layer_index],
                cache_indices=slots,
                query_start_loc=qsl,
                intermediate_states=records[layer_index],
                intermediate_state_indices=rows,
            )
            for request in range(BATCH):
                tokens = slice(request * WIDTH, (request + 1) * WIDTH)
                want, states = _serial(
                    reference[layer_index, slots[request]],
                    q[0, tokens],
                    k[0, tokens],
                    v[0, tokens],
                    a[tokens],
                    b[tokens],
                    layer.A_log,
                    layer.dt_bias,
                    trees[request],
                )
                torch.testing.assert_close(out[0, tokens].float(), want.float(), atol=0.002, rtol=0.02)
                reference[layer_index, slots[request]] = states[accepted[request]]

        steps = torch.tensor(accepted, dtype=torch.int32, device=device)
        assert extension.commit_state(state, slots, rows, steps, parent)
        torch.testing.assert_close(state, reference, atol=1e-4, rtol=1e-3)

    assert torch.equal(state[:, 0], reference[:, 0]) and torch.equal(state[:, -1], reference[:, -1])
    assert bool((records[:, -1] == -999.0).all()), "the padding record row changed"


def test_record_width_follows_the_commit_seam(monkeypatch):
    """Records in place of snapshots only where the commit may replay them and the record kernels take the shape."""
    pytest.importorskip("sglang")
    from eagle_harness.sglang import algorithm

    algo = algorithm.EagleHarnessSpecAlgo(name="EAGLE_HARNESS_TEST", factory=None)
    serving = SimpleNamespace(disable_radix_cache=True, pp_size=1)

    def width(steps=16, value_dim=V, key_dim=K, **args):
        monkeypatch.setattr(algorithm, "get_spec", lambda: SimpleNamespace(speculative_num_draft_tokens=steps))
        server_args = SimpleNamespace(**{**vars(serving), **args})
        return algo.linear_attn_verify_record_width(server_args, head_v_dim=value_dim, head_k_dim=key_dim)

    assert width() == V + K + 4
    assert width(steps=64) == V + K + 4
    assert width(disable_radix_cache=False) == 0  # radix tracking commits snapshots of tracked steps
    assert width(pp_size=2) == 0
    assert width(steps=65) == 0
    assert width(value_dim=32) == 0
    assert width(key_dim=96) == 0


@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
def test_pool_records_replay_paths_where_the_chunk_is_slower(monkeypatch, ssm_dtype):
    """A pool with records and no snapshots: every verify replays root paths unless the platform prefers the fp32
    chunk (AMD); the record kernels' layout limits raise instead of falling back to snapshots that do not exist."""
    LinearAttnStateLayout, _ = _layout_types()
    from eagle_harness.kernels import arch
    from eagle_harness.sglang.gdn_tree_verify import GDNTreeVerify

    def create(conv_state_len=3, chunk=False):
        monkeypatch.setattr(arch, "gdn_records_chunk", lambda device=None: chunk)
        return GDNTreeVerify.create(
            LinearAttnStateLayout(
                conv_state_len=conv_state_len,
                conv_dtype=torch.bfloat16,
                ssm_dtype=ssm_dtype,
                num_v_heads=HV,
                has_state_snapshots=False,
                device=torch.device("cpu"),
                can_commit_state=True,
                verify_records=torch.zeros(1),
            )
        )

    extension = create()
    for batch, width in ((1, 8), (8, 16), (32, 32), (64, 64)):
        assert extension._uses_path_replay(batch, width)
    assert create(chunk=True)._uses_path_replay(1, 16) is (ssm_dtype != torch.float32)
    with pytest.raises(ValueError, match="replay records"):
        create(conv_state_len=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("width,max_depth,batch", [(16, 12, 2), (64, 8, 1)])
@pytest.mark.parametrize("value_tile", [8, 32])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_fp32_records_are_plain_decoding(width, max_depth, batch, value_tile, strided):
    """An fp32 pool's records verify computes what plain greedy decoding computes: every node's output and the state its
    records commit equal the target's decode kernel (sglang's packed Triton decode) run token by token along the
    node's root path. Bitwise with the Triton the engine pins (3.8, torch 2.14), which contracts the decode kernel's
    rank-1 update into the FMA the clone writes; within rounding with older Triton."""
    decode_kernel = pytest.importorskip("sglang.srt.layers.attention.linear.kernels.gdn_triton")
    from eagle_harness.kernels.gdn_tree import gdn_tree_ancestors, gdn_tree_paths_buffer
    from eagle_harness.kernels.gdn_tree_records import (
        gdn_tree_records_commit,
        gdn_tree_records_decode_order_update,
        gdn_tree_records_update,
    )
    from eagle_harness.sglang.gdn_tree_verify import record_width

    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(width + value_tile)
    rng = random.Random(width)
    import triton

    if tuple(int(part) for part in triton.__version__.split(".")[:2]) >= (3, 8):
        same = torch.equal
    else:
        def same(got, want):
            torch.testing.assert_close(got, want)
            return True

    def rand(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=device) * scale).to(dtype)

    trees = [_tree(rng, width, max_depth) for _ in range(batch)]
    tokens = batch * width
    if strided:
        # serving hands the verify token-strided views of the layer's input projection (reads_strided_inputs)
        projection = rand(tokens, 2 * H * K + HV * V + 64)
        q, k, v = (projection[:, start : start + size].unflatten(1, shape).unsqueeze(0)
                   for start, size, shape in ((0, H * K, (H, K)), (H * K, H * K, (H, K)), (2 * H * K, HV * V, (HV, V))))
    else:
        q, k, v = rand(1, tokens, H, K), rand(1, tokens, H, K), rand(1, tokens, HV, V)
    a, b = rand(tokens, HV), rand(tokens, HV)
    A_log, dt_bias = rand(HV, dtype=torch.float32, scale=0.5), rand(HV, dtype=torch.float32, scale=0.5)
    pool = rand(batch + 2, HV, V, K, dtype=torch.float32, scale=0.05)
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    rows = torch.arange(batch, dtype=torch.int32, device=device)
    qsl = torch.arange(batch + 1, dtype=torch.int32, device=device) * width

    # plain decoding: one decode call per tree level, every node in a slot seeded from its parent's
    states = torch.cat((pool, torch.zeros(tokens, HV, V, K, device=device)))
    node_slots = torch.arange(tokens, device=device) + batch + 2  # int64: torch indexes with it
    want = torch.empty_like(v)
    mixed = torch.cat((q[0].flatten(1), k[0].flatten(1), v[0].flatten(1)), dim=1)
    depth = [[0] * width for _ in trees]
    for request, tree in enumerate(trees):
        for node in range(1, width):
            depth[request][node] = depth[request][tree[node]] + 1
    for level in range(max_depth + 1):
        level_rows = [r * width + n for r in range(batch) for n in range(width) if depth[r][n] == level]
        if not level_rows:
            break
        for row in level_rows:
            request, node = divmod(row, width)
            up = trees[request][node]
            source = slots[request].long() if up < 0 else node_slots[request * width + up]
            states[node_slots[row]] = states[source]
        index = torch.tensor(level_rows, dtype=torch.int64, device=device)
        out = decode_kernel.TritonGDNKernel().packed_decode(
            mixed_qkv=mixed[index].contiguous(),
            a=a[index].contiguous(),
            b=b[index].contiguous(),
            A_log=A_log,
            dt_bias=dt_bias,
            scale=K**-0.5,
            ssm_states=states,
            cache_indices=node_slots[index].to(torch.int32),
            num_v_heads=HV,
            head_v_dim=V,
        )
        want[0, index] = out.view(len(level_rows), HV, V)

    parent = torch.tensor(trees, dtype=torch.int32, device=device)
    first = torch.tensor([_links(tree)[0] for tree in trees], dtype=torch.int32, device=device)
    sibling = torch.tensor([_links(tree)[1] for tree in trees], dtype=torch.int32, device=device)
    paths = gdn_tree_paths_buffer(batch, width, device)
    ancestors = torch.empty(batch, 3, width, dtype=torch.int32, device=device)
    gdn_tree_ancestors(first, sibling, parent, slots, width, ancestors, query_start_loc=qsl, paths=paths)
    records = torch.zeros(batch + 1, width, HV, record_width(V, K), device=device)
    inputs = (A_log, dt_bias, q, k, v, a, b)
    tables = dict(state=pool, state_indices=slots, query_start_loc=qsl, tree_paths=paths, record_indices=rows)
    got = gdn_tree_records_decode_order_update(*inputs, **tables, records=records, value_tile=value_tile)
    assert same(got, want)
    # the dispatcher picks this verify for an fp32 pool with a path table
    dispatched = gdn_tree_records_update(*inputs, **tables, records=records.clone(), parent=parent, width=width)
    assert same(dispatched, want)
    for node in range(width):
        committed = pool.clone().unsqueeze(0)
        steps = torch.full((batch,), node, dtype=torch.int32, device=device)
        gdn_tree_records_commit(committed, records.unsqueeze(0), slots, rows, steps, parent, False)
        for request in range(batch):
            assert same(committed[0, slots[request].long()], states[node_slots[request * width + node]])
