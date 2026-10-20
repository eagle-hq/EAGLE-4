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
