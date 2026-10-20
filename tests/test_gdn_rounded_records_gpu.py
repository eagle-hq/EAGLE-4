# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""ROCm replay records against independently rounded parent states."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="ROCm GPU required")
@pytest.mark.parametrize("width,batch", [(1, 1), (8, 2), (16, 1), (32, 2)])
@pytest.mark.parametrize("seed", [11, 41, 97])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rounded", [True, False], ids=["path_replay", "chunk"])
def test_rounded_records_and_commit(width, batch, seed, state_dtype, rounded):
    """The rounded serving path and unrounded diagnostic chunk match their respective references.

    The chunk is not a numerically equivalent substitute for low-precision parent rounding.
    """
    from eagle_harness.kernels.gdn_tree_records import gdn_tree_chunk_update
    from eagle_harness.kernels.gdn_tree_records import gdn_tree_records_commit
    from eagle_harness.kernels.gdn_tree_records import gdn_tree_records_update

    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    h, hv, kdim, vdim = 4, 12, 128, 128

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, device=device, dtype=dtype) * 0.2

    q, k = rand(1, batch * width, h, kdim), rand(1, batch * width, h, kdim)
    v = rand(1, batch * width, hv, vdim)
    a, b = rand(batch * width, hv), rand(batch * width, hv)
    A, bias = rand(hv, dtype=torch.float32), rand(hv, dtype=torch.float32)
    state = rand(batch + 2, hv, vdim, kdim, dtype=state_dtype)
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device=device).flip(0)
    rows = torch.arange(batch, dtype=torch.int32, device=device).flip(0)
    qsl = torch.arange(batch + 1, dtype=torch.int32, device=device) * width
    parent = torch.zeros(batch, width + 7, dtype=torch.int32, device=device)
    for node in range(1, width):
        parent[:, node] = torch.randint(node, (batch,), generator=gen, device=device, dtype=torch.int32)
    rec = torch.full((batch + 1, width + 3, hv, vdim + kdim + 4), -999.0, device=device)
    expected = torch.empty_like(v)
    snapshots = []
    for request in range(batch):
        states = []
        for node in range(width):
            token = request * width + node
            prior = state[slots[request]].float() if node == 0 else states[int(parent[request, node])]
            if rounded and node > 0:
                prior = prior.to(state.dtype).float()
            key = k[0, token].float()
            key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            query = q[0, token].float()
            query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt() / kdim**0.5
            key, query = key.repeat_interleave(hv // h, 0), query.repeat_interleave(hv // h, 0)
            decay = (-A.exp() * torch.nn.functional.softplus(a[token].float() + bias)).exp()
            beta = b[token].float().sigmoid().to(b.dtype).float()
            decayed = prior * decay[:, None, None]
            delta = (v[0, token].float() - (decayed * key[:, None, :]).sum(-1)) * beta[:, None]
            current = decayed + delta[:, :, None] * key[:, None, :]
            states.append(current)
            expected[0, token] = (current * query[:, None, :]).sum(-1).to(v.dtype)
        snapshots.append(states)
    paths = torch.zeros(batch, width, width + 2, dtype=torch.int32, device=device)
    for request in range(batch):
        for node in range(width):
            chain = [node]
            while chain[-1] > 0:
                chain.append(int(parent[request, chain[-1]]))
            chain.reverse()
            paths[request, node, 0] = len(chain)
            paths[request, node, 1] = len(chain) - 1
            paths[request, node, 2 : 2 + len(chain)] = torch.tensor(chain, device=device)
    update = gdn_tree_records_update if rounded else gdn_tree_chunk_update
    path_args = {"tree_paths": paths} if rounded else {}
    out = update(
        A,
        bias,
        q,
        k,
        v,
        a,
        b,
        state=state,
        state_indices=slots,
        query_start_loc=qsl,
        parent=parent,
        records=rec,
        record_indices=rows,
        width=width,
        **path_args,
    )
    torch.testing.assert_close(out.float(), expected.float(), atol=0.002, rtol=0.02)
    assert bool((rec[-1] == -999.0).all()), "unused record row changed"
    accepted = torch.full((batch,), width - 1, dtype=torch.int32, device=device)
    committed = state.clone().unsqueeze(0)
    gdn_tree_records_commit(committed, rec.unsqueeze(0), slots, rows, accepted, parent, rounded)
    for request in range(batch):
        torch.testing.assert_close(
            committed[0, slots[request]].float(),
            snapshots[request][-1].to(state.dtype).float(),
            atol=0.002,
            rtol=0.02,
        )
    assert torch.equal(committed[0, 0], state[0])
    assert torch.equal(committed[0, -1], state[-1])
    # A negative accepted step must not touch any persistent state.
    before = committed.clone()
    gdn_tree_records_commit(committed, rec.unsqueeze(0), slots, rows, -torch.ones_like(accepted), parent, rounded)
    assert torch.equal(committed, before)
    # sglang's eager verify (no decode graph, or a batch beyond the captured sizes) builds the parent table int64.
    from_int64 = state.clone().unsqueeze(0)
    gdn_tree_records_commit(from_int64, rec.unsqueeze(0), slots, rows, accepted, parent.to(torch.int64), rounded)
    assert torch.equal(from_int64, committed)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="ROCm GPU required")
def test_parent_rounding_does_not_accumulate_sub_ulp_decay():
    from eagle_harness.kernels.gdn_tree_records import gdn_tree_records_commit, gdn_tree_records_update

    device, width, dim = "cuda", 32, 128
    state = torch.ones(1, 1, dim, dim, device=device, dtype=torch.bfloat16)
    q = torch.zeros(1, width, 1, dim, device=device, dtype=torch.bfloat16)
    q[..., 0] = 1
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    a = torch.full((width, 1), -7.0, device=device, dtype=torch.bfloat16)
    b = torch.zeros_like(a)
    A = torch.zeros(1, device=device)
    slot = torch.zeros(1, device=device, dtype=torch.int32)
    qsl = torch.tensor([0, width], device=device, dtype=torch.int32)
    parent = (torch.arange(width, device=device, dtype=torch.int32) - 1).view(1, width)
    paths = torch.cat(
        (
            torch.tensor([width, 0], device=device, dtype=torch.int32),
            torch.arange(width, device=device, dtype=torch.int32),
        )
    ).view(1, 1, width + 2)
    records = torch.empty(1, width, 1, dim * 2 + 4, device=device)
    out = gdn_tree_records_update(
        A,
        A,
        q,
        k,
        v,
        a,
        b,
        state=state,
        state_indices=slot,
        query_start_loc=qsl,
        parent=parent,
        tree_paths=paths,
        records=records,
        record_indices=slot,
        width=width,
    )
    # The per-node decay is below half a BF16 ulp: every parent rounds back to one.
    assert torch.equal(out, out[:, :1].expand_as(out))
    committed = state.clone().unsqueeze(0)
    steps = torch.full_like(slot, width - 1)
    gdn_tree_records_commit(committed, records.unsqueeze(0), slot, slot, steps, parent, True)
    assert torch.equal(committed[0], state)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="ROCm GPU required")
@pytest.mark.parametrize("width,batch", [(16, 1), (32, 8)])
def test_value_tile_does_not_change_results(monkeypatch, width, batch):
    """The V block is a launch-geometry choice (one wave per program at every width): outputs and records are
    bitwise equal for 8, 16 and 32 columns."""
    from eagle_harness.kernels import gdn_tree_records as records_module

    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(5)
    h, hv, kdim, vdim = 4, 12, 128, 128

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, device=device, dtype=dtype)

    q, k, v = rand(1, batch * width, h, kdim), rand(1, batch * width, h, kdim), rand(1, batch * width, hv, vdim) * 0.2
    a, b = rand(batch * width, hv), rand(batch * width, hv)
    A, bias = rand(hv, dtype=torch.float32), rand(hv, dtype=torch.float32)
    state = rand(batch, hv, vdim, kdim) * 0.2
    slots = torch.arange(batch, dtype=torch.int32, device=device)
    qsl = torch.arange(batch + 1, dtype=torch.int32, device=device) * width
    parent = torch.zeros(batch, width, dtype=torch.int32, device=device)
    for node in range(1, width):
        parent[:, node] = torch.randint(node, (batch,), generator=gen, device=device, dtype=torch.int32)
    paths = torch.zeros(batch, width, width + 2, dtype=torch.int32, device=device)
    for request in range(batch):
        for node in range(width):
            chain = [node]
            while chain[-1] > 0:
                chain.append(int(parent[request, chain[-1]]))
            chain.reverse()
            paths[request, node, 0], paths[request, node, 1] = len(chain), len(chain) - 1
            paths[request, node, 2 : 2 + len(chain)] = torch.tensor(chain, device=device)
    results = {}
    for tile in (8, 16, 32):
        monkeypatch.setattr(records_module, "_records_value_tile", lambda *args, tile=tile: tile)
        rec = torch.zeros((batch, width, hv, vdim + kdim + 4), device=device)
        out = records_module.gdn_tree_records_update(
            A,
            bias,
            q,
            k,
            v,
            a,
            b,
            state=state,
            state_indices=slots,
            query_start_loc=qsl,
            parent=parent,
            tree_paths=paths,
            records=rec,
            record_indices=slots,
            width=width,
        )
        results[tile] = (out.clone(), rec)
    for tile in (16, 32):
        assert torch.equal(results[tile][0], results[8][0]), f"outputs differ at V block {tile}"
        assert torch.equal(results[tile][1], results[8][1]), f"records differ at V block {tile}"


def test_value_tile_widens_with_the_program_count(monkeypatch):
    """8 columns while the programs fit two rounds of the device's waves, wider (up to 32) beyond that."""
    from eagle_harness.kernels import gdn_tree_records as records_module

    monkeypatch.setattr(records_module, "_wave_slots", lambda device: 8192)  # 256 CUs x 32 waves (MI355X)
    tile = records_module._records_value_tile
    assert tile(1 * 48 * 16, 128, "cuda") == 8  # batch 1, width 16: 12288 programs at 8 columns
    assert tile(1 * 48 * 32, 128, "cuda") == 16  # batch 1, width 32: 24576 -> 12288
    assert tile(8 * 48 * 16, 128, "cuda") == 32  # batch 8: 98304 -> 24576, capped at 32
    assert tile(32 * 48 * 32, 128, "cuda") == 32
    assert tile(1, 4, "cuda") == 4  # a value dim below 8 keeps one block
