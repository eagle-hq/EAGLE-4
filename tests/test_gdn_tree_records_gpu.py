# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""ROCm chunk records against an independent fp32 tree recurrence."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="ROCm GPU required")
@pytest.mark.parametrize("width,batch", [(1, 1), (8, 2), (16, 1), (32, 2)])
@pytest.mark.parametrize("seed", [11, 41, 97])
def test_chunk_records_and_commit(width, batch, seed):
    from eagle_harness.kernels.gdn_tree_records import gdn_tree_chunk_update, gdn_tree_records_commit

    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    h, hv, kdim, vdim = 4, 12, 128, 128

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, device=device, dtype=dtype) * 0.2

    q, k = rand(1, batch * width, h, kdim), rand(1, batch * width, h, kdim)
    v = rand(1, batch * width, hv, vdim)
    a, b = rand(batch * width, hv), rand(batch * width, hv)
    A, bias = rand(hv, dtype=torch.float32), rand(hv, dtype=torch.float32)
    state = rand(batch + 2, hv, vdim, kdim)
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
    for fly in [False, True]:
        out = gdn_tree_chunk_update(
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
            use_flydsl=fly,
        )
        torch.testing.assert_close(out.float(), expected.float(), atol=0.002, rtol=0.02)
        assert bool((rec[-1] == -999.0).all()), "unused record row changed"
        accepted = torch.full((batch,), width - 1, dtype=torch.int32, device=device)
        committed = state.clone().unsqueeze(0)
        gdn_tree_records_commit(committed, rec.unsqueeze(0), slots, rows, accepted, parent, False)
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
        gdn_tree_records_commit(committed, rec.unsqueeze(0), slots, rows, -torch.ones_like(accepted), parent, False)
        assert torch.equal(committed, before)
