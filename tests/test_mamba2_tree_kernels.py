# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Mamba2 tree verify (kernel and vLLM extension) against dense references (CPU through the Triton interpreter, or a GPU).

Every row must continue from its parent's state; on a chain the kernel must be vLLM's chain verify, and a committed
tree round must leave vLLM's chain kernels resuming exactly as if only the accepted path had been verified.
"""

import math
import os
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

from eagle_harness.kernels.mamba2_tree import mamba2_tree_state_update  # noqa: E402
from tests.test_gdn_tree_kernels import CHAIN, HALF, TREE, _requests, random_tree, tree_conv  # noqa: E402
from tests.triton_probe import needs_triton  # noqa: E402

pytestmark = needs_triton

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM's own kernels need a GPU")


def _params(heads, groups, head_dim, dstate, tokens, seed):
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(DEV)

    return SimpleNamespace(
        x=rand(tokens, heads, head_dim),
        dt=rand(tokens, heads, scale=0.5),
        A=-torch.exp(rand(heads, scale=0.5)),
        B=rand(tokens, groups, dstate),
        C=rand(tokens, groups, dstate),
        D=rand(heads),
        dt_bias=rand(heads, scale=0.5),
    )


def _reference(p, state, *, slots, parents, qsl, accepted):
    """Row by row: a row whose parent is the previous row continues in fp32, any other row from the parent's stored
    state; every row stores its state in its slot (in the state's dtype)."""
    state, out = state.clone(), torch.empty_like(p.x)
    heads_per_group = p.x.shape[1] // p.B.shape[1]
    for b in range(qsl.shape[0] - 1):
        bos, eos = int(qsl[b]), int(qsl[b + 1])
        carried = state[slots[b, int(accepted[b]) - 1]].float()
        for t in range(eos - bos):
            parent = int(parents[b, t])
            if t > 0 and parent != t - 1:
                carried = state[slots[b, parent]].float()
            tok = bos + t
            dt = p.dt[tok].float() + p.dt_bias  # [heads]; softplus and exp as the kernel writes them
            dt = torch.where(dt <= 20.0, torch.log(torch.exp(dt) + 1), dt)
            dA = torch.exp2(1.4426950408889634 * (p.A * dt))
            Bh = p.B[tok].repeat_interleave(heads_per_group, 0)  # [heads, dstate]
            Ch = p.C[tok].repeat_interleave(heads_per_group, 0)
            carried = carried * dA[:, None, None] + (Bh * dt[:, None])[:, None, :] * p.x[tok][:, :, None]
            state[slots[b, t]] = carried.to(state.dtype)
            out[tok] = (carried * Ch[:, None, :]).sum(-1) + p.x[tok] * p.D[:, None]
    return out, state


@pytest.mark.parametrize(
    "parent_lists", [[TREE, CHAIN], [CHAIN, CHAIN], [TREE[:3], TREE], [random_tree(16, 5), random_tree(16, 6)]]
)
@pytest.mark.parametrize("state_dtype", [torch.float32, HALF])
@pytest.mark.parametrize("dims", [(4, 2, 8, 16), (8, 1, 64, 128)])  # heads, groups, head_dim, dstate
def test_every_row_continues_from_its_parent(parent_lists, state_dtype, dims):
    heads, groups, head_dim, dstate = dims
    width = max(map(len, parent_lists))
    parents, _, qsl = _requests(parent_lists, width)
    tokens = int(qsl[-1])
    p = _params(heads, groups, head_dim, dstate, tokens, seed=len(parent_lists[0]) + heads)
    slots = torch.arange(1, 1 + 2 * width, dtype=torch.int32, device=DEV).view(2, width)
    accepted = torch.tensor([1, 3], dtype=torch.int32, device=DEV)
    gen = torch.Generator().manual_seed(heads * 100 + width)
    state0 = (torch.randn(2 * width + 1, heads, head_dim, dstate, generator=gen) * 0.3).to(DEV, state_dtype)
    want_out, want_state = _reference(p, state0, slots=slots, parents=parents, qsl=qsl, accepted=accepted)
    state, out = state0.clone(), torch.empty_like(p.x)
    mamba2_tree_state_update(
        state,
        p.x,
        p.dt,
        p.A,
        p.B,
        p.C,
        p.D,
        p.dt_bias,
        out,
        state_indices=slots,
        parents=parents,
        query_start_loc=qsl,
        num_accepted_tokens=accepted,
    )
    # A 16-bit state's last bit can round either way where the kernel and the reference differ in the last fp32 bit
    # (fused multiply-adds, exp2); one ulp of state, and the outputs' dstate sums of such flips.
    state_tol, out_tol = {torch.float32: (1e-4, 1e-4), torch.float16: (2e-3, 1e-2), torch.bfloat16: (2e-2, 1e-1)}[
        state_dtype
    ]
    torch.testing.assert_close(out, want_out, rtol=out_tol, atol=out_tol)
    torch.testing.assert_close(state.float(), want_state.float(), rtol=state_tol, atol=state_tol)
    assert torch.equal(state[0], state0[0])  # the null slot is never written


def test_padded_requests_are_skipped():
    """A request whose slots are the pad slot (a CUDA graph's padding) reads zeros and writes no state."""
    heads, groups, head_dim, dstate, width = 2, 1, 8, 16, 4
    parents, _, qsl = _requests([CHAIN[:width], CHAIN[:width]], width)
    p = _params(heads, groups, head_dim, dstate, int(qsl[-1]), seed=9)
    slots = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]], dtype=torch.int32, device=DEV)
    state0 = torch.randn(5, heads, head_dim, dstate, generator=torch.Generator().manual_seed(1)).to(DEV)
    state, out = state0.clone(), torch.empty_like(p.x)
    mamba2_tree_state_update(
        state,
        p.x,
        p.dt,
        p.A,
        p.B,
        p.C,
        p.D,
        p.dt_bias,
        out,
        state_indices=slots,
        parents=parents,
        query_start_loc=qsl,
        pad_slot_id=0,
    )
    assert torch.equal(state[0], state0[0])
    want_out, _ = _reference(
        p,
        torch.cat([torch.zeros_like(state0[:1]), state0[1:]]),
        slots=slots,
        parents=parents,
        qsl=qsl,
        accepted=torch.ones(2, dtype=torch.int32),
    )
    torch.testing.assert_close(out, want_out, rtol=1e-4, atol=1e-4)


def _mamba2_rows(
    p, xbc, conv_state, weight, bias, ssm_state, slots, accepted, qsl, out, heads, head_dim, groups, dstate
):
    return SimpleNamespace(
        layer_name="layers.0",
        hidden_states_B_C=xbc,
        dt=p.dt,
        conv_state=conv_state,
        conv_weight=weight,
        conv_bias=bias,
        activation="silu",
        ssm_state=ssm_state,
        A=p.A,
        D=p.D,
        dt_bias=p.dt_bias,
        num_heads=heads,
        head_dim=head_dim,
        n_groups=groups,
        ssm_state_size=dstate,
        state_indices=slots,
        num_accepted_tokens=accepted,
        query_start_loc=qsl,
        out=out,
    )


def test_spec_rows_read_the_projection_in_place_and_write_the_layer_output():
    """``TreeVerify.forward_mamba2_spec_rows`` on strided views of the projection == tree conv + the tree SSU on
    contiguous inputs, bit for bit; the output lands in the layer's buffer and the publishing layer's slots are
    copied out."""
    from eagle_harness.vllm.gdn_tree_verify import TreeVerify
    from eagle_harness.vllm.verify_layout import set_stage

    heads, groups, head_dim, dstate = 4, 2, 8, 16
    inner, dim = heads * head_dim, heads * head_dim + 2 * groups * dstate
    plists = [random_tree(8, 7), TREE]
    width = max(map(len, plists))
    parents, ancestors, qsl = _requests(plists, width)
    tokens = int(qsl[-1])
    torch.manual_seed(4)
    proj = torch.randn(tokens, inner + dim + heads, device=DEV, dtype=HALF)  # [gate, xBC, dt]
    xbc, dt = proj[:, inner : inner + dim], proj[:, inner + dim :]
    p = _params(heads, groups, head_dim, dstate, tokens, seed=1)
    p.dt = dt
    weight, bias = torch.randn(dim, 4, device=DEV, dtype=HALF), torch.randn(dim, device=DEV, dtype=HALF)
    slots = torch.arange(1, 1 + 2 * width, dtype=torch.int32, device=DEV).view(2, width)
    accepted = torch.tensor([1, 2], dtype=torch.int32, device=DEV)
    conv0 = torch.randn(2 * width + 1, dim, 3 + width - 1, device=DEV, dtype=HALF)
    ssm0 = torch.randn(2 * width + 1, heads, head_dim, dstate, device=DEV) * 0.1
    published = torch.zeros(2, width, dtype=torch.int32, device=DEV)

    conv_state, ssm_state = conv0.clone(), ssm0.clone()
    conv = tree_conv(
        xbc.contiguous(),
        conv_state,
        weight,
        bias,
        "silu",
        slots=slots[:, 0],
        accepted=accepted,
        query_start_loc=qsl,
        ancestors=ancestors,
    )
    x, B, C = (t.contiguous() for t in conv.split([inner, groups * dstate, groups * dstate], dim=-1))
    want = torch.empty(tokens, heads, head_dim, device=DEV, dtype=HALF)
    mamba2_tree_state_update(
        ssm_state,
        x.view(tokens, heads, head_dim),
        dt.contiguous(),
        p.A,
        B.view(tokens, groups, dstate),
        C.view(tokens, groups, dstate),
        p.D,
        p.dt_bias,
        want,
        state_indices=slots,
        parents=parents,
        query_start_loc=qsl,
        num_accepted_tokens=accepted,
    )

    set_stage(SimpleNamespace(ancestors=ancestors, parents=parents, state_slots={"layers.0": published}))
    try:
        got_conv, got_ssm = conv0.clone(), ssm0.clone()
        out = torch.zeros(tokens + 2, inner, device=DEV, dtype=HALF)[:tokens]  # the layer's buffer, cut to the rows
        TreeVerify(None).forward_mamba2_spec_rows(
            _mamba2_rows(
                p, xbc, got_conv, weight, bias, got_ssm, slots, accepted, qsl, out, heads, head_dim, groups, dstate
            )
        )
    finally:
        set_stage(None)
    assert torch.equal(out, want.view(tokens, inner))
    assert torch.equal(got_ssm, ssm_state) and torch.equal(got_conv, conv_state)
    assert torch.equal(published, slots)


@needs_gpu
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("accepted", [[1, 1], [2, 5], [8, 3]])
def test_a_chain_is_what_vllm_verifies(state_dtype, accepted):
    """On chain-shaped trees the tree SSU computes vLLM's chain ``selective_state_update`` (outputs and every row's
    state)."""
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

    heads, groups, head_dim, dstate, width = 16, 2, 64, 128, 8
    parents, _, qsl = _requests([CHAIN + [4, 5, 6], CHAIN + [4, 5, 6]], width)
    tokens = int(qsl[-1])
    p = _params(heads, groups, head_dim, dstate, tokens, seed=11)
    p.x, p.B, p.C = p.x.to(HALF), p.B.to(HALF), p.C.to(HALF)
    p.dt, p.D, p.dt_bias = p.dt.to(HALF), p.D, p.dt_bias
    slots = torch.arange(1, 1 + 2 * width, dtype=torch.int32, device=DEV).view(2, width)
    acc = torch.tensor(accepted, dtype=torch.int32, device=DEV)
    state0 = (torch.randn(2 * width + 1, heads, head_dim, dstate, device=DEV) * 0.3).to(state_dtype)
    ours_state, ours = state0.clone(), torch.empty_like(p.x)
    mamba2_tree_state_update(
        ours_state,
        p.x,
        p.dt,
        p.A,
        p.B,
        p.C,
        p.D,
        p.dt_bias,
        ours,
        state_indices=slots,
        parents=parents,
        query_start_loc=qsl,
        num_accepted_tokens=acc,
    )
    theirs_state, theirs = state0.clone(), torch.empty_like(p.x)
    selective_state_update(
        theirs_state,
        p.x,
        p.dt[:, :, None].expand(-1, -1, head_dim),
        p.A[:, None, None].expand(-1, head_dim, dstate),
        p.B,
        p.C,
        p.D[:, None].expand(-1, head_dim),
        p.dt_bias[:, None].expand(-1, head_dim),
        dt_softplus=True,
        state_batch_indices=slots,
        dst_state_batch_indices=slots,
        out=theirs,
        num_accepted_tokens=acc,
        cu_seqlens=qsl,
    )
    torch.testing.assert_close(ours.float(), theirs.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ours_state.float(), theirs_state.float(), rtol=1e-5, atol=1e-5)


@needs_gpu
@pytest.mark.parametrize(
    "parent_lists, paths",
    [
        ([TREE, TREE], [[0, 2, 5], [0]]),
        ([random_tree(16, 8), CHAIN], [None, [0, 1, 2]]),
    ],
)
def test_a_committed_tree_round_continues_like_the_accepted_chain(parent_lists, paths):
    """Tree verify through the extension + ``LinearStateCommit``, then vLLM's own chain kernels resume exactly as
    after verifying only the accepted path as a chain."""
    from tests.test_gdn_tree_kernels import deepest_path
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

    from eagle_harness.vllm.gdn_tree_verify import LinearStateCommit, TreeVerify
    from eagle_harness.vllm.verify_layout import TreeStore, set_stage

    paths = [path or deepest_path(plist) for path, plist in zip(paths, parent_lists)]
    heads, groups, head_dim, dstate = 8, 2, 64, 128
    inner, dim = heads * head_dim, heads * head_dim + 2 * groups * dstate
    reqs, width = len(parent_lists), max(map(len, parent_lists))
    parents, ancestors, qsl = _requests(parent_lists, width)
    tokens = int(qsl[-1])
    torch.manual_seed(5)
    bf16 = torch.bfloat16
    p = _params(heads, groups, head_dim, dstate, tokens, seed=13)
    p.dt = p.dt.to(bf16)
    xbc = torch.randn(tokens, dim, device=DEV, dtype=bf16)
    weight, bias = torch.randn(dim, 4, device=DEV, dtype=bf16) * 0.5, torch.randn(dim, device=DEV, dtype=bf16)
    slots = torch.arange(1, 1 + reqs * width, dtype=torch.int32, device=DEV).view(reqs, width)
    first = torch.tensor([1, 2][:reqs], dtype=torch.int32, device=DEV)  # accepted before the round
    dim_first = is_conv_state_dim_first()
    state_len = 3 + width - 1
    conv_raw = torch.randn(
        (reqs * width + 1, dim, state_len) if dim_first else (reqs * width + 1, state_len, dim), device=DEV, dtype=bf16
    )
    window = lambda raw: raw if dim_first else raw.transpose(-1, -2)  # noqa: E731
    ssm0 = torch.randn(reqs * width + 1, heads, head_dim, dstate, device=DEV) * 0.1

    # tree side
    tree_conv_raw, tree_ssm = conv_raw.clone(), ssm0.clone()
    published = torch.zeros(reqs, width, dtype=torch.int32, device=DEV)
    set_stage(SimpleNamespace(ancestors=ancestors, parents=parents, state_slots={"layers.0": published}))
    try:
        TreeVerify(None).forward_mamba2_spec_rows(
            _mamba2_rows(
                p,
                xbc,
                window(tree_conv_raw),
                weight,
                bias,
                tree_ssm,
                slots,
                first,
                qsl,
                torch.empty(tokens, inner, device=DEV, dtype=bf16),
                heads,
                head_dim,
                groups,
                dstate,
            )
        )
    finally:
        set_stage(None)
    store = TreeStore(8, width, DEV)
    for r, path in enumerate(paths):
        store.path[r, : len(path)] = torch.tensor(path, dtype=torch.int32, device=DEV)
        store.path_len[r] = len(path)

    class Layer:
        kv_cache = (tree_conv_raw, tree_ssm)

    LinearStateCommit([Layer], [published], DEV).commit(
        torch.arange(reqs, dtype=torch.int64, device=DEV), store.path, store.path_len
    )

    # reference side: vLLM verifies a chain of `width` rows per request, the accepted path first
    def chain_step(conv_raw_side, ssm_side, xbc_rows, dt_rows, lens, accepted, p_rows):
        y = causal_conv1d_update(
            xbc_rows.clone(),
            window(conv_raw_side),
            weight,
            bias,
            "silu",
            conv_state_indices=slots[:, 0].contiguous(),
            num_accepted_tokens=accepted,
            query_start_loc=lens,
            max_query_len=width,
            validate_data=False,
        )
        x, B, C = y.split([inner, groups * dstate, groups * dstate], dim=-1)
        out = torch.empty(y.shape[0], heads, head_dim, device=DEV, dtype=bf16)
        selective_state_update(
            ssm_side,
            x.reshape(-1, heads, head_dim),
            dt_rows[:, :, None].expand(-1, -1, head_dim),
            p_rows.A[:, None, None].expand(-1, head_dim, dstate),
            B.reshape(-1, groups, dstate),
            C.reshape(-1, groups, dstate),
            p_rows.D[:, None].expand(-1, head_dim),
            p_rows.dt_bias[:, None].expand(-1, head_dim),
            dt_softplus=True,
            state_batch_indices=slots,
            dst_state_batch_indices=slots,
            out=out,
            num_accepted_tokens=accepted,
            cu_seqlens=lens,
        )
        return out

    ref_conv_raw, ref_ssm = conv_raw.clone(), ssm0.clone()
    order = [path + [r for r in range(len(pl)) if r not in path] for path, pl in zip(paths, parent_lists)]
    rows = torch.cat([qsl[r] + torch.tensor(o, device=DEV) for r, o in enumerate(order)])
    chain_step(ref_conv_raw, ref_ssm, xbc[rows], p.dt[rows], qsl, first, p)

    # the next step, a chain of 4 rows resuming after the accepted path, must agree
    accepted = torch.tensor([len(path) for path in paths], dtype=torch.int32, device=DEV)
    nxt = 4
    nqsl = torch.arange(0, (reqs + 1) * nxt, nxt, dtype=torch.int32, device=DEV)
    xbc2 = torch.randn(reqs * nxt, dim, device=DEV, dtype=bf16)
    dt2 = torch.randn(reqs * nxt, heads, device=DEV, dtype=bf16) * 0.5
    outs = [
        chain_step(c, s, xbc2, dt2, nqsl, accepted, p) for c, s in ((tree_conv_raw, tree_ssm), (ref_conv_raw, ref_ssm))
    ]
    assert torch.equal(outs[0], outs[1])
    assert not math.isnan(outs[0].float().sum().item())
