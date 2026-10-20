# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The Mamba2 tree verify extension against sglang's own Mamba2 verify kernels, on the shapes it serves (GPU).

For a batch of draft trees (chains, stars, random trees, a padded request) at the per-rank Mamba2 shapes of Nemotron 3
Nano (TP1, TP4) and Nemotron 3.5 Super (TP4, bf16 SSM state), one Mamba2 layer's verify runs twice on
the same inputs: sglang's ``causal_conv1d_update`` following the tree links, and the extension (tables built in
``begin_forward``, then ``conv_update``); then sglang's ``selective_state_update`` on each path's parent table.
Parent table, conv windows, conv state, conv output, SSM outputs and per-step SSM snapshots must be identical.

Run in a GPU job: ``pytest -q tests/test_mamba2_tree_verify_gpu.py``.
"""

import random
import types
import zlib

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

conv_triton = pytest.importorskip("sglang.kernels.ops.mamba.causal_conv1d_triton")
mamba_ssm = pytest.importorskip("sglang.kernels.ops.mamba.triton_ops.mamba_ssm")
verify_extension = pytest.importorskip("sglang.srt.layers.attention.linear.verify_extension")

from eagle_harness.sglang.mamba2_tree_verify import Mamba2TreeVerify  # noqa: E402

DEV = torch.device("cuda")
PAD = -1
# (heads, head_dim, groups, state, SSM state dtype) per rank: Nemotron 3 Nano at TP1 and TP4, and Nemotron 3.5 Super
# at TP4 with the bf16 SSM state it is served with (--mamba-ssm-dtype bfloat16).
SHAPES = {
    "tp1": (64, 64, 8, 128, torch.float32),
    "tp4": (16, 64, 2, 128, torch.float32),
    "super-tp4": (32, 64, 2, 128, torch.bfloat16),
}


def _tree(kind: str, width: int, rng: random.Random) -> list[int]:
    """Parents in node order (-1 for the root)."""
    if kind == "chain":
        return [-1] + list(range(width - 1))
    if kind == "star":
        return [-1] + [0] * (width - 1)
    return [-1] + [rng.randrange(r) for r in range(1, width)]


def _links(parent_lists, width):
    """sglang's first-child / next-sibling links, [bs, width], -1 where absent."""
    nxt = torch.full((len(parent_lists), width), -1, dtype=torch.int64)
    sib = torch.full_like(nxt, -1)
    for b, parents in enumerate(parent_lists):
        last_child = {}
        for node, p in enumerate(parents[1:], start=1):
            if p in last_child:
                sib[b, last_child[p]] = node
            else:
                nxt[b, p] = node
            last_child[p] = node
    return nxt.to(DEV), sib.to(DEV)


def _verify(case, use_extension):
    """One Mamba2 layer verify; returns every tensor the commit or the next layer reads."""
    heads, head_dim, groups, dstate, ssm_dtype = SHAPES[case["shape"]]
    bs, width = len(case["trees"]), case["width"]
    dim = heads * head_dim + 2 * groups * dstate
    g = torch.Generator(device=DEV).manual_seed(case["seed"])

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=g, device=DEV, dtype=torch.float32).to(dtype)

    x = rnd(bs * width, dim)
    weight, bias = rnd(dim, 1, 4), rnd(dim)
    conv_state = rnd(bs + 2, dim, 3)
    slots = torch.arange(1, bs + 1, dtype=torch.int32, device=DEV)
    if case["pad"]:
        slots[-1] = PAD
    inter_idx = torch.arange(bs, dtype=torch.int32, device=DEV)
    windows = torch.zeros(bs, width, dim, 3, device=DEV, dtype=torch.bfloat16)
    nxt, sib = _links(case["trees"], width)
    parents = torch.full((bs, width), -7, dtype=torch.int64, device=DEV)
    x_view = x.view(bs, width, dim).transpose(1, 2)

    if use_extension:
        ext = Mamba2TreeVerify.create(
            verify_extension.LinearAttnStateLayout(
                conv_state_len=3,
                conv_dtype=torch.bfloat16,
                ssm_dtype=ssm_dtype,
                num_v_heads=heads,
                has_state_snapshots=True,
                device=DEV,
            )
        )
        qsl = torch.arange(0, bs * width + 1, width, dtype=torch.int32, device=DEV)
        ext.begin_forward(verify_extension.LinearAttnVerifyLayout(nxt, sib, parents, slots, qsl, width), in_graph=False)
        mixer = types.SimpleNamespace(conv1d=types.SimpleNamespace(weight=weight, bias=bias), activation="silu")
        conv_out = ext.conv_update(
            mixer,
            x_view,
            conv_states=conv_state,
            cache_indices=slots,
            intermediate_conv_window=windows,
            intermediate_state_indices=inter_idx,
        )
    else:
        conv_out = conv_triton.causal_conv1d_update(
            x_view,
            conv_state,
            weight.view(dim, 4),
            bias,
            "silu",
            conv_state_indices=slots,
            intermediate_conv_window=windows,
            intermediate_state_indices=inter_idx,
            retrieve_next_token=nxt,
            retrieve_next_sibling=sib,
            retrieve_parent_token=parents,
        )
    tokens = conv_out.transpose(1, 2).reshape(bs * width, dim)
    xs, B, C = torch.split(tokens, [heads * head_dim, groups * dstate, groups * dstate], dim=-1)
    ssm_state = rnd(bs + 2, heads, head_dim, dstate, dtype=ssm_dtype)
    snapshots = torch.zeros(bs, width, heads, head_dim, dstate, device=DEV, dtype=ssm_dtype)
    out = torch.empty(bs, width, heads, head_dim, device=DEV, dtype=torch.bfloat16)
    A = -torch.rand(heads, generator=g, device=DEV)
    dt = rnd(bs * width, heads)
    dt_bias, D = rnd(heads, dtype=torch.float32), rnd(heads, dtype=torch.float32)
    mamba_ssm.selective_state_update(
        ssm_state,
        xs.reshape(bs, width, heads, head_dim),
        dt[:, :, None].expand(-1, -1, head_dim).reshape(bs, width, heads, head_dim),
        A[:, None, None].expand(-1, head_dim, dstate),
        B.reshape(bs, width, groups, dstate),
        C.reshape(bs, width, groups, dstate),
        D[:, None].expand(-1, head_dim),
        z=None,
        dt_bias=dt_bias[:, None].expand(-1, head_dim),
        dt_softplus=True,
        state_batch_indices=slots,
        out=out,
        disable_state_update=True,
        intermediate_states_buffer=snapshots,
        cache_steps=width,
        retrieve_parent_token=parents,
        intermediate_state_indices=inter_idx,
    )
    real = slice(0, bs - 1) if case["pad"] else slice(0, bs)
    return {
        "parents": parents[real],
        "windows": windows[real],
        "conv_state": conv_state,
        "conv_out": conv_out[real],
        "ssm_out": out[real],
        "snapshots": snapshots[real],
        "ssm_state": ssm_state,
    }


def _cases():
    cases = []
    for shape in SHAPES:
        for width in (2, 5, 16):
            for kinds, pad in ((("chain",), False), (("star", "random"), False), (("random",) * 4, True)):
                rng = random.Random(zlib.crc32(f"{shape}-{width}-{kinds}".encode()))
                trees = [_tree(k, width, rng) for k in kinds]
                cases.append(
                    dict(shape=shape, width=width, trees=trees, pad=pad, seed=zlib.crc32(repr(trees).encode()))
                )
    return cases


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"{c['shape']}-w{c['width']}-bs{len(c['trees'])}")
def test_extension_matches_sglangs_mamba2_tree_verify(case):
    want, got = _verify(case, use_extension=False), _verify(case, use_extension=True)
    for name in want:
        assert torch.equal(got[name], want[name]), f"{name} differs"
