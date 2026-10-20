# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The Gluon GDN tree verify against the Triton walk it replaces, bitwise, on the trees serving builds (GPU).

The dispatcher's run-time self-check draws one random tree per launch signature, which misses a last-bit difference
in one snapshot element in 10^7 (Triton 3.8 fusing other multiply-adds in the two kernels). This sweeps chains, stars,
random trees and EAGLE-style top-k levels, with and without PDL, at the TP4 and TP1 shapes of Qwen3.8-27B
(H 16 / HV 48, K = V = 128, split by TP 1 to 8), in both slot layouts: sglang's (one bf16 snapshot scratch row per request)
and vLLM's (one state slot per verify row, bf16 or fp32 state, several requests with resume offsets, a padded
request, token-strided views of one packed projection, a published slot table). Run it on every Triton version the
package supports.

Run in a GPU job: ``pytest -q tests/test_gdn_tree_gluon_gpu.py``.
"""

import random
import zlib

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from eagle_harness.kernels import arch  # noqa: E402
from eagle_harness.kernels.gdn_tree_recurrence import gdn_tree_recurrent_update  # noqa: E402

if not arch.gluon_gdn():
    pytest.skip("the Gluon GDN kernel does not serve this GPU", allow_module_level=True)

from eagle_harness.kernels.gdn_tree_gluon import gluon_admits, gluon_recurrent_update  # noqa: E402


def _chain(n):
    return [0] + list(range(n - 1))


def _star(n):
    return [0] * n


def _random(n, seed):
    rng = random.Random(seed)
    return [0] + [rng.randrange(i) for i in range(1, n)]


def _topk_levels(n, k=4):
    """Root, then levels of k children hanging off the previous level's first node (EAGLE top-k expansion)."""
    parents, prev_first = [0], 0
    while len(parents) < n:
        first = len(parents)
        parents += [prev_first] * min(k, n - len(parents))
        prev_first = first
    return parents


TREES = {
    "chain": _chain,
    "star": _star,
    "random1": lambda n: _random(n, 1),
    "random2": lambda n: _random(n, 2),
    "topk4": _topk_levels,
}


def _activations(tp, tokens, g):
    """q, k, v, a, b, A_log, dt_bias at a TP shape; q / k / v are token-strided views of one packed projection and
    a / b column slices of another, as both engines hand them over."""
    H, HV, d = 16 // tp, 48 // tp, 128
    dev = "cuda"

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device=dev, generator=g) * scale

    packed = torch.nn.functional.silu(randn(tokens, (2 * H + HV) * d + 16)).bfloat16()
    q, k, v = packed[:, 16:].split([H * d, H * d, HV * d], dim=-1)
    ab = randn(tokens, 2 * HV + 3).bfloat16()
    return dict(
        A_log=torch.log(torch.rand(HV, device=dev, generator=g) * 15 + 1),
        dt_bias=randn(HV, scale=0.5).bfloat16(),
        q=q.view(1, tokens, H, d),
        k=k.view(1, tokens, H, d),
        v=v.view(1, tokens, HV, d),
        a=ab[:, 1 : 1 + HV],
        b=ab[:, 2 + HV : 2 + 2 * HV],
    )


def _per_request_inputs(tp, width, parents, seed):
    """sglang: one request, the pool's initial state, a bf16 scratch row of snapshots."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    act = _activations(tp, width, g)
    HV, d = act["v"].shape[2], 128
    i32 = dict(dtype=torch.int32, device="cuda")
    inputs = dict(
        act,
        initial_state_source=(torch.randn(2, HV, d, d, device="cuda", generator=g) * 0.05).bfloat16(),
        initial_state_indices=torch.tensor([1], **i32),
        cu_seqlens=torch.tensor([0, width], **i32),
        intermediate_state_indices=torch.tensor([0], **i32),
        retrieve_parent_token=torch.tensor([parents], **i32),
    )
    states = dict(intermediate_states_buffer=torch.randn(1, width, HV, d, d, device="cuda", generator=g).bfloat16())
    return inputs, states


def _per_row_inputs(tp, width, parents, seed, state_dtype):
    """vLLM: the tree, a shorter chain resuming at row 2, and a padded request (null slot 0) of one row; one state
    slot per row in a pool that is also the initial state, the output and the published table in caller buffers."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    chain = [-1] + list(range(max(width - 3, 1) - 1))
    rows = [[-1] + parents[1:], chain, [-1]]
    tokens = sum(map(len, rows))
    act = _activations(tp, tokens, g)
    HV, d = act["v"].shape[2], 128
    i32 = dict(dtype=torch.int32, device="cuda")
    table = torch.full((len(rows), width), -1, **i32)
    for r, plist in enumerate(rows):
        table[r, : len(plist)] = torch.tensor(plist, **i32)
    slots = torch.arange(1, 1 + len(rows) * (width + 1), **i32).view(len(rows), width + 1)
    slots[2] = 0
    qsl = torch.tensor([0, len(rows[0]), len(rows[0]) + len(rows[1]), tokens], **i32)
    inputs = dict(
        act,
        initial_state_indices=slots,
        num_accepted_tokens=torch.tensor([1, 2, 1], **i32),
        cu_seqlens=qsl,
        intermediate_state_indices=slots,
        retrieve_parent_token=table,
        pad_slot_id=0,
    )
    pool = (torch.randn(slots.numel() + 2, HV, d, d, device="cuda", generator=g) * 0.05).to(state_dtype)
    states = dict(
        pool=pool,
        out=torch.zeros(1, tokens, HV, d, device="cuda", dtype=torch.bfloat16),
        publish=torch.full((len(rows), width + 3), -7, **i32),
    )
    return inputs, states


def _launch(fn, inputs, states, **kw):
    states = {n: t.clone() for n, t in states.items()}
    if "pool" in states:
        kw.update(
            initial_state_source=states["pool"],
            intermediate_states_buffer=states["pool"],
            out=states["out"],
            publish=states["publish"],
        )
    else:
        kw.update(intermediate_states_buffer=states["intermediate_states_buffer"])
    o = fn(**inputs, **kw)
    return o, states


LAYOUTS = {
    "per-request": lambda tp, width, parents, seed: _per_request_inputs(tp, width, parents, seed),
    "per-row-bf16": lambda tp, width, parents, seed: _per_row_inputs(tp, width, parents, seed, torch.bfloat16),
    "per-row-fp32": lambda tp, width, parents, seed: _per_row_inputs(tp, width, parents, seed, torch.float32),
}


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("tp", [8, 4, 2, 1])
@pytest.mark.parametrize("width", [4, 8, 16, 32])
@pytest.mark.parametrize("tree", sorted(TREES))
def test_gluon_equals_the_triton_walk_bitwise(layout, tp, width, tree):
    parents = TREES[tree](width)
    inputs, states = LAYOUTS[layout](tp, width, parents, seed=zlib.crc32(f"{layout}-{tp}-{width}-{tree}".encode()))
    pool0 = inputs.get("initial_state_source", torch.empty(0)).clone()
    o_ref, ref = _launch(gdn_tree_recurrent_update, inputs, states, gluon=False)
    live, _ = _launch(lambda **kw: kw, inputs, states)
    assert gluon_admits(**{n: t for n, t in live.items() if isinstance(t, torch.Tensor)})
    bad = []
    for use_pdl in (False, True):
        o, got = _launch(gluon_recurrent_update, inputs, states, use_pdl=use_pdl)
        diffs = {"o": int((o.view(torch.int16) != o_ref.view(torch.int16)).sum())}
        for name in got:
            x, y = got[name].contiguous(), ref[name].contiguous()
            view = torch.int32 if x.element_size() == 4 else torch.int16
            diffs[name] = int((x.view(view) != y.view(view)).sum())
        if any(diffs.values()):
            bad.append(f"pdl={use_pdl}: elements differ {diffs}")
    if "pool" in states:  # the null slot is never written, the table is published
        assert torch.equal(ref["pool"][0], states["pool"][0])
        assert torch.equal(ref["publish"][:, : width + 1], inputs["intermediate_state_indices"])
    else:  # the pool is only read
        assert torch.equal(inputs["initial_state_source"], pool0)
    assert not bad, f"tree {tree} {parents}: " + "; ".join(bad)


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_the_dispatcher_serves_gluon_after_its_self_check(layout):
    """gdn_tree_recurrent_update(gluon=True) self-checks the launch signature (in the layout's slot tables) and serves
    the Gluon kernel: same output and states as the walk, and the kernel stays enabled."""
    from eagle_harness.kernels import gdn_tree_gluon

    gdn_tree_gluon.reset_gdn_tree_gluon_state()
    inputs, states = LAYOUTS[layout](4, 16, _topk_levels(16), seed=zlib.crc32(layout.encode()))
    o_ref, ref = _launch(gdn_tree_recurrent_update, inputs, states, gluon=False)
    o, got = _launch(gdn_tree_recurrent_update, inputs, states, gluon=True)
    assert not gdn_tree_gluon.gdn_tree_gluon_disabled()
    assert len(gdn_tree_gluon._STATE.checked) == 1
    assert torch.equal(o, o_ref) and all(torch.equal(got[n], ref[n]) for n in got)


def test_rows_that_triton_cannot_vectorize_keep_the_walk():
    """q / k rows at a token stride Triton does not know to be aligned reduce the norms in another order in the walk
    (a last-bit difference in some snapshots), so the launcher does not admit them."""
    inputs, states = _per_request_inputs(4, 16, _topk_levels(16), seed=3)
    tokens, H, HV = 16, inputs["q"].shape[2], inputs["v"].shape[2]
    packed = torch.zeros(tokens, (2 * H + HV) * 128 + 8, device="cuda", dtype=torch.bfloat16)[:, 8:]
    q, k, v = packed.split([H * 128, H * 128, HV * 128], dim=-1)
    live = dict(inputs, q=q.view(1, tokens, H, 128), k=k.view(1, tokens, H, 128), v=v.view(1, tokens, HV, 128))
    assert not gluon_admits(**live, intermediate_states_buffer=states["intermediate_states_buffer"])
    assert gluon_admits(**inputs, intermediate_states_buffer=states["intermediate_states_buffer"])
