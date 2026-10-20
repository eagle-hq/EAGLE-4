# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""GDN tree kernels against dense torch references (CPU through the Triton interpreter, or a GPU).

Chain trees must reproduce vLLM's own chain kernels' semantics; a branched tree must make every row see
only its ancestors, and the conv window must roll as the chain kernel rolls it.
"""

import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

from eagle_harness.kernels.gdn_tree_conv import gdn_tree_conv1d_update  # noqa: E402
from eagle_harness.kernels.gdn_tree_recurrence import gdn_tree_recurrent_update  # noqa: E402
from tests.triton_probe import needs_triton  # noqa: E402

pytestmark = needs_triton

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the conv takes 16-bit inputs; the Triton interpreter computes bf16 products wrongly, so fp16 there
HALF = torch.bfloat16 if torch.cuda.is_available() else torch.float16
TREE = [-1, 0, 0, 1, 1, 2, 3]
CHAIN = [-1, 0, 1, 2, 3]


def tree_conv(x, conv_state, weight, bias, activation, *, slots, accepted, query_start_loc, ancestors):
    """The shared tree conv in vLLM's layout (x [tokens, dim], the window in ``conv_state[slots[b]]`` from column
    ``accepted - 1``), as the extension calls it."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    gdn_tree_conv1d_update(
        x.t()[None],
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices=slots,
        intermediate_conv_window=None,
        intermediate_state_indices=None,
        tree_ancestors=ancestors,
        pad_slot_id=0,
        out=out.t()[None],
        num_accepted_tokens=accepted,
        query_start_loc=query_start_loc,
    )
    return out


def tree_recurrence(
    A_log, a, b, dt_bias, q, k, v, state, *, parents, slots, accepted, query_start_loc, kernel="triton", **kw
):
    """The shared tree recurrence in vLLM's layout (one state slot per row, resume at ``accepted - 1``), as the
    extension calls it; ``kernel`` "gluon" launches the Gluon kernel directly (no dispatch checks)."""
    args = dict(
        initial_state_source=state,
        initial_state_indices=slots,
        num_accepted_tokens=accepted,
        cu_seqlens=query_start_loc,
        intermediate_states_buffer=state,
        intermediate_state_indices=slots,
        retrieve_parent_token=parents,
        pad_slot_id=0,
        **kw,
    )
    if kernel == "gluon":
        from eagle_harness.kernels import arch

        if not arch.gluon_gdn():
            pytest.skip("the Gluon GDN kernel does not serve this GPU")
        from eagle_harness.kernels.gdn_tree_gluon import gluon_recurrent_update

        return gluon_recurrent_update(A_log, dt_bias, q, k, v, a, b, **args)
    return gdn_tree_recurrent_update(A_log, dt_bias, q, k, v, a, b, **args)


def random_tree(width, seed):
    """A tree in node order (a parent before its children): row ``r`` hangs off a random earlier row."""
    gen = torch.Generator().manual_seed(seed)
    return [-1] + [int(torch.randint(0, r, (1,), generator=gen)) for r in range(1, width)]


def deepest_path(parents):
    """Rows from the root to a deepest row of the tree."""
    best = []
    for row in range(len(parents)):
        path, node = [], row
        while node >= 0:
            path.append(node)
            node = parents[node]
        best = max(best, path[::-1], key=len)
    return best


def _ancestors(parents):
    """Per row: parent, grandparent, great-grandparent; negative codes: the old window (-1 is its last input)."""
    anc = [[-1, -2, -3]]
    for row in range(1, len(parents)):
        p = parents[row]
        anc.append([p, anc[p][0], anc[p][1]])
    return anc


def _requests(parent_lists, width):
    parents = torch.zeros(len(parent_lists), width, dtype=torch.int32)
    ancestors = torch.zeros(len(parent_lists), 3, width, dtype=torch.int32)
    qsl = [0]
    for b, plist in enumerate(parent_lists):
        parents[b, : len(plist)] = torch.tensor(plist)
        ancestors[b, :, : len(plist)] = torch.tensor(_ancestors(plist), dtype=torch.int32).t()
        qsl.append(qsl[-1] + len(plist))
    return parents.to(DEV), ancestors.to(DEV), torch.tensor(qsl, dtype=torch.int32, device=DEV)


@pytest.mark.parametrize(
    "parent_lists", [[TREE, CHAIN], [CHAIN, CHAIN], [TREE[:4], TREE], [random_tree(16, 0), random_tree(16, 1)]]
)
def test_conv_matches_reference_and_rolls_the_window(parent_lists):
    torch.manual_seed(0)
    width = max(map(len, parent_lists))
    dim, state_len, num_slots = 40, 3 + width, 8
    parents, ancestors, qsl = _requests(parent_lists, width)
    tokens = int(qsl[-1])
    bf16 = HALF
    x = torch.randn(tokens, dim, device=DEV, dtype=bf16)
    weight, bias = torch.randn(dim, 4, device=DEV, dtype=bf16), torch.randn(dim, device=DEV, dtype=bf16)
    state = torch.randn(num_slots, dim, state_len, device=DEV, dtype=bf16)
    slots = torch.tensor([2, 5][: len(parent_lists)], dtype=torch.int32, device=DEV)
    accepted = torch.tensor([1, 3][: len(parent_lists)], dtype=torch.int32, device=DEV)
    old = state.clone()
    out = tree_conv(
        x, state, weight, bias, "silu", slots=slots, accepted=accepted, query_start_loc=qsl, ancestors=ancestors
    )

    for b, plist in enumerate(parent_lists):
        window = old[int(slots[b]), :, int(accepted[b]) - 1 : int(accepted[b]) + 2]  # [dim, 3]: oldest first
        rows = x[int(qsl[b]) : int(qsl[b + 1])]
        for row, codes in enumerate(_ancestors(plist)):
            inputs = [(rows[c] if c >= 0 else window[:, 3 + c]) for c in reversed(codes)] + [rows[row]]
            want = bias.float() + sum((w * v).float() for w, v in zip(weight.t(), inputs))
            want = want * torch.sigmoid(want)
            torch.testing.assert_close(out[int(qsl[b]) + row].float(), want, atol=2e-2, rtol=2e-2)
        rolled = state[int(slots[b])]
        torch.testing.assert_close(rolled[:, :2], old[int(slots[b]), :, int(accepted[b]) : int(accepted[b]) + 2])
        torch.testing.assert_close(rolled[:, 2 : 2 + len(plist)], rows.t())


def _gate(a, b, A_log, dt_bias):
    g = -torch.exp(A_log) * torch.nn.functional.softplus(a + dt_bias)
    return g, torch.sigmoid(b)


def _reference(q, k, v, a, b, A_log, dt_bias, init, parents, heads, value_heads):
    """Per row: ``(output [HV, V], state [HV, V, K])``, each row continuing from its parent's state."""
    outs, states = [], []
    for row, parent in enumerate(parents):
        state = (init if parent < 0 else states[parent]).clone()
        qn, kn = q[row], k[row]
        qn = qn / torch.sqrt((qn * qn).sum(-1, keepdim=True) + 1e-6)
        kn = kn / torch.sqrt((kn * kn).sum(-1, keepdim=True) + 1e-6)
        qn, kn = qn * q.shape[-1] ** -0.5, kn
        g, beta = _gate(a[row], b[row], A_log, dt_bias)
        out = torch.zeros_like(v[row])
        for hv in range(value_heads):
            h = hv // (value_heads // heads)
            s = state[hv] * torch.exp(g[hv])
            delta = (v[row, hv] - s @ kn[h]) * beta[hv]
            s = s + delta[:, None] * kn[h][None, :]
            state[hv] = s
            out[hv] = s @ qn[h]
        outs.append(out)
        states.append(state)
    return torch.stack(outs), states


@pytest.mark.parametrize("parent_lists", [[TREE, CHAIN], [TREE, TREE[:5]], [random_tree(16, 2), random_tree(16, 3)]])
@pytest.mark.parametrize("dims", [(2, 4, 8, 8), (2, 6, 128, 128)])  # heads, value heads, key dim, value dim
def test_recurrence_continues_each_row_from_its_parent(parent_lists, dims):
    torch.manual_seed(1)
    width = max(map(len, parent_lists))
    (heads, value_heads, dk, dv), num_slots = dims, 40
    parents, _, qsl = _requests(parent_lists, width)
    tokens = int(qsl[-1])
    q, k = torch.randn(1, tokens, heads, dk, device=DEV), torch.randn(1, tokens, heads, dk, device=DEV)
    v = torch.randn(1, tokens, value_heads, dv, device=DEV)
    a, b = torch.randn(tokens, value_heads, device=DEV), torch.randn(tokens, value_heads, device=DEV)
    A_log, dt_bias = torch.randn(value_heads, device=DEV), torch.randn(value_heads, device=DEV)
    state = torch.randn(num_slots, value_heads, dv, dk, device=DEV) * 0.1
    slots = torch.arange(1, 1 + len(parent_lists) * (width + 1), dtype=torch.int32, device=DEV).view(
        len(parent_lists), width + 1
    )
    accepted = torch.tensor([1, 2][: len(parent_lists)], dtype=torch.int32, device=DEV)
    init = [state[int(slots[i, int(accepted[i]) - 1])].clone() for i in range(len(parent_lists))]

    out = tree_recurrence(
        A_log, a, b, dt_bias, q, k, v, state, parents=parents, slots=slots, accepted=accepted, query_start_loc=qsl
    )

    for i, plist in enumerate(parent_lists):
        lo, hi = int(qsl[i]), int(qsl[i + 1])
        want_out, want_states = _reference(
            q[0, lo:hi],
            k[0, lo:hi],
            v[0, lo:hi],
            a[lo:hi],
            b[lo:hi],
            A_log,
            dt_bias,
            init[i],
            plist,
            heads,
            value_heads,
        )
        torch.testing.assert_close(out[0, lo:hi], want_out, atol=1e-4, rtol=1e-4)
        for row, want in enumerate(want_states):
            torch.testing.assert_close(state[int(slots[i, row])], want, atol=1e-4, rtol=1e-4)


def test_spec_rows_read_the_packed_projection_in_place_and_write_the_layer_output():
    """``TreeVerify.forward_spec_rows`` on strided views of the projections == tree_conv + contiguous
    q/k/v/a/b + tree_recurrence, bit for bit; the output lands in the layer's buffer and the publishing layer's
    slots are copied out."""
    from types import SimpleNamespace

    pytest.importorskip("vllm")
    from vllm.model_executor.layers.mamba.gdn.verify_extension import GDNSpecRows

    from eagle_harness.vllm.gdn_tree_verify import TreeVerify
    from eagle_harness.vllm.verify_layout import set_stage

    torch.manual_seed(3)
    plists = [random_tree(8, 4), TREE]
    width = max(map(len, plists))
    heads, value_heads, dk, dv = 2, 4, 16, 16
    key_dim, value_dim = heads * dk, value_heads * dv
    dim = 2 * key_dim + value_dim
    parents, ancestors, qsl = _requests(plists, width)
    tokens = int(qsl[-1])
    a_log, dt_bias = torch.randn(value_heads, device=DEV), torch.randn(value_heads, device=DEV)
    published = torch.zeros(len(plists), width + 1, dtype=torch.int32, device=DEV)
    set_stage(SimpleNamespace(ancestors=ancestors, parents=parents, state_slots={"layers.0": published}))
    slots = torch.arange(1, 1 + len(plists) * (width + 1), dtype=torch.int32, device=DEV).view(len(plists), -1)
    accepted = torch.tensor([1, 2], dtype=torch.int32, device=DEV)
    bf16 = HALF
    qkvz = torch.randn(tokens, dim + value_dim, device=DEV, dtype=bf16)  # the projection: [q, k, v, z]
    ba = torch.randn(tokens, 2 * value_heads, device=DEV, dtype=bf16)
    b, a = ba.chunk(2, dim=-1)
    weight, bias = torch.randn(dim, 4, device=DEV, dtype=bf16), torch.randn(dim, device=DEV, dtype=bf16)
    conv0 = torch.randn(slots.numel() + 1, dim, 3 + width, device=DEV, dtype=bf16)
    ssm0 = torch.randn(slots.numel() + 1, value_heads, dv, dk, device=DEV) * 0.1
    try:
        conv_state, ssm_state = conv0.clone(), ssm0.clone()
        conv = tree_conv(
            qkvz[:, :dim].contiguous(),
            conv_state,
            weight,
            bias,
            "silu",
            slots=slots[:, 0],
            accepted=accepted,
            query_start_loc=qsl,
            ancestors=ancestors,
        )
        q, k, v = (t.contiguous() for t in conv.split([key_dim, key_dim, value_dim], dim=-1))
        want = tree_recurrence(
            a_log,
            a.contiguous(),
            b.contiguous(),
            dt_bias,
            q.view(1, tokens, heads, dk),
            k.view(1, tokens, heads, dk),
            v.view(1, tokens, value_heads, dv),
            ssm_state,
            parents=parents,
            slots=slots,
            accepted=accepted,
            query_start_loc=qsl,
        )
        got_conv, got_ssm = conv0.clone(), ssm0.clone()
        out = torch.zeros(tokens, value_heads, dv, device=DEV, dtype=bf16)
        TreeVerify(None).forward_spec_rows(
            GDNSpecRows(
                layer_name="layers.0",
                mixed_qkv=qkvz[:, :dim],
                a=a,
                b=b,
                conv_state=got_conv,
                conv_weight=weight,
                conv_bias=bias,
                activation="silu",
                ssm_state=got_ssm,
                A_log=a_log,
                dt_bias=dt_bias,
                num_k_heads=heads,
                num_v_heads=value_heads,
                head_k_dim=dk,
                head_v_dim=dv,
                state_indices=slots,
                num_accepted_tokens=accepted,
                query_start_loc=qsl,
                out=out,
            )
        )
    finally:
        set_stage(None)
    assert torch.equal(out, want.view(tokens, value_heads, dv))
    assert torch.equal(got_ssm, ssm_state) and torch.equal(got_conv, conv_state)
    assert torch.equal(published, slots)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM's own kernels need a GPU")
@pytest.mark.parametrize("dim_first", [True, False])
def test_a_chain_is_what_vllm_verifies(dim_first):
    """On a chain the tree kernels must be vLLM's own chain kernels (outputs and the states they leave)."""
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fused_sigmoid_gating_delta_rule_update

    torch.manual_seed(2)
    width, dim, state_len, num_slots, reqs = 6, 64, 3 + 5, 40, 3
    parents, ancestors, qsl = _requests([list(range(-1, width - 1))] * reqs, width)
    tokens = reqs * width
    accepted = torch.tensor([1, 3, 6][:reqs], dtype=torch.int32, device=DEV)
    x = torch.randn(tokens, dim, device=DEV, dtype=torch.bfloat16)
    weight, bias = (
        torch.randn(dim, 4, device=DEV, dtype=torch.bfloat16),
        torch.randn(dim, device=DEV, dtype=torch.bfloat16),
    )
    raw = torch.randn(
        (num_slots, dim, state_len) if dim_first else (num_slots, state_len, dim), device=DEV, dtype=torch.bfloat16
    )
    view = raw if dim_first else raw.transpose(-1, -2)
    slots = torch.arange(1, 1 + reqs * (width + 1), dtype=torch.int32, device=DEV).view(reqs, width + 1)
    ours, theirs = view.clone(), view.clone()
    got = tree_conv(
        x,
        ours,
        weight,
        bias,
        "silu",
        slots=slots[:, 0].contiguous(),
        accepted=accepted,
        query_start_loc=qsl,
        ancestors=ancestors,
    )
    want = causal_conv1d_update(
        x,
        theirs,
        weight,
        bias,
        "silu",
        conv_state_indices=slots[:, 0].contiguous(),
        num_accepted_tokens=accepted,
        query_start_loc=qsl,
        max_query_len=width + 1,
        validate_data=False,
    )
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(ours[slots[:, 0].long()], theirs[slots[:, 0].long()])

    heads, value_heads, dk, dv = 2, 4, 128, 128
    q, k = (torch.randn(1, tokens, heads, dk, device=DEV, dtype=torch.bfloat16) for _ in range(2))
    v = torch.randn(1, tokens, value_heads, dv, device=DEV, dtype=torch.bfloat16)
    a, b = (torch.randn(tokens, value_heads, device=DEV, dtype=torch.bfloat16) for _ in range(2))
    A_log, dt_bias = torch.randn(value_heads, device=DEV), torch.randn(value_heads, device=DEV, dtype=torch.bfloat16)
    state = torch.randn(num_slots, value_heads, dv, dk, device=DEV) * 0.1
    ours, theirs = state.clone(), state.clone()
    got = tree_recurrence(
        A_log, a, b, dt_bias, q, k, v, ours, parents=parents, slots=slots, accepted=accepted, query_start_loc=qsl
    )
    want, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=theirs,
        inplace_final_state=True,
        cu_seqlens=qsl,
        ssm_state_indices=slots,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(ours, theirs, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM's own kernels need a GPU")
@pytest.mark.parametrize(
    "parent_lists,paths",
    [
        ([TREE, TREE], [[0, 1, 4], [0, 2, 5]]),  # accepted rows that are not a prefix of the verify rows
        ([TREE, TREE], [[0], [0, 1, 3]]),
        ([random_tree(16, 4), random_tree(16, 5)], [[0, 1], [0, 1, 2]]),
        (
            [random_tree(16, 8), random_tree(16, 9)],
            [deepest_path(random_tree(16, 8)), deepest_path(random_tree(16, 9))],
        ),
        (
            [[-1, 0, 1, 2, 3, 3, 4, 5], [-1, 0, 1, 2, 3, 3, 4, 5]],
            [[0, 1, 2, 3, 5, 7], [0, 1, 2, 3, 4, 6]],
        ),  # long paths ending off the prefix
    ],
)
def test_a_committed_tree_round_continues_like_the_accepted_chain(parent_lists, paths):
    """After tree verify + commit, vLLM's own chain kernels resume exactly as if only the accepted path had been verified."""
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fused_sigmoid_gating_delta_rule_update
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    from eagle_harness.vllm.gdn_tree_verify import LinearStateCommit
    from eagle_harness.vllm.verify_layout import TreeStore

    torch.manual_seed(3)
    reqs = len(parent_lists)
    width = max(map(len, parent_lists))
    dim, state_len, num_slots = 48, width + 2, 64
    heads, value_heads, dk, dv = 2, 4, 64, 64
    bf16 = torch.bfloat16
    parents, ancestors, qsl = _requests(parent_lists, width)
    tokens = int(qsl[-1])
    slots = torch.arange(1, 1 + reqs * (width + 1), dtype=torch.int32, device=DEV).view(reqs, width + 1)
    first = torch.tensor([1, 2][:reqs], dtype=torch.int32, device=DEV)  # accepted before the round

    x = torch.randn(tokens, dim, device=DEV, dtype=bf16)
    weight, bias = torch.randn(dim, 4, device=DEV, dtype=bf16), torch.randn(dim, device=DEV, dtype=bf16)
    q, k = (torch.randn(1, tokens, heads, dk, device=DEV, dtype=bf16) for _ in range(2))
    v = torch.randn(1, tokens, value_heads, dv, device=DEV, dtype=bf16)
    a, b = (torch.randn(tokens, value_heads, device=DEV, dtype=bf16) for _ in range(2))
    A_log, dt_bias = torch.randn(value_heads, device=DEV), torch.randn(value_heads, device=DEV, dtype=bf16)

    dim_first = is_conv_state_dim_first()
    conv_raw = torch.randn(
        (num_slots, dim, state_len) if dim_first else (num_slots, state_len, dim), device=DEV, dtype=bf16
    )
    ssm0 = torch.randn(num_slots, value_heads, dv, dk, device=DEV) * 0.1
    window = lambda raw: raw if dim_first else raw.transpose(-1, -2)  # noqa: E731

    # tree side: verify the whole tree, then commit the accepted path
    tree_conv_raw, tree_ssm = conv_raw.clone(), ssm0.clone()
    tree_conv(
        x,
        window(tree_conv_raw),
        weight,
        bias,
        "silu",
        slots=slots[:, 0].contiguous(),
        accepted=first,
        query_start_loc=qsl,
        ancestors=ancestors,
    )
    tree_recurrence(
        A_log, a, b, dt_bias, q, k, v, tree_ssm, parents=parents, slots=slots, accepted=first, query_start_loc=qsl
    )
    store = TreeStore(8, width, DEV)
    for r, path in enumerate(paths):
        store.path[r, : len(path)] = torch.tensor(path, dtype=torch.int32, device=DEV)
        store.path_len[r] = len(path)

    class Layer:
        kv_cache = (tree_conv_raw, tree_ssm)

    LinearStateCommit([Layer], [slots], DEV).commit(
        torch.arange(reqs, dtype=torch.int64, device=DEV), store.path, store.path_len
    )

    # reference side: vLLM verifies a chain of `width` rows per request, the accepted path first; the rows after
    # the accepted ones are never read again
    ref_conv_raw, ref_ssm = conv_raw.clone(), ssm0.clone()
    order = [path + [r for r in range(width) if r not in path] for path in paths]
    rows = torch.cat([qsl[r] + torch.tensor(o, device=DEV) for r, o in enumerate(order)])
    lens = qsl
    causal_conv1d_update(
        x[rows],
        window(ref_conv_raw),
        weight,
        bias,
        "silu",
        conv_state_indices=slots[:, 0].contiguous(),
        num_accepted_tokens=first,
        query_start_loc=lens,
        max_query_len=width + 1,
        validate_data=False,
    )
    fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a[rows],
        b=b[rows],
        dt_bias=dt_bias,
        q=q[:, rows],
        k=k[:, rows],
        v=v[:, rows],
        initial_state=ref_ssm,
        inplace_final_state=True,
        cu_seqlens=lens,
        ssm_state_indices=slots,
        num_accepted_tokens=first,
        use_qk_l2norm_in_kernel=True,
    )

    # the next step, a chain of 4 rows resuming after the accepted path, must agree
    accepted = torch.tensor([len(p) for p in paths], dtype=torch.int32, device=DEV)
    nxt = 4
    nqsl = torch.arange(0, (reqs + 1) * nxt, nxt, dtype=torch.int32, device=DEV)
    x2 = torch.randn(reqs * nxt, dim, device=DEV, dtype=bf16)
    q2, k2 = (torch.randn(1, reqs * nxt, heads, dk, device=DEV, dtype=bf16) for _ in range(2))
    v2 = torch.randn(1, reqs * nxt, value_heads, dv, device=DEV, dtype=bf16)
    a2, b2 = (torch.randn(reqs * nxt, value_heads, device=DEV, dtype=bf16) for _ in range(2))
    outs = []
    for conv_raw_side, ssm_side in ((tree_conv_raw, tree_ssm), (ref_conv_raw, ref_ssm)):
        y = causal_conv1d_update(
            x2.clone(),
            window(conv_raw_side),
            weight,
            bias,
            "silu",  # (vLLM's conv writes over its input)
            conv_state_indices=slots[:, 0].contiguous(),
            num_accepted_tokens=accepted,
            query_start_loc=nqsl,
            max_query_len=width + 1,
            validate_data=False,
        )
        o, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a2,
            b=b2,
            dt_bias=dt_bias,
            q=q2,
            k=k2,
            v=v2,
            initial_state=ssm_side,
            inplace_final_state=True,
            cu_seqlens=nqsl,
            ssm_state_indices=slots,
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
        )
        outs.append((y.float(), o.float()))
    torch.testing.assert_close(outs[0][0], outs[1][0], atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(outs[0][1], outs[1][1], atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM's own kernels need a GPU")
@pytest.mark.parametrize("accepted", [[1, 1], [2, 5], [4, 9], [16, 7]])
@pytest.mark.parametrize("kernel", ["triton", "gluon"])
def test_every_row_equals_vllm_verifying_its_path_as_a_chain(accepted, kernel):
    """A tree row computes what vLLM's chain kernels compute for the chain root -> .. -> row (any resume offset)."""
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fused_sigmoid_gating_delta_rule_update
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    torch.manual_seed(5)
    parent_lists = [random_tree(16, 6), random_tree(16, 7)]
    reqs, width = 2, 16
    dim, state_len, num_slots = 96, width + 2, 80
    heads, value_heads, dk, dv = 4, 12, 128, 128
    bf16 = torch.bfloat16
    parents, ancestors, qsl = _requests(parent_lists, width)
    tokens = int(qsl[-1])
    slots = torch.arange(1, 1 + reqs * width, dtype=torch.int32, device=DEV).view(reqs, width)
    slots = torch.cat([slots, slots[:, -1:] + 1], dim=1)
    acc = torch.tensor(accepted, dtype=torch.int32, device=DEV)
    x = torch.randn(tokens, dim, device=DEV, dtype=bf16)
    weight, bias = torch.randn(dim, 4, device=DEV, dtype=bf16), torch.randn(dim, device=DEV, dtype=bf16)
    q, k = (torch.randn(1, tokens, heads, dk, device=DEV, dtype=bf16) for _ in range(2))
    v = torch.randn(1, tokens, value_heads, dv, device=DEV, dtype=bf16)
    a, b = (torch.randn(tokens, value_heads, device=DEV, dtype=bf16) for _ in range(2))
    A_log, dt_bias = torch.randn(value_heads, device=DEV), torch.randn(value_heads, device=DEV, dtype=bf16)
    dim_first = is_conv_state_dim_first()
    conv_raw = torch.randn(
        (num_slots, dim, state_len) if dim_first else (num_slots, state_len, dim), device=DEV, dtype=bf16
    )
    ssm0 = torch.randn(num_slots, value_heads, dv, dk, device=DEV, dtype=bf16) * 0.1
    window = lambda raw: raw if dim_first else raw.transpose(-1, -2)  # noqa: E731

    tree_raw, tree_ssm = conv_raw.clone(), ssm0.clone()
    got_conv = tree_conv(
        x,
        window(tree_raw),
        weight,
        bias,
        "silu",
        slots=slots[:, 0].contiguous(),
        accepted=acc,
        query_start_loc=qsl,
        ancestors=ancestors,
    )
    got_rec = tree_recurrence(
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        tree_ssm,
        parents=parents,
        slots=slots,
        accepted=acc,
        query_start_loc=qsl,
        kernel=kernel,
    )

    for r, plist in enumerate(parent_lists):
        for row in range(width):
            chain, node = [], row
            while node >= 0:
                chain.append(node)
                node = plist[node]
            chain = chain[::-1]
            idx = qsl[r] + torch.tensor(chain, device=DEV)
            n = len(chain)
            one = torch.tensor([0, n], dtype=torch.int32, device=DEV)
            raw, ssm = conv_raw.clone(), ssm0.clone()
            y = causal_conv1d_update(
                x[idx].clone(),
                window(raw),
                weight,
                bias,
                "silu",
                conv_state_indices=slots[r : r + 1, 0].contiguous(),
                num_accepted_tokens=acc[r : r + 1],
                query_start_loc=one,
                max_query_len=width + 1,
                validate_data=False,
            )
            o, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                a=a[idx],
                b=b[idx],
                dt_bias=dt_bias,
                q=q[:, idx],
                k=k[:, idx],
                v=v[:, idx],
                initial_state=ssm,
                inplace_final_state=True,
                cu_seqlens=one,
                ssm_state_indices=slots[r : r + 1],
                num_accepted_tokens=acc[r : r + 1],
                use_qk_l2norm_in_kernel=True,
            )
            torch.testing.assert_close(got_conv[int(qsl[r]) + row].float(), y[-1].float(), atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(got_rec[0, int(qsl[r]) + row].float(), o[0, -1].float(), atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(
                tree_ssm[int(slots[r, row])].float(), ssm[int(slots[r, n - 1])].float(), atol=3e-2, rtol=3e-2
            )


@pytest.mark.parametrize(
    "kernel",
    ["triton", pytest.param("gluon", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"))],
)
def test_recurrence_publishes_every_requests_slots(kernel):
    """``publish`` receives each request's state slots from the kernel, a padded request's too (no rows, so its
    programs do no recurrence), and leaves the columns past the slots alone; the outputs do not change."""
    torch.manual_seed(5)
    heads, value_heads, d, width, bf16 = 1, 2, 128, 4, torch.bfloat16
    parents = torch.tensor([[-1, 0, 0, 1], [-1, 0, 1, 2], [0, 0, 0, 0]], dtype=torch.int32, device=DEV)
    qsl = torch.tensor([0, 4, 8, 8], dtype=torch.int32, device=DEV)  # request 2 is padding: no rows
    tokens = 8
    q, k = (torch.randn(1, tokens, heads, d, device=DEV, dtype=bf16) for _ in range(2))
    v = torch.randn(1, tokens, value_heads, d, device=DEV, dtype=bf16)
    a, b = (torch.randn(tokens, value_heads, device=DEV, dtype=bf16) for _ in range(2))
    A_log, dt_bias = torch.zeros(value_heads, device=DEV), torch.zeros(value_heads, device=DEV, dtype=bf16)
    slots = torch.arange(1, 1 + 3 * (width + 1), dtype=torch.int32, device=DEV).view(3, width + 1)
    slots[2] = 0
    accepted = torch.tensor([1, 2, 1], dtype=torch.int32, device=DEV)
    state0 = torch.randn(slots.numel() + 1, value_heads, d, d, device=DEV) * 0.05
    results = []
    for publish in (None, torch.full((3, width + 3), -7, dtype=torch.int32, device=DEV)):
        state = state0.clone()
        out = tree_recurrence(
            A_log,
            a,
            b,
            dt_bias,
            q,
            k,
            v,
            state,
            parents=parents,
            slots=slots,
            accepted=accepted,
            query_start_loc=qsl,
            publish=publish,
            kernel=kernel,
        )
        results.append((out, state, publish))
    (out0, state0_after, _), (out1, state1_after, published) = results
    assert torch.equal(published[:, : width + 1], slots)
    assert torch.all(published[:, width + 1 :] == -7)
    assert torch.equal(out1, out0) and torch.equal(state1_after, state0_after)
