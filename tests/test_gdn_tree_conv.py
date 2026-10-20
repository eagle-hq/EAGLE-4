# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The GDN tree conv in both engines' layouts against a torch reference (CPU through the Triton interpreter, or a GPU).

sglang: fixed-width requests, history in conv-state columns 0-2, every node's window in the scratch, the state shifted
to the last 3 inputs. vLLM: packed requests of any width, history at column ``num_accepted - 1``, the state rolled to
``[history[1:], x_0, .., x_{n-1}]``. Both kernels (gather, serial).
"""

import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

from eagle_harness.kernels import gdn_tree_conv  # noqa: E402
from tests.triton_probe import needs_triton  # noqa: E402

pytestmark = needs_triton

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# serving's bf16 on a GPU; the Triton interpreter computes bf16 products wrongly, so fp16 there
BF16 = torch.bfloat16 if torch.cuda.is_available() else torch.float16


def _ancestor_codes(parents):
    """Per node: parent, grandparent, great-grandparent; -1/-2/-3 = history columns 2/1/0."""
    anc = [[-1, -2, -3]]
    for node in range(1, len(parents)):
        p = parents[node]
        anc.append([p, anc[p][0], anc[p][1]])
    return anc


def _reference(x_nodes, history, weight, bias, parents):
    """Node outputs ([nodes, dim], fp32) and windows ([nodes, dim, 3]: grandparent, parent, self)."""
    outs, windows = [], []
    for node, codes in enumerate(_ancestor_codes(parents)):
        vals = [x_nodes[c] if c >= 0 else history[:, 3 + c] for c in codes]  # parent, grandparent, great-grandparent
        y = bias.float() + (x_nodes[node] * weight[:, 3]).float()
        for v, w in zip(vals, (weight[:, 2], weight[:, 1], weight[:, 0])):
            y = y + (v * w).float()
        outs.append(y * torch.sigmoid(y))
        windows.append(torch.stack([vals[1], vals[0], x_nodes[node]], dim=-1))
    return torch.stack(outs), torch.stack(windows)


@pytest.fixture(params=["gather", "serial"])
def kernel(request, monkeypatch):
    if request.param == "serial":
        monkeypatch.setattr(gdn_tree_conv, "CONV_SERIAL_MIN_BATCH_DIM", 0)
    return request.param


@pytest.mark.parametrize("width", [1, 2, 5])
def test_sglang_layout_writes_windows_and_shifts_the_state(kernel, width):
    torch.manual_seed(1)
    batch, dim = 3, 24
    trees = [[0] + [torch.randint(0, n, (1,)).item() for n in range(1, width)] for _ in range(batch)]
    x = torch.randn(batch * width, dim, device=DEV, dtype=BF16)  # token-major, viewed as (batch, dim, width)
    weight, bias = torch.randn(dim, 4, device=DEV, dtype=BF16), torch.randn(dim, device=DEV, dtype=BF16)
    state0 = torch.randn(5, dim, 3, device=DEV, dtype=BF16)
    idx = torch.tensor([3, -1, 1], dtype=torch.int32, device=DEV)  # request 1 is padding
    inter = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEV)
    anc = torch.tensor([list(zip(*_ancestor_codes(t))) for t in trees], dtype=torch.int32, device=DEV)
    windows = torch.zeros(3, width, dim, 3, device=DEV, dtype=BF16)
    state = state0.clone()
    out = gdn_tree_conv.gdn_tree_conv1d_update(
        x.view(batch, width, dim).transpose(1, 2), state, weight, bias, True, idx, windows, inter, anc.contiguous()
    )
    for n in (0, 2):
        nodes = x[n * width : (n + 1) * width]
        want, want_win = _reference(nodes, state0[idx[n]], weight, bias, trees[n])
        torch.testing.assert_close(out[n].t().float(), want, atol=2e-2, rtol=2e-2)
        assert torch.equal(windows[n], want_win)
        assert torch.equal(state[idx[n]], torch.cat([state0[idx[n]], nodes.t()], dim=1)[:, -3:])
    assert torch.equal(state[[0, 2, 4]], state0[[0, 2, 4]])


def test_vllm_layout_reads_the_resume_column_and_rolls_the_state(kernel):
    torch.manual_seed(2)
    width, dim, state_len = 6, 24, 3 + 6
    trees = [[-1, 0, 0, 1, 2, 4], [-1, 0, 1], [-1]]
    qsl = torch.tensor([0, 6, 9, 10], dtype=torch.int32, device=DEV)
    x = torch.randn(10, dim, device=DEV, dtype=BF16)
    weight, bias = torch.randn(dim, 4, device=DEV, dtype=BF16), torch.randn(dim, device=DEV, dtype=BF16)
    state0 = torch.randn(6, state_len, dim, device=DEV, dtype=BF16).transpose(1, 2)  # vLLM's other conv layout
    slots = torch.tensor([4, 2, 0], dtype=torch.int32, device=DEV)  # request 2 is padding (null block 0)
    accepted = torch.tensor([1, 3, 1], dtype=torch.int32, device=DEV)
    anc = torch.zeros(4, 3, width, dtype=torch.int32, device=DEV)  # the stage is wider than the batch
    for n, t in enumerate(trees):
        anc[n, :, : len(t)] = torch.tensor(list(zip(*_ancestor_codes(t))), dtype=torch.int32)
    state = state0.clone()
    out = torch.zeros_like(x)
    gdn_tree_conv.gdn_tree_conv1d_update(
        x.t()[None],
        state,
        weight,
        bias,
        "silu",
        slots,
        None,
        None,
        anc,
        pad_slot_id=0,
        out=out.t()[None],
        num_accepted_tokens=accepted,
        query_start_loc=qsl,
    )
    for n in (0, 1):
        lo, hi, s, a = int(qsl[n]), int(qsl[n + 1]), int(slots[n]), int(accepted[n])
        history = state0[s, :, a - 1 : a + 2]
        want, _ = _reference(x[lo:hi], history, weight, bias, trees[n])
        torch.testing.assert_close(out[lo:hi].float(), want, atol=2e-2, rtol=2e-2)
        rolled = torch.cat([history[:, 1:], x[lo:hi].t()], dim=1)
        assert torch.equal(state[s, :, : rolled.shape[1]], rolled)
        assert torch.equal(state[s, :, rolled.shape[1] :], state0[s, :, rolled.shape[1] :])
    assert torch.equal(out[9], torch.zeros_like(out[9])) and torch.equal(state[0], state0[0])
