# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""DraftExtend on CPU: staging, rows per verify width, the next round's lengths and tree begin in the same body."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("sglang")
from eagle_harness.sglang import draft_extend  # noqa: E402

K, H, IN, W, MAX_BS = 2, 4, 6, 4, 2


class _Model:
    def candidate_topk(self, normed, k):
        return normed[:, :k].float(), normed[:, :k].long()


def _extend():
    tree = SimpleNamespace(
        max_bs=MAX_BS,
        shape=SimpleNamespace(top_k=K),
        root_probs=torch.zeros(MAX_BS, K),
        root_tokens=torch.zeros(MAX_BS, K, dtype=torch.long),
        root_hidden=torch.zeros(MAX_BS, H),
    )
    began = []
    steps = SimpleNamespace(
        tree=tree,
        sampled=None,
        seq_lens=torch.zeros(MAX_BS, dtype=torch.long),
        req_pool_indices=torch.zeros(MAX_BS, dtype=torch.long),
        begin_body=began.append,
    )
    runner = SimpleNamespace(
        model=_Model(),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(4 * 64).view(4, 64)),
        dtype=torch.float32,
    )
    attention = mock.Mock()

    def extend_rows(seq_lens, req, accept_lens, slots, positions, last, *, bs, width):  # the kernel, in torch
        table = runner.req_to_token_pool.req_to_token
        pos = seq_lens[:, None] + torch.arange(width)
        positions.copy_(pos.reshape(-1))
        slots.copy_(table[req[:, None], pos].reshape(-1))
        last.copy_(torch.arange(bs) * width + accept_lens - 1)

    attention.build_extend_rows.side_effect = extend_rows
    extend = draft_extend.DraftExtend(
        runner, attention, steps, width=W, input_hidden_size=IN, widths=lambda bs: [W, 2, 1]
    )
    return extend, steps, began, attention


def test_stage_copies_the_results_and_skips_hidden_already_in_place():
    extend, *_ = _extend()
    hidden = extend.hidden_rows(2)
    hidden.copy_(torch.full((2, IN), 7.0))
    extend.stage(torch.tensor([5, 6], dtype=torch.int32), hidden, torch.tensor([2]))
    assert extend.accept_lens[0] == 2
    assert extend.tokens[:2].tolist() == [5, 6] and torch.equal(extend.input_hidden[:2], torch.full((2, IN), 7.0))
    extend.stage(torch.tensor([1, 2]), torch.ones(2, IN), torch.tensor([1]))
    assert torch.equal(extend.input_hidden[:2], torch.ones(2, IN))


def test_the_body_runs_the_verify_width_and_begins_the_next_round():
    extend, steps, began, attention = _extend()
    steps.seq_lens[0], steps.req_pool_indices[0] = 10, 3  # the round's batch, staged by the draft
    extend.stage(torch.tensor([5, 6]), torch.arange(2 * IN).view(2, IN).float(), torch.tensor([2]))
    calls = {}

    def fake_rows(model, attn, *, input_ids, positions, out_cache_loc, hidden_states):
        calls.update(input_ids=input_ids, positions=positions, slots=out_cache_loc, hidden=hidden_states)
        rows = input_ids.shape[0]
        normed = torch.arange(rows * H).view(rows, H).float()
        return normed, normed + 100

    with mock.patch.object(draft_extend, "forward_draft_rows", fake_rows):
        probs, tokens, hidden = extend.run(1, 2)
    assert calls["input_ids"].tolist() == [5, 6] and calls["positions"].tolist() == [10, 11]
    assert calls["slots"].tolist() == [3 * 64 + 10, 3 * 64 + 11] and calls["hidden"].shape == (2, IN)
    attention.build_extend_rows.assert_called_once()
    assert attention.build_extend_rows.call_args.kwargs == {"bs": 1, "width": 2}
    assert tokens[0].tolist() == [4, 5] and hidden[0].tolist() == [104.0, 105.0, 106.0, 107.0]  # row accept_len - 1
    assert steps.seq_lens[0] == 12 and steps.req_pool_indices[0] == 3  # committed: 10 + 2 accepted
    assert began == [1]


def test_the_fused_tail_compacts_into_the_extend_and_stages_it():
    from eagle_harness.sglang import verify_step

    extend, *_ = _extend()
    accepted = SimpleNamespace(
        accept_index=torch.tensor([[0, 2]]),
        accept_lens=torch.tensor([2], dtype=torch.int32),
        predict=torch.tensor([8, 9, 1, 1], dtype=torch.int32),
        committed_slots=None,
        accepted_slots=None,
    )
    compacted = {}

    def fake_compact(rows, accept_index, out=None):
        compacted["out"] = out
        out.copy_(rows.flip(0))
        return out

    spec = SimpleNamespace(
        draft_token=None, retrieve_index=None, retrieve_next_token=None, retrieve_next_sibling=None, max_tree_depth=2
    )
    rows = torch.arange(W * IN).view(W, IN).float()
    with (
        mock.patch.object(verify_step, "accept_tail", return_value=accepted),
        mock.patch.object(verify_step, "compact_rows", fake_compact),
    ):
        _, out = verify_step._accept(spec, torch.zeros(W, 3), rows, None, None, None, None, mock.Mock(), extend)
    assert out.data_ptr() == extend.input_hidden.data_ptr() and compacted["out"].shape == (W, IN)
    assert torch.equal(extend.input_hidden[:W], rows.flip(0))
    assert extend.accept_lens[0] == 2
    assert extend.tokens[:W].tolist() == [8, 9, 1, 1]
