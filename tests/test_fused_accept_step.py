# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The fused accept step plugged into sglang's ``run_eagle_verify(accept_step=...)``."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("sglang")
from sglang.srt.speculative.eagle_worker_common import VerifyAcceptInput  # noqa: E402

from eagle_harness.sglang import verify_step  # noqa: E402


def _step_result(*, static: bool):
    accepted = SimpleNamespace(
        predict=torch.tensor([5, 6]),
        accept_lens=torch.tensor([2]),
        accept_index=torch.tensor([[0, 3]]),
        new_seq_lens=torch.tensor([9]),
        bonus_tokens=torch.tensor([6]),
    )
    compacted = torch.ones(16, 3)
    graphs = SimpleNamespace(run=lambda key, body: body())
    staged = (torch.tensor([7]), torch.tensor([0]))
    step = verify_step.fused_accept_step(
        req_to_token_pool=SimpleNamespace(req_to_token=None),
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
        num_draft_tokens=16,
        tail_graphs=graphs if static else None,
        staged=staged if static else None,
    )
    spec = SimpleNamespace(prepared_out_cache_loc=torch.zeros(16), max_tree_depth=4, draft_token=torch.zeros(16))
    inp = VerifyAcceptInput(
        verify_input=spec,
        next_token_ids=torch.zeros(16),
        next_token_logits=None,
        hidden_states=torch.zeros(16, 3),
        seq_lens=torch.tensor([7]),
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=torch.arange(16),
        can_run_cuda_graph=True,
    )
    with mock.patch.object(verify_step, "_accept", return_value=(accepted, compacted)) as accept:
        result = step(inp)
    return result, accepted, compacted, inp, staged, accept


@pytest.mark.parametrize("static", [False, True])
def test_step_returns_the_accepted_path_and_the_compacted_rows(static):
    result, accepted, compacted, inp, staged, accept = _step_result(static=static)
    accept.assert_called_once()
    args = accept.call_args.args
    # A replayed tail reads the draft's static lengths and slots; an eager one the batch's.
    if static:
        assert args[3] is staged[0] and args[4] is staged[1] and args[5] is inp.verify_input.prepared_out_cache_loc
    else:
        assert args[3] is inp.seq_lens and args[4] is inp.req_pool_indices and args[5] is inp.out_cache_loc
    assert result.next_token_ids is accepted.predict
    assert result.accept_lens is accepted.accept_lens
    assert result.accept_index is accepted.accept_index  # sglang commits the states of this path
    assert result.bonus_tokens is accepted.bonus_tokens
    assert result.hidden_states is compacted
    assert torch.equal(result.new_seq_lens, accepted.new_seq_lens)
    # A replayed tail's lengths are a static buffer: the scheduler gets a copy.
    assert (result.new_seq_lens is accepted.new_seq_lens) is (not static)
