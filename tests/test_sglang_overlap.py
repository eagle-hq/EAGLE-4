# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Scheduler handoff and graph-output ownership on CPU."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("sglang.srt.speculative.eagle_worker_v2")
from eagle_harness.sglang import worker as module  # noqa: E402
from eagle_harness.sglang.chain_rounds import ChainRounds  # noqa: E402


@pytest.mark.parametrize("overlap", [False, True])
def test_decode_handoff_uses_the_scheduler_stream_contract(monkeypatch, overlap):
    calls = []
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False, is_idle=lambda: False),
        is_extend_in_batch=False,
        seq_lens=torch.tensor([10]),
        spec_info=None,
    )
    result = SimpleNamespace(
        new_seq_lens=torch.tensor([12]),
        next_token_ids=torch.tensor([1, 2]),
        accept_lens=torch.tensor([2]),
        logits_output=SimpleNamespace(hidden_states=torch.zeros(2, 4)),
    )
    drafter = SimpleNamespace(
        draft=lambda b: calls.append("draft"),
        last_width=4,
        seeding=False,  # the round served the cost model's tree, not a live-cost seed width
        timer=None,
        prelaunch=Mock(side_effect=lambda b, **kwargs: (calls.append("prelaunch"), "fence")[1]),
    )
    worker = SimpleNamespace(
        _drafter=drafter,
        _draft_worker=SimpleNamespace(extend_after_verify=lambda *args: calls.append("extend")),
        _draft_context=nullcontext,
        _verify=lambda *args: (result, True),
        _verified=SimpleNamespace(record=lambda: calls.append("verified")),
        _chain_rounds=ChainRounds(None, None, lambda batch: 10),
        _native_decode=SimpleNamespace(select=lambda batch: False),
    )
    monkeypatch.setattr(module, "spec_stage_span", lambda name: nullcontext())
    monkeypatch.setattr(module, "spin_wait", lambda event: calls.append("wait"))
    monkeypatch.setattr(module.torch.cuda, "current_stream", lambda: "forward")
    publish = (lambda lens: calls.append("publish")) if overlap else None
    out = module.EagleHarnessWorker.forward_batch_generation(worker, batch, publish)
    # The round's result, wrapped with the prelaunch fence the scheduler's copies wait on.
    assert isinstance(out, module.PrelaunchResult) and out.prelaunch_done == "fence"
    assert out.next_token_ids is result.next_token_ids and out.accept_lens is result.accept_lens
    assert drafter.prelaunch.call_args.kwargs["stream"] == ("forward" if overlap else None)
    assert calls == (
        ["draft", "verified", "publish", "prelaunch"] if overlap else ["draft", "verified", "prelaunch", "wait"]
    )


@pytest.mark.parametrize("overlap", [False, True])
def test_overlap_readback_owns_fused_tail_outputs(monkeypatch, overlap):
    tokens, lengths, hidden = torch.tensor([11, 12]), torch.tensor([2]), torch.ones(2, 4)
    result = SimpleNamespace(
        next_token_ids=tokens,
        accept_lens=lengths,
        logits_output=SimpleNamespace(hidden_states=hidden),
    )
    batch = SimpleNamespace(spec_info=None, enable_overlap=overlap, return_hidden_states=True)
    worker = SimpleNamespace(
        topk=4,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
        _draft_worker=SimpleNamespace(
            steps=SimpleNamespace(seq_lens=torch.tensor([10]), req_pool_indices=torch.tensor([0])),
            extend=None,
        ),
        req_to_token_pool=None,
        _tail_graphs=None,
        target_worker=None,
        plan_stream=None,
        plan_stream_ctx=None,
        device="cpu",
    )
    monkeypatch.setattr(module, "fused_accept_supported", lambda *args: True)
    monkeypatch.setattr(module, "fused_accept_step", lambda **kwargs: Mock())
    monkeypatch.setattr(module, "run_eagle_verify", lambda *args, **kwargs: result)
    output, staged = module.EagleHarnessWorker._verify(worker, batch, 4, None)
    assert staged
    tokens.zero_()
    lengths.zero_()
    hidden.zero_()
    assert output.next_token_ids.tolist() == ([11, 12] if overlap else [0, 0])
    assert output.accept_lens.tolist() == ([2] if overlap else [0])
    assert output.logits_output.hidden_states.sum().item() == (8 if overlap else 0)
