# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Scheduler compaction must not race a prelaunched draft's published roots."""

import copy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from sglang.srt.speculative.eagle_info import EagleDraftInput

from eagle_harness.sglang.drafter import DynamicDrafter, PrelaunchResult


def _copy_result(done):
    result = PrelaunchResult(
        next_token_ids=torch.zeros(1, dtype=torch.int64, device="cuda"),
        copy_done=torch.cuda.Event(),
        prelaunch_done=done,
    )
    result.copy_to_cpu(return_logprob=False, return_hidden_states=False)


@pytest.fixture(params=[None, 1, 2, 4, 8], ids=["tree", "w1", "w2", "w4", "w8"])
def chain_width(request):
    return request.param


@pytest.fixture(params=["default", "nondefault"])
def caller_stream(request):
    stream = torch.cuda.default_stream() if request.param == "default" else torch.cuda.Stream()
    with torch.cuda.stream(stream):
        yield stream
    stream.synchronize()


def _prelaunch_drafter(chain_width):
    drafter = DynamicDrafter.__new__(DynamicDrafter)
    drafter._device = torch.device("cuda")
    drafter.live = None
    drafter._prelaunch_stream = None
    drafter._pending = None
    drafter._chain_width = (lambda bs: chain_width) if chain_width is not None else None
    drafter.policy = lambda bs: None
    return drafter


@pytest.mark.parametrize("consumer", ["filter", "merge_into", "merge_from"])
def test_prelaunch_roots_are_ready_for_scheduler_compaction(consumer, caller_stream, chain_width):
    pending = EagleDraftInput(
        topk_p=torch.zeros(2, 2, device="cuda"),
        topk_index=torch.zeros(2, 2, dtype=torch.int64, device="cuda"),
        hidden_states=torch.zeros(2, 3, device="cuda"),
        bonus_tokens=torch.zeros(2, dtype=torch.int64, device="cuda"),
    )
    other = EagleDraftInput(
        topk_p=torch.zeros(1, 2, device="cuda"),
        topk_index=torch.zeros(1, 2, dtype=torch.int64, device="cuda"),
        hidden_states=torch.zeros(1, 3, device="cuda"),
        bonus_tokens=torch.zeros(1, dtype=torch.int64, device="cuda"),
    )
    indices = torch.tensor([1], device="cuda")
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(kv=SimpleNamespace(req_pool_idx=i)) for i in range(2)],
        req_pool_indices=torch.arange(2, device="cuda"),
    )
    drafter = _prelaunch_drafter(chain_width)
    drafter._queue_until_decision = lambda bs, policy, context=0: 1
    drafter._queue_chain = lambda bs, width: max(0, width - 1)

    def compact():
        result = copy.copy(pending)
        if consumer == "filter":
            result.filter_batch(indices, [1])
        elif consumer == "merge_into":
            result.merge_batch(copy.copy(other))
        else:
            incoming, result = result, copy.copy(other)
            result.merge_batch(incoming)
        return result

    # Warm the indexing/concatenation and allocator before delaying the producer.
    for _ in range(3):
        compact()
    torch.cuda.synchronize()

    def extend():
        torch.cuda._sleep(100_000_000)
        pending.topk_p.fill_(1)
        pending.topk_index.fill_(2)
        pending.hidden_states.fill_(3)

    done = drafter.prelaunch(batch, extend=extend, inputs=())
    _copy_result(done)
    result = compact()  # The scheduler does this before calling draft() again.
    drafter.cancel_prelaunch()
    torch.cuda.synchronize()

    for name, value in (("topk_p", 1), ("topk_index", 2), ("hidden_states", 3)):
        actual = getattr(result, name)
        if consumer == "merge_into":
            actual = actual[:2]
        elif consumer == "merge_from":
            actual = actual[1:]
        torch.testing.assert_close(actual, torch.full_like(actual, value), rtol=0, atol=0)


def test_prelaunch_finishes_reading_slots_before_scheduler_reuses_them(caller_stream, chain_width):
    slots = torch.ones(2, dtype=torch.int64, device="cuda")
    observed = torch.zeros_like(slots)
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(kv=SimpleNamespace(req_pool_idx=i)) for i in range(2)],
        req_pool_indices=torch.arange(2, device="cuda"),
    )
    drafter = _prelaunch_drafter(chain_width)

    def queue_steps(bs):
        # A fence after extend alone is too early: draft steps still read slot mappings.
        torch.cuda._sleep(100_000_000)
        observed.copy_(slots)
        return 1

    drafter._queue_until_decision = lambda bs, policy, context=0: queue_steps(bs)
    drafter._queue_chain = lambda bs, width: queue_steps(bs)
    torch.cuda.synchronize()
    done = drafter.prelaunch(batch, extend=lambda: None, inputs=(slots,))
    _copy_result(done)
    slots.zero_()  # A finished request's slots are reused before the next worker entry.
    drafter.cancel_prelaunch()
    caller_stream.synchronize()
    torch.testing.assert_close(observed, torch.ones_like(observed), rtol=0, atol=0)


def test_scheduler_result_copy_does_not_wait_for_prelaunch(caller_stream, chain_width):
    from sglang.srt.managers.utils import GenerationBatchResult

    batch = SimpleNamespace(
        reqs=[SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0))],
        req_pool_indices=torch.zeros(1, dtype=torch.int64, device="cuda"),
    )
    result = GenerationBatchResult(
        next_token_ids=torch.tensor([7], device="cuda"),
        accept_lens=torch.ones(1, dtype=torch.int32, device="cuda"),
        copy_done=torch.cuda.Event(),
    )
    drafter = _prelaunch_drafter(chain_width)
    finished = torch.cuda.Event()

    def queue_steps(bs):
        torch.cuda._sleep(100_000_000)
        finished.record()
        return 1

    drafter._queue_until_decision = lambda bs, policy, context=0: queue_steps(bs)
    drafter._queue_chain = lambda bs, width: queue_steps(bs)
    torch.cuda.synchronize()
    done = drafter.prelaunch(batch, extend=lambda: None, inputs=())
    result = PrelaunchResult(**vars(result), prelaunch_done=done)
    result.copy_to_cpu(return_logprob=False, return_hidden_states=False)
    result.copy_done.synchronize()
    still_running = not finished.query()
    drafter.cancel_prelaunch()
    caller_stream.synchronize()
    assert still_running, "scheduler result copy serialized behind the next draft"
    assert result.next_token_ids.tolist() == [7]
