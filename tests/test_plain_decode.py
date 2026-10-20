# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""No-spec pricing and request-lifetime native decode contracts."""

from types import SimpleNamespace

import pytest

from eagle_harness.core.chain_width import AcceptProfile, save_round_times, served_chain_width


def test_real_no_spec_is_a_separate_price_from_width_one():
    profile = AcceptProfile.from_counts(10, [8, 5, 2])
    prices = {0: 100, 1: 150, 2: 300, 4: 400}
    assert served_chain_width(prices, profile, 4, allow_no_spec=True) == 0
    # Adapters without a native decode route retain their old action set.
    assert served_chain_width(prices, profile, 4) == 1
    assert served_chain_width({**prices, 4: 200}, profile, 4, allow_no_spec=True) == 4


def test_no_spec_round_measurement_can_be_saved(tmp_path):
    save_round_times(str(tmp_path), [{"batch_size": 64, "width": 0, "round_us": 100, "count": 20}])


def test_calibration_times_off_last_and_never_resumes_stale_draft_kv(tmp_path):
    from collections import Counter

    import torch

    from eagle_harness.sglang.chain_rounds import MIN_ROUNDS, ChainCalibration

    calibration = ChainCalibration(str(tmp_path), [1, 2, 4], torch.device("cpu"), writer=False, native_decode=True)
    choices = [calibration.width(8) for _ in range(100)]
    assert set(choices) == {0, 1, 2, 4}
    assert all(Counter(choices)[width] >= MIN_ROUNDS + 2 for width in (1, 2, 4))
    assert set(choices[choices.index(0) :]) == {0}
    calibration.interrupt()
    assert calibration.width(8) == 1


def test_native_decode_has_a_separate_graph_from_width_one():
    pytest.importorskip("sglang")
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner.decode_graph_variants import DecodeGraphVariant

    from eagle_harness.core.limits import ComputeIntensity
    from eagle_harness.sglang.verify_graphs import VerifyWidthGraphs

    if not hasattr(DecodeGraphVariant, "forward_mode"):
        pytest.skip("host has no per-variant forward mode")
    graphs = VerifyWidthGraphs(ComputeIntensity.CHAIN, 8, None, plain_decode=True)
    variants = list(graphs.capture_variants(8))
    native = [v for v in variants if v.forward_mode is ForwardMode.DECODE]
    assert len(native) == 1 and native[0].num_tokens_per_req == 1
    assert any(v.num_tokens_per_req == 1 and v.forward_mode is None for v in variants)
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, spec_info=None)
    assert graphs.replay_variant(batch) == native[0]


def test_disabled_requests_never_resume_with_stale_draft_kv():
    pytest.importorskip("sglang")
    from eagle_harness.sglang.plain_decode import NoSpecRequests

    class Request:
        pass

    a, b, fresh = Request(), Request(), Request()
    policy = NoSpecRequests()
    assert not policy.select([a], 8)
    assert policy.select([a, b], 0)
    assert policy.select([b], 8)  # Shrinking the batch cannot revive stale KV.
    assert policy.select([b, fresh], 8)  # Mixed batches follow the same route.
    assert policy.select([fresh], 8)
    assert not policy.select([Request()], 8)


def test_worker_off_dispatch_skips_draft_verify_and_prelaunch(monkeypatch):
    pytest.importorskip("sglang")
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    from eagle_harness.sglang import plain_decode as plain_module
    from eagle_harness.sglang import worker as module
    from eagle_harness.sglang.chain_rounds import ChainRounds

    class Request:
        pass

    calls = []
    worker = module.EagleHarnessWorker.__new__(module.EagleHarnessWorker)
    worker._chain_rounds = ChainRounds([(1, 256, 0)], None, lambda batch: 0)
    worker._drafter = SimpleNamespace(cancel_prelaunch=lambda: calls.append("cancel"))
    worker._verified = SimpleNamespace(record=lambda: calls.append("record"), query=lambda: True)
    worker._target_worker = object()
    worker.req_to_token_pool = SimpleNamespace(req_to_token=object())
    worker._native_decode = plain_module.NativeDecode(
        worker.target_worker, worker.req_to_token_pool.req_to_token, worker._chain_rounds, enabled=True
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, is_extend_in_batch=False, reqs=[Request()])
    result = SimpleNamespace(new_seq_lens=[5], new_seq_lens_cpu=[5])
    monkeypatch.setattr(plain_module, "plain_decode", lambda *args: calls.append("decode") or result)
    worker._native_decode._requests.publishes_host_lengths = True
    on_publish = lambda lengths, **host: calls.append(("publish", lengths, host))  # noqa: E731
    assert worker.forward_batch_generation(batch, on_publish=on_publish) is result
    assert calls == ["cancel", "decode", "record", ("publish", [5], {"new_seq_lens_cpu": [5]})]


def test_off_publish_sends_host_lengths_only_to_a_host_that_reads_them():
    pytest.importorskip("sglang")
    from eagle_harness.sglang.plain_decode import NoSpecRequests

    calls = []
    policy = NoSpecRequests()
    publish = lambda lengths, **host: calls.append((lengths, host))  # noqa: E731
    policy.publishes_host_lengths = True
    policy.publish(publish, SimpleNamespace(new_seq_lens=[5], new_seq_lens_cpu=[5]))
    policy.publish(publish, SimpleNamespace(new_seq_lens=[6], new_seq_lens_cpu=None))
    policy.publishes_host_lengths = False
    policy.publish(publish, SimpleNamespace(new_seq_lens=[7], new_seq_lens_cpu=[7]))
    assert calls == [([5], {"new_seq_lens_cpu": [5]}), ([6], {}), ([7], {})]


def test_plain_decode_consumes_reserved_slot_once_and_preserves_roots():
    pytest.importorskip("sglang")
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    from eagle_harness.sglang.plain_decode import plain_decode

    draft = SimpleNamespace(bonus_tokens=torch.tensor([17, 19], dtype=torch.int32), hidden_states=object())
    algorithm = object()
    batch = SimpleNamespace(
        spec_info=draft,
        spec_algorithm=algorithm,
        req_pool_indices=torch.tensor([3, 1]),
        seq_lens=torch.tensor([5, 7]),
        seq_lens_cpu=torch.tensor([5, 7]),
        return_logprob=False,
    )
    slots = torch.arange(50).reshape(5, 10)
    calls = []

    def forward(actual, **kwargs):
        calls.append(actual)
        assert actual.forward_mode is ForwardMode.DECODE
        assert actual.spec_algorithm is SpeculativeAlgorithm.NONE
        assert actual.spec_info is None
        assert actual.input_ids.tolist() == [17, 19]
        assert actual.out_cache_loc.tolist() == [35, 17]
        assert actual.seq_lens.tolist() == actual.seq_lens_cpu.tolist() == [6, 8]
        return SimpleNamespace(next_token_ids=torch.tensor([23, 29]))

    result = plain_decode(batch, SimpleNamespace(forward_batch_generation=forward), slots)
    assert len(calls) == 1
    assert result.new_seq_lens.tolist() == [6, 8]
    assert result.new_seq_lens_cpu.tolist() == [6, 8]  # Host-known: the relay needs no device pull.
    assert result.accept_lens.tolist() == [1, 1]
    assert result.accept_index.tolist() == [[0], [1]]
    assert result.next_draft_input.bonus_tokens.tolist() == [23, 29]
    assert result.next_draft_input.bonus_tokens is result.next_token_ids
    assert result.next_draft_input.hidden_states is draft.hidden_states
    assert draft.bonus_tokens.tolist() == [17, 19]
    assert batch.spec_info is draft and batch.spec_algorithm is algorithm


def test_plain_decode_logprobs_are_consumed_as_one_accepted_token():
    pytest.importorskip("sglang")
    import torch
    from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor

    from eagle_harness.sglang.plain_decode import plain_decode

    batch = SimpleNamespace(
        spec_info=SimpleNamespace(bonus_tokens=torch.tensor([17, 19])),
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([2, 3]),
        seq_lens_cpu=None,
        return_logprob=True,
    )
    logits = SimpleNamespace(
        next_token_logprobs=torch.tensor([-0.25, -0.5]),
        next_token_top_logprobs_val=[[-0.25, -1.0], [-0.5, -2.0]],
        next_token_top_logprobs_idx=[[23, 7], [29, 11]],
        next_token_token_ids_logprobs_val=[[-3.0], [-4.0]],
        next_token_token_ids_logprobs_idx=[[13], [13]],
    )
    native = SimpleNamespace(next_token_ids=torch.tensor([23, 29]), logits_output=logits)
    result = plain_decode(
        batch,
        SimpleNamespace(forward_batch_generation=lambda *args, **kwargs: native),
        torch.arange(10).reshape(2, 5),
    )
    assert result.new_seq_lens_cpu is None  # No host lengths without a host mirror to advance.
    for i, token in enumerate(result.next_token_ids.tolist()):
        logprob = SimpleNamespace(
            top_logprobs_num=2,
            token_ids_logprob=[13],
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
            output_top_logprobs_val=[],
            output_top_logprobs_idx=[],
            output_token_ids_logprobs_val=[],
            output_token_ids_logprobs_idx=[],
        )
        SchedulerBatchResultProcessor._apply_decode_logprobs(
            None,
            req=SimpleNamespace(logprob=logprob),
            i=i,
            batch=batch,
            next_token_id=[token],
            next_token_logprobs=result.logits_output.next_token_logprobs.tolist(),
            logits_output=result.logits_output,
        )
        assert logprob.output_token_logprobs_val == [[-0.25], [-0.5]][i]
        assert logprob.output_token_logprobs_idx == [token]
        assert logprob.output_top_logprobs_val == [logits.next_token_top_logprobs_val[i]]
        assert logprob.output_top_logprobs_idx == [logits.next_token_top_logprobs_idx[i]]
        assert logprob.output_token_ids_logprobs_val == [logits.next_token_token_ids_logprobs_val[i]]
        assert logprob.output_token_ids_logprobs_idx == [[13]]


def test_off_lifetime_survives_retraction_and_ignores_reused_pool_slot():
    pytest.importorskip("sglang")
    from eagle_harness.sglang.plain_decode import NoSpecRequests

    class Request:
        def __init__(self, slot):
            self.req_pool_idx = slot

    retracted, other = Request(3), Request(7)
    policy = NoSpecRequests()
    assert policy.select([retracted, other], 0)
    assert policy.select([other], 8)
    fresh = Request(3)
    assert not policy.select([fresh], 8)
    retracted.req_pool_idx = 9
    assert policy.select([retracted], 8)
    assert policy.select([fresh, retracted], 8)
    assert policy.select([fresh], 8)


def test_finished_off_requests_are_not_kept_alive():
    import gc
    import weakref

    pytest.importorskip("sglang")
    from eagle_harness.sglang.plain_decode import NoSpecRequests

    class Request:
        pass

    request = Request()
    reference = weakref.ref(request)
    policy = NoSpecRequests()
    assert policy.select([request], 0)
    del request
    gc.collect()
    assert reference() is None
    assert not policy.select([Request()], 8)


def test_plain_decode_restores_spec_state_when_target_raises():
    pytest.importorskip("sglang")
    import torch

    from eagle_harness.sglang.plain_decode import plain_decode

    draft = SimpleNamespace(bonus_tokens=torch.tensor([17], dtype=torch.int32))
    algorithm = object()
    batch = SimpleNamespace(
        spec_info=draft,
        spec_algorithm=algorithm,
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([2]),
        seq_lens_cpu=None,
    )

    def forward(actual, **kwargs):
        assert actual.spec_info is None
        assert actual.seq_lens_cpu is None
        raise ValueError("target forward failed")

    with pytest.raises(ValueError, match="target forward failed"):
        plain_decode(batch, SimpleNamespace(forward_batch_generation=forward), torch.arange(4).reshape(1, 4))
    assert batch.spec_info is draft and batch.spec_algorithm is algorithm
    assert draft.bonus_tokens.tolist() == [17]
