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
    monkeypatch.setattr(plain_module, "plain_decode", lambda *args, **kwargs: calls.append("decode") or result)
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


@pytest.mark.parametrize("tracking", [False, True])
def test_plain_decode_saves_mamba_states_on_the_track_grid_like_a_no_spec_decode(monkeypatch, tracking):
    pytest.importorskip("sglang")
    import torch
    from sglang.srt import runtime_context
    from sglang.srt.speculative import spec_utils

    from eagle_harness.sglang.plain_decode import plain_decode

    monkeypatch.setattr(runtime_context, "mamba_track_grid", lambda page_size: 8, raising=False)
    planned = []

    def prepare(batch):  # the engine's verify-time track plan: slots only, no mask
        planned.append(batch.seq_lens.tolist())
        batch.mamba_track_indices = torch.tensor([40, 41])
        batch.mamba_track_mask = None

    monkeypatch.setattr(spec_utils, "prepare_mamba_track_for_verify", prepare, raising=False)
    draft = SimpleNamespace(bonus_tokens=torch.tensor([17, 19], dtype=torch.int32))
    batch = SimpleNamespace(
        spec_info=draft,
        spec_algorithm=object(),
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([5, 7]),
        seq_lens_cpu=torch.tensor([5, 7]),
        return_logprob=False,
        tree_cache=SimpleNamespace(page_size=1),
        mamba_track_indices=None,
        mamba_track_mask=None,
    )
    seen = {}

    def forward(actual, **kwargs):
        seen["indices"], seen["mask"] = actual.mamba_track_indices, actual.mamba_track_mask
        return SimpleNamespace(next_token_ids=torch.tensor([23, 29]))

    target = SimpleNamespace(forward_batch_generation=forward)
    plain_decode(batch, target, torch.arange(20).reshape(2, 10), track_mamba=tracking)
    if tracking:
        # The new lengths are 6 and 8: only the second row lands on the grid of 8, after its token is in the state.
        assert planned == [[6, 8]]
        assert seen["indices"].tolist() == [40, 41]
        assert seen["mask"].tolist() == [False, True]
    else:
        assert planned == [] and seen == {"indices": None, "mask": None}


@pytest.mark.parametrize("engine_tracks", [False, True])
def test_native_decode_needs_tracking_graphs_only_with_the_radix_cache(monkeypatch, engine_tracks):
    pytest.importorskip("sglang")
    from sglang.srt import runtime_context

    from eagle_harness.sglang import plain_decode as plain_module
    from eagle_harness.sglang import verify_graphs

    monkeypatch.setattr(verify_graphs, "native_decode_graphs", lambda: True)
    monkeypatch.setattr(plain_module, "native_decode_tracks_mamba", lambda: engine_tracks)
    monkeypatch.setattr(runtime_context, "get_parallel", lambda: SimpleNamespace(dp_size=1))
    for tracking in (False, True):
        monkeypatch.setattr(
            runtime_context,
            "get_exec",
            lambda tracking=tracking: SimpleNamespace(mamba=SimpleNamespace(enable_mamba_extra_buffer=tracking)),
        )
        available = engine_tracks or not tracking
        assert plain_module.native_decode_available() is available
        assert plain_module.native_decode_blocked() is not available


@pytest.mark.parametrize("native_route", [False, True])
@pytest.mark.parametrize("intensity,default", [("high_intensity", 32), ("chain_intensity", 48)])
def test_without_a_running_cap_admission_follows_memory_where_large_batches_decode_natively(
    native_route, intensity, default
):
    pytest.importorskip("sglang")
    from eagle_harness.core.config import harness_config_from_dict
    from eagle_harness.core.limits import max_batch_size
    from eagle_harness.sglang.algorithm import default_batch_limits

    config = harness_config_from_dict({"intensity": intensity})
    running, verify = default_batch_limits(config, native_route)
    if native_route:
        # The verify states cover the old default only; the engine sizes admission as without speculation.
        assert (running, verify) == (max_batch_size(config.intensity), default)
    else:
        assert (running, verify) == (default, None)
    explicit = harness_config_from_dict({"intensity": intensity, "max_verify_batch_size": 8})
    assert default_batch_limits(explicit, native_route) == (
        (max_batch_size(config.intensity), 8) if native_route else (8, 8)
    )


def test_batches_above_the_verify_limit_decode_natively_without_asking_the_policy():
    pytest.importorskip("sglang")
    from eagle_harness.sglang.plain_decode import NativeDecode

    asked = []
    rounds = SimpleNamespace(plain_width=lambda bs: asked.append(bs) or 4)
    native = NativeDecode(None, None, rounds, enabled=True, verify_limit=2)

    class Request:
        pass

    def batch(n):
        return SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False), reqs=[Request() for _ in range(n)])

    assert not native.select(batch(2)) and asked == [2]
    assert native.select(batch(3)) and asked == [2]


def test_batches_above_the_verify_limit_capture_only_the_native_decode_graph():
    pytest.importorskip("sglang")
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner.decode_graph_variants import DecodeGraphVariant

    from eagle_harness.core.limits import ComputeIntensity
    from eagle_harness.sglang.verify_graphs import VerifyWidthGraphs

    if not hasattr(DecodeGraphVariant, "forward_mode"):
        pytest.skip("host has no per-variant forward mode")
    graphs = VerifyWidthGraphs(ComputeIntensity.CHAIN, 8, None, plain_decode=True, max_verify_bs=8)
    assert len(graphs.capture_variants(8)) > 1
    (native,) = graphs.capture_variants(16)
    assert native.forward_mode is ForwardMode.DECODE


@pytest.mark.parametrize("enabled", [True, False])
def test_the_native_route_builds_its_decode_position_kernel_before_serving(monkeypatch, enabled):
    """The engine computes native decode's positions with a JIT module (nvcc, about 3 s); a server with the native route
    builds it at startup, not at its first native round mid-serving."""
    pytest.importorskip("sglang")
    import torch

    from sglang.kernels.ops.attention import clamp_position as engine_positions

    from eagle_harness.sglang import plain_decode as plain_module

    built = []
    monkeypatch.setattr(engine_positions, "clamp_position", lambda seq_lens: built.append(seq_lens.dtype))
    monkeypatch.setattr(plain_module.torch, "ones", lambda *shape, dtype, device: torch.zeros(*shape, dtype=dtype))
    on_gpu = SimpleNamespace(device=torch.device("cuda"))
    plain_module.NativeDecode(None, on_gpu, None, enabled=enabled)
    assert built == ([torch.int64] if enabled else [])
    plain_module.NativeDecode(None, SimpleNamespace(device=torch.device("cpu")), None, enabled=enabled)
    plain_module.NativeDecode(None, None, None, enabled=enabled)
    assert built == ([torch.int64] if enabled else [])


def test_prefills_joining_a_batch_above_the_verify_limit_skip_the_draft_for_life():
    pytest.importorskip("sglang")
    from eagle_harness.sglang.plain_decode import NativeDecode

    native = NativeDecode(None, None, SimpleNamespace(plain_width=lambda bs: 4), enabled=True, verify_limit=2)

    class Request:
        done = False

        def finished(self):
            return self.done

    def batch(reqs):
        return SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False), reqs=reqs)

    a, b, c, d, e, f = Request(), Request(), Request(), Request(), Request(), Request()
    assert not native.skips_draft(batch([a])) and not native.skips_draft(batch([b]))  # running 1, 2
    assert native.skips_draft(batch([c]))  # running 3 > 2: c is off before its first decode
    assert native.select(batch([a, b, c]))
    assert native.select(batch([a]))  # a decoded natively above the limit: off for life
    assert not native.skips_draft(batch([d]))  # running 1 + d = 2: drafts again
    assert not native.skips_draft(batch([d]))  # d's next chunk is not counted twice
    assert native.skips_draft(batch([c]))  # a chunk of a request that is off skips at any load
    assert not native.select(batch([d]))
    assert native.select(batch([a, b, c]))  # a burst's last decode step: every row finishes in it
    a.done = b.done = c.done = True
    assert not native.skips_draft(batch([e]))  # the batch e joins is empty: e drafts
    assert not native.skips_draft(batch([f]))  # e + f = 2


def test_stale_draft_kv_counts_reused_slots_the_draft_window_reads():
    import torch

    from eagle_harness.sglang.draft_kv import StaleDraftKV

    req_to_token = torch.arange(1, 17).view(2, 8)  # row 0: slots 1-8, row 1: slots 9-16
    kv = StaleDraftKV(20, "cpu")
    kv.target_only(torch.tensor([3, 4, 12]))
    kv.drafted(torch.tensor([4]))  # a later drafted write of slot 4
    kv.reused(req_to_token, rows=[0, 1], prefix_lens=[4, 0], seq_lens=[6, 3])  # row 0 reads slots 1-4
    assert kv._reused.tolist() == [1, 1]
    windowed = StaleDraftKV(20, "cpu", window=3)
    windowed.target_only(torch.tensor([3]))
    windowed.reused(req_to_token, rows=[0], prefix_lens=[4], seq_lens=[6])  # first drafted key: position 3 (slot 4)
    assert windowed._reused.tolist() == [0, 0]


def test_stale_draft_kv_never_waits_for_the_gpu():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from eagle_harness.sglang.draft_kv import StaleDraftKV

    kv = StaleDraftKV(1 << 16, "cuda")
    req_to_token = torch.arange(1, 1 + 4 * 512, device="cuda").view(4, 512)
    slots = torch.arange(100, 213, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        kv.target_only(slots)
        kv.drafted(slots[:8])
        kv.reused(req_to_token, rows=[0, 1], prefix_lens=[300, 0], seq_lens=[400, 10])
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert kv._reused.tolist() == [1, 105]


def test_a_kv_only_prefill_shifts_the_ids_writes_the_draft_kv_and_relays_a_placeholder(monkeypatch):
    pytest.importorskip("sglang.srt.speculative.eagle_info")
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from eagle_harness.sglang.draft_kv import kv_only_prefill

    calls = []
    out = SimpleNamespace(next_token_ids=torch.tensor([7, 8]), kv_loc_plan="plan",
                          logits_output=SimpleNamespace(hidden_states="target hidden"))

    def forward(batch, pp_proxy_tensors=None, capture_hidden_mode=None, return_kv_loc_plan=False):
        calls.append(("target", capture_hidden_mode.name, return_kv_loc_plan))
        return out

    def init_new(batch, runner, **kw):
        calls.append(("draft batch", batch.spec_info.hidden_states, kw["kv_loc_plan"]))
        return SimpleNamespace(input_ids=batch.input_ids.clone(), extend_seq_lens=torch.tensor([3, 2]))

    monkeypatch.setattr(ForwardBatch, "init_new", init_new)
    runner = SimpleNamespace(token_to_kv_pool="draft pool", model=SimpleNamespace(
        write_prefill_kv=lambda fb, pool: calls.append(("kv", fb.input_ids.tolist(), pool))))
    batch = SimpleNamespace(reqs=[object(), object()], seq_lens=torch.tensor([5, 9]), input_ids=torch.tensor([1, 2, 3, 4, 5]),
                            chunked_req_next_prompt_token=None, chunked_req=None)
    result = kv_only_prefill(
        batch, SimpleNamespace(forward_batch_generation=forward), runner, topk=4, hidden=(16, torch.bfloat16),
        on_publish=lambda lengths: calls.append(("publish", lengths.tolist())),
    )
    assert calls == [("target", "FULL", True), ("publish", [5, 9]), ("draft batch", "target hidden", "plan"),
                     ("kv", [2, 3, 7, 5, 8], "draft pool")]  # each position's next token; the last: sampled tokens
    assert result is out and out.kv_loc_plan is None
    draft = result.next_draft_input
    assert draft.bonus_tokens is out.next_token_ids
    assert draft.topk_p.shape == (2, 4) and draft.topk_index.shape == (2, 4) and draft.hidden_states.shape == (2, 16)


def test_the_kv_only_pass_writes_the_keys_and_values_the_attention_would():
    pytest.importorskip("sglang")
    import torch

    memory_pool = pytest.importorskip("sglang.srt.mem_cache.memory_pool")
    if not hasattr(memory_pool, "KVWriteLoc"):
        pytest.skip("engine without KV write plans")
    from eagle_harness.sglang.models.llama_eagle4 import LlamaForCausalLMEagle4

    torch.manual_seed(0)
    heads, dim, hidden = 2, 4, 8  # MHA: q, k and v each 8 wide; the layer reads concat(embeds, hidden) = 16
    weight = torch.randn(3 * heads * dim, 2 * hidden)
    written = {}
    radix = SimpleNamespace(is_cross_attention=False, k_scale=None, v_scale=None)
    attn = SimpleNamespace(
        qkv_proj=SimpleNamespace(weight=weight), q_size=heads * dim, kv_size=heads * dim, head_dim=dim,
        num_kv_heads=heads, attn=radix, rotary_emb=lambda positions, q, k: (q, k + positions[:, None]),
    )
    layer = SimpleNamespace(input_layernorm=lambda x: x, hidden_norm=lambda x: 2 * x, attn_norm=False, self_attn=attn)
    embeds = torch.randn(3, hidden)
    model = SimpleNamespace(hidden_size=hidden, layers=[layer], _embed=lambda ids: embeds[ids])
    pool = SimpleNamespace(set_kv_buffer=lambda layer, loc, k, v, ks, vs: written.update(layer=layer, loc=loc, k=k, v=v))
    target = torch.randn(3, hidden)
    fb = SimpleNamespace(spec_info=SimpleNamespace(hidden_states=target), input_ids=torch.tensor([2, 0, 1]),
                         positions=torch.tensor([10.0, 11.0, 12.0]),
                         out_cache_loc=torch.tensor([5, 6, 7]), out_cache_loc_is_physical=True)
    draft = SimpleNamespace(model=model)
    LlamaForCausalLMEagle4.write_prefill_kv(draft, fb, pool)
    _, k, v = (torch.cat([embeds[fb.input_ids], 2 * target], -1) @ weight.T).split(heads * dim, -1)
    assert written["layer"] is radix and torch.equal(written["loc"].loc, fb.out_cache_loc) and written["loc"].physical
    torch.testing.assert_close(written["k"], (k + fb.positions[:, None]).view(3, heads, dim))
    torch.testing.assert_close(written["v"], v.view(3, heads, dim))


@pytest.mark.parametrize("skip", [True, False])
def test_the_worker_prefill_routes_and_marks_the_written_slots(monkeypatch, skip):
    pytest.importorskip("sglang")
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative import eagle_utils

    from eagle_harness.sglang import worker as module
    from eagle_harness.sglang.draft_kv import StaleDraftKV

    calls = []
    worker = module.EagleHarnessWorker.__new__(module.EagleHarnessWorker)
    worker._stale_kv = StaleDraftKV(16, "cpu")
    worker._stale_kv.target_only(torch.tensor([2]))  # slot 2: decoded natively earlier
    worker._native_decode = SimpleNamespace(skips_draft=lambda batch: skip)
    worker.req_to_token_pool = SimpleNamespace(req_to_token=torch.arange(16).view(2, 8))
    worker.topk, worker._target_worker = 4, "target"
    worker._draft_worker = SimpleNamespace(draft_runner="draft")
    worker._kv_only_prefill = True
    monkeypatch.setattr(module.EagleHarnessWorker, "_draft_context", lambda self: __import__("contextlib").nullcontext())
    monkeypatch.setattr(eagle_utils, "get_draft_recurrent_hidden_state_spec", lambda runner: (8, torch.float32))
    monkeypatch.setattr(module, "kv_only_prefill", lambda batch, target, runner, **kw: calls.append(("kv only", target, runner, kw["hidden"])) or "t")
    monkeypatch.setattr(module.EAGLEWorkerV2, "forward_batch_generation", lambda self, *a: calls.append("drafted") or "d")
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, is_extend_in_batch=False, return_hidden_states=False,
                            out_cache_loc=torch.tensor([4, 5]), reqs=[SimpleNamespace(multimodal_inputs=None)], prefix_lens=[4], extend_lens=[2],
                            req_pool_indices_cpu=torch.tensor([0]))
    assert worker._prefill(batch, None, None, None) == ("t" if skip else "d")
    assert calls == ([("kv only", "target", "draft", (8, torch.float32))] if skip else ["drafted"])
    assert worker._stale_kv.stale[[4, 5]].tolist() == [False, False]  # both routes write the draft KV
    assert worker._stale_kv._reused.tolist() == ([0, 0] if skip else [1, 1])  # the drafted prefill read slot 2


def test_the_prefill_reads_batch_fields_the_engine_has():
    """The engine fields the prefill routing and the KV-only pass read (the fakes above cannot catch a renamed
    field): ScheduleBatch, and ForwardBatch where the engine has KV write plans."""
    pytest.importorskip("sglang")
    import dataclasses

    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from eagle_harness.sglang.draft_kv import kv_only_prefill_supported

    fields = {f.name for f in dataclasses.fields(ScheduleBatch)}
    assert {"forward_mode", "is_extend_in_batch", "return_hidden_states", "out_cache_loc", "reqs", "prefix_lens",
            "extend_lens", "seq_lens", "input_ids", "chunked_req"} <= fields
    assert "req_pool_indices_cpu" in fields or hasattr(Req, "req_pool_idx")
    if kv_only_prefill_supported():
        fb = {f.name for f in dataclasses.fields(ForwardBatch)}
        assert {"input_ids", "extend_seq_lens", "positions", "spec_info", "out_cache_loc",
                "out_cache_loc_is_physical"} <= fb
