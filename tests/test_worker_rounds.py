# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Worker round ordering on CPU, with only the engine/GPU boundary replaced."""

import contextlib
import importlib.util
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from eagle_harness.sglang.chain_rounds import BLOCK_ROUNDS, MIN_ROUNDS, ChainCalibration, ChainRounds


class Mode(Enum):
    DECODE = 0
    EXTEND = 1
    IDLE = 2

    def is_extend(self):
        return self is Mode.EXTEND

    def is_idle(self):
        return self is Mode.IDLE


class Request:
    pass


@pytest.fixture
def modules(monkeypatch):
    copies = []

    def stub(name, **attrs):
        if "." in name and name.rpartition(".")[0] not in sys.modules:
            stub(name.rpartition(".")[0])
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    def load(name):
        path = Path(__file__).parents[1] / (name.replace(".", "/") + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    @dataclass
    class Result:
        new_seq_lens: object
        next_token_ids: object
        accept_lens: object
        logits_output: object
        next_draft_input: object

        def copy_to_cpu(self, return_logprob, return_hidden_states=True):
            copies.append(("copy", return_logprob, return_hidden_states))

    class BaseWorker:
        @property
        def target_worker(self):
            return self._target_worker

    stub("sglang")
    stub("sglang.version", __version__="fixture")
    stub("sglang.srt.distributed.parallel_state", get_tp_group=lambda: SimpleNamespace(
        rank_in_group=0, broadcast_object=lambda cells, src: cells
    ))
    stub(
        "sglang.srt.layers.moe.utils",
        speculative_moe_a2a_backend_context=contextlib.nullcontext,
        speculative_moe_backend_context=contextlib.nullcontext,
    )
    stub("sglang.srt.model_executor.cuda_graph_config", Backend=object, Phase=object, check_cuda_graph_backend=object)
    stub("sglang.srt.model_executor.runner", get_batch_sizes_to_capture=lambda runner: ([1, 8], None))
    stub(
        "sglang.srt.runtime_context", **dict.fromkeys(("get_exec", "get_model", "get_parallel", "get_schedule", "get_spec"), object)
    )
    stub("sglang.srt.speculative.draft_utils", DraftBackendFactory=object)
    stub("sglang.srt.speculative.eagle_utils", default_tree_mask_mode=object)
    stub("sglang.srt.speculative.eagle_worker_common", run_eagle_verify=object, build_eagle_verify_input=object)
    stub("sglang.srt.speculative.eagle_worker_v2", EagleDraftWorker=object, EAGLEWorkerV2=BaseWorker)
    stub(
        "sglang.srt.speculative.spec_utils",
        draft_tp_context=contextlib.nullcontext,
        spec_stage_span=lambda name: contextlib.nullcontext(),
    )
    stub(
        "sglang.srt.model_executor.forward_batch_info", ForwardMode=Mode, CaptureHiddenMode=SimpleNamespace(FULL="full")
    )
    stub("sglang.srt.speculative.spec_info", SpeculativeAlgorithm=SimpleNamespace(NONE="none"))
    stub("sglang.srt.managers.utils", GenerationBatchResult=Result)
    stub("sglang.srt.managers.overlap_utils", FutureMap=SimpleNamespace(publish=lambda self, lengths: None))
    for name, attrs in {
        "draft_extend": ("DraftExtend",),
        "draft_steps": ("DraftSteps",),
        "l2_warm": ("MlpL2WarmAhead",),
        "row_attention": ("DraftRowAttention",),
        "verify_step": ("AcceptTailGraphs", "fused_accept_step", "fused_accept_supported", "warm_up_kv_moves"),
        "verify_graphs": ("verify_widths", "round_up_width"),
        "verify_input": ("HarnessVerifyInput", "build_attn_args"),
        "verify_tail": ("VerifyTail",),
    }.items():
        stub(f"eagle_harness.sglang.{name}", **dict.fromkeys(attrs, object))
    plain = load("eagle_harness.sglang.plain_decode")
    drafter = load("eagle_harness.sglang.drafter")
    worker = load("eagle_harness.sglang.worker")
    return SimpleNamespace(worker=worker, plain=plain, drafter=drafter, base=BaseWorker, Result=Result, copies=copies)


def make_batch(*requests, mode=Mode.DECODE, mixed=False):
    return SimpleNamespace(
        forward_mode=mode,
        is_extend_in_batch=mixed,
        return_logprob=False,
        reqs=list(requests),
        seq_lens=torch.tensor([5] * len(requests)),
        seq_lens_cpu=torch.tensor([5] * len(requests)),
        req_pool_indices=torch.arange(len(requests)),
        spec_info=SimpleNamespace(bonus_tokens=torch.tensor([17] * len(requests)), hidden_states=object()),
        spec_algorithm="eagle",
    )


def make_worker(modules, monkeypatch, *, calibration=None, schedule=None, native=True):
    calls = []
    worker = modules.worker.EagleHarnessWorker.__new__(modules.worker.EagleHarnessWorker)
    worker._verified = SimpleNamespace(
        record=lambda: calls.append("record"), query=lambda: calls.append("wait") or True
    )
    worker._draft_context = contextlib.nullcontext
    worker.req_to_token_pool = SimpleNamespace(req_to_token=torch.arange(80).reshape(8, 10))

    def draft(batch):
        calls.append("draft")
        if calibration is not None:
            drafter.last_width = worker._chain_rounds.draft_width(len(batch.reqs))
        return "verify-input"

    fence = object()
    scheduler_stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: scheduler_stream)

    def prelaunch(batch, *, extend, inputs, stream=None):
        assert stream is None or stream is scheduler_stream
        calls.append("prelaunch")
        assert inputs[0] is batch.seq_lens
        extend()
        if calibration is not None:
            worker._chain_rounds.draft_width(len(batch.reqs))
        return fence

    drafter = SimpleNamespace(
        cancel_prelaunch=lambda: calls.append("cancel"),
        average_context=lambda batch: calls.append("context") or 5,
        draft=draft,
        last_width=4,
        timer=None,
        seeding=False,
        prelaunch=prelaunch,
    )
    worker._drafter = drafter
    worker._draft_worker = SimpleNamespace(extend_after_verify=lambda *args: calls.append("extend"))
    worker._build_trivial_verify_input = lambda batch: calls.append("trivial") or "trivial"

    def verify(batch, width, grammar_barrier):
        calls.append(("verify", width))
        if grammar_barrier:
            grammar_barrier()
        return modules.Result(
            batch.seq_lens + 1,
            torch.tensor([23] * len(batch.reqs)),
            torch.tensor([width] * len(batch.reqs)),
            SimpleNamespace(hidden_states=torch.zeros(1)),
            object(),
        ), True

    worker._verify = verify

    def target(batch, **kwargs):
        calls.append("decode")
        assert kwargs == {"capture_hidden_mode": "full"}
        assert batch.spec_algorithm == "none" and batch.spec_info is None
        assert batch.input_ids.tolist() == [17] * len(batch.reqs)
        assert batch.out_cache_loc.tolist() == [i * 10 + 5 for i in range(len(batch.reqs))]
        return SimpleNamespace(next_token_ids=torch.tensor([23] * len(batch.reqs)))

    worker._target_worker = SimpleNamespace(forward_batch_generation=target)
    worker._chain_rounds = ChainRounds(schedule, calibration, drafter.average_context)
    worker._native_decode = modules.plain.NativeDecode(
        worker.target_worker, worker.req_to_token_pool.req_to_token, worker._chain_rounds, enabled=native
    )
    if calibration is not None:
        for name in ("interrupt", "round_start", "accepted"):
            method = getattr(calibration, name)

            def record(*args, _name=name, _method=method):
                calls.append((_name, *args[:2]) if _name == "round_start" else _name)
                return _method(*args)

            monkeypatch.setattr(calibration, name, record)
    return worker, calls, fence


def test_sampled_tree_follows_the_batch_and_never_verifies_roots_drafted_without_it(modules, monkeypatch):
    worker, _, _ = make_worker(modules, monkeypatch, native=False)
    worker.topk = 4
    invalidated = []
    sampled = SimpleNamespace(active=False, root_valid=torch.ones(8, dtype=torch.bool), invalidate=invalidated.append)
    worker._draft_worker.steps = SimpleNamespace(sampled=sampled)
    monkeypatch.setattr(modules.base, "forward_batch_generation", lambda self, *args: "prefill", raising=False)

    def batch(temperature, mode=Mode.DECODE):
        out = make_batch(Request(), Request(), mode=mode)
        for request in out.reqs:
            request.sampling_params = SimpleNamespace(temperature=temperature)
        out.has_grammar = False
        out.sampling_info = SimpleNamespace(
            is_all_greedy=False, acc_additive_penalties=None, acc_scaling_penalties=None, has_watermark_candidates=False
        )
        return out

    worker.forward_batch_generation(batch(1.0))
    assert sampled.active and not sampled.root_valid.any()  # the roots came from the greedy draft extend
    sampled.root_valid.fill_(True)  # this round's sampled extend recorded the next roots
    worker.forward_batch_generation(batch(1.0))
    assert sampled.active and sampled.root_valid.all()
    worker.forward_batch_generation(batch(0.7))
    assert not sampled.active and not sampled.root_valid.any()
    prefill = batch(1.0, mode=Mode.EXTEND)
    assert worker.forward_batch_generation(prefill) == "prefill"
    assert invalidated == [prefill.req_pool_indices]


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("calibrating", [False, True])
def test_native_decode_orders_barrier_calibration_publication_and_wait(modules, monkeypatch, tmp_path, calibrating, overlap):
    cal = (
        ChainCalibration(str(tmp_path), [1, 4], torch.device("cpu"), writer=False, native_decode=True)
        if calibrating
        else None
    )
    if cal is not None:
        cal.last_width = 0
    worker, calls, _ = make_worker(modules, monkeypatch, calibration=cal, schedule=[(1, 8, 0)])
    batch = make_batch(Request(), Request())
    root = batch.spec_info
    result = worker.forward_batch_generation(
        batch,
        on_publish=(lambda lengths: calls.append(("publish", lengths.tolist()))) if overlap else None,
        grammar_barrier=lambda: calls.append("barrier"),
    )
    expected = ["cancel", "barrier"]
    if calibrating:
        expected += ["context", ("round_start", 2, 0)]
        assert cal._planned == 0  # Dispatch reads the selected width without advancing calibration.
    assert calls == expected + ["decode", "record"] + ([("publish", [6, 6])] if overlap else ["wait"])
    assert result.accept_lens.tolist() == [1, 1]
    assert result.accept_index.tolist() == [[0], [1]]
    assert result.speculative_num_draft_tokens == 1
    assert result.next_draft_input.bonus_tokens is result.next_token_ids
    assert result.next_draft_input.hidden_states is root.hidden_states
    assert batch.spec_info is root and batch.spec_algorithm == "eagle"
    assert batch.seq_lens_cpu.tolist() == [6, 6] and batch.seq_lens_sum is None
    assert not hasattr(result, "prelaunch_done")


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("calibrating", [False, True])
def test_speculative_round_keeps_calibration_and_prelaunch_order(modules, monkeypatch, tmp_path, calibrating, overlap):
    cal = (
        ChainCalibration(str(tmp_path), [4], torch.device("cpu"), writer=False, native_decode=True)
        if calibrating
        else None
    )
    worker, calls, fence = make_worker(modules, monkeypatch, calibration=cal, schedule=[(1, 8, 4)], native=calibrating)
    result = worker.forward_batch_generation(
        make_batch(Request()),
        on_publish=(lambda lengths: calls.append("publish")) if overlap else None,
        grammar_barrier=lambda: calls.append("barrier"),
    )
    expected = ["draft"]
    if calibrating:
        expected += ["context", ("round_start", 1, 4)]
    expected += [("verify", 4), "barrier", "record"]
    if calibrating:
        expected += ["accepted"]
        assert cal._accept.tolist() == [0, 0, 0, 0, 1]
        assert cal._planned == 2
    assert calls == expected + (["publish"] if overlap else []) + ["prelaunch", "extend"] + ([] if overlap else ["wait"])
    assert isinstance(result, modules.drafter.PrelaunchResult)
    assert result.prelaunch_done is fence


@pytest.mark.parametrize("mode,mixed", [(Mode.EXTEND, False), (Mode.DECODE, True)])
def test_prefill_cancels_and_interrupts_before_delegating(modules, monkeypatch, tmp_path, mode, mixed):
    cal = ChainCalibration(str(tmp_path), [1, 4], torch.device("cpu"), writer=False, native_decode=True)
    cal.last_width, cal._planned = 0, 100
    worker, calls, _ = make_worker(modules, monkeypatch, calibration=cal, schedule=[(1, 8, 0)])
    batch = make_batch(Request(), mode=mode, mixed=mixed)
    args = (object(), object(), object())
    sentinel = object()

    def prefill(self, actual, *actual_args):
        assert actual is batch and actual_args == args
        calls.append("prefill")
        return sentinel

    monkeypatch.setattr(modules.base, "forward_batch_generation", prefill, raising=False)
    assert worker.forward_batch_generation(batch, *args) is sentinel
    assert calls == ["cancel", "interrupt", "prefill"]
    assert cal.last_width == 1 and cal._planned == 0


def test_idle_cancels_and_interrupts_without_calibrating_or_prelaunching(modules, monkeypatch, tmp_path):
    cal = ChainCalibration(str(tmp_path), [1, 4], torch.device("cpu"), writer=False, native_decode=True)
    cal.last_width = 0
    worker, calls, _ = make_worker(modules, monkeypatch, calibration=cal, schedule=[(1, 8, 0)])
    result = worker.forward_batch_generation(
        make_batch(mode=Mode.IDLE), on_publish=lambda lengths: calls.append("publish")
    )
    assert calls == ["cancel", "trivial", "interrupt", ("verify", 1), "record", "publish"]
    assert not hasattr(result, "prelaunch_done")


def test_sticky_off_survives_reset_shrinking_and_mixed_batches(modules, monkeypatch, tmp_path):
    cal = ChainCalibration(str(tmp_path), [1, 4], torch.device("cpu"), writer=False, native_decode=True)
    cal.last_width = 0
    worker, calls, _ = make_worker(modules, monkeypatch, calibration=cal)
    a, b, fresh = Request(), Request(), Request()
    for requests in ([a, b], [b], [b, fresh], [fresh]):
        worker.forward_batch_generation(make_batch(*requests))
        cal.interrupt()
    assert calls.count("decode") == 4
    assert "draft" not in calls
    calls.clear()
    worker.forward_batch_generation(make_batch(Request()))
    assert "draft" in calls and "decode" not in calls


def test_native_decode_error_restores_spec_state_without_publishing(modules, monkeypatch):
    worker, calls, _ = make_worker(modules, monkeypatch, schedule=[(1, 8, 0)])
    batch = make_batch(Request())
    root = batch.spec_info

    def fail(*args, **kwargs):
        raise RuntimeError("target failed")

    worker._target_worker.forward_batch_generation = fail
    with pytest.raises(RuntimeError, match="target failed"):
        worker.forward_batch_generation(batch, on_publish=lambda lengths: calls.append("publish"))
    assert batch.spec_info is root and batch.spec_algorithm == "eagle"
    assert calls == ["cancel"]


def test_prelaunch_off_choice_takes_effect_without_advancing_calibration(modules, monkeypatch, tmp_path):
    cal = ChainCalibration(str(tmp_path), [1], torch.device("cpu"), writer=False, native_decode=True)
    cal._planned = BLOCK_ROUNDS + MIN_ROUNDS - 1
    worker, calls, _ = make_worker(modules, monkeypatch, calibration=cal)
    request = Request()
    worker.forward_batch_generation(make_batch(request))
    assert ("verify", 1) in calls and cal.last_width == 0
    planned = cal._planned
    calls.clear()
    worker.forward_batch_generation(make_batch(request))
    assert calls == ["cancel", "context", ("round_start", 1, 0), "decode", "record", "wait"]
    assert cal._planned == planned


def test_result_copies_precede_scheduler_prelaunch_fence(modules, monkeypatch):
    worker, _, fence = make_worker(modules, monkeypatch, native=False)
    result = worker.forward_batch_generation(make_batch(Request()))
    stream = SimpleNamespace(wait_event=lambda event: modules.copies.append(("fence", event)))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    result.copy_to_cpu(True, False)
    assert modules.copies == [("copy", True, False), ("fence", fence)]


@pytest.mark.parametrize("kind", ["tree", "schedule", "calibrate", "served", "served-tree"])
def test_attention_initialization_wires_the_same_width_plan_into_both_routes(modules, monkeypatch, tmp_path, kind):
    class Draft:
        average_context = staticmethod(lambda batch: 5)

        def __init__(self, *args, chain_width, **kwargs):
            self.chain_width = chain_width

    module = modules.worker
    monkeypatch.setattr(modules.base, "init_attention_backends", lambda self: None, raising=False)
    monkeypatch.setattr(module, "DynamicDrafter", Draft)
    monkeypatch.setattr(module, "verify_widths", lambda *args: [4, 1])
    worker = module.EagleHarnessWorker.__new__(module.EagleHarnessWorker)
    monkeypatch.setattr(module, "get_schedule", lambda: SimpleNamespace(max_running_requests=8))
    monkeypatch.setattr(module, "cost_key", lambda runner: SimpleNamespace(digest="setup", to_dict=lambda: {}))
    monkeypatch.setattr(module, "packaged_choices", lambda kind, key, runner: None)  # an unmeasured setup
    worker.config = SimpleNamespace(
        mode=module.Mode.CALIBRATE if kind == "calibrate" else module.Mode.SERVE,
        intensity=module.ComputeIntensity.CHAIN,
        data_path=None if kind == "served" else str(tmp_path),
        cost_cache_dir=str(tmp_path),
    )
    worker._chain_schedule = [(1, 1, 4), (2, 8, 0)] if kind == "schedule" else None
    worker._served_chain = kind == "served"
    worker._served_tree = kind == "served-tree"
    worker._native_decode_enabled = kind != "tree"
    worker._target_worker = SimpleNamespace(model_runner=object())
    worker.fixed = object()
    worker.live = None
    worker.req_to_token_pool = SimpleNamespace(req_to_token=torch.zeros(8, 10))
    worker.speculative_num_draft_tokens = 4
    worker.device = "cpu"
    worker.cost_model = worker.speculative_algorithm = object()
    worker._draft_worker = SimpleNamespace(steps=object(), tree_mask_mode=None)
    worker.init_attention_backends()
    one, two = make_batch(Request()), make_batch(Request(), Request())
    assert not worker._native_decode.select(one)
    width = worker._drafter.chain_width
    if kind == "tree":
        assert width is None and not worker._native_decode.select(two)
    elif kind == "schedule":
        assert width(1) == 4 and width(2) == 1
        assert worker._native_decode.select(two)
    elif kind == "served":
        # Every batch size starts its search at the root-only width, and native decode is never explored.
        assert width(1) == width(2) == 1
        assert not worker._native_decode.select(two)
    elif kind == "served-tree":
        # The cost model sizes every round; a batch size speculates before native decode is measured against it.
        assert width is None and not worker._native_decode.select(two)
        assert isinstance(worker._chain_rounds._policy, module.ServedTreePolicy)
    else:
        block = BLOCK_ROUNDS + MIN_ROUNDS
        widths = [width(1) for _ in range(2 * block + 1)]
        assert widths == [1] * block + [4] * block + [1]
        assert worker._native_decode.select(one)


def test_packaged_choices_look_up_this_servers_setup(modules, monkeypatch):
    module = modules.worker
    monkeypatch.setattr(module, "get_model", lambda: SimpleNamespace(model_path="/target", quantization=None))
    monkeypatch.setattr(module, "get_spec", lambda: SimpleNamespace(speculative_draft_model_path="/draft"))
    seen = {}
    monkeypatch.setattr(module, "packaged_served_choices", lambda kind, **setup: seen.update(setup, kind=kind) or {16: 4})
    key = SimpleNamespace(gpu="NVIDIA B300 SXM6 AC", gpu_count=4)
    assert module.packaged_choices("tree", key, SimpleNamespace(dtype=torch.bfloat16)) == {16: 4}
    assert seen == dict(kind="tree", device_name="NVIDIA B300 SXM6 AC", model_path="/target", draft_model_path="/draft", tp_size=4,
                        dtype="torch.bfloat16", quantization=None)


@pytest.mark.parametrize("dp_size,tracking", [(2, False), (1, True), (2, True)])
def test_native_decode_still_rejects_dp_and_mamba_tracking(modules, monkeypatch, dp_size, tracking):
    module = modules.worker
    spec = SimpleNamespace(
        speculative_algorithm_config="{}",
        speculative_eagle_topk=1,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
    )
    monkeypatch.setattr(module, "get_spec", lambda: spec)
    monkeypatch.setattr(module, "parse_harness_config", lambda value: SimpleNamespace(mode=module.Mode.CALIBRATE))
    monkeypatch.setattr(module, "build_cost_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(module.EagleHarnessWorker, "_fixed_chain_widths", lambda *args: None)
    monkeypatch.setattr(
        module, "get_exec", lambda: SimpleNamespace(mamba=SimpleNamespace(enable_mamba_extra_buffer=tracking))
    )
    monkeypatch.setattr(module, "get_parallel", lambda: SimpleNamespace(dp_size=dp_size))
    target = SimpleNamespace(model_runner=SimpleNamespace(model_config=object()))
    with pytest.raises(ValueError, match="requires DP1 and disabled Mamba prefix-cache tracking"):
        module.EagleHarnessWorker(None, 0, 0, target)
