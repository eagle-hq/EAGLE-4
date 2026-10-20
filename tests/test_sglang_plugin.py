# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The sglang plugin on branch eagle-hooks: discovery, registration, hooks H2/H5/H7/H10/H11.

Needs sglang importable (the eagle-hooks tree on PYTHONPATH) and, for the
entry-point test, the package installed (``pip install -e .``).
"""

from __future__ import annotations

import ast
import json
import os
from importlib.metadata import PackageNotFoundError, distribution, entry_points
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang.srt.speculative.spec_registry")

from sglang.srt.arg_groups.overrides import resolution_result  # noqa: E402
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding  # noqa: E402
from sglang.srt.mem_cache.allocation_sizing import get_req_to_token_extra_context_len  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.model_executor.runner.decode_graph_variants import DecodeGraphRunnerInfo  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.srt.speculative.spec_registry import _REGISTRY  # noqa: E402

import eagle_harness  # noqa: E402
from eagle_harness.core.config import parse_harness_config  # noqa: E402
from eagle_harness.sglang import plugin  # noqa: E402
from eagle_harness.sglang.algorithm import ALGORITHM_NAME, EagleHarnessSpecAlgo  # noqa: E402
from eagle_harness.sglang.l2_warm import MlpL2WarmAhead  # noqa: E402
from eagle_harness.sglang.verify_graphs import VerifyWidthGraphs, round_up_width, verify_widths  # noqa: E402
from eagle_harness.sglang.worker import EagleHarnessWorker  # noqa: E402

CONFIG = json.dumps({"data_path": "/nonexistent/cost-data"})
STATIC_CONFIG = json.dumps({"mode": "static"})


@pytest.fixture
def registered():
    snapshot = _REGISTRY.copy()
    _REGISTRY.pop(ALGORITHM_NAME, None)
    plugin.register()
    yield SpeculativeAlgorithm.from_string(ALGORITHM_NAME)
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def _server_args(**overrides) -> ServerArgs:
    args = ServerArgs(model_path="dummy")
    fields = dict(
        speculative_algorithm=ALGORITHM_NAME,
        speculative_algorithm_config=CONFIG,
        speculative_num_steps=12,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=16,
    )
    fields.update(overrides)
    for name, value in fields.items():
        setattr(args, name, value)
    return args


# ------------------------------------------------------------------ discovery
def test_entry_point_is_installed():
    try:
        dist = distribution("eagle-harness")
    except PackageNotFoundError:
        pytest.skip("eagle-harness is not installed (pip install -e .)")
    eps = [ep for ep in entry_points(group="sglang.srt.plugins") if ep.name == "eagle_harness"]
    assert len(eps) == 1 and eps[0].value == "eagle_harness.sglang.plugin:register"
    assert dist.version == eagle_harness.__version__


def test_load_plugins_registers_the_algorithm():
    try:
        distribution("eagle-harness")
    except PackageNotFoundError:
        pytest.skip("eagle-harness is not installed (pip install -e .)")
    import sglang.srt.plugins as plugins

    snapshot = _REGISTRY.copy()
    _REGISTRY.pop(ALGORITHM_NAME, None)
    plugins._plugins_loaded = False
    try:
        plugins.load_plugins()
        algo = SpeculativeAlgorithm.from_string("eagle_harness")
        assert isinstance(algo, EagleHarnessSpecAlgo)
        assert algo.name == ALGORITHM_NAME and algo.supports_overlap is True
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)


def test_register_is_idempotent(registered):
    plugin.register()
    assert SpeculativeAlgorithm.from_string(ALGORITHM_NAME) is registered


def _worker_args(config: str = CONFIG, overlap_off: bool = True) -> SimpleNamespace:
    return SimpleNamespace(disable_overlap_schedule=overlap_off, speculative_algorithm_config=config)


@pytest.mark.parametrize("overlap_off", [False, True])
def test_worker_class_supports_both_schedulers(registered, overlap_off):
    assert registered.create_worker(_worker_args(overlap_off=overlap_off)) is EagleHarnessWorker
    assert issubclass(EagleHarnessWorker, BaseSpecWorker)


def test_static_mode_serves_with_the_harness_worker(registered):
    """A fixed shape runs the harness round (its graphs, kernels and prelaunch), not sglang's generic worker."""
    assert registered.create_worker(_worker_args(STATIC_CONFIG)) is EagleHarnessWorker


@pytest.mark.parametrize("batch_size", [64, 128, 256, 512])
def test_explicit_tree_batch_limit_and_graph_variants(registered, batch_size):
    args = _server_args(max_running_requests=batch_size)
    handle_speculative_decoding(args)
    assert args.max_running_requests == batch_size
    assert verify_widths(batch_size, parse_harness_config(CONFIG).intensity, 16) == [16, 8, 4, 1]


def test_algorithm_joins_the_eagle_family(registered):
    assert registered.is_eagle() and registered.uses_aux_hidden_states()
    # Native EAGLE3 initialization preserves the learned head and d2t token map.
    assert registered.is_eagle3()
    assert registered.need_topk() and registered.carries_draft_hidden_states()


def test_plugin_registers_the_draft_model(registered):
    from sglang.srt.models.registry import ModelRegistry

    from eagle_harness.sglang.models.llama_eagle4 import LlamaForCausalLMEagle4

    model_cls, arch = ModelRegistry.resolve_model_cls(["LlamaForCausalLMEagle4"])
    assert model_cls is LlamaForCausalLMEagle4 and arch == plugin.DRAFT_ARCHITECTURE
    plugin.register()  # a second registration in the same process is a no-op


# ------------------------------------------------------- H11 + server args
@pytest.mark.parametrize("overlap_off", [False, True])
def test_server_args_accept_eagle_harness_and_config(registered, overlap_off):
    args = _server_args(disable_overlap_schedule=overlap_off)
    handle_speculative_decoding(args)
    assert resolution_result(args, "speculative_algorithm") == ALGORITHM_NAME
    assert resolution_result(args, "disable_overlap_schedule") is overlap_off
    assert resolution_result(args, "scheduler_recv_interval") == 4
    assert resolution_result(args, "max_running_requests") == 32
    assert resolution_result(args, "enable_mixed_chunk") is False


@pytest.mark.parametrize(
    "overrides, error, message",
    [
        ({"speculative_algorithm_config": None}, ValueError, "needs --speculative-algorithm-config"),
        ({"speculative_algorithm_config": '{"data_path": "/x", "bogus": 1}'}, ValueError, "unknown keys"),
        ({"speculative_eagle_topk": 9}, NotImplementedError, "topk"),
        ({"speculative_num_draft_tokens": 65}, NotImplementedError, "verify_num"),
        ({"max_running_requests": 513}, ValueError, "batch limit 512"),
        ({"speculative_num_steps": None, "speculative_eagle_topk": None}, ValueError, "num-steps"),
        ({"speculative_token_map": "/x/map.pt"}, NotImplementedError, "token-map"),
    ],
)
def test_server_args_rejections(registered, overrides, error, message):
    with pytest.raises(error, match=message):
        handle_speculative_decoding(_server_args(**overrides))


@pytest.mark.parametrize(
    "vendor,backend,aiter_env,warns",
    [
        ("amd", "aiter", "1", False),
        ("amd", "aiter", None, True),
        ("amd", "triton", None, False),
        ("nvidia", "aiter", None, False),
    ],
)
def test_rocm_environment_is_logged_and_checked(registered, monkeypatch, caplog, vendor, backend, aiter_env, warns):
    from eagle_harness.kernels import arch

    monkeypatch.setattr(arch, "gpu", lambda device=None: arch.Gpu(0, "fake", vendor, "gfx950", 0))
    monkeypatch.setattr(arch, "require_supported", lambda device=None: None)
    for name in ("SGLANG_USE_AITER", "SGLANG_SPLITKV_VERIFY_FLYDSL", "SGLANG_SPLITKV_VERIFY_FLYDSL_COMBINE"):
        monkeypatch.delenv(name, raising=False)
    if aiter_env is not None:
        monkeypatch.setenv("SGLANG_USE_AITER", aiter_env)
    monkeypatch.setenv("SGLANG_SPLITKV_VERIFY_FLYDSL", "0")
    with caplog.at_level("INFO", logger="eagle_harness.sglang.algorithm"):
        handle_speculative_decoding(_server_args(attention_backend=backend))
    logged = caplog.text
    if vendor == "nvidia":
        assert "SGLANG_USE_AITER" not in logged
    else:
        assert "SGLANG_USE_AITER=" + (aiter_env or "0") in logged and "SGLANG_SPLITKV_VERIFY_FLYDSL=0" in logged
    assert ("without SGLANG_USE_AITER=1" in logged) is warns


def test_config_still_rejected_for_builtins(registered):
    with pytest.raises(ValueError, match="algorithm-config"):
        handle_speculative_decoding(_server_args(speculative_algorithm="EAGLE"))


# ------------------------------------------------------------ H2 admission
def test_request_admission(registered):
    def req(logprob):
        return SimpleNamespace(return_logprob=logprob)

    assert registered.supports_mm_inputs() is False
    assert registered.validate_request(req(False), has_mm_inputs=False, enable_overlap=False) is None
    assert "multimodal" in registered.validate_request(req(False), has_mm_inputs=True, enable_overlap=False)
    # Logprob requests are served: sglang's verify returns the accepted path's logprobs (test_sglang_logprobs.py).
    assert registered.validate_request(req(True), has_mm_inputs=False, enable_overlap=False) is None


# ------------------------------------------------- H7 verify-width graphs
def _runner_info(is_draft: bool) -> DecodeGraphRunnerInfo:
    return DecodeGraphRunnerInfo(is_draft, ForwardMode.TARGET_VERIFY, 16, torch.device("cpu"))


def test_verify_graph_variants_only_on_the_target_in_cost_model_modes(registered, monkeypatch):
    hooked = []
    monkeypatch.setattr(MlpL2WarmAhead, "install", classmethod(lambda cls, model, plan, device: hooked.append(model)))
    monkeypatch.setattr(registered, "target_verify_graphs", None)
    # Every mode gets the per-width verify graphs: static mode runs the harness round at a fixed shape too.
    registered.__dict__["config"] = parse_harness_config(STATIC_CONFIG)
    with get_context().override_server_args(speculative_num_draft_tokens=16):
        assert registered.create_decode_graph_variants(_runner_info(True)) is None
        assert registered.target_verify_graphs is None
        provider = registered.create_decode_graph_variants(_runner_info(False))
    assert isinstance(provider, VerifyWidthGraphs)
    # The algorithm keeps the provider for the drafter, and installs nothing on the model (the worker does).
    assert registered.target_verify_graphs is provider
    assert hooked == []


@pytest.mark.parametrize(
    "hf_config, expected_moe",
    [
        (SimpleNamespace(architectures=["MoEForCausalLM"]), False),
        (SimpleNamespace(text_config=SimpleNamespace(n_routed_experts=8)), True),
    ],
)
def test_worker_hooks_the_l2_warm_ahead_before_the_graphs(monkeypatch, hf_config, expected_moe):
    """The cost-model worker hooks the warm-ahead onto the target model when it is built: sglang builds every
    worker before it captures any CUDA graph, so the target's verify graphs record the prefetch."""
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

    from eagle_harness.sglang import worker as worker_mod

    hooked = []
    cost_inputs = []
    built = []

    def build_cost_model(*args, **kwargs):
        cost_inputs.append(kwargs["target_is_moe"])
        built.append(kwargs)
        return SimpleNamespace(tables={"bs1": None})

    monkeypatch.setattr(MlpL2WarmAhead, "install", classmethod(lambda cls, model, plan, device: hooked.append(model)))
    monkeypatch.setattr(worker_mod, "build_cost_model", build_cost_model)
    monkeypatch.setattr(worker_mod, "cost_key", lambda runner: "key")
    monkeypatch.setattr(EAGLEWorkerV2, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(torch.cuda, "Event", lambda *a, **k: None)  # CPU-only torch has no events
    target = SimpleNamespace(
        model_runner=SimpleNamespace(model="target", device="cpu", model_config=SimpleNamespace(hf_config=hf_config))
    )
    with get_context().override_server_args(
        speculative_algorithm_config=CONFIG,
        speculative_eagle_topk=4,
        speculative_num_steps=12,
        speculative_num_draft_tokens=16,
    ):
        EagleHarnessWorker(None, 0, 0, target)
    assert hooked == ["target"]
    assert cost_inputs == [expected_moe]
    assert built[0]["key"] == "key" and built[0]["tables"] is None


def test_live_chain_does_not_read_offline_round_files(registered):
    from eagle_harness.core.config import HarnessConfig
    from eagle_harness.core.limits import ComputeIntensity

    config = HarnessConfig(intensity=ComputeIntensity.CHAIN)
    worker = SimpleNamespace(config=config)
    spec = SimpleNamespace(speculative_eagle_topk=1, speculative_num_draft_tokens=8)
    registered.__dict__["config"] = config
    with get_context().override_server_args(
        speculative_num_draft_tokens=8, speculative_eagle_topk=1, max_running_requests=512
    ):
        assert EagleHarnessWorker._fixed_chain_widths(worker, spec) is None
        provider = registered.create_decode_graph_variants(_runner_info(False))
    assert isinstance(provider, VerifyWidthGraphs)


def test_verify_widths_follow_the_cost_buckets():
    from eagle_harness.core.limits import ComputeIntensity

    assert verify_widths(1, ComputeIntensity.HIGH, 16) == [16, 8, 1]
    assert verify_widths(1, ComputeIntensity.HIGH, 12) == [12, 8, 1]
    assert round_up_width(2, [16, 8, 1]) == 8 and round_up_width(1, [16, 8, 1]) == 1
    with pytest.raises(ValueError):
        round_up_width(17, [16, 8, 1])


def test_verify_graph_provider_pins_the_width():
    from eagle_harness.core.limits import ComputeIntensity

    from sglang.srt.speculative.eagle_info import EagleVerifyInput

    tail_mask = object()
    provider = VerifyWidthGraphs(ComputeIntensity.HIGH, 16, tail_mask)
    variants = provider.capture_variants(1)
    # The greedy graphs of the tree widths come last.
    assert [(v.label, v.num_tokens_per_req) for v in variants] == [
        ("w16", 16),
        ("w8", 8),
        ("w1", 1),
        ("a16", 16),
        ("a8", 8),
    ]
    assert [v.label for v in provider.capture_variants(2)] == ["w16", "w8", "w1"]
    backend = object()
    for variant, greedy in ((variants[1], False), (variants[4], True)):
        idle = EagleVerifyInput.create_idle_input(topk=4, spec_steps=3, num_verify_tokens=16, device="cpu")
        batch = SimpleNamespace(spec_info=idle)
        assert provider.begin_capture(variant, batch, backend, None) is backend
        spec = batch.spec_info  # the harness's input, captured at the variant's width
        assert (spec.draft_token_num, spec.num_tokens_per_req, spec.num_tokens_for_logprob_per_req) == (8, 8, 8)
        assert spec.greedy_tokens_only is greedy and spec.tail_mask is tail_mask
        assert provider.replay_variant(batch) is variant
    spec.draft_token_num = 4  # no graph: eager
    assert provider.replay_variant(batch) is None


# ------------------------------------------------------ H10 GDN tree verify
def test_gdn_tree_verify_follows_the_state_layout(registered):
    from sglang.srt.layers.attention.linear.verify_extension import LinearAttnStateLayout

    from eagle_harness.sglang.gdn_tree_verify import GDNTreeVerify

    registered.__dict__["config"] = parse_harness_config(CONFIG)
    state = dict(
        conv_state_len=3,
        conv_dtype=torch.bfloat16,
        ssm_dtype=torch.float32,
        num_v_heads=12,
        has_state_snapshots=True,
        device=torch.device("cpu"),
    )
    ext = registered.create_linear_attn_verify_extension(LinearAttnStateLayout(**state))
    assert isinstance(ext, GDNTreeVerify) and ext.fp32_state and ext.num_v_heads == 12
    for change in ({"has_state_snapshots": False}, {"conv_state_len": 2}, {"conv_dtype": torch.float32}):
        assert registered.create_linear_attn_verify_extension(LinearAttnStateLayout(**{**state, **change})) is None


def test_mamba2_tree_verify_follows_the_state_layout(registered):
    from sglang.srt.layers.attention.linear.verify_extension import LinearAttnStateLayout

    from eagle_harness.sglang.gdn_tree_verify import GDNTreeVerify
    from eagle_harness.sglang.mamba2_tree_verify import Mamba2TreeVerify

    if "kind" not in LinearAttnStateLayout.__dataclass_fields__:
        pytest.skip("this sglang asks GDN backends only")
    registered.__dict__["config"] = parse_harness_config(CONFIG)
    state = dict(
        conv_state_len=3,
        conv_dtype=torch.bfloat16,
        ssm_dtype=torch.float32,
        num_v_heads=16,
        has_state_snapshots=True,
        device=torch.device("cpu"),
        kind="mamba2",
        head_dim=64,
        ssm_state_size=128,
        n_groups=2,
    )
    assert isinstance(registered.create_linear_attn_verify_extension(LinearAttnStateLayout(**state)), Mamba2TreeVerify)
    gdn = registered.create_linear_attn_verify_extension(LinearAttnStateLayout(**{**state, "kind": "gdn"}))
    assert isinstance(gdn, GDNTreeVerify)
    for change in ({"has_state_snapshots": False}, {"conv_state_len": 2}, {"conv_dtype": torch.float32}):
        assert registered.create_linear_attn_verify_extension(LinearAttnStateLayout(**{**state, **change})) is None
    registered.__dict__["config"] = parse_harness_config(STATIC_CONFIG)  # static mode verifies with it too
    assert isinstance(registered.create_linear_attn_verify_extension(LinearAttnStateLayout(**state)), Mamba2TreeVerify)
    state = {**state, "kind": "gdn"}
    assert isinstance(registered.create_linear_attn_verify_extension(LinearAttnStateLayout(**state)), GDNTreeVerify)


# ------------------------------------------------------------ H5 headroom
def test_block_table_headroom(registered):
    with get_context().override_server_args(
        speculative_algorithm=ALGORITHM_NAME,
        speculative_num_steps=12,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=16,
        page_size=1,
    ):
        # draft span 4 * 12 = 48 > default 4 + 16
        assert get_req_to_token_extra_context_len() == 48
    with get_context().override_server_args(
        speculative_algorithm=ALGORITHM_NAME,
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        page_size=1,
    ):
        assert get_req_to_token_extra_context_len() == 4 + 4  # the default wins


# ------------------------------------------------------------- real hooks
def test_no_monkeypatching_in_the_package():
    """The package may register through sglang's APIs only: no HookRegistry, no
    attribute writes onto sglang modules or classes."""
    root = os.path.dirname(eagle_harness.__file__)
    offenders = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            src = open(path, encoding="utf-8").read()
            if "HookRegistry" in src or "hook_registry" in src or "monkey" in src.lower():
                offenders.append(path)
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setattr":
                    offenders.append(f"{path}:{node.lineno}")
    assert offenders == []


# ------------------------------------------------------------- plain decoding
def test_trivial_verify_input_hands_the_greedy_kernel_int64_tokens():
    """Width 1 (plain decoding) verifies the root alone; sglang's greedy verify kernel takes int64 candidates, and
    sglang's builder passes the int32 bonus tokens (the kernel rejected them and the scheduler died at batch size 8)."""
    backend = SimpleNamespace(verify_mask=None, max_context_len=64)
    worker = SimpleNamespace(
        topk=1, device="cpu", _target_worker=SimpleNamespace(model_runner=SimpleNamespace(attn_backend=backend))
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=SimpleNamespace(bonus_tokens=torch.tensor([5, 7, 9], dtype=torch.int32)),
        seq_lens=torch.tensor([10, 11, 12]),
        seq_lens_sum=33,
        seq_lens_cpu=None,
    )
    verify_input = EagleHarnessWorker._build_trivial_verify_input(worker, batch)
    assert verify_input.draft_token.dtype == torch.int64
    assert verify_input.draft_token.tolist() == [5, 7, 9]
    assert verify_input.retrieve_index.dtype == torch.int64 and verify_input.draft_token_num == 1


def test_draft_attention_splits_keep_the_reference_waves(monkeypatch):
    """The swept B300 TP4 launches (64 / 24 splits for 10 heads on 148 SMs) are reproduced exactly; fewer heads per
    rank (TP8) or SMs get more splits, more heads (TP1) fewer, never above 128; CPU keeps the reference."""
    import torch

    from eagle_harness.sglang import row_attention as ra

    class Props:
        def __init__(self, sms):
            self.multi_processor_count = sms

    sms = {"b300": 148, "h100": 132}
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: Props(sms[device.type if device.type != "cuda" else "b300"]))
    cuda = torch.device("cuda")
    step, extend = ra.draft_launches(10, cuda)
    assert (step.splits, extend.splits) == (ra.STEP_LAUNCH.splits, ra.EXTEND_LAUNCH.splits) == (64, 24)
    assert (step.block_n, extend.block_n) == (64, 128)
    assert [ra.draft_launches(h, cuda)[0].splits for h in (40, 20, 5)] == [16, 32, 128]  # TP1, TP2, TP8
    assert [ra.draft_launches(h, cuda)[1].splits for h in (40, 20, 5)] == [8, 16, 48]
    assert ra.draft_launches(1, cuda)[0].splits == ra.draft_launches(1, cuda)[1].splits == 128  # the cap
    assert ra.draft_launches(10, torch.device("cpu")) == (ra.STEP_LAUNCH, ra.EXTEND_LAUNCH)
