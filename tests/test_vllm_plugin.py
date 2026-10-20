# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The vLLM plugin: registration, method_config parsing, the verify schedule, hook map.

CPU only. Needs vLLM with the speculative-method hooks (branch ``eagle-hooks``)
importable; without them the module is skipped (the plugin itself then switches
itself off, which ``test_register_is_off_without_the_vllm_hooks`` covers).
"""

from __future__ import annotations

import inspect
import logging
import sys
from importlib.metadata import PackageNotFoundError, distribution, entry_points
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm.v1.spec_decode.registry")

import vllm  # noqa: E402
from vllm.config import SpeculativeConfig, VllmConfig  # noqa: E402
from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402
from vllm.v1.spec_decode import registry as vllm_registry  # noqa: E402
from vllm.v1.spec_decode.dynamic.utils import validate_and_normalize_dynamic_sd_schedule  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator, DraftModelSpeculator  # noqa: E402

from eagle_harness.core.limits import ComputeIntensity  # noqa: E402
from eagle_harness.vllm import plugin  # noqa: E402
from eagle_harness.vllm.adapter import HOOK_MAP  # noqa: E402
from eagle_harness.vllm.config import (  # noqa: E402
    METHOD_NAME,
    num_lookahead_tokens,
    parse_method_config,
    verify_width_schedule,
)

COST = {"data_path": "/nonexistent/cost-data"}
STATIC = {"mode": "static"}
CHAIN = {"mode": "static", "node_top_k": 1, "tree_depth": 5, "max_verify_width": 6}


@pytest.fixture
def registered():
    snapshot = dict(vllm_registry._REGISTRY)
    vllm_registry._REGISTRY.pop(METHOD_NAME, None)
    assert plugin.register() is True
    yield vllm_registry.get_speculative_method(METHOD_NAME)
    vllm_registry._REGISTRY.clear()
    vllm_registry._REGISTRY.update(snapshot)


# ------------------------------------------------------------------ registration
def test_register_adds_the_method(registered):
    assert registered.speculator_cls == "eagle_harness.vllm.speculator:HarnessSpeculator"
    assert registered.uses_target_hidden_states is True
    assert registered.uses_aux_hidden_states is True
    if "supports_async_scheduling" in vllm_registry.SpeculativeMethodSpec.__dataclass_fields__:  # 0.1.0 engine
        assert registered.supports_async_scheduling is True
    assert registered.configure is not None
    assert registered.num_lookahead_tokens is num_lookahead_tokens
    assert registered.linear_attn_verify_extension == "eagle_harness.vllm.gdn_tree_verify:create_extension"


def test_register_is_idempotent(registered):
    assert plugin.register() is True
    assert vllm_registry.get_speculative_method(METHOD_NAME) == registered


def test_register_adds_the_draft_architecture(registered):
    assert plugin.DRAFT_ARCHITECTURE in ModelRegistry.get_supported_archs()


def test_register_is_off_without_the_vllm_hooks(monkeypatch):
    # An import of the registry module fails, as on a vLLM without the hooks.
    monkeypatch.setitem(sys.modules, "vllm.v1.spec_decode.registry", None)
    assert plugin.register() is False


def test_speculator_class_resolves(registered):
    from eagle_harness.vllm.speculator import HarnessSpeculator

    assert registered.resolve_speculator_cls() is HarnessSpeculator
    assert issubclass(HarnessSpeculator, DraftModelSpeculator)


def test_entry_point_is_declared():
    tomllib = pytest.importorskip("tomllib")
    root = Path(__file__).resolve().parents[1]
    declared = tomllib.loads((root / "pyproject.toml").read_text())["project"]["entry-points"]
    assert declared["vllm.general_plugins"] == {"eagle_harness": "eagle_harness.vllm.plugin:register"}
    assert declared["sglang.srt.plugins"] == {"eagle_harness": "eagle_harness.sglang.plugin:register"}


def test_installed_entry_point_is_found_by_vllm():
    try:
        distribution("eagle-harness")
    except PackageNotFoundError:
        pytest.skip("eagle-harness is not installed (pip install -e .)")
    from vllm.plugins import load_plugins_by_group

    found = [ep for ep in entry_points(group="vllm.general_plugins") if ep.name == "eagle_harness"]
    assert len(found) == 1 and found[0].value == "eagle_harness.vllm.plugin:register"
    assert load_plugins_by_group("vllm.general_plugins")["eagle_harness"] is plugin.register


# ------------------------------------------------------------------ method_config
def test_parse_defaults():
    parsed = parse_method_config(COST)
    assert parsed.harness.data_path == COST["data_path"]
    assert parsed.max_verify_width == 32 and parsed.num_speculative_tokens == 31
    assert (parsed.node_top_k, parsed.tree_depth) == (8, 12)
    # The draft tree writes node_top_k * tree_depth KV slots, more than the verify does.
    assert parsed.draft_kv_lookahead == 96


def test_parse_chain_defaults():
    parsed = parse_method_config({**COST, "intensity": "chain_intensity"})
    assert (parsed.max_verify_width, parsed.node_top_k, parsed.tree_depth) == (16, 1, 15)
    assert parsed.draft_kv_lookahead == 16  # a chain verifies its 15 drafts plus the root


def test_parse_overrides():
    parsed = parse_method_config({**COST, "max_verify_width": 64, "node_top_k": 4, "tree_depth": 10})
    assert parsed.num_speculative_tokens == 63
    assert parsed.draft_kv_lookahead == 64  # the 64 verify rows beat the 40 draft slots


def test_parse_depth_stop_gain():
    assert parse_method_config(COST).depth_stop_gain == "derived"  # default: the measured cost of a step
    assert parse_method_config({**COST, "depth_stop_gain": "derived"}).depth_stop_gain == "derived"
    assert parse_method_config({**COST, "depth_stop_gain": 0}).depth_stop_gain == 0.0  # override: lossless
    assert parse_method_config({**COST, "depth_stop_gain": 0.05}).depth_stop_gain == 0.05
    assert parse_method_config({**COST, "depth_stop_gain": 1}).depth_stop_gain == 1.0
    assert parse_method_config({**COST, "depth_stop_gain": None}).depth_stop_gain is None  # always tree_depth


@pytest.mark.parametrize(
    "bad, error",
    [
        ({**COST, "no_such_key": 1}, ValueError),
        ({**COST, "max_verify_width": 65}, NotImplementedError),
        ({**COST, "max_verify_width": 1}, ValueError),
        ({**COST, "max_verify_width": "32"}, ValueError),
        ({**COST, "node_top_k": 9}, NotImplementedError),
        ({**COST, "depth_stop_gain": -0.1}, ValueError),
        ({**COST, "depth_stop_gain": True}, ValueError),
        ({**COST, "depth_stop_gain": "0"}, ValueError),
        ({**COST, "intensity": "chain_intensity", "node_top_k": 2}, NotImplementedError),
    ],
)
def test_parse_rejects(bad, error):
    with pytest.raises(error):
        parse_method_config(bad)


# ------------------------------------------------------------------ schedule
def test_schedule_is_valid_for_vllm_and_capped():
    for intensity in ComputeIntensity:
        for width in (4, 16, 32, 64):
            if intensity is ComputeIntensity.CHAIN and width > 16:
                continue
            schedule = verify_width_schedule(intensity, width, allow_unpriced=True)
            assert validate_and_normalize_dynamic_sd_schedule(schedule) == schedule
            assert max(k for _, _, k in schedule) <= width - 1


def test_schedule_follows_the_verify_buckets():
    # high_intensity: 64 wide up to batch 8, 32 wide up to the shared batch limit.
    schedule = verify_width_schedule(ComputeIntensity.HIGH, 64)
    assert schedule[0] == (1, 8, 63)
    assert schedule[1] == (9, 512, 31)
    assert schedule[-1][2] == 0 and schedule[-1][0] == 513
    assert verify_width_schedule(ComputeIntensity.LOW, 64)[:2] == [(1, 2, 63), (3, 8, 31)]
    # Chains reach batch 512; without measured costs they verify the widest chain at every batch size.
    chain = verify_width_schedule(ComputeIntensity.CHAIN, 16, allow_unpriced=True)
    assert chain == [(1, 512, 15), (513, 1 << 16, 0)]


def test_chain_schedule_is_derived_from_the_tables_and_the_profile(tmp_path):
    from tests.test_chain_width import _table, write_tables

    from eagle_harness.core.chain_width import save_accept_profile

    write_tables(
        str(tmp_path), {1: _table(1, lambda w: 5000.0 + w, 20.0), 512: _table(512, lambda w: 3000.0 * w, 300.0)}
    )
    # Tables alone do not price a chain: automatic serving refuses missing acceptance.
    with pytest.raises(ValueError, match="acceptance profile"):
        verify_width_schedule(ComputeIntensity.CHAIN, 16, str(tmp_path))
    save_accept_profile(str(tmp_path), 1000, [900 - 50 * d for d in range(15)])
    schedule = verify_width_schedule(ComputeIntensity.CHAIN, 16, str(tmp_path))
    assert schedule == [(1, 256, 15), (257, 512, 0), (513, 1 << 16, 0)]
    assert validate_and_normalize_dynamic_sd_schedule(schedule) is not None
    # Trees keep the bucket ceilings whatever the data path holds.
    assert verify_width_schedule(ComputeIntensity.HIGH, 64, str(tmp_path)) == verify_width_schedule(
        ComputeIntensity.HIGH, 64
    )


# ------------------------------------------------------------------ SpeculativeConfig
def test_speculative_config_is_configured_by_the_method(registered):
    config = SpeculativeConfig(method=METHOD_NAME, method_config={**COST, "max_verify_width": 64})
    assert config.num_speculative_tokens == 63
    assert config.num_speculative_tokens_per_batch_size[0] == (1, 8, 63)
    assert config.method_config["max_verify_width"] == 64
    assert config.use_eagle() is True


def test_num_speculative_tokens_is_the_widest_scheduled_verify(registered, tmp_path):
    # vLLM reserves 1 + K state slots per request: K is the widest chain the measured costs ever pick, not 15.
    from tests.test_chain_width import _table, write_tables

    from eagle_harness.core.chain_width import save_accept_profile

    write_tables(str(tmp_path), {1: _table(1, lambda w: 5000.0 + w, 20.0)})
    save_accept_profile(str(tmp_path), 1000, [900 - 50 * d for d in range(7)])  # measured up to width 8
    chain = {"mode": "serve", "intensity": "chain_intensity", "data_path": str(tmp_path)}
    config = SpeculativeConfig(method=METHOD_NAME, method_config=chain)
    assert config.num_speculative_tokens_per_batch_size[0] == (1, 512, 7)
    assert config.num_speculative_tokens == 7
    # A config re-created with the derived K (e.g. copied to another process) still validates.
    again = SpeculativeConfig(
        method=METHOD_NAME,
        method_config=chain,
        num_speculative_tokens=7,
        num_speculative_tokens_per_batch_size=config.num_speculative_tokens_per_batch_size,
    )
    assert again.num_speculative_tokens == 7


def test_served_costs_choose_no_drafts_without_overreserving(registered, tmp_path):
    from eagle_harness.core.chain_width import save_accept_profile, save_round_times

    save_accept_profile(str(tmp_path), 100, [90, 80, 70])
    save_round_times(
        str(tmp_path),
        [
            {"batch_size": 1, "width": 1, "round_us": 1000, "count": 10},
            {"batch_size": 1, "width": 4, "round_us": 1200, "count": 10},
            {"batch_size": 512, "width": 1, "round_us": 1000, "count": 10},
            {"batch_size": 512, "width": 4, "round_us": 4000, "count": 10},
        ],
    )
    config = SpeculativeConfig(
        method=METHOD_NAME,
        method_config={"mode": "serve", "intensity": "chain_intensity", "data_path": str(tmp_path)},
    )
    assert config.num_speculative_tokens == 3
    schedule = config.num_speculative_tokens_per_batch_size
    assert next(k for first, last, k in schedule if first <= 1 <= last) == 3
    assert next(k for first, last, k in schedule if first <= 512 <= last) == 0


def test_native_schedule_survives_real_speculative_config_serialization(registered, tmp_path):
    import json
    from tests.test_chain_no_spec import _native_schedule_config

    if not hasattr(SpeculativeConfig, "native_decode_batch_sizes"):
        pytest.skip("Host lacks the optional native-decode capability")
    expected = _native_schedule_config(tmp_path)
    config = SpeculativeConfig(method=METHOD_NAME, method_config=expected.method_config)
    assert config.native_decode_batch_sizes == expected.native_decode_batch_sizes
    fields = (
        "method",
        "method_config",
        "num_speculative_tokens",
        "num_speculative_tokens_per_batch_size",
        "native_decode_batch_sizes",
    )
    again = SpeculativeConfig(**json.loads(json.dumps({name: getattr(config, name) for name in fields})))
    assert again.native_decode_batch_sizes == config.native_decode_batch_sizes
    assert again.num_speculative_tokens_per_batch_size == config.num_speculative_tokens_per_batch_size


def test_trees_refuse_native_decode_ranges(registered):
    if not hasattr(SpeculativeConfig, "native_decode_batch_sizes"):
        pytest.skip("Host lacks the optional native-decode capability")
    with pytest.raises(ValueError, match="native decode.*chain"):
        SpeculativeConfig(method=METHOD_NAME, method_config=STATIC, native_decode_batch_sizes=[(1, 64)])


def test_static_mode_has_no_batch_schedule(registered):
    config = SpeculativeConfig(method=METHOD_NAME, method_config=STATIC)
    assert config.num_speculative_tokens == 31
    assert config.num_speculative_tokens_per_batch_size is None


def test_measured_tree_rounds_narrow_only_the_losing_batch_sizes(registered, tmp_path):
    from eagle_harness.core.chain_width import merge_tree_rounds

    def row(bs, width, round_us, accept):
        return {"batch_size": bs, "width": width, "round_us": round_us, "accept": accept, "count": 50}

    merge_tree_rounds(
        str(tmp_path),
        [row(8, 16, 12_500, 4.43), row(8, 4, 9_000, 2.9), row(64, 16, 47_900, 3.96), row(64, 4, 20_000, 2.9)],
    )
    tree = {"mode": "static", "node_top_k": 4, "tree_depth": 14, "max_verify_width": 16, "data_path": str(tmp_path)}
    config = SpeculativeConfig(method=METHOD_NAME, method_config=tree)
    schedule = config.num_speculative_tokens_per_batch_size
    assert schedule is not None, "measured rounds must give the tree a batch-size schedule"
    assert validate_and_normalize_dynamic_sd_schedule(schedule) == schedule
    k_at = lambda bs: next(k for first, last, k in schedule if first <= bs <= last)  # noqa: E731
    assert (k_at(1), k_at(8), k_at(64), k_at(512)) == (15, 15, 3, 3)
    assert config.num_speculative_tokens == 15
    # The same tree without measured rounds keeps one width (the release default).
    plain = SpeculativeConfig(method=METHOD_NAME, method_config={**tree, "data_path": str(tmp_path / "none")})
    assert plain.num_speculative_tokens_per_batch_size is None


def test_record_rounds_needs_a_data_path():
    assert parse_method_config({"mode": "static", "record_rounds": True, "data_path": "/x"}).record_rounds is True
    assert parse_method_config({"mode": "static"}).record_rounds is False
    with pytest.raises(ValueError, match="record_rounds"):
        parse_method_config({"mode": "static", "record_rounds": True})
    with pytest.raises(ValueError, match="record_rounds"):
        parse_method_config({"mode": "static", "record_rounds": "yes", "data_path": "/x"})


def test_user_schedule_wins(registered):
    mine = [(1, 4, 7)]
    config = SpeculativeConfig(method=METHOD_NAME, method_config=COST, num_speculative_tokens_per_batch_size=mine)
    assert config.num_speculative_tokens_per_batch_size == mine


def test_explicit_num_speculative_tokens_must_agree(registered):
    ok = SpeculativeConfig(method=METHOD_NAME, method_config=COST, num_speculative_tokens=31)
    assert ok.num_speculative_tokens == 31
    with pytest.raises(ValueError, match="disagrees"):
        SpeculativeConfig(method=METHOD_NAME, method_config=COST, num_speculative_tokens=8)


def test_typo_in_method_config_fails_at_config_time(registered):
    with pytest.raises(ValueError, match="tre_depth"):
        SpeculativeConfig(method=METHOD_NAME, method_config={**COST, "tre_depth": 3})


def test_lookahead_reaches_the_vllm_config(registered):
    config = SpeculativeConfig(method=METHOD_NAME, method_config=COST)
    fake = SimpleNamespace(speculative_config=config, num_speculative_tokens=config.num_speculative_tokens)
    assert VllmConfig.num_lookahead_tokens.fget(fake) == 96


def test_the_method_name_is_unknown_before_the_plugin_runs(monkeypatch):
    monkeypatch.delitem(vllm_registry._REGISTRY, METHOD_NAME, raising=False)
    with pytest.raises(ValueError, match="Unknown speculative method"):
        SpeculativeConfig(method=METHOD_NAME, method_config=STATIC)


# ------------------------------------------------------------------ hook map
def test_every_mapped_hook_exists_in_vllm():
    from eagle_harness.vllm.adapter import missing_hooks

    assert missing_hooks() == []


def test_register_refuses_a_vllm_without_a_mapped_hook(monkeypatch):
    from eagle_harness.vllm import adapter

    monkeypatch.setattr(
        adapter, "HOOK_MAP", adapter.HOOK_MAP + (adapter.HookUse("x", "vllm.config:NoSuchHook", "gpu"),)
    )
    assert adapter.missing_hooks() == ["vllm.config:NoSuchHook"]
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logging.getLogger("eagle_harness.compat").addHandler(handler)  # the plugin's loggers may not propagate to root
    try:
        assert plugin.register() is False
    finally:
        logging.getLogger("eagle_harness.compat").removeHandler(handler)
    (message,) = [r.getMessage() for r in records if r.levelno == logging.ERROR]
    assert f"vllm {vllm.__version__} lacks 1 of the hooks it needs (vllm.config:NoSuchHook)" in message
    assert "Fix: install vllm with the eagle-harness hooks" in message


@pytest.mark.parametrize("layout", [0, 1], ids=["review-head", "release-0.1.0"])
def test_verify_extension_module_follows_the_engine_layout(monkeypatch, layout):
    import types

    from eagle_harness.vllm import adapter

    present = types.ModuleType(adapter.VERIFY_EXTENSION_MODULES[layout])
    for i, name in enumerate(adapter.VERIFY_EXTENSION_MODULES):
        monkeypatch.setitem(sys.modules, name, present if i == layout else None)  # None: import raises
    assert adapter.verify_extension_module() is present


def test_verify_extension_hook_counts_either_layout(monkeypatch):
    from eagle_harness.vllm import adapter

    (hook,) = [use.hook for use in HOOK_MAP if "LinearAttnVerifyExtension" in use.hook]
    for layout in adapter.VERIFY_EXTENSION_MODULES:
        monkeypatch.setattr(adapter, "_exists_one", lambda path, layout=layout: path.startswith(layout + ":"))
        assert adapter._exists(hook)
    monkeypatch.setattr(adapter, "_exists_one", lambda path: False)
    assert not adapter._exists(hook)


@pytest.mark.parametrize("has_field", [True, False], ids=["release-0.1.0", "review-head"])
def test_async_scheduling_opt_in_only_where_the_engine_has_the_field(has_field):
    import dataclasses

    fields = [("speculator_cls", str)] + ([("supports_async_scheduling", bool, False)] if has_field else [])
    spec_cls = dataclasses.make_dataclass("Spec", fields)
    opt_in = plugin._async_scheduling_opt_in(spec_cls)
    assert opt_in == ({"supports_async_scheduling": True} if has_field else {})
    spec_cls("x", **opt_in)  # the engine's spec accepts what the plugin passes


def test_hook_map_statuses_are_known():
    assert {use.status for use in HOOK_MAP} == {"cpu", "gpu", "none"}
    assert all(not use.hook for use in HOOK_MAP if use.status == "none")


# ------------------------------------------------------------------ skeleton honesty
SPECULATOR_HOOKS = ("make_rejection_sampler", "prepare_verify_inputs", "post_accept")


@pytest.mark.parametrize("name", SPECULATOR_HOOKS)
def test_speculator_hooks_match_the_vllm_signatures(name):
    from eagle_harness.vllm.speculator import HarnessSpeculator

    ours = inspect.signature(getattr(HarnessSpeculator, name)).parameters
    theirs = inspect.signature(getattr(BaseSpeculator, name)).parameters
    assert list(ours) == list(theirs)


def test_gdn_extension_factory_follows_the_mode():
    from eagle_harness.vllm.adapter import verify_extension_module
    from eagle_harness.vllm.gdn_tree_verify import TreeVerify, create_extension

    LinearAttnVerifyExtension = verify_extension_module().LinearAttnVerifyExtension

    def config(method_config):
        return SimpleNamespace(speculative_config=SimpleNamespace(method_config=method_config))

    assert create_extension(config(CHAIN)) is None  # a chain is verified by vLLM's own kernels
    extension = create_extension(config(STATIC))
    assert isinstance(extension, TreeVerify) and isinstance(extension, LinearAttnVerifyExtension)


def test_tree_mode_refuses_linear_layers_the_extension_cannot_reach(monkeypatch):
    """A Mamba2 layer reaches the tree verify only through vLLM's Mamba2 hook; without it (or for any other linear
    layer) vLLM's kernels would verify every tree as a chain, so the speculator refuses the target."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    from eagle_harness.vllm.gdn_tree_verify import LinearAttnVerifyExtension, check_tree_verifiable

    gdn, mamba2 = object.__new__(QwenGatedDeltaNetAttention), object.__new__(MambaMixer2)
    check_tree_verifiable([gdn, gdn])
    with pytest.raises(RuntimeError, match="object layers"):
        check_tree_verifiable([gdn, object()])

    def mamba2_hook(self, rows):
        raise AssertionError("not called")

    monkeypatch.setattr(LinearAttnVerifyExtension, "forward_mamba2_spec_rows", mamba2_hook, raising=False)
    check_tree_verifiable([gdn, mamba2])
    monkeypatch.delattr(LinearAttnVerifyExtension, "forward_mamba2_spec_rows")
    with pytest.raises(RuntimeError, match="MambaMixer2 layers.*forward_mamba2_spec_rows"):
        check_tree_verifiable([gdn, mamba2])


def test_draft_aux_norms_run_on_the_kernel_ops(monkeypatch):
    """``combine_hidden_states`` (called eagerly by the draft prefill) runs inside ``_kernel_ops``, so the per-chunk
    RMSNorms dispatch to vLLM's CUDA kernel rather than the native torch code."""
    import contextlib

    from eagle_harness.vllm import models

    events = []

    @contextlib.contextmanager
    def record():
        events.append("enter")
        yield
        events.append("exit")

    monkeypatch.setattr(models, "_kernel_ops", record)
    monkeypatch.setattr(
        models.Eagle3LlamaForCausalLM, "combine_hidden_states", lambda self, h: events.append("combine") or h
    )
    model = object.__new__(models.LlamaForCausalLMEagle4)
    hidden = object()
    assert model.combine_hidden_states(hidden) is hidden
    assert events == ["enter", "combine", "exit"]


def test_package_logs_through_vllms_handler():
    import logging

    from eagle_harness.vllm.plugin import _log_through_vllm

    ours, theirs = logging.getLogger("eagle_harness"), logging.getLogger("vllm")
    saved = (list(ours.handlers), ours.level, ours.propagate, list(theirs.handlers))
    try:
        ours.handlers.clear()
        handler = logging.StreamHandler()
        theirs.addHandler(handler)
        _log_through_vllm()
        assert handler in ours.handlers and not ours.propagate
    finally:
        ours.handlers[:], ours.level, ours.propagate = saved[0], saved[1], saved[2]
        theirs.handlers[:] = saved[3]


# A reduced-size Nemotron EAGLE-4 draft configuration with the same layer structures.
NEMOTRON35_DRAFT = dict(
    architectures=["LlamaForCausalLMEagle4"],
    num_hidden_layers=1,
    hidden_size=256,
    num_attention_heads=4,
    num_key_value_heads=4,
    intermediate_size=512,
    vocab_size=1000,
    efficient_lm_head_config={"type": "low_rank", "rank": 32, "num_candidates": 64},
    attn_norm=True,
    use_rope=False,
    layer_types=["sliding_attention"],
    sliding_window=64,
    hidden_act="relu2",
    enable_moe=True,
    moe_architecture="nemotron_latent_relu2",
    n_routed_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=64,
    moe_latent_size=32,
    n_shared_experts=1,
    moe_shared_expert_intermediate_size=128,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
    routed_scaling_factor=5.0,
    norm_topk_prob=True,
    fc_norm=True,
    norm_output=True,
    eagle_config={"eagle_aux_hidden_state_layer_ids": [7, 47, 78]},
    max_position_embeddings=1048576,
)
NEMOTRON35_MAX_POSITIONS = 1048576


def test_base_layer_config_restores_after_constructor_failure():
    from eagle_harness.vllm.models import _base_layer_config

    config = SimpleNamespace(
        hidden_act="relu2",
        max_position_embeddings=4096,
        enable_moe=True,
        moe_architecture="nemotron_latent_relu2",
        use_rope=False,
    )
    original = vars(config).copy()
    with pytest.raises(RuntimeError, match="layer init failed"):
        with _base_layer_config(config):
            assert (config.hidden_act, config.max_position_embeddings) == ("silu", 1)
            raise RuntimeError("layer init failed")
    assert vars(config) == original


def test_vllm_builds_the_nemotron_draft_structures():
    """NoPE, attn_norm and the latent ReLU^2 MoE are built on vLLM; a SwiGLU MoE and the sandwich norm are not."""
    import transformers

    from eagle_harness.vllm.models import _base_layer_config, check_supported, uses_rope

    config = transformers.LlamaConfig(**NEMOTRON35_DRAFT)
    check_supported(config)
    assert not uses_rope(config) and uses_rope(transformers.LlamaConfig())
    with _base_layer_config(config):  # vLLM's EAGLE-3 layer builds a SiLU MLP and a 1-position rotary placeholder
        assert (config.hidden_act, config.max_position_embeddings) == ("silu", 1)
    assert config.hidden_act == "relu2" and config.max_position_embeddings == NEMOTRON35_MAX_POSITIONS
    swiglu_moe = {**NEMOTRON35_DRAFT, "moe_architecture": "sparse_swiglu", "hidden_act": "silu"}
    with pytest.raises(NotImplementedError, match="nemotron_latent_relu2"):
        check_supported(transformers.LlamaConfig(**swiglu_moe))
    with pytest.raises(NotImplementedError, match="enable_post_mlp_norm"):
        check_supported(transformers.LlamaConfig(**{**NEMOTRON35_DRAFT, "enable_post_mlp_norm": True}))


def test_attn_norm_modules_are_registered_on_the_draft_layer():
    """``eh_proj`` / ``pre_attention_layernorm`` must resolve to the registered submodules (a class-level ``None``
    default once shadowed them and the compiled prefill fed the 2H input to the H-wide qkv projection)."""
    import torch

    from eagle_harness.vllm.models import _Eagle4DecoderLayer

    layer = _Eagle4DecoderLayer.__new__(_Eagle4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.eh_proj = layer.pre_attention_layernorm = None  # as _build_structures starts
    assert layer.eh_proj is None
    proj = torch.nn.Linear(4, 2)
    layer.eh_proj = proj
    assert layer.eh_proj is proj and dict(layer.named_children())["eh_proj"] is proj


@pytest.mark.parametrize("host_supports_stream_control", [True, False])
def test_nemotron_draft_uses_public_shared_expert_stream_control(monkeypatch, host_supports_stream_control):
    from types import SimpleNamespace

    import torch

    from eagle_harness.vllm import models

    calls = []
    shared = SimpleNamespace()
    if host_supports_stream_control:
        shared.disable_aux_stream = lambda: calls.append("disabled")
    mlp = torch.nn.Module()
    mlp.experts = SimpleNamespace(shared_experts=shared)
    monkeypatch.setattr(models, "_latent_moe", lambda *args: mlp)
    model = models.LlamaForCausalLMEagle4.__new__(models.LlamaForCausalLMEagle4)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=4, enable_moe=True, moe_architecture=models.LATENT_MOE)
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = SimpleNamespace(attn=SimpleNamespace(layer_name="model.layers.0.self_attn.attn"))
    model.model.layers = torch.nn.ModuleList([layer])
    if host_supports_stream_control:
        model._build_structures(None)
        assert calls == ["disabled"]
    else:
        with pytest.raises(NotImplementedError, match="SharedExperts.disable_aux_stream"):
            model._build_structures(None)
