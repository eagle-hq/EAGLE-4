# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The TokenSpeed API the adapter is written against, checked against the installed TokenSpeed.

:func:`hook_problems` lists what the running TokenSpeed lacks: the plugin refuses to register on such a TokenSpeed
(:func:`eagle_harness.tokenspeed.plugin.register`) and ``eagle-harness-check`` reports it, so a TokenSpeed refactor that
drops or renames a hook fails at load time instead of in a GPU run.

Two TokenSpeed lines carry the hooks, told apart by :func:`contract`:

- ``capabilities``: TokenSpeed advertises ``PLUGIN_CAPABILITIES``, one name per additive plugin contract, and bumps
  ``PLUGIN_API_VERSION`` only for a breaking change. The adapter requires :data:`REQUIRED_CAPABILITIES` and an API
  version in :data:`CAPABILITY_API_VERSIONS`, then pins the names it calls. ``Eagle``'s tree policy is public there
  (:data:`PUBLIC_POLICY`).
- ``release``: the eagle-harness release engine (TokenSpeed ``0.1.1+eagle.N``) has no capability set; its
  ``PLUGIN_API_VERSION`` (at least :data:`RELEASE_MIN_API_VERSION`) counts every contract, and the tree policy has the
  private names of :data:`RELEASE_POLICY`.

On both, the adapter needs the plugin registries, the algorithm config and the target's aux hidden states for a plugin
algorithm, tree layout for a plugin chain, sliding-window tree lanes (the EAGLE-4 draft layer has a window), the
drafter round protocol and tree lanes, the tree policy of TokenSpeed's ``Eagle`` drafter that
:class:`~eagle_harness.tokenspeed.drafter.HarnessDrafter` replaces. The harness owns the dynamic tree and device policy.
Recorded-proposal verification is checked separately when sampled trees are enabled.
"""

from __future__ import annotations

import importlib
import inspect
from typing import NamedTuple

RELEASE_MIN_API_VERSION = 4
CAPABILITY_API_VERSIONS = frozenset({3})
REQUIRED_CAPABILITIES = frozenset(
    {
        "drafter.algorithm_config",
        "drafter.aux_hidden_capture",
        "drafter.tree_layout",
        "attention.windowed_tree_lanes",
        "drafter.round_protocol",
        "drafter.tree_lanes",
        "drafter.tree_policy",
        "sampling.tree_sampler",
    }
)


class Hook(NamedTuple):
    path: str  # "module:attribute[.attribute]"
    params: tuple[str, ...] = ()  # parameter names the callable must accept


class TreePolicy(NamedTuple):
    """The names of the tree-policy methods :class:`~eagle_harness.tokenspeed.drafter.HarnessDrafter` overrides."""

    candidates: str  # DynamicDrafter: each draft row's best K candidates
    seed: str  # Eagle: the first lanes
    expand: str  # Eagle: the next lanes
    window: str  # Eagle: the next verify window and its parents


PUBLIC_POLICY = TreePolicy("round_candidates", "seed_tree", "expand_tree", "verify_window")
RELEASE_POLICY = TreePolicy("_candidates", "_seed_tree_lanes", "_expand_tree_lanes", "_draft_window")

_EAGLE = "tokenspeed.runtime.execution.drafter.eagle:Eagle"


def _policy_hooks(names: TreePolicy) -> tuple[Hook, ...]:
    return (
        Hook(f"{_EAGLE}.{names.seed}", ("bs", "logits_output", "frontier")),
        Hook(f"{_EAGLE}.{names.expand}", ("bs", "step", "logits_output")),
        Hook(f"{_EAGLE}.{names.window}", ("bs", "next_tokens")),
    )


HOOKS: tuple[Hook, ...] = (
    Hook("tokenspeed.runtime.plugins:PLUGIN_API_VERSION"),
    Hook("tokenspeed.runtime.sampling.backends.base:SamplingBackend.bind_tree_sampler", ("factory",)),
    Hook("tokenspeed.runtime.sampling.backends.base:SamplingBackend.supports_tree_sampler"),
    Hook(
        "tokenspeed.runtime.plugins.registry:register_drafter",
        ("algorithm", "drafter_cls", "model_cls", "captures_aux_hidden_states", "uses_tree_layout"),
    ),
    Hook("tokenspeed.runtime.plugins.registry:register_model", ("cls", "architectures")),
    Hook("tokenspeed.runtime.configs.model_profile:ModelProfile"),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.configure_algorithm", ("algorithm_config",)),
    Hook("tokenspeed.runtime.execution.drafter:drafter_captures_aux_hidden_states", ("algorithm",)),
    Hook(f"{_EAGLE}.bind_tree", ("tree_spec",)),
    Hook(f"{_EAGLE}.bind_tree_lanes", ("tree_spec",)),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.prepare_round_sampling", ("sampling_params_list",)),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.replay_decode_graph", ("graph", "bs", "graph_bs")),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.finish_round", ("ctx", "graph_bs")),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.complete_round", ()),
    Hook(
        "tokenspeed.runtime.execution.drafter.base:BaseDrafter.capture_round_graphs",
        ("graph_bs", "lo_bs", "stream", "pool"),
    ),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.has_round_graphs", ("graph_bs",)),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.release_round_graphs", ()),
    Hook("tokenspeed.runtime.layers.attention.backends.base:AttentionBackend.begin_tree_lanes", ("bs", "live_bs")),
    Hook("tokenspeed.runtime.layers.attention.backends.paged.tree_verify:TreeDraftInputs.set_frontier"),
    Hook("tokenspeed.runtime.models.llama_eagle3:LlamaForCausalLMEagle3.exit_logits"),
    Hook("tokenspeed.runtime.execution.context:ForwardContext.gather_ids"),
    Hook("tokenspeed.runtime.distributed.comm_ops:all_gather", ("tensor", "group", "dim")),
)
CAPABILITY_HOOKS: tuple[Hook, ...] = _policy_hooks(PUBLIC_POLICY)
RELEASE_HOOKS: tuple[Hook, ...] = (
    *_policy_hooks(RELEASE_POLICY),
    # the release engine names no capability for the sliding-window lane attention the EAGLE-4 draft layer needs
    Hook(
        "tokenspeed.runtime.layers.attention.backends.paged.trtllm:TRTLLMMHAAttnBackend._tree_lane_window_cascade",
        ("lanes", "step"),
    ),
)


SAMPLED_HOOKS = (
    Hook("tokenspeed.runtime.sampling.tree_verify:DraftTreeQ.zeros", ("rows", "num_slots", "num_candidates", "num_nodes", "device")),
    Hook("tokenspeed.runtime.sampling.tree_verify:TreeVerifyBatch.draft_q"),
    Hook("tokenspeed.runtime.execution.runtime_states:RuntimeStates.init_draft_tree_q", ("num_nodes", "num_slots", "num_candidates")),
    Hook("tokenspeed.runtime.sampling.backends.base:SamplingBackend.verifies_draft_tree_q"),
    Hook("tokenspeed.runtime.sampling.backends.base:SamplingBackend.speculative_sampling_pools"),
)


def sampled_hook_problems() -> list[str]:
    """Recorded-proposal hooks required only when sampled trees are enabled."""
    return [problem for hook in SAMPLED_HOOKS if (problem := _problem(hook))]


def contract() -> str | None:
    """``"capabilities"``, ``"release"``, or None when TokenSpeed has no plugin API."""
    try:
        plugins = importlib.import_module("tokenspeed.runtime.plugins")
    except ImportError:
        return None
    return "capabilities" if hasattr(plugins, "PLUGIN_CAPABILITIES") else "release"


def tree_policy() -> TreePolicy:
    """The tree-policy names of the installed TokenSpeed (the public ones when it has neither)."""
    return RELEASE_POLICY if contract() == "release" else PUBLIC_POLICY


def hook_problems() -> list[str]:
    """What this TokenSpeed lacks of the adapter's contract (empty: every hook the adapter uses is there)."""
    kind = contract()
    problems: list[str] = []
    if kind == "capabilities":
        from tokenspeed.runtime.plugins import PLUGIN_API_VERSION, PLUGIN_CAPABILITIES

        if PLUGIN_API_VERSION not in CAPABILITY_API_VERSIONS:
            problems.append(
                f"PLUGIN_API_VERSION {PLUGIN_API_VERSION} is not one of {sorted(CAPABILITY_API_VERSIONS)} "
                "(a breaking plugin API change)"
            )
        missing = sorted(REQUIRED_CAPABILITIES - set(PLUGIN_CAPABILITIES))
        if missing:
            problems.append(f"PLUGIN_CAPABILITIES lacks {', '.join(missing)}")
    hooks = HOOKS + (CAPABILITY_HOOKS if kind == "capabilities" else RELEASE_HOOKS)
    problems += [p for p in map(_problem, hooks) if p]
    if not problems and kind == "release":
        from tokenspeed.runtime.plugins import PLUGIN_API_VERSION

        if PLUGIN_API_VERSION < RELEASE_MIN_API_VERSION:
            problems.append(f"PLUGIN_API_VERSION {PLUGIN_API_VERSION} < {RELEASE_MIN_API_VERSION}")
    return problems


def _problem(hook: Hook) -> str | None:
    module_name, _, attr_path = hook.path.partition(":")
    try:
        obj = importlib.import_module(module_name)
    except ImportError:
        return f"{hook.path} (module {module_name} not importable)"
    for part in attr_path.split("."):
        if not hasattr(obj, part):  # a dataclass field without a class-level default still counts
            return None if part in getattr(obj, "__dataclass_fields__", {}) else hook.path
        obj = getattr(obj, part)
    if hook.params:
        accepted = inspect.signature(obj).parameters
        missing = [name for name in hook.params if name not in accepted]
        if missing:
            return f"{hook.path} (no parameter {', '.join(missing)})"
    return None
