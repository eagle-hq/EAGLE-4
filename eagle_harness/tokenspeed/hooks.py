# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The TokenSpeed API the adapter is written against, checked against the installed TokenSpeed.

:func:`hook_problems` lists what the running TokenSpeed lacks: the plugin refuses to register on such a TokenSpeed
(:func:`eagle_harness.tokenspeed.plugin.register`) and ``eagle-harness-check`` reports it, so a TokenSpeed refactor that
drops or renames a hook fails at load time instead of in a GPU run.

Three groups:

- the plugin API of TokenSpeed main (``tokenspeed.plugins`` entry points, the drafter and model registries);
- the generic hooks of TokenSpeed branch ``pr-spec-plugin-hooks`` (``pr/TOKENSPEED_HOOKS.md``): H1 the algorithm config, H2 the
  target's aux hidden states for a plugin algorithm, H3 tree drafting by a plugin drafter, and the sliding-window
  attention of tree lanes (the EAGLE-4 draft layer has a window; TokenSpeed main refuses it in a tree);
- the tree policy of TokenSpeed's ``Eagle`` drafter that :class:`~eagle_harness.tokenspeed.drafter.HarnessDrafter`
  replaces (seed, expand, finalize of the draft tree). These are not public API yet: the check pins their names and
  parameters so a change shows up here first.
"""

from __future__ import annotations

import importlib
import inspect
from typing import NamedTuple

MIN_PLUGIN_API_VERSION = 4


class Hook(NamedTuple):
    path: str  # "module:attribute[.attribute]"
    params: tuple[str, ...] = ()  # parameter names the callable must accept


HOOKS: tuple[Hook, ...] = (
    Hook("tokenspeed.runtime.plugins:PLUGIN_API_VERSION"),
    Hook(
        "tokenspeed.runtime.plugins.registry:register_drafter",
        ("algorithm", "drafter_cls", "model_cls", "captures_aux_hidden_states", "uses_tree_layout"),
    ),
    Hook("tokenspeed.runtime.plugins.registry:register_model", ("cls", "architectures")),
    Hook("tokenspeed.runtime.configs.model_profile:ModelProfile"),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.configure_algorithm", ("algorithm_config",)),  # H1
    Hook("tokenspeed.runtime.execution.drafter:drafter_captures_aux_hidden_states", ("algorithm",)),  # H2
    Hook("tokenspeed.runtime.execution.drafter.eagle:Eagle.bind_tree", ("tree_spec",)),
    Hook("tokenspeed.runtime.execution.drafter.eagle:Eagle.bind_tree_lanes", ("tree_spec",)),
    Hook(
        "tokenspeed.runtime.execution.drafter.tree_round.dynamic:DynamicDrafter.bind_dynamic_tree",
        ("tree_spec", "policy"),
    ),
    Hook("tokenspeed.runtime.execution.drafter.tree_round.device_policy:DevicePolicy", ("costs", "device")),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.replay_decode_graph", ("graph", "bs", "graph_bs")),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.finish_round", ("ctx", "graph_bs")),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.complete_round", ()),
    Hook(
        "tokenspeed.runtime.execution.drafter.base:BaseDrafter.capture_round_graphs",
        ("graph_bs", "lo_bs", "stream", "pool"),
    ),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.has_round_graphs", ("graph_bs",)),
    Hook("tokenspeed.runtime.execution.drafter.base:BaseDrafter.release_round_graphs", ()),
    Hook("tokenspeed.runtime.execution.drafter.eagle:Eagle._seed_tree_lanes", ("bs", "logits_output", "frontier")),
    Hook("tokenspeed.runtime.execution.drafter.eagle:Eagle._expand_tree_lanes", ("bs", "step", "logits_output")),
    Hook("tokenspeed.runtime.execution.drafter.eagle:Eagle._draft_window", ("bs", "next_tokens")),
    Hook("tokenspeed.runtime.layers.attention.backends.base:AttentionBackend.begin_tree_lanes", ("bs", "live_bs")),
    Hook("tokenspeed.runtime.layers.attention.backends.paged.tree_verify:TreeDraftInputs.set_frontier"),
    # the EAGLE-4 draft layer has a sliding window: tree lanes need the hooks branch's sliding-window lane attention
    Hook(
        "tokenspeed.runtime.layers.attention.backends.paged.trtllm:TRTLLMMHAAttnBackend._tree_lane_window_cascade",
        ("lanes", "step"),
    ),
    Hook("tokenspeed.runtime.models.llama_eagle3:LlamaForCausalLMEagle3.exit_logits"),
    Hook("tokenspeed.runtime.execution.context:ForwardContext.gather_ids"),
    Hook("tokenspeed.runtime.distributed.comm_ops:all_gather", ("tensor", "group", "dim")),
)


def hook_problems() -> list[str]:
    """What this TokenSpeed lacks of :data:`HOOKS` (empty: every hook the adapter uses is there)."""
    problems = [_problem(hook) for hook in HOOKS]
    problems = [p for p in problems if p]
    if not problems:
        from tokenspeed.runtime.plugins import PLUGIN_API_VERSION

        if PLUGIN_API_VERSION < MIN_PLUGIN_API_VERSION:
            problems.append(f"PLUGIN_API_VERSION {PLUGIN_API_VERSION} < {MIN_PLUGIN_API_VERSION}")
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
