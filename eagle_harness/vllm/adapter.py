# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Which vLLM hook carries which :class:`EngineAdapter` need (``eagle_core_map.md`` section 3).

:func:`missing_hooks` lists the named hooks the running vLLM lacks: the plugin refuses to register on such a vLLM
(``plugin.register``) and the CPU tests fail on it, so a vLLM refactor that drops a hook fails at load time and in
tests instead of in a GPU run. ``status`` is where the harness side
stands: ``cpu`` (implemented, CPU-tested), ``gpu`` (implemented, tested in GPU
jobs only), ``none`` (vLLM offers no hook; see VLLM_HOOKS.md open questions).
"""

from __future__ import annotations

import importlib
from typing import NamedTuple, Tuple


class HookUse(NamedTuple):
    need: str  # EngineAdapter method or engine-facing need
    hook: str  # "module:attribute[.attribute]" in vLLM, or "" when vLLM has none
    status: str


HOOK_MAP: Tuple[HookUse, ...] = (
    HookUse("algorithm registration", "vllm.v1.spec_decode.registry:register_speculative_method", "cpu"),
    HookUse("H11 opaque algorithm config", "vllm.config.speculative:SpeculativeConfig.method_config", "cpu"),
    HookUse(
        "method configure / max tree budget / per-batch K",
        "vllm.v1.spec_decode.registry:SpeculativeMethodSpec.configure",
        "cpu",
    ),
    HookUse(
        "H5 KV headroom (lookahead)", "vllm.v1.spec_decode.registry:SpeculativeMethodSpec.num_lookahead_tokens", "cpu"
    ),
    HookUse(
        "draft + KV slots (draft_forward, alloc_slots)",
        "vllm.v1.worker.gpu.spec_decode.speculator:DraftModelSpeculator.propose",
        "gpu",
    ),
    HookUse(
        "target_verify positions and tree mask (H9)",
        "vllm.v1.worker.gpu.spec_decode.speculator:BaseSpeculator.prepare_verify_inputs",
        "gpu",
    ),
    HookUse(
        "verify accept (device_accept)",
        "vllm.v1.worker.gpu.spec_decode.speculator:BaseSpeculator.make_rejection_sampler",
        "gpu",
    ),
    HookUse(
        "accepted-path KV compaction (compact_accept_kv)",
        "vllm.v1.worker.gpu.spec_decode.speculator:BaseSpeculator.post_accept",
        "gpu",
    ),
    HookUse(
        "H10 GDN tree verify of the spec-decode rows",
        "vllm.model_executor.layers.mamba.gdn.verify_extension:LinearAttnVerifyExtension.forward_spec_rows",
        "gpu",
    ),
    HookUse(
        "H7 per-width verify graphs",
        "vllm.v1.worker.gpu.spec_decode.speculator:BaseSpeculator.uses_varlen_verify_graphs",
        "gpu",
    ),
    HookUse(
        "tree attention backend (H9 plan)", "vllm.v1.attention.backends.registry:AttentionBackendEnum.CUSTOM", "gpu"
    ),
    HookUse("H2 request admission (no logprobs, text only)", "", "none"),
    HookUse("H1 worker idle / batch released (deferred frees)", "", "none"),
)


def missing_hooks() -> list[str]:
    """The ``HOOK_MAP`` hooks this vLLM does not have (empty: every hook the harness uses is there)."""
    return [use.hook for use in HOOK_MAP if use.hook and not _exists(use.hook)]


def _exists(path: str) -> bool:
    module_name, _, attr_path = path.partition(":")
    try:
        obj = importlib.import_module(module_name)
    except ImportError:
        return False
    for part in attr_path.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:  # a dataclass / pydantic field without a class-level default still counts
            return part in getattr(obj, "__dataclass_fields__", {}) or part in getattr(obj, "model_fields", {})
    return True
