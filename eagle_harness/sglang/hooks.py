# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The sglang API the plugin is written against, checked when it registers.

Each :class:`Hook` names an sglang attribute and the parameters the plugin expects:

- ``override``: the plugin overrides it, so the parameter names must match exactly (a parameter sglang adds would
  never reach the override);
- ``call``: the plugin calls it, so the expected names must be among sglang's (sglang may add optional ones);
- ``field``: a field of an sglang argument group (``get_spec()`` and friends) or data class, or a class-level
  default the plugin's subclass sets.

:func:`hook_problems` lists the mismatches of the installed sglang; :func:`eagle_harness.sglang.plugin.register`
refuses to register when there is one, so an sglang refactor fails at load time with the reason instead of in a
GPU run. The CPU tests check that every override in the package has the signature listed here.
"""

from __future__ import annotations

import importlib
import inspect
from typing import NamedTuple

OVERRIDE, CALL, FIELD = "override", "call", "field"

_ALGO = "sglang.srt.speculative.spec_registry:CustomSpecAlgo"
_WORKER = "sglang.srt.speculative.eagle_worker_v2:EAGLEWorkerV2"
_DRAFT = "sglang.srt.speculative.eagle_worker_v2:EagleDraftWorker"
_HARNESS_ALGO = "eagle_harness.sglang.algorithm:EagleHarnessSpecAlgo"
_HARNESS_WORKER = "eagle_harness.sglang.worker:EagleHarnessWorker"
_HARNESS_DRAFT = "eagle_harness.sglang.worker:HarnessDraftWorker"
_SPEC_INPUT = "sglang.srt.speculative.spec_info:SpecInput"
_VERIFY_INPUT = "sglang.srt.speculative.eagle_info:EagleVerifyInput"
_EXT = "sglang.srt.layers.attention.linear.verify_extension:LinearAttnVerifyExtension"
_HARNESS_EXT = "eagle_harness.sglang.gdn_tree_verify:GDNTreeVerify"
_HARNESS_MAMBA2_EXT = "eagle_harness.sglang.mamba2_tree_verify:Mamba2TreeVerify"
_GRAPH_RUNNER_INFO = "sglang.srt.model_executor.runner.decode_graph_variants:DecodeGraphRunnerInfo"
_EXT_CONV = (
    "layer",
    "mixed_qkv",
    "conv_states",
    "cache_indices",
    "intermediate_conv_window",
    "intermediate_state_indices",
)
_EXT_RECURRENT = (
    "layer",
    "query",
    "key",
    "value",
    "a",
    "b",
    "ssm_states",
    "cache_indices",
    "query_start_loc",
    "intermediate_states",
    "intermediate_state_indices",
)

_RUN_VERIFY = (
    "batch",
    "target_worker",
    "req_to_token_pool",
    "token_to_kv_pool_allocator",
    "plan_stream",
    "plan_stream_ctx",
    "topk",
    "num_draft_tokens",
    "device",
    "metadata_ready_pre_pad",
    "finalize_tree_path",
    "grammar_barrier",
    "accept_step",
)
_ACCEPT_INPUT = (
    "verify_input",
    "next_token_ids",
    "next_token_logits",
    "hidden_states",
    "seq_lens",
    "req_pool_indices",
    "out_cache_loc",
    "can_run_cuda_graph",
)
_ACCEPT_RESULT = ("next_token_ids", "accept_lens", "accept_index", "new_seq_lens", "bonus_tokens", "hidden_states")


class Hook(NamedTuple):
    path: str  # "module:Class.attribute" in sglang
    params: tuple[str, ...]  # parameter names after self / cls
    kind: str  # OVERRIDE, CALL or FIELD
    owner: str = ""  # "module:Class" in this package that overrides it (OVERRIDE only)
    optional: bool = False  # an absent capability keeps the existing fallback


HOOKS: tuple[Hook, ...] = (
    # Registration and the algorithm's EAGLE-family membership.
    Hook(
        "sglang.srt.speculative.spec_info:SpeculativeAlgorithm.register",
        ("name", "supports_overlap", "spec_class"),
        CALL,
    ),
    Hook("sglang.srt.speculative.spec_info:SpeculativeAlgorithm.from_string", ("name",), CALL),
    Hook(f"{_ALGO}.is_eagle", (), OVERRIDE, _HARNESS_ALGO),
    Hook(f"{_ALGO}.uses_aux_hidden_states", (), OVERRIDE, _HARNESS_ALGO),
    Hook(f"{_ALGO}.is_eagle3", (), OVERRIDE, _HARNESS_ALGO),
    Hook(f"{_ALGO}.handle_server_args", ("server_args",), OVERRIDE, _HARNESS_ALGO),
    # H2 request admission, H5 req_to_token headroom, H11 algorithm config.
    Hook(f"{_ALGO}.supports_mm_inputs", (), OVERRIDE, _HARNESS_ALGO),
    Hook(f"{_ALGO}.req_to_token_extra_len", ("page_size",), OVERRIDE, _HARNESS_ALGO),
    Hook("sglang.srt.arg_groups.fields.spec:Spec.speculative_algorithm_config", (), FIELD),
    # H7 per-width verify graphs, H10 GDN / Mamba2 verify extensions.
    Hook(f"{_ALGO}.create_decode_graph_variants", ("info",), OVERRIDE, _HARNESS_ALGO),
    Hook(f"{_GRAPH_RUNNER_INFO}.is_draft_worker", (), FIELD),
    Hook(f"{_GRAPH_RUNNER_INFO}.device", (), FIELD),
    Hook(f"{_ALGO}.create_linear_attn_verify_extension", ("state",), OVERRIDE, _HARNESS_ALGO),
    Hook(f"{_EXT}.init_cuda_graph_state", ("max_bs", "max_num_tokens"), OVERRIDE, _HARNESS_EXT),
    Hook(f"{_EXT}.begin_forward", ("layout", "in_graph"), OVERRIDE, _HARNESS_EXT),
    Hook(f"{_EXT}.reads_strided_inputs", (), OVERRIDE, _HARNESS_EXT),
    Hook(f"{_EXT}.conv_update", _EXT_CONV, OVERRIDE, _HARNESS_EXT),
    Hook(f"{_EXT}.recurrent_update", _EXT_RECURRENT, OVERRIDE, _HARNESS_EXT),
    Hook(
        f"{_EXT}.commit_state",
        ("state", "state_indices", "record_indices", "commit_steps", "parent"),
        OVERRIDE,
        _HARNESS_EXT,
        optional=True,
    ),
    Hook(f"{_EXT}.init_cuda_graph_state", ("max_bs", "max_num_tokens"), OVERRIDE, _HARNESS_MAMBA2_EXT),
    Hook(f"{_EXT}.begin_forward", ("layout", "in_graph"), OVERRIDE, _HARNESS_MAMBA2_EXT),
    Hook(f"{_EXT}.conv_update", _EXT_CONV, OVERRIDE, _HARNESS_MAMBA2_EXT),
    # Verify input: greedy tokens from a vocab-parallel argmax, the draft's KV slots, the tail-mask FlashInfer variant.
    Hook(f"{_SPEC_INPUT}.greedy_tokens_only", (), FIELD),
    Hook("sglang.srt.layers.logits_processor:LogitsProcessorOutput.next_token_ids", (), FIELD),
    Hook(f"{_VERIFY_INPUT}.prepared_out_cache_loc", (), FIELD),
    Hook(
        f"{_SPEC_INPUT}.flashinfer_verify_variant",
        ("bs", "num_tokens", "q_dtype", "kv_dtype", "head_dim"),
        OVERRIDE,
        "eagle_harness.sglang.verify_input:HarnessVerifyInput",
    ),
    # The verify (worker.py) and the fused accept step plugged into it (verify_step.py); sglang commits the states.
    Hook(
        "sglang.srt.speculative.eagle_worker_common:run_eagle_verify",
        _RUN_VERIFY,
        CALL,
    ),
    Hook("sglang.srt.speculative.eagle_worker_common:VerifyAcceptInput.__init__", _ACCEPT_INPUT, CALL),
    Hook("sglang.srt.speculative.eagle_worker_common:VerifyAcceptResult.__init__", _ACCEPT_RESULT, CALL),
    # The EAGLE V2 worker the cost-model worker extends (prefill, pools, backends stay sglang's).
    Hook(f"{_WORKER}.__init__", ("server_args", "gpu_id", "nccl_port", "target_worker"), CALL),
    Hook(
        f"{_WORKER}.create_draft_worker",
        ("server_args", "gpu_id", "nccl_port", "target_worker"),
        OVERRIDE,
        _HARNESS_WORKER,
    ),
    Hook(f"{_WORKER}.init_attention_backends", (), OVERRIDE, _HARNESS_WORKER),
    Hook(f"{_WORKER}._build_trivial_verify_input", ("batch",), OVERRIDE, _HARNESS_WORKER),
    Hook(
        f"{_WORKER}.forward_batch_generation",
        ("batch", "on_publish", "grammar_barrier", "pp_proxy_tensors"),
        OVERRIDE,
        _HARNESS_WORKER,
    ),
    Hook(f"{_DRAFT}.__init__", ("server_args", "gpu_id", "nccl_port", "target_worker"), CALL),
    Hook(f"{_DRAFT}.init_attention_backend", (), OVERRIDE, _HARNESS_DRAFT),
    Hook(f"{_DRAFT}._capture_cuda_graphs", (), OVERRIDE, _HARNESS_DRAFT),
)


def resolve(path: str):
    """The object at ``"module:Class.attribute"``; raises ImportError / AttributeError."""
    module_name, _, attr_path = path.partition(":")
    obj = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def parameter_names(func) -> tuple[str, ...]:
    """Parameter names of ``func`` after ``self`` / ``cls`` (``*args`` / ``**kwargs`` excluded)."""
    params = [
        p
        for p in inspect.signature(func).parameters.values()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
    return tuple(p.name for p in params)


def _problem(hook: Hook) -> str | None:
    if hook.kind == FIELD:
        cls_path, _, name = hook.path.rpartition(".")
        try:
            cls = resolve(cls_path)
            fields = (*getattr(cls, "__struct_fields__", ()), *getattr(cls, "__dataclass_fields__", ()))
        except (ImportError, AttributeError):
            return f"{cls_path} is missing"
        return None if name in fields or name in vars(cls) else f"{hook.path} is missing"
    try:
        found = parameter_names(resolve(hook.path))
    except (ImportError, AttributeError):
        return None if hook.optional else f"{hook.path} is missing"
    if hook.kind == OVERRIDE and found != hook.params:
        return f"{hook.path}{found} is not {hook.params}"
    if hook.kind == CALL and not set(hook.params) <= set(found):
        return f"{hook.path}{found} lacks {tuple(p for p in hook.params if p not in found)}"
    return None


def hook_problems() -> list[str]:
    """What the installed sglang lacks or has changed among :data:`HOOKS` (empty: everything matches)."""
    return [problem for problem in map(_problem, HOOKS) if problem is not None]
