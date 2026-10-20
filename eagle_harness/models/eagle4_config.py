# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-4 draft config validation and sliding-window resolution.

Checks supported keys and structural constraints in a transformers config.
Pure Python: no engine, no GPU.

Sliding-window semantics, as trained: at draft depth d a query attends to at
most W context keys ending at it plus the d draft tokens of its own chain.
Activation: an explicit ``layer_types`` wins, then ``use_sliding_window``, and
a positive ``sliding_window`` alone enables the window. Single-layer drafts only.
"""

from __future__ import annotations

from numbers import Integral
from typing import Mapping, Optional

from eagle_harness.models.eagle4_structure import LATENT_MOE_ARCHITECTURE, draft_moe_architecture, draft_uses_rope

EAGLE4_ARCH = "LlamaForCausalLMEagle4"
LAYER_TYPES = ("full_attention", "sliding_attention")


def resolve_draft_sliding_window(config) -> Optional[int]:
    """Return the sliding window of an EAGLE4 draft config, or None.

    Non-EAGLE4 configs always return None. Invalid window settings raise.
    """
    architectures = getattr(config, "architectures", None) or ()
    if EAGLE4_ARCH not in architectures:
        return None

    window = getattr(config, "sliding_window", None)
    if window is not None:
        if isinstance(window, bool) or not isinstance(window, Integral) or window < 1:
            raise ValueError(f"EAGLE4 draft sliding_window must be a positive integer, got {window!r}.")
        window = int(window)

    enabled = getattr(config, "use_sliding_window", None)
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError(f"EAGLE4 draft use_sliding_window must be a boolean, got {enabled!r}.")

    num_layers = getattr(config, "num_hidden_layers", None)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        if not isinstance(layer_types, (list, tuple)) or len(layer_types) != num_layers:
            raise ValueError(
                "EAGLE4 draft layer_types must list one entry per draft layer, got "
                f"{layer_types!r} for num_hidden_layers={num_layers}."
            )
        unknown = [kind for kind in layer_types if kind not in LAYER_TYPES]
        if unknown:
            raise ValueError(f"EAGLE4 draft layer_types supports only {list(LAYER_TYPES)}, got {unknown!r}.")
        if "sliding_attention" in layer_types and window is None:
            raise ValueError("EAGLE4 draft layer_types has sliding_attention but no positive sliding_window.")
        enabled = layer_types[0] == "sliding_attention"
    elif enabled is None:
        enabled = window is not None
    if enabled and window is None:
        raise ValueError("EAGLE4 draft use_sliding_window requires a positive sliding_window.")
    if not enabled:
        return None
    if num_layers != 1:
        raise ValueError(f"EAGLE4 sliding-window drafts must have num_hidden_layers=1, got {num_layers}.")
    return window


# Config keys an EAGLE4 draft may carry on top of the standard LlamaConfig
# attributes. Every key that selects draft structure is read by this file (or
# by the EAGLE workers); anything outside both sets fails construction, so a
# new draft structure cannot be dropped silently.
_STRUCTURAL_KEYS = frozenset(
    {
        # aux-hidden-state input
        "draft_vocab_size",
        "target_hidden_size",
        "num_aux_hidden_states",
        "eagle_config",
        "share_draft_embedding_from_target",
        "fc_norm",
        "enable_fc_norm",
        "bias",
        # layer-0 input fusion
        "input_fuse_type",
        "attn_norm",
        # attention
        "use_rope",
        "position_embedding_type",
        "partial_rotary_factor",
        "rope_is_neox_style",
        "sliding_window",
        "use_sliding_window",
        "layer_types",
        # output norms
        "norm_output",
        "enable_post_mlp_norm",
        # MoE
        "enable_moe",
        "moe_architecture",
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "n_shared_experts",
        "moe_shared_expert_intermediate_size",
        "moe_latent_size",
        "n_group",
        "topk_group",
        "norm_topk_prob",
        "routed_scaling_factor",
        "scoring_func",
        "topk_method",
        "expert_bias_update_rate",
        "moe_router_bias_update_rate",
        # candidate lm_head
        "efficient_lm_head_config",
        # retired structures, accepted only when off
        "enable_2hs",
        "enable_defa",
    }
)
# Hyperparameters and provenance keys: no effect on the served structure.
# rope_interleave / max_window_layers / moe_router_dtype are copied from the
# target config and do not change the draft structure.
_NON_STRUCTURAL_KEYS = frozenset(
    {
        "_torchspec_version",
        "load_balance_loss_alpha",
        "seq_aux_loss_alpha",
        "aux_loss_alpha",
        "moe_seq_aux_loss_coeff",
        "expert_parallel_size",
        "moe_expert_parallel_size",
        "moe_initialization",
        "moe_mtp_layer_index",
        "moe_mtp_source_revision",
        "moe_router_dtype",
        "rope_interleave",
        "max_window_layers",
        "quantization_config",
    }
)
# Attributes sglang's ModelConfig sets on every hf_config.
_RUNTIME_KEYS = frozenset({"context_len", "encoder_only", "language_only", "language_model_only"})
# ModelConfig also fills these in from head_dim. The draft attention uses
# head_dim for q, k and v, so a config may carry them only at that value.
_DERIVED_HEAD_DIM_KEYS = frozenset({"v_head_dim", "swa_head_dim", "swa_v_head_dim"})
_DENSE_MOE_ARCHITECTURES = ("sparse_swiglu",)
_EFFICIENT_LM_HEAD_KEYS = frozenset({"type", "rank", "num_candidates", "smooth_factor"})


def _expect(config, name: str, allowed: tuple) -> None:
    """If the config sets ``name``, its value must be one of ``allowed``."""
    if not hasattr(config, name):
        return
    value = getattr(config, name)
    # `is` for booleans: 1 == True must not pass as a boolean flag.
    if isinstance(value, bool) or any(isinstance(a, bool) for a in allowed):
        ok = any(value is a for a in allowed)
    else:
        ok = value in allowed
    if not ok:
        raise ValueError(f"EAGLE4 draft config {name}={value!r} is not supported (expected one of {list(allowed)}).")


def validate_eagle4_draft_config(config) -> None:
    """Refuse EAGLE4 draft configs whose structure this model does not build."""
    standard = set(type(config)().to_dict())
    keys = set(config.to_dict()) - standard - _RUNTIME_KEYS - _DERIVED_HEAD_DIM_KEYS
    unknown = sorted(keys - _STRUCTURAL_KEYS - _NON_STRUCTURAL_KEYS)
    if unknown:
        raise ValueError(
            f"EAGLE4 draft config has keys this model does not implement: {unknown}. "
            "A key that changes the draft structure needs support in "
            "models/llama_eagle4.py; a non-structural key goes into its "
            "_NON_STRUCTURAL_KEYS."
        )
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    for name in sorted(_DERIVED_HEAD_DIM_KEYS):
        value = getattr(config, name, None)
        if value and value != head_dim:
            raise ValueError(
                f"EAGLE4 draft config {name}={value} differs from head_dim={head_dim}; "
                "the draft attention uses one head dim for q, k and v."
            )

    for name in (
        "fc_norm",
        "enable_fc_norm",
        "attn_norm",
        "use_rope",
        "norm_output",
        "enable_post_mlp_norm",
        "enable_moe",
        "share_draft_embedding_from_target",
        "norm_topk_prob",
    ):
        _expect(config, name, (True, False))
    for name in ("bias", "attention_bias", "mlp_bias", "enable_2hs", "enable_defa"):
        _expect(config, name, (False,))
    _expect(config, "rope_is_neox_style", (True,))
    _expect(config, "partial_rotary_factor", (None, 0, 1))
    _expect(config, "input_fuse_type", ("concat",))
    _expect(config, "num_aux_hidden_states", (3,))
    _expect(config, "moe_architecture", _DENSE_MOE_ARCHITECTURES + (LATENT_MOE_ARCHITECTURE,))

    eagle_config = getattr(config, "eagle_config", None) or {}
    extra = set(eagle_config) - {"use_aux_hidden_state", "eagle_aux_hidden_state_layer_ids"}
    layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
    if extra or (layer_ids is not None and len(layer_ids) != 3):
        raise ValueError(
            f"EAGLE4 draft eagle_config={eagle_config!r} is not supported: the fc "
            "input takes exactly three aux hidden states."
        )

    head = getattr(config, "efficient_lm_head_config", None)
    if head is not None and (
        not isinstance(head, dict)
        or head.get("type") not in ("low_rank", "weighted_low_rank")
        or set(head) - _EFFICIENT_LM_HEAD_KEYS
    ):
        raise ValueError(f"EAGLE4 draft efficient_lm_head_config={head!r} is not supported.")

    moe_architecture = draft_moe_architecture(config)
    latent_moe = moe_architecture == LATENT_MOE_ARCHITECTURE
    _expect(config, "hidden_act", ("relu2",) if latent_moe else ("silu",))
    if moe_architecture is not None:
        # The served routers are sigmoid over bias-corrected scores.
        _expect(config, "scoring_func", ("sigmoid",))
        _expect(config, "topk_method", ("noaux_tc",))
    if latent_moe:
        # The engine block reads its expert count, top-k and widths from the checkpoint (no harness knob).
        required = ["n_routed_experts", "num_experts_per_tok", "moe_intermediate_size", "moe_latent_size"]
        if getattr(config, "n_shared_experts", 0):
            required.append("moe_shared_expert_intermediate_size")
        missing = [name for name in required if not getattr(config, name, None)]
        if missing:
            raise ValueError(f"EAGLE4 latent MoE draft config is missing {missing}.")

    single_layer_only = {
        "sliding window": resolve_draft_sliding_window(config) is not None,
        "attn_norm": getattr(config, "attn_norm", False),
        "NoPE": not draft_uses_rope(config),
        "latent MoE": latent_moe,
    }
    used = [name for name, on in single_layer_only.items() if on]
    if used and config.num_hidden_layers != 1:
        raise ValueError(
            f"EAGLE4 draft structures {used} are implemented for single-layer "
            f"drafts, got num_hidden_layers={config.num_hidden_layers}."
        )


# Structures the config validator accepts that an adapter does not build unless it passes its own set
# (sglang builds attn_norm, NoPE and the latent MoE; vLLM builds dense one-layer drafts with rotary positions).
# All ports are one-layer drafts with the low-rank candidate head.
UNPORTED = {
    "enable_moe": True,
    "attn_norm": True,
    "enable_post_mlp_norm": True,
    "use_rope": False,
    "position_embedding_type": "none",
    "partial_rotary_factor": 0,
}


def check_ported(config, unported: Mapping[str, object] = UNPORTED) -> None:
    """Validate the config and refuse draft structures the adapters do not build.

    ``unported`` maps a config key to the value that is not built.
    """
    validate_eagle4_draft_config(config)
    used = [f"{key}={value!r}" for key, value in unported.items() if getattr(config, key, None) == value]
    if used:
        raise NotImplementedError(
            f"EAGLE-4 drafts with {used} are not ported to eagle-harness yet (dense drafts only)."
        )
    if config.num_hidden_layers != 1:
        raise NotImplementedError(f"EAGLE-4 drafts with {config.num_hidden_layers} layers are not ported (1 only).")
    if getattr(config, "efficient_lm_head_config", None) is None:
        raise NotImplementedError("EAGLE-4 drafts without efficient_lm_head_config are not ported.")
    if (getattr(config, "draft_vocab_size", None) or config.vocab_size) != config.vocab_size:
        raise NotImplementedError("EAGLE-4 drafts with a reduced vocab (d2t) are not ported.")
