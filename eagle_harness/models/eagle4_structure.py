# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-4 structure decisions and config preparation, independent of engines and torch."""

from __future__ import annotations

import copy
from typing import Any

LATENT_MOE_ARCHITECTURE = "nemotron_latent_relu2"


def draft_moe_architecture(config: Any) -> str | None:
    """Return the enabled draft MoE architecture, or None for a dense draft."""
    if not getattr(config, "enable_moe", False):
        return None
    return getattr(config, "moe_architecture", "sparse_swiglu")


def draft_uses_rope(config: Any) -> bool:
    """RoPE is off if any of the checkpoint's three position settings disables it."""
    return not (
        getattr(config, "use_rope", True) is False
        or getattr(config, "position_embedding_type", None) == "none"
        or getattr(config, "partial_rotary_factor", None) == 0
    )


def base_layer_overrides(config: Any) -> dict[str, Any]:
    """Temporary settings for a Llama base layer whose MLP or RoPE will be replaced.

    Apply to a copy, or restore after construction. NoPE needs only a one-position
    placeholder because engines retain rotary tables in process-wide caches.
    """
    overrides = {}
    if draft_moe_architecture(config) == LATENT_MOE_ARCHITECTURE:
        overrides["hidden_act"] = "silu"
    if not draft_uses_rope(config):
        overrides["max_position_embeddings"] = 1
    return overrides


def latent_moe_config(config: Any) -> Any:
    """Shallow-copy a validated latent-MoE draft config with the engine block's defaults.

    Preserve the config type and checkpoint fields; module construction and
    parallelism checks belong to the engine adapter.
    """
    moe = copy.copy(config)
    moe.mlp_hidden_act = config.hidden_act
    moe.mlp_bias = False
    moe.n_shared_experts = int(getattr(config, "n_shared_experts", 0) or 0)
    moe.n_group = int(getattr(config, "n_group", 1) or 1)
    moe.topk_group = int(getattr(config, "topk_group", moe.n_group) or moe.n_group)
    moe.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
    moe.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))
    return moe
