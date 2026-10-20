# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Target model properties shared by engine cost policies."""


def target_is_moe(hf_config: object) -> bool:
    """Detect routed experts from the text model's config, without architecture-name heuristics."""
    config = getattr(hf_config, "text_config", None) or hf_config
    return any((getattr(config, name, 0) or 0) > 0 for name in ("num_experts", "num_local_experts", "n_routed_experts"))
