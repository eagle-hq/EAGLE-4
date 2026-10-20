# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
"""Packaged serving costs for measured device and checkpoint configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

MI355X_PROFILE = Path(__file__).parent / "profiles" / "mi355x-qwen3.8-27b-tp1"


def packaged_cost_profile(
    *,
    device_name: str,
    model_path: str,
    draft_model_path: str,
    tp_size: int,
    dtype: str,
    quantization: Optional[str],
) -> Optional[str]:
    """Select the MI355X BF16 TP1 profile only for its measured checkpoint configs.

    Model directory names do not participate in identity. Unknown devices,
    unresolved Hub paths, different configs, TP sizes or dtypes keep the caller's
    ordinary calibration path; a profile for one model must not price another.
    """
    if device_name not in ("AMD Instinct MI355X", "AMD Instinct MI355X VF"):
        return None
    if tp_size != 1 or dtype not in ("auto", "bfloat16", "torch.bfloat16") or quantization is not None:
        return None
    for model, name in ((model_path, "target"), (draft_model_path, "draft")):
        if not model:
            return None
        config_path = Path(model) / "config.json"
        if not config_path.is_file():
            return None
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
        with (MI355X_PROFILE / f"{name}_config.json").open(encoding="utf-8") as file:
            measured = json.load(file)
        if config != measured:
            return None
    return str(MI355X_PROFILE)
