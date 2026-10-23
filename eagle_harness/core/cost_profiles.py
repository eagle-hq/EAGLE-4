# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
"""Packaged serving costs and choices for measured device and checkpoint configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

MI355X_PROFILE = Path(__file__).parent / "profiles" / "mi355x-qwen3.8-27b-tp1"
B300_SERVED_PROFILE = Path(__file__).parent / "profiles" / "b300-qwen3.8-27b-tp4-chain"


def _measured_on(
    profile: Path,
    devices: Sequence[str],
    tp: int,
    *,
    device_name: str,
    model_path: str,
    draft_model_path: str,
    tp_size: int,
    dtype: str,
    quantization: Optional[str],
) -> bool:
    """Whether a setup is the one ``profile`` was measured on: one of ``devices``, ``tp`` ranks, BF16 without
    quantization, and the target and draft ``config.json`` of the profile (parsed: directory names do not matter)."""
    if device_name not in devices or tp_size != tp:
        return False
    if dtype not in ("auto", "bfloat16", "torch.bfloat16") or quantization is not None:
        return False
    for model, name in ((model_path, "target"), (draft_model_path, "draft")):
        if not model:
            return False
        config_path = Path(model) / "config.json"
        if not config_path.is_file():
            return False
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
        with (profile / f"{name}_config.json").open(encoding="utf-8") as file:
            measured = json.load(file)
        if config != measured:
            return False
    return True


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

    Unknown devices, unresolved Hub paths, different configs, TP sizes or dtypes keep the caller's
    ordinary calibration path; a profile for one model must not price another.
    """
    if _measured_on(
        MI355X_PROFILE,
        ("AMD Instinct MI355X", "AMD Instinct MI355X VF"),
        1,
        device_name=device_name,
        model_path=model_path,
        draft_model_path=draft_model_path,
        tp_size=tp_size,
        dtype=dtype,
        quantization=quantization,
    ):
        return str(MI355X_PROFILE)
    return None


def packaged_served_choices(
    kind: str,
    *,
    device_name: str,
    model_path: str,
    draft_model_path: str,
    tp_size: int,
    dtype: str,
    quantization: Optional[str],
) -> Optional[Dict[int, int]]:
    """The served choice per batch key measured on this setup, or None: for a ``"chain"`` the verify width, for a
    ``"tree"`` 1 to speculate (the cost model's tree) or 0 to decode natively.

    Only the B300 BF16 TP4 setup of the profile's checkpoint pair has them. A served chain or tree
    (:class:`~eagle_harness.sglang.chain_rounds.ServedChainPolicy`, ``ServedTreePolicy``) starts from them where it
    has no choices of its own.
    """
    if not _measured_on(
        B300_SERVED_PROFILE,
        ("NVIDIA B300 SXM6 AC",),
        4,
        device_name=device_name,
        model_path=model_path,
        draft_model_path=draft_model_path,
        tp_size=tp_size,
        dtype=dtype,
        quantization=quantization,
    ):
        return None
    with (B300_SERVED_PROFILE / f"{kind}.json").open(encoding="utf-8") as file:
        return {int(key): int(choice) for key, choice in json.load(file)[kind].items()}
