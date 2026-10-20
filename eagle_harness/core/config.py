# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""HarnessConfig: every EAGLE-Harness option in one typed object.

sglang passes the raw ``--speculative-algorithm-config`` string through unparsed.
:func:`parse_harness_config` accepts inline JSON or a path to a JSON file.
Configuration does not read environment variables.

Example::

    --speculative-algorithm EAGLE_HARNESS \\
    --speculative-algorithm-config '{"data_path": "/data/cost/qwen38-tp4", "intensity": "high_intensity"}'

Unknown keys are rejected so a typo cannot silently fall back to a default.
The migration tables live in ``docs/migration.md``.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import typing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

from eagle_harness.core.cost_model import ChainCoef, CostModel, CostModelParams, depth_loss_ratio
from eagle_harness.core.cost_table import BatchCostTable, cost_model_problem, load_cost_tables
from eagle_harness.core.limits import ComputeIntensity

if TYPE_CHECKING:
    from eagle_harness.core.live_costs import CostKey


class Mode(str, enum.Enum):
    """Execution modes: static, serve and calibrate."""

    STATIC = "static"  # the fixed --speculative-num-steps / -eagle-topk / -num-draft-tokens tree, no cost model
    SERVE = "serve"
    CALIBRATE = "calibrate"

    @property
    def uses_cost_model(self) -> bool:
        """Whether rounds are sized by the cost model (every mode but ``static``)."""
        return self is not Mode.STATIC


class DraftFamily(str, enum.Enum):
    """The draft model family; it sets the cost model's depth loss."""

    EAGLE4 = "eagle4"
    EAGLE3 = "eagle3"
    EAGLE = "eagle"


@dataclass(frozen=True)
class DraftModelOptions:
    """EAGLE-4 draft model switches (legacy environment variables: ``SGLANG_EAGLE4_*``, ``SGLANG_EAGLE_DRAFT_EMBED_*``)."""

    cuda_candidate_kernels: bool = True
    embed_full: bool = True  # full draft embedding table on every TP rank
    embed_triton: bool = True  # draft embedding lookups on sglang's Triton kernel (a full table on the rank)


@dataclass(frozen=True)
class L2Warm:
    """Weight prefetches into L2 during the target-verify graphs' idle HBM windows
    (:mod:`eagle_harness.sglang.l2_warm`). Each is ``(share of the weight's bytes, CTAs)``, None: off."""

    gate_up_gdn: Optional[Tuple[float, int]] = (0.42, 16)  # MLP gate_up weight, after a GDN layer's input projection
    gate_up_attention: Optional[Tuple[float, int]] = (0.3, 74)  # ... after an attention layer's QKV projection
    down_proj: Optional[Tuple[float, int]] = None  # MLP down weight, after the mixer's output projection
    next_input: Optional[Tuple[float, int]] = None  # next layer's input projection, after the MLP's down projection


@dataclass(frozen=True)
class CostModelConfig:
    """Cost-model constants, configurable through JSON fields."""

    threshold: float = 2.5
    async_layer_ratio: float = 0.9
    # None: derived from the draft family and target (see cost_model.depth_loss_ratio).
    depth_loss_ratio: Optional[float] = None
    chain_tiny: Tuple[float, float, float] = (2.60, 0.24, 0.00)
    chain_small: Tuple[float, float, float] = (2.85, 0.24, 0.00)
    chain_medium: Tuple[float, float, float] = (2.54, 0.24, 0.70)
    chain_large: Tuple[float, float, float] = (1.55, 0.24, 0.00)
    chain_scale_tiny: float = 1.25
    chain_tiny_max_bs: int = 2
    chain_scale_small: float = 1.45
    chain_scale_medium: float = 2.00

    def params(self, depth_loss_ratio: float) -> CostModelParams:
        """The constants as the cost model's parameters, with the given depth loss ratio."""
        return CostModelParams(
            threshold=self.threshold,
            async_layer_ratio=self.async_layer_ratio,
            depth_loss_ratio=self.depth_loss_ratio if self.depth_loss_ratio is not None else depth_loss_ratio,
            chain_coef=tuple(
                ChainCoef(*c) for c in (self.chain_tiny, self.chain_small, self.chain_medium, self.chain_large)
            ),
            chain_scale_tiny=self.chain_scale_tiny,
            chain_tiny_max_bs=self.chain_tiny_max_bs,
            chain_scale_small=self.chain_scale_small,
            chain_scale_medium=self.chain_scale_medium,
        )


@dataclass(frozen=True)
class HarnessConfig:
    """The harness's whole configuration (``--speculative-algorithm-config``)."""

    data_path: Optional[str] = None  # offline calibration root (cost/, key.json); None: costs measured on served rounds
    cost_cache: Optional[str] = None  # directory of the live-cost cache files; None: ~/.cache/eagle-harness/costs
    mode: Mode = Mode.SERVE
    intensity: ComputeIntensity = ComputeIntensity.HIGH
    draft_family: DraftFamily = DraftFamily.EAGLE4
    cost_model: CostModelConfig = field(default_factory=CostModelConfig)
    draft_model: DraftModelOptions = field(default_factory=DraftModelOptions)
    l2_warm: L2Warm = field(default_factory=L2Warm)

    def validate(self) -> "HarnessConfig":
        """Check the config; return it."""
        if self.mode is Mode.CALIBRATE and not self.data_path:
            raise ValueError("calibrate mode needs data_path for its output")
        return self

    @property
    def cost_dir(self) -> str:
        """Directory of the fitted offline cost tables under ``data_path``."""
        assert self.data_path is not None
        return os.path.join(self.data_path, "cost")

    @property
    def cost_cache_dir(self) -> str:
        """Directory of the live-cost cache files."""
        return self.cost_cache or os.path.join(os.path.expanduser("~"), ".cache", "eagle-harness", "costs")


def build_cost_model(
    config: HarnessConfig,
    *,
    node_top_k: int,
    tree_depth: int,
    verify_num: int,
    target_is_moe: bool,
    tables: Optional[Dict[int, BatchCostTable]] = None,
    key: Optional["CostKey"] = None,
) -> CostModel:
    """The worker's cost model: with ``tables``, else the offline tables under ``config.data_path`` (checked
    against ``key``, the serving setup)."""
    loss = depth_loss_ratio(config.draft_family.value, target_is_moe=target_is_moe, intensity=config.intensity)
    if tables is None:
        tables = load_cost_tables(config.cost_dir, key)
        for bs, table in sorted(tables.items()):
            problem = cost_model_problem(table, target_is_moe=target_is_moe)
            if problem is not None:
                raise ValueError(f"Cost tables under {config.cost_dir}/bs_{bs} cannot drive the cost model: {problem}.")
    return CostModel(
        tables=tables,
        params=config.cost_model.params(loss),
        intensity=config.intensity,
        node_top_k=node_top_k,
        tree_depth=tree_depth,
        verify_num=verify_num,
        target_is_moe=target_is_moe,
    )


def _convert(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, Mapping):
            raise ValueError(f"{where}: expected an object, got {type(value).__name__}.")
        return _from_dict(tp, value, where)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        try:
            return tp(value)
        except ValueError as e:
            choices = ", ".join(m.value for m in tp)
            raise ValueError(f"{where}: {value!r} is not one of {choices}.") from e
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if value is None:
            return None
        return _convert(args[0], value, where)
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, (list, tuple)) or len(value) != len(args):
            raise ValueError(f"{where}: expected a list of {len(args)} values.")
        return tuple(_convert(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    if tp is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected true/false, got {value!r}.")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: expected an integer, got {value!r}.")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: expected a number, got {value!r}.")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ValueError(f"{where}: expected a string, got {value!r}.")
        return value
    raise TypeError(f"{where}: unsupported field type {tp!r}.")


# Options removed in 0.1.1: no engine adapter turned them on. A config that still sets one is refused by name.
REMOVED_IN_0_1_1 = (
    "mode: eval",
    "ragged",
    "cost_model.ragged_draft_relax",
    "cost_model.ragged_verify_threshold",
    "draft_model.moe_gate_acc",
)


def _from_dict(cls: type, data: Mapping[str, Any], where: str) -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    keys = [f"{where}.{k}" if where else k for k in data]
    if not where and data.get("mode") == "eval":
        keys.append("mode: eval")
    removed = [key for key in keys if key in REMOVED_IN_0_1_1]
    if removed:
        raise ValueError(f"{', '.join(removed)}: removed in eagle-harness 0.1.1 (no engine used it); delete it.")
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"{where or 'config'}: unknown keys {unknown}; valid keys are {sorted(names)}.")
    kwargs = {k: _convert(hints[k], v, f"{where}.{k}" if where else k) for k, v in data.items()}
    return cls(**kwargs)


def harness_config_from_dict(data: Mapping[str, Any]) -> HarnessConfig:
    """A validated :class:`HarnessConfig` from a (possibly partial) dict."""
    return _from_dict(HarnessConfig, data, "").validate()


def parse_harness_config(value: Optional[str], *, default_data_path: Optional[str] = None) -> HarnessConfig:
    """Parse JSON or a JSON file, filling an absent data_path from a device-selected profile."""
    if value is None or not value.strip():
        if default_data_path is not None:
            return harness_config_from_dict({"data_path": default_data_path})
        raise ValueError(
            "EAGLE_HARNESS needs --speculative-algorithm-config (JSON or a path to JSON), "
            'e.g. {"data_path": "<calibration dir>"} or {"mode": "static"}.'
        )
    text = value.strip()
    if not text.startswith(("{", "[")):
        with open(os.path.expanduser(text), encoding="utf-8") as f:
            text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"--speculative-algorithm-config is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("--speculative-algorithm-config must be a JSON object.")
    if (
        default_data_path is not None
        and data.get("data_path") is None
        and data.get("mode", Mode.SERVE.value) == Mode.SERVE.value
    ):
        data["data_path"] = default_data_path
    return harness_config_from_dict(data)


def config_to_dict(config: HarnessConfig) -> Dict[str, Any]:
    """The config as plain JSON-able values (enums by value, tuples as lists)."""

    def plain(v: Any) -> Any:
        """One value as plain JSON data."""
        if isinstance(v, enum.Enum):
            return v.value
        if isinstance(v, tuple):
            return [plain(x) for x in v]
        return v

    def walk(obj: Any) -> Any:
        """A dataclass tree as nested dicts."""
        return {
            f.name: (
                walk(getattr(obj, f.name))
                if dataclasses.is_dataclass(getattr(obj, f.name))
                else plain(getattr(obj, f.name))
            )
            for f in dataclasses.fields(obj)
        }

    return walk(config)
