# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Reader for the calibrated target and draft cost tables.

On-disk layout::

    <data_path>/cost/bs_<N>/model_llm/model_interval_<i>.json   (target verify)
    <data_path>/cost/bs_<N>/model_slm/model_interval_<i>.json   (draft step)

Each interval file holds ``value`` / ``ratio`` / ``index`` / ``array_query`` /
``array_cache`` lists plus the grid fields written by
:func:`eagle_harness.core.calibration.fit_model_costs`.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from eagle_harness.core.live_costs import CostKey

logger = logging.getLogger(__name__)

# What the cost model's decisions need of a table (its own checks raise mid-serving otherwise): the target / draft
# cost ratio it prices a draft step by, and the verify cost at 8 tokens relative to 1 token (its batch size class).
MODEL_RATIO_RANGE = (5.0, 100.0)
VERIFY8_RATIO_RANGE = (0.70, 4.0)  # 0.85 for dense targets; the cost model applies the exact bound


@dataclass(frozen=True)
class CostCurve:
    """Cost of one model at one context interval, per measured query length.

    ``index[i]`` is the query length minus one that ``value[i]`` (microseconds)
    and ``ratio[i]`` (``value[i] / value[0]``) were measured at.
    """

    value: Tuple[float, ...]
    ratio: Tuple[float, ...]
    index: Tuple[int, ...]
    array_query: Tuple[float, ...]
    array_cache: Tuple[float, ...]

    @classmethod
    def from_json(cls, data: dict) -> "CostCurve":
        """A curve from an on-disk JSON object."""
        return cls(
            value=tuple(data["value"]),
            ratio=tuple(data["ratio"]),
            index=tuple(data["index"]),
            array_query=tuple(data["array_query"]),
            array_cache=tuple(data["array_cache"]),
        )


@dataclass(frozen=True)
class IntervalCost:
    """Target and draft cost curves of one cache-length interval."""

    target: CostCurve  # "llm" in the cost-table files
    draft: CostCurve  # "slm" in the cost-table files


@dataclass(frozen=True)
class BatchCostTable:
    """The cost curves of one batch size over the cache-length intervals."""

    batch_size: int
    cache_length_start: int
    cache_length_step: int
    target_max_qo_len_query: int
    target_max_qo_len_cache: int
    draft_max_qo_len: int
    intervals: Tuple[IntervalCost, ...]
    # target.value[0] / draft.value[0] per interval; -1.0 when the draft cost is 0.
    model_ratio: Tuple[float, ...]
    _warned_context_clamp: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def max_interval_id(self) -> int:
        """Index of the last cache-length interval."""
        return len(self.intervals) - 1

    def interval_index(self, average_cache_len: int) -> int:
        """Context interval for an average cached length, clamped to the table."""
        step = self.cache_length_step
        index = (average_cache_len - step) // step
        if index > self.max_interval_id and not self._warned_context_clamp:
            logger.warning(
                "Cost table for batch size %d clamps context %d at calibration context_stop=%d; "
                "recalibrate for the served max_model_len.",
                self.batch_size,
                average_cache_len,
                (len(self.intervals) + 1) * step,
            )
            object.__setattr__(self, "_warned_context_clamp", True)
        return min(max(index, 0), self.max_interval_id)


def _read_intervals(path: str) -> Dict[int, dict]:
    intervals: Dict[int, dict] = {}
    try:
        names = os.listdir(path)
    except FileNotFoundError:
        return intervals
    for name in names:
        if not (name.startswith("model_interval_") and name.endswith(".json")):
            continue
        try:
            interval_id = int(name.split("_")[-1].split(".")[0])
            with open(os.path.join(path, name), encoding="utf-8") as f:
                intervals[interval_id] = json.load(f)
        except (ValueError, IndexError, json.JSONDecodeError):
            logger.warning("Could not parse cost interval file %s", name)
    return intervals


def _load_batch(batch_dir: str, batch_size: int) -> BatchCostTable:
    target = _read_intervals(os.path.join(batch_dir, "model_llm"))
    draft = _read_intervals(os.path.join(batch_dir, "model_slm"))
    if not target and not draft:
        raise RuntimeError(f"No cost intervals for batch size {batch_size} in {batch_dir}.")
    max_interval_id = max(max(target, default=-1), max(draft, default=-1))
    intervals: List[IntervalCost] = []
    ratios: List[float] = []
    for i in range(max_interval_id + 1):
        llm, slm = target.get(i), draft.get(i)
        if not llm or not slm:
            raise RuntimeError(
                f"Missing interval {i} for batch size {batch_size}: target={llm is not None}, draft={slm is not None}."
            )
        intervals.append(IntervalCost(CostCurve.from_json(llm), CostCurve.from_json(slm)))
        llm_v0 = llm["value"][0] if llm["value"] else 0
        slm_v0 = slm["value"][0] if slm["value"] else 0
        ratios.append(round(llm_v0 / slm_v0, 4) if slm_v0 > 0 else -1.0)
    first_llm = next(iter(target.values()))
    first_slm = next(iter(draft.values()))
    return BatchCostTable(
        batch_size=batch_size,
        cache_length_start=first_llm["cache_length_start"],
        cache_length_step=first_llm["cache_length_step"],
        target_max_qo_len_query=first_llm["max_query_length_query"],
        target_max_qo_len_cache=first_llm["max_query_length_cache"],
        draft_max_qo_len=first_slm["max_query_length_query"],
        intervals=tuple(intervals),
        model_ratio=tuple(ratios),
    )


def table_problem(table: BatchCostTable) -> Optional[str]:
    """Why ``table`` is not a cost table, None when it is: every curve finite and positive with a strictly
    increasing query index from 0 (what every consumer, the depth stop included, reads)."""
    for i, interval in enumerate(table.intervals):
        for name, curve in (("target", interval.target), ("draft", interval.draft)):
            if not curve.value or len(curve.value) != len(curve.index) or len(curve.ratio) != len(curve.index):
                return f"interval {i}: {name} curve has mismatched lengths"
            if any(not math.isfinite(v) or v <= 0 for v in curve.value):
                return f"interval {i}: {name} cost is not finite and positive: {curve.value}"
            if curve.index[0] != 0 or any(b <= a for a, b in zip(curve.index, curve.index[1:])):
                return f"interval {i}: {name} query index must increase from 0: {curve.index}"
    return None


def cost_model_problem(table: BatchCostTable, *, target_is_moe: bool = False) -> Optional[str]:
    """Why the cost model could not decide from ``table``, None when it can: the target / draft cost ratio in
    :data:`MODEL_RATIO_RANGE` and the target's cost at 8 tokens relative to 1 token in :data:`VERIFY8_RATIO_RANGE`
    (the model's own checks raise in a decode round otherwise)."""
    problem = table_problem(table)
    if problem is not None:
        return problem
    for i, interval in enumerate(table.intervals):
        ratio = table.model_ratio[i]
        if not MODEL_RATIO_RANGE[0] <= ratio <= MODEL_RATIO_RANGE[1]:
            return f"interval {i}: target / draft cost ratio {ratio} outside {MODEL_RATIO_RANGE}"
        target = interval.target
        at8 = next((target.ratio[j] for j, idx in enumerate(target.index) if idx >= 7), target.ratio[-1])
        bounds = (VERIFY8_RATIO_RANGE[0] if target_is_moe else 0.85, VERIFY8_RATIO_RANGE[1])
        if not bounds[0] <= at8 <= bounds[1]:
            return f"interval {i}: target cost at 8 tokens / at 1 token = {at8} outside {bounds}"
    return None


def load_cost_tables(cost_dir: str, key: Optional["CostKey"] = None) -> Dict[int, BatchCostTable]:
    """Load every ``bs_<N>`` table under ``cost_dir``; empty if it does not exist.

    A batch directory missing ``model_llm`` or ``model_slm`` is skipped with a
    warning; an incomplete interval set, or a table the cost model could
    not decide from (:func:`table_problem`), is an error. With ``key``, the tables' ``key.json`` (in the parent
    of ``cost_dir``) must name the same setup (:func:`eagle_harness.core.live_costs.check_key`).
    """
    try:
        entries = os.listdir(cost_dir)
    except FileNotFoundError:
        logger.error("Cost data path not found: %s", cost_dir)
        return {}
    if key is not None:
        from eagle_harness.core.live_costs import check_key

        check_key(os.path.dirname(os.path.normpath(cost_dir)), key)
    tables: Dict[int, BatchCostTable] = {}
    for entry in entries:
        batch_dir = os.path.join(cost_dir, entry)
        if not (entry.startswith("bs_") and os.path.isdir(batch_dir)):
            continue
        try:
            batch_size = int(entry.split("_")[1])
        except (IndexError, ValueError) as e:
            raise RuntimeError(f"Unexpected batch directory name: {entry}") from e
        if not (
            os.path.isdir(os.path.join(batch_dir, "model_llm")) and os.path.isdir(os.path.join(batch_dir, "model_slm"))
        ):
            logger.warning("Missing model cost directory for batch size %d; skipped.", batch_size)
            continue
        table = _load_batch(batch_dir, batch_size)
        problem = table_problem(table)
        if problem is not None:
            raise RuntimeError(f"Cost tables under {batch_dir} cannot drive the cost model: {problem}.")
        tables[batch_size] = table
        logger.debug("Loaded cost data for batch size %d", batch_size)
    return tables
