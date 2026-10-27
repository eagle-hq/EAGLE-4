# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Pack caller-computed tree and chain costs into the device policy ABI.

Records are structural: callers need no engine base class. Policy formulas,
measurement selection and tier decisions belong to the producer, not here.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Protocol

import torch


class TreeIntervalCosts(Protocol):
    """One measured context interval, including this live batch's verify decision."""

    @property
    def verify_widths(self) -> tuple[int, ...]: ...

    @property
    def verify_cost_deltas(self) -> tuple[tuple[float, ...], ...]: ...

    @property
    def draft_cost(self) -> float: ...

    @property
    def model_cost_ratio(self) -> float: ...

    @property
    def depth_thresholds(self) -> tuple[tuple[float, ...], ...]: ...

    @property
    def verify_tier(self) -> int: ...

    @property
    def verify_thresholds(self) -> tuple[float, float]: ...


class TreeBatchCosts(Protocol):
    """Costs for one live batch; cost_batch_size identifies its measured row."""

    @property
    def cost_batch_size(self) -> int: ...

    @property
    def cache_length_step(self) -> int: ...

    @property
    def intervals(self) -> tuple[TreeIntervalCosts, ...]: ...


class TreePolicyCosts(Protocol):
    """Immutable host records for live batches 1..32 and five tier variants."""

    @property
    def batches(self) -> tuple[TreeBatchCosts, ...]: ...

    @property
    def history_floors(self) -> tuple[int, ...]: ...

    @property
    def initial_widths(self) -> tuple[int, ...]: ...

    @property
    def async_layer_ratio(self) -> float: ...


class ChainStopTerms(Protocol):
    """Core terms: gain_scale and cliff_gain_scale are denominators, not reciprocals."""

    @property
    def gain_scale(self) -> float: ...

    @property
    def cost(self) -> float: ...

    @property
    def threshold(self) -> float: ...

    @property
    def cliff_gain_scale(self) -> float: ...

    @property
    def cliff_cost(self) -> float: ...


class ChainPolicyBatch(Protocol):
    @property
    def cache_length_step(self) -> int: ...

    @property
    def intervals(self) -> tuple[tuple[ChainStopTerms, ...], ...]: ...


class ChainPolicyCosts(Protocol):
    """Core-computed terms indexed by live batch 1..512, interval and depth."""

    @property
    def batches(self) -> tuple[ChainPolicyBatch, ...]: ...


@dataclass(frozen=True)
class PolicyTables:
    batch_table: torch.Tensor  # int64 [max_bs, 5]; chain cost ID is live batch - 1
    interval_i32: torch.Tensor  # int32 [costs, intervals, 1 + measured widths]
    interval_f64: torch.Tensor  # float64 [costs, intervals, 2 + 5*depth + widths**2]
    tier_table: torch.Tensor  # int32 [9]: five history floors, four draft widths
    verify_i32: torch.Tensor  # int32 [32, intervals]: tier
    verify_f64: torch.Tensor  # float64 [32, intervals, 2]: stability thresholds
    chain_f64: torch.Tensor  # float64 [512, intervals, 2, depth, 5]; empty for trees


def _shared_row(batch: TreeBatchCosts) -> tuple:
    # Verify decisions depend on the live batch even when its measurement is shared.
    return (
        batch.cache_length_step,
        tuple(
            (
                interval.verify_widths,
                interval.verify_cost_deltas,
                interval.draft_cost,
                interval.model_cost_ratio,
                interval.depth_thresholds,
            )
            for interval in batch.intervals
        ),
    )


def _check_costs(costs: TreePolicyCosts) -> int:
    if len(costs.batches) != 32:
        raise ValueError("tree policy costs need 32 live batch records")
    if len(costs.history_floors) != 5 or len(costs.initial_widths) != 4:
        raise ValueError(
            "tree policy costs need five history floors and four initial widths"
        )
    if any(type(v) is not int or not 1 <= v < 2**31 for v in costs.history_floors):
        raise ValueError("history floors must be positive int32 values")
    if any(type(v) is not int or not 1 <= v <= 8 for v in costs.initial_widths):
        raise ValueError("initial widths must be integers in 1..8")
    if not math.isfinite(costs.async_layer_ratio) or costs.async_layer_ratio <= 0:
        raise ValueError("async_layer_ratio must be positive and finite")
    depth = None
    for batch in costs.batches:
        if type(batch.cost_batch_size) is not int or batch.cost_batch_size < 1:
            raise ValueError("measured batch sizes must be positive integers")
        if (
            type(batch.cache_length_step) is not int
            or not 0 < batch.cache_length_step < 2**63
        ):
            raise ValueError("cache_length_step must be a positive int64")
        if not batch.intervals:
            raise ValueError("each batch needs at least one interval")
        for interval in batch.intervals:
            widths = interval.verify_widths
            if (
                not 1 <= len(widths) <= 32
                or any(type(w) is not int or not 0 <= w < 2**31 - 1 for w in widths)
                or widths[0] != 0
                or any(a >= b for a, b in itertools.pairwise(widths))
            ):
                raise ValueError(
                    "interval needs 1..32 increasing widths starting at index 0"
                )
            if (
                not math.isfinite(interval.draft_cost)
                or interval.draft_cost <= 0
                or not math.isfinite(interval.model_cost_ratio)
                or not 5 <= interval.model_cost_ratio <= 100
            ):
                raise ValueError(
                    "interval needs positive finite draft cost and model ratio in [5, 100]"
                )
            thresholds = interval.depth_thresholds
            if len(thresholds) != 5 or not 2 <= len(thresholds[0]) <= 32:
                raise ValueError("depth thresholds need five variants of 2..32 depths")
            if depth is None:
                depth = len(thresholds[0])
            if any(
                len(row) != depth or any(not math.isfinite(v) for v in row)
                for row in thresholds
            ):
                raise ValueError(
                    "depth thresholds must be finite with the same depth in every interval"
                )
            deltas = interval.verify_cost_deltas
            if len(deltas) != len(widths) or any(
                len(row) != len(widths)
                or any(not math.isfinite(v) or v < 0 for v in row)
                for row in deltas
            ):
                raise ValueError(
                    "verify cost deltas must be a finite nonnegative square matrix over widths"
                )
            if not 0 <= interval.verify_tier <= 3:
                raise ValueError("verify tier must be in 0..3")
            if len(interval.verify_thresholds) != 2 or any(
                not math.isfinite(v) or v <= 0 for v in interval.verify_thresholds
            ):
                raise ValueError("verify thresholds must be two positive finite values")
    assert depth is not None
    return depth


def build_tables(costs: TreePolicyCosts, *, device: torch.device) -> PolicyTables:
    """Pack structural cost records on CPU and upload each completed tensor once.

    Args:
        costs: Records for live batches 1..32. Repeated measured batch IDs must
            carry identical shared interval data; verify decisions may differ.
        device: Destination device; CPU supports exact packing checks.

    Returns:
        Contiguous tables preserving the tree ABI and float64 values exactly.
        Padding and reserved chain columns remain zero; chain_f64 is empty.
    """
    depth = _check_costs(costs)
    unique: dict[int, TreeBatchCosts] = {}
    for batch in costs.batches:
        previous = unique.setdefault(batch.cost_batch_size, batch)
        if _shared_row(previous) != _shared_row(batch):
            raise ValueError(
                f"conflicting costs for measured batch {batch.cost_batch_size}"
            )
    ids = {measured: i for i, measured in enumerate(unique)}
    intervals = max(len(batch.intervals) for batch in unique.values())
    max_widths = max(
        len(row.verify_widths) for batch in unique.values() for row in batch.intervals
    )
    batch_table = torch.zeros((32, 5), dtype=torch.int64)
    interval_i32 = torch.zeros(
        (len(unique), intervals, 1 + max_widths), dtype=torch.int32
    )
    interval_f64 = torch.zeros(
        (len(unique), intervals, 2 + 5 * depth + max_widths**2), dtype=torch.float64
    )
    verify_i32 = torch.zeros((32, intervals), dtype=torch.int32)
    verify_f64 = torch.zeros((32, intervals, 2), dtype=torch.float64)
    for measured, batch in unique.items():
        for i, interval in enumerate(batch.intervals):
            n = len(interval.verify_widths)
            interval_i32[ids[measured], i, : 1 + n] = torch.tensor(
                [n, *interval.verify_widths], dtype=torch.int32
            )
            packed = interval_f64[ids[measured], i]
            packed[: 2 + 5 * depth] = torch.tensor(
                [
                    interval.draft_cost,
                    interval.model_cost_ratio,
                    *itertools.chain.from_iterable(interval.depth_thresholds),
                ],
                dtype=torch.float64,
            )
            packed[2 + 5 * depth :].view(max_widths, max_widths)[:n, :n] = torch.tensor(
                interval.verify_cost_deltas, dtype=torch.float64
            )
    for i, batch in enumerate(costs.batches):
        batch_table[i] = torch.tensor(
            [
                ids[batch.cost_batch_size],
                0,
                len(batch.intervals),
                batch.cache_length_step,
                0,
            ],
            dtype=torch.int64,
        )
        for j, interval in enumerate(batch.intervals):
            verify_i32[i, j] = interval.verify_tier
            verify_f64[i, j] = torch.tensor(
                interval.verify_thresholds, dtype=torch.float64
            )
    return PolicyTables(
        batch_table=batch_table.to(device),
        interval_i32=interval_i32.to(device),
        interval_f64=interval_f64.to(device),
        tier_table=torch.tensor(
            (*costs.history_floors, *costs.initial_widths),
            dtype=torch.int32,
            device=device,
        ),
        verify_i32=verify_i32.to(device),
        verify_f64=verify_f64.to(device),
        chain_f64=torch.zeros(0, dtype=torch.float64, device=device),
    )


def build_chain_tables(
    costs: ChainPolicyCosts | None,
    *,
    tree_depth: int,
    verify_num: int,
    chain_width_table: tuple[tuple[int, int], ...],
    device: torch.device,
) -> PolicyTables:
    """Pack chain terms unchanged; no cost model or tier classification runs here.

    Args:
        costs: Structural records for every live batch 1..512. Each interval
            has one term record per depth, zero at depths without a decision.
            ``gain_scale`` is the denominator in ``raw_gain * 2 / gain_scale``;
            the cliff pair is either both positive or both zero.
        tree_depth: Chain depth in 2..15, matching every interval.
        verify_num: Fixed verify capacity in 2..16.
        chain_width_table: Increasing positive ``(min_batch_size, width)`` pairs.
            The greatest covered threshold supplies the round's fixed width;
            uncovered batches use the confidence rule.
        device: Destination device; CPU supports packing and validation tests.

    Returns:
        Contiguous tables with cost ID ``live_batch - 1`` and tier row zero.
        Both ABI tier rows contain identical terms. Interval padding is zero;
        tree-only tensors are empty.
    """
    if (
        type(tree_depth) is not int
        or not 2 <= tree_depth <= 15
        or type(verify_num) is not int
        or not 2 <= verify_num <= 16
    ):
        raise ValueError("chain policy needs depth in 2..15 and verify_num in 2..16")
    previous = 0
    for entry in chain_width_table:
        if (
            not isinstance(entry, tuple)
            or len(entry) != 2
            or any(type(v) is not int for v in entry)
            or not previous < entry[0]
            or not 2 <= entry[1] <= min(tree_depth + 1, verify_num)
        ):
            raise ValueError(
                "chain width table needs increasing positive integer thresholds "
                "and widths in 2..min(depth + 1, verify_num)"
            )
        previous = entry[0]
    if costs is None:
        if not chain_width_table or chain_width_table[0][0] != 1:
            raise ValueError(
                "an uncalibrated chain width table must cover every batch starting at 1"
            )
        batch_table = torch.zeros((512, 5), dtype=torch.int64)
        width = chain_width_table[0][1]
        for i in range(512):
            for minimum, candidate in chain_width_table:
                if i + 1 < minimum:
                    break
                width = candidate
            batch_table[i] = torch.tensor([0, 0, 1, 1, width])
        # Table rounds return before reading any cost. These are unused ABI buffers,
        # not fabricated timings; uncovered batches above are rejected.
        return PolicyTables(
            batch_table=batch_table.to(device),
            interval_i32=torch.zeros(0, dtype=torch.int32, device=device),
            interval_f64=torch.zeros(0, dtype=torch.float64, device=device),
            tier_table=torch.zeros(0, dtype=torch.int32, device=device),
            verify_i32=torch.zeros(0, dtype=torch.int32, device=device),
            verify_f64=torch.zeros(0, dtype=torch.float64, device=device),
            chain_f64=torch.zeros(
                (512, 1, 2, tree_depth, 5), dtype=torch.float64, device=device
            ),
        )
    if len(costs.batches) != 512:
        raise ValueError("chain policy costs need 512 live batch records")
    rows = []
    for batch in costs.batches:
        if (
            type(batch.cache_length_step) is not int
            or not 0 < batch.cache_length_step < 2**63
            or not batch.intervals
        ):
            raise ValueError(
                "chain batch needs intervals and a positive int64 cache step"
            )
        intervals = []
        for interval in batch.intervals:
            if len(interval) != tree_depth:
                raise ValueError("chain cost records must match the policy tree_depth")
            packed = []
            for depth, terms in enumerate(interval):
                values = (
                    terms.gain_scale,
                    terms.cost,
                    terms.threshold,
                    terms.cliff_gain_scale,
                    terms.cliff_cost,
                )
                if any(not math.isfinite(v) for v in values):
                    raise ValueError("chain stop terms must be finite")
                if depth >= 2 and depth % 2 == 0 and depth < tree_depth - 1:
                    if any(v <= 0 for v in values[:3]) or not (
                        values[3] == values[4] == 0 or values[3] > 0 and values[4] > 0
                    ):
                        raise ValueError(
                            "chain decision needs positive denominator, cost and threshold; "
                            "cliff terms must both be positive or both zero"
                        )
                elif any(v != 0 for v in values):
                    raise ValueError("unused chain depth slots must be zero")
                packed.append(values)
            intervals.append(packed)
        rows.append(intervals)
    max_intervals = max(len(row) for row in rows)
    chain_f64 = torch.zeros((512, max_intervals, 2, tree_depth, 5), dtype=torch.float64)
    batch_table = torch.zeros((512, 5), dtype=torch.int64)
    for i, (batch, intervals) in enumerate(zip(costs.batches, rows)):
        n = len(intervals)
        chain_f64[i, :n] = torch.tensor(intervals, dtype=torch.float64).unsqueeze(1)
        width = 0
        for min_batch_size, candidate in chain_width_table:
            if i + 1 < min_batch_size:
                break
            width = candidate
        batch_table[i] = torch.tensor(
            [i, 0, n, batch.cache_length_step, width], dtype=torch.int64
        )
    return PolicyTables(
        batch_table=batch_table.to(device),
        interval_i32=torch.zeros(0, dtype=torch.int32, device=device),
        interval_f64=torch.zeros(0, dtype=torch.float64, device=device),
        tier_table=torch.zeros(0, dtype=torch.int32, device=device),
        verify_i32=torch.zeros(0, dtype=torch.int32, device=device),
        verify_f64=torch.zeros(0, dtype=torch.float64, device=device),
        chain_f64=chain_f64.to(device),
    )
