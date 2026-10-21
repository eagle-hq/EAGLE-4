# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Online cost model: decides draft tree width / chain depth and the verify budget.

Decisions are checked against golden fixtures on fixed inputs
(``tests/test_cost_golden.py``).

Everything here is host-side Python over lists. The GPU parts (launching the
score-delta kernel, event spins, pinned copies, verify-count tensors) belong
to the caller, which hands in the host lists.

Calibration mode never consults this class: forced shapes come from the
calibration plan.
"""

from __future__ import annotations

import enum
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from eagle_harness.core.cost_table import BatchCostTable
from eagle_harness.core.limits import (
    MAX_BATCH_SIZE_CHAIN,
    MAX_BATCH_SIZE_TREE,
    MAX_CHAIN_DEPTH,
    MAX_CHAIN_VERIFY_NUM,
    MAX_NODE_TOP_K,
    MAX_VERIFY_NUM,
    ComputeIntensity,
    check_shape,
    max_batch_size,
    nearest_cost_batch_size,
)


class SizeClass(enum.IntEnum):
    """Batch cost class from the measured verify cost at 8 tokens."""

    SMALL = 0
    MEDIUM = 1
    LARGE = 2
    HUGE = 3


class ChainTier(enum.IntEnum):
    """Size class of a chain verify, by the target's cost curve."""

    TINY = 0
    SMALL = 1
    MEDIUM = 2
    LARGE = 3


@dataclass(frozen=True)
class ChainCoef:
    """Threshold, depth loss and cost power of one chain tier."""

    threshold: float
    depth_loss: float
    cost_power: float


@dataclass(frozen=True)
class CostModelParams:
    """Tuned constants for draft and verify cost estimates."""

    threshold: float = 2.5
    async_layer_ratio: float = 0.9
    depth_loss_ratio: float = 0.04
    chain_coef: tuple[ChainCoef, ChainCoef, ChainCoef, ChainCoef] = (
        ChainCoef(2.60, 0.24, 0.00),
        ChainCoef(2.85, 0.24, 0.00),
        ChainCoef(2.54, 0.24, 0.70),
        ChainCoef(1.55, 0.24, 0.00),
    )
    chain_ref_cost: float = 2.43
    chain_cliff_min_cost: float = 0.15
    chain_cliff_ratio: float = 1.5
    chain_scale_tiny: float = 1.25
    chain_tiny_max_bs: int = 2
    chain_scale_small: float = 1.45
    chain_scale_medium: float = 2.00


def depth_loss_ratio(draft_family: str, *, target_is_moe: bool, intensity: ComputeIntensity) -> float:
    """Per-depth acceptance loss used by the tree-width criterion."""
    ratio = {"eagle4": 0.04, "eagle3": 0.08, "eagle": 0.64}.get(draft_family, 0.16)
    if target_is_moe or intensity is ComputeIntensity.CHAIN:
        ratio *= 2.0
    if intensity is ComputeIntensity.CHAIN:
        ratio *= 0.5
    return ratio


@dataclass(frozen=True)
class DraftCostRow:
    """Draft-side cost inputs of one round."""

    cost_values: tuple[float, ...]
    cost_index: tuple[int, ...]
    cost_state: float
    adjust_cost_ratio: float
    model_cost_ratio: float
    batch_size_ratio: float


@dataclass(frozen=True)
class VerifyCost:
    """Target verify cost curve and the gain threshold of the current class."""

    value: tuple[float, ...]
    index: tuple[int, ...]
    threshold: float


@dataclass(frozen=True)
class ChainStopTerms:
    """Costs and gain divisors for one chain stopping decision.

    Evaluate gain as ``raw_gain * 2.0 / gain_scale``. When ``cliff_cost > 0``,
    lookahead adds ``raw_gain * 2.0 / cliff_gain_scale`` to that gain.
    Keeping the divisors and the two gains separate preserves host rounding.
    All fields are zero at depths without a decision; disabled cliffs have
    zero ``cliff_gain_scale`` and ``cliff_cost``.
    """

    gain_scale: float
    cost: float
    threshold: float
    cliff_gain_scale: float
    cliff_cost: float


@dataclass(frozen=True)
class ChainPolicyBatch:
    """Costs selected for a live batch; each context interval is indexed by depth."""

    cache_length_step: int
    intervals: tuple[tuple[ChainStopTerms, ...], ...]


@dataclass(frozen=True)
class ChainPolicyCosts:
    """Immutable chain costs for live batches 1..512, indexed by batch size minus one."""

    batches: tuple[ChainPolicyBatch, ...]


@dataclass(frozen=True)
class TreePolicyInterval:
    """Core costs for one context interval; tier rows are SMALL..HUGE, then tiny.

    Verify widths are query lengths minus one, as in ``CostCurve.index``.
    Delta entry [i][j] is the usable cost increase from width i to width j.
    Verify thresholds are ordered for the <= and > majority-history cases.
    """

    verify_widths: tuple[int, ...]
    verify_cost_deltas: tuple[tuple[float, ...], ...]
    draft_cost: float
    model_cost_ratio: float
    depth_thresholds: tuple[tuple[float, ...], ...]
    verify_tier: int
    verify_thresholds: tuple[float, float]


@dataclass(frozen=True)
class TreePolicyBatch:
    """Measured costs selected for a live batch size."""

    cost_batch_size: int
    cache_length_step: int
    intervals: tuple[TreePolicyInterval, ...]


@dataclass(frozen=True)
class TreePolicyCosts:
    """Immutable host data for all supported tree batch sizes, without engine dependencies."""

    batches: tuple[TreePolicyBatch, ...]
    history_floors: tuple[int, ...]
    initial_widths: tuple[int, ...]
    async_layer_ratio: float


@dataclass(frozen=True)
class WidthDecision:
    """One draft step's decision: next width, verify budget, and whether the tree stops."""

    width: int
    verify_num: int
    stop: bool


class WidthStep(enum.Enum):
    """What ``tree_width`` needs at a depth."""

    KEEP = "keep"  # tiny batch, depth < 2 (a chain: depth 0): keep the top-k width
    LAST = "last"  # last depth: stop
    LAUNCH = "launch"  # even depth (a chain: odd): launch the score-delta copy, keep width
    DECIDE = "decide"  # odd depth (a chain: even): decide from the host deltas


def find_majority(nums: Sequence[int]) -> int:
    """Strict-majority element, scanning in order; -1 when there is none."""
    counts: Counter = Counter()
    limit = len(nums) // 2
    for x in nums:
        counts[x] += 1
        if counts[x] > limit:
            return x
    return -1


def verify_cost_delta(previous: float, current: float) -> float:
    """Cost increase used by verify pruning; negligible increases are ignored."""
    delta = current - previous
    return delta if delta > 0.04 else 0.0


def stable_verify_thresholds(threshold: float) -> tuple[float, float]:
    """Verify thresholds for a majority budget <= or > the candidate budget."""
    return threshold * 0.95, threshold * 1.05


def _adjust_cost_ratio(model_cost_ratio: float) -> float:
    if model_cost_ratio < 5:
        raise ValueError(
            f"Invalid model_cost_ratio: {model_cost_ratio:.2f}. The ratio is too low "
            "(below 5), which is not suitable for speculative sampling."
        )
    if model_cost_ratio > 100:
        raise ValueError(
            f"Invalid model_cost_ratio: {model_cost_ratio:.2f}. The ratio is too high "
            "(above 100), which may indicate poor draft model performance."
        )
    for bound, ratio in ((8, 1.30), (12, 1.10), (17, 1.30), (23, 1.50), (32, 1.80), (64, 2.20)):
        if model_cost_ratio < bound:
            return ratio
    return 2.70


_VERIFY_THRESHOLDS = {  # size class -> (high_intensity, other intensities)
    SizeClass.SMALL: (2.40, 2.00),
    SizeClass.MEDIUM: (2.10, 1.80),
    SizeClass.LARGE: (1.80, 1.60),
    SizeClass.HUGE: (1.50, 1.40),
}
_BATCH_SIZE_RATIO = {SizeClass.SMALL: 1.25, SizeClass.MEDIUM: 1.50, SizeClass.LARGE: 1.75, SizeClass.HUGE: 2.00}
_MIN_HISTORY_VERIFY = {SizeClass.SMALL: 16, SizeClass.MEDIUM: 8, SizeClass.LARGE: 4, SizeClass.HUGE: 2}
_CHAIN_HISTORY_SLACK = {SizeClass.SMALL: 4, SizeClass.MEDIUM: 2, SizeClass.LARGE: 2, SizeClass.HUGE: 0}
_CHAIN_COST_INDEX = (0, 1, 3, 5, 7, 9, 11, 13, 15)


@dataclass
class _TierState:
    # tiny: batch size 1 (set by the width decision); size: set by verify_cost.
    tiny: bool = False
    size: SizeClass = SizeClass.SMALL


@dataclass
class CostModel:
    """Stateful per-worker cost model. Not thread safe; one per EAGLE-Harness worker."""

    tables: dict[int, BatchCostTable]
    params: CostModelParams
    intensity: ComputeIntensity
    node_top_k: int
    tree_depth: int
    verify_num: int
    target_is_moe: bool = False

    tier: _TierState = field(default_factory=_TierState)
    interval_index: int = 0
    # Majority vote window of the paged verify budget.
    recent_verify_nums: list[int] = field(default_factory=lambda: [-5, -4, -3, -2, -1])

    def __post_init__(self) -> None:
        self.max_batch_size = max_batch_size(self.intensity)
        self.set_tables(self.tables)
        # Per-batch-size history of chosen verify numbers.
        self._history = [[self.verify_num] * 16 for _ in range(self.max_batch_size)]

    # ------------------------------------------------------------------ tables
    def set_tables(self, tables: dict[int, BatchCostTable]) -> None:
        """Replace the cost tables (live costs refresh them between rounds); history and state are kept."""
        self.tables = tables
        measured = sorted(tables)
        self._bs_lut = (
            [0] + [nearest_cost_batch_size(bs, measured) for bs in range(1, self.max_batch_size + 1)]
            if measured
            else []
        )

    def table(self, batch_size: int) -> BatchCostTable | None:
        """Cost table of the nearest measured batch size; None without cost data."""
        if not self._bs_lut:
            return None
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError(f"batch_size={batch_size} is outside 1..{self.max_batch_size}.")
        return self.tables[self._bs_lut[batch_size]]

    def _require_table(self, batch_size: int) -> BatchCostTable:
        table = self.table(batch_size)
        if table is None:
            raise ValueError(f"No cost data found for batch_size={batch_size}.")
        return table

    def update_interval(self, batch_size: int, average_cache_len: int) -> BatchCostTable:
        """Select the cost interval of the average cache length; return the batch size's table."""
        table = self._require_table(batch_size)
        self.interval_index = table.interval_index(average_cache_len)
        return table

    def draft_cost_row(self, batch_size: int, average_cache_len: int) -> DraftCostRow:
        """Draft cost inputs of a round; also selects the context interval."""
        table = self.update_interval(batch_size, average_cache_len)
        model_cost_ratio = table.model_ratio[self.interval_index]
        adjust = _adjust_cost_ratio(model_cost_ratio)
        draft = table.intervals[self.interval_index].draft
        if self.tier.tiny:
            bs_ratio = 1.00
        else:
            bs_ratio = _BATCH_SIZE_RATIO[self.tier.size]
        bs_ratio = max(1.0, bs_ratio / ((20 / model_cost_ratio + 1) / 2))
        return DraftCostRow(
            cost_values=draft.ratio,
            cost_index=draft.index,
            cost_state=draft.ratio[-1],
            adjust_cost_ratio=adjust,
            model_cost_ratio=model_cost_ratio,
            batch_size_ratio=bs_ratio,
        )

    def verify_cost(self, batch_size: int, average_cache_len: int) -> VerifyCost | None:
        """Target verify cost curve; classifies the batch (small .. huge) as a side effect."""
        table = self.table(batch_size)
        if table is None:
            return None
        target = table.intervals[table.interval_index(average_cache_len)].target
        value, index = target.ratio, target.index
        cost_state = value[-1]
        for i, idx in enumerate(index):
            if idx >= 7:
                cost_state = value[i]
                break
        if cost_state <= 1.25 or batch_size == 1:
            size = SizeClass.SMALL
            if cost_state < 0.85 and (cost_state < 0.70 or not self.target_is_moe):
                raise ValueError(f"EAGLE-Harness: cost_state={cost_state} is too low, which is not expected.")
        elif cost_state < 1.60:
            size = SizeClass.MEDIUM
        elif cost_state < 2.20:
            size = SizeClass.LARGE
        else:
            size = SizeClass.HUGE
            if cost_state > 4.0:
                raise ValueError(f"EAGLE-Harness: cost_state={cost_state} is too high, which is not expected.")
        self.tier.size = size
        high, other = _VERIFY_THRESHOLDS[size]
        threshold = high if self.intensity is ComputeIntensity.HIGH else other
        return VerifyCost(value=value, index=index, threshold=threshold)

    def tree_policy_costs(self) -> TreePolicyCosts:
        """Export core policy inputs without changing this worker's online state.

        Each live batch uses :meth:`table`'s nearest measured batch mapping.
        Depth thresholds cover all depths and all five possible draft tiers.
        History floors use a fresh history with verify_num=1. Invalid or missing
        tables, nonfinite costs and unsupported shapes are rejected.
        """
        self._validate_tree_policy_inputs()
        model = replace(
            self,
            tier=_TierState(),
            interval_index=0,
            recent_verify_nums=list(self.recent_verify_nums),
        )
        batches = []
        for batch_size in range(1, MAX_BATCH_SIZE_TREE + 1):
            table = model._require_table(batch_size)
            intervals = []
            for interval_id in range(len(table.intervals)):
                context = (interval_id + 1) * table.cache_length_step
                verify = model._require_verify_cost(batch_size, context)
                verify_tier = int(model.tier.size)
                thresholds = []
                for variant in range(5):
                    model.tier = _TierState(tiny=variant == 4, size=SizeClass(variant % 4))
                    row = model.draft_cost_row(batch_size, context)
                    thresholds.append(tuple(model.tree_depth_threshold(d, row) for d in range(self.tree_depth)))
                if not all(math.isfinite(value) for tier in thresholds for value in tier):
                    raise ValueError(f"Nonfinite depth thresholds for batch_size={batch_size}, interval={interval_id}.")
                intervals.append(
                    TreePolicyInterval(
                        verify_widths=tuple(verify.index),
                        verify_cost_deltas=tuple(
                            tuple(verify_cost_delta(a, b) for b in verify.value) for a in verify.value
                        ),
                        draft_cost=row.cost_values[-1],
                        model_cost_ratio=row.model_cost_ratio,
                        depth_thresholds=tuple(thresholds),
                        verify_tier=verify_tier,
                        verify_thresholds=stable_verify_thresholds(verify.threshold),
                    )
                )
            batches.append(TreePolicyBatch(table.batch_size, table.cache_length_step, tuple(intervals)))

        floors = []
        widths = []
        model = replace(model, verify_num=1)
        for variant in range(5):
            model.tier = _TierState(tiny=variant == 4, size=SizeClass(variant % 4))
            floors.append(model.history_verify_num(1))
            if variant < 4:
                row = model.draft_cost_row(1, 0)
                widths.append(model.tree_width(0, 1, 1, row).width)
        return TreePolicyCosts(tuple(batches), tuple(floors), tuple(widths), self.params.async_layer_ratio)

    def _validate_tree_policy_inputs(self) -> None:
        for name, lower in (("tree_depth", 2), ("node_top_k", 1), ("verify_num", 2)):
            value = getattr(self, name)
            if type(value) is not int or value < lower:
                raise ValueError(f"{name}={value!r} must be an integer >= {lower} for tree policy export.")
        check_shape(self.intensity, self.node_top_k, self.tree_depth, self.verify_num, MAX_BATCH_SIZE_TREE)
        for name in ("threshold", "async_layer_ratio", "depth_loss_ratio"):
            value = getattr(self.params, name)
            if not math.isfinite(value) or value < 0 or (name != "depth_loss_ratio" and value == 0):
                raise ValueError(f"Invalid tree policy parameter {name}={value!r}.")
        self._validate_policy_tables(chain=False)

    def _validate_policy_tables(self, *, chain: bool) -> None:
        if not self.tables:
            raise ValueError(f"No cost data found for {'chain' if chain else 'tree'} policy export.")
        for batch_size, table in self.tables.items():
            if (
                type(batch_size) is not int
                or batch_size < 1
                or (chain and batch_size > MAX_BATCH_SIZE_CHAIN)
                or type(table.batch_size) is not int
                or table.batch_size != batch_size
            ):
                raise ValueError(f"Invalid measured batch_size={batch_size!r}; table reports {table.batch_size!r}.")
            if type(table.cache_length_step) is not int or table.cache_length_step <= 0:
                raise ValueError(f"Invalid cache_length_step={table.cache_length_step!r} for batch_size={batch_size}.")
            if not table.intervals or len(table.intervals) != len(table.model_ratio):
                raise ValueError(f"Missing cost intervals or model ratios for batch_size={batch_size}.")
            for interval_id, (interval, ratio) in enumerate(zip(table.intervals, table.model_ratio)):
                location = f"batch_size={batch_size}, interval={interval_id}"
                if not math.isfinite(ratio) or ratio <= 0:
                    raise ValueError(f"Model cost ratio must be finite and positive for {location}.")
                if not chain:
                    _adjust_cost_ratio(ratio)
                for name, curve, bound in (
                    ("target", interval.target, MAX_VERIFY_NUM),
                    ("draft", interval.draft, MAX_NODE_TOP_K),
                ):
                    count = len(curve.index)
                    if count == 0 or (name == "target" and not 2 <= count <= 32):
                        raise ValueError(f"Invalid {name} width count {count} for {location}.")
                    if len(curve.value) != count or len(curve.ratio) != count:
                        raise ValueError(f"Mismatched {name} cost lengths for {location}.")
                    if (
                        curve.index[0] != 0
                        or any(type(index) is not int or not 0 <= index < bound for index in curve.index)
                        or any(a >= b for a, b in zip(curve.index, curve.index[1:]))
                    ):
                        raise ValueError(f"Invalid {name} cost indices {curve.index!r} for {location}.")
                    if chain and name == "target" and tuple(curve.index) != _CHAIN_COST_INDEX:
                        raise ValueError(f"Invalid chain cost indices {curve.index!r} for {location}.")
                    if any(not math.isfinite(value) or value <= 0 for value in (*curve.value, *curve.ratio)):
                        raise ValueError(f"{name} costs must be finite and positive for {location}.")

    # ----------------------------------------------------------------- history
    def _floored_history(self, batch_size: int) -> int:
        recent = max(self._history[batch_size - 1])
        return max(recent, 32 if self.tier.tiny else _MIN_HISTORY_VERIFY[self.tier.size])

    def history_verify_num(self, batch_size: int) -> int:
        """Verify-number cap a new round starts from."""
        if self.intensity is ComputeIntensity.CHAIN:
            slack = 4 if self.tier.tiny else _CHAIN_HISTORY_SLACK[self.tier.size]
            return min(max(self._history[batch_size - 1]) + slack, self.verify_num)
        return self._floored_history(batch_size)

    @property
    def max_history_verify_num(self) -> int:
        """Upper bound of :meth:`history_verify_num`: the verify budgets the work deltas need.

        History entries never exceed ``verify_num`` (a round verifies at most
        that many tokens), but the tree-mode floor (32 at batch size 1) can.
        """
        if self.intensity is ComputeIntensity.CHAIN:
            return self.verify_num
        return max(self.verify_num, 32, *_MIN_HISTORY_VERIFY.values())

    @property
    def chain_rounds(self) -> bool:
        """Rounds of a top-1 chain at chain intensity: sized by :meth:`chain_stop`, not the tree rule, and every
        drafted token is verified.

        The chain rule prices the full 16-token chain (its cost index); a narrower chain keeps the tree rule.
        """
        return (
            self.intensity is ComputeIntensity.CHAIN
            and self.node_top_k == 1
            and self.verify_num == _CHAIN_COST_INDEX[-1] + 1
        )

    @property
    def verifies_whole_draft(self) -> bool:
        """Whether a round verifies every drafted token, without the paged verify budget."""
        return self.chain_rounds

    def deltas_budget(self, past_verify_num: int) -> int:
        """The verify budget whose work deltas a ``DECIDE`` step reads: the round's history budget, or the whole
        chain for chain rounds."""
        return self.verify_num if self.chain_rounds else past_verify_num

    @property
    def launch_depths(self) -> tuple[int, ...]:
        """Depths at which :meth:`width_step` can return ``LAUNCH`` (the depths whose work deltas are read)."""
        return tuple(range(1 if self.chain_rounds else 0, self.tree_depth - 1, 2))

    def record_verify_num(self, batch_size: int, verify_num: int) -> None:
        """Push this round's verify budget into the batch size's history."""
        history = self._history[batch_size - 1]
        history.append(verify_num)
        history.pop(0)

    def verify_count(self, batch_size: int, draft_count: int) -> int:
        """How many top-scored draft nodes the verify considers (a chain verifies all)."""
        if self.intensity is ComputeIntensity.CHAIN:
            return draft_count
        return min(self.verify_num, draft_count, max(self._floored_history(batch_size) * 2, 16))

    # ------------------------------------------------------------ tree width
    def _width_cap(self) -> int:
        if self.tier.size in (SizeClass.SMALL, SizeClass.MEDIUM):
            return min(self.node_top_k, 8)
        return min(self.node_top_k, 4)

    def width_step(self, depth: int) -> WidthStep:
        """How the frontier width changes after depth ``depth``.

        A chain round follows :meth:`chain_stop`'s pattern: depth 0 keeps, odd depths launch, even depths decide.
        """
        if self.chain_rounds:
            if depth == self.tree_depth - 1:
                return WidthStep.LAST
            if depth == 0:
                return WidthStep.KEEP
            return WidthStep.LAUNCH if depth % 2 == 1 else WidthStep.DECIDE
        if self.tier.tiny and depth < 2:
            return WidthStep.KEEP
        if depth == self.tree_depth - 1:
            return WidthStep.LAST
        if depth % 2 == 0:
            return WidthStep.LAUNCH
        return WidthStep.DECIDE

    def tree_depth_threshold(self, depth: int, row: DraftCostRow) -> float:
        """Minimum gain per cost at this depth, shared by host and device policies."""
        bs_ratio = row.batch_size_ratio
        if depth <= 1:
            bs_ratio = (bs_ratio + 1.00) / 2
        elif depth > 5 and bs_ratio > 1.3:
            bs_ratio = bs_ratio * 1.1
        p = self.params
        return p.threshold * (depth * p.depth_loss_ratio + 1.0) * row.adjust_cost_ratio * bs_ratio

    def tree_width(
        self,
        depth: int,
        batch_size: int,
        past_verify_num: int,
        row: DraftCostRow,
        deltas: Sequence[float] | None = None,
    ) -> WidthDecision:
        """Width of the next draft level, or stop.

        ``deltas`` are the host copy of the top work-score deltas launched at
        the previous (even; a chain: odd) depth; only the ``DECIDE`` step reads them.
        A chain round decides with :meth:`chain_stop` and keeps width 1.
        """
        step = self.width_step(depth)
        if self.chain_rounds and step is not WidthStep.LAST:
            if step is WidthStep.DECIDE and deltas is None:
                raise ValueError(f"tree_width at depth {depth} needs the host score deltas.")
            stop = step is WidthStep.DECIDE and self.chain_stop(depth, batch_size, deltas)
            return WidthDecision(1, past_verify_num, stop)
        if step is WidthStep.KEEP:
            return WidthDecision(self.node_top_k, past_verify_num, False)
        if step is WidthStep.LAST:
            return WidthDecision(-1, -1, True)
        if step is WidthStep.LAUNCH:
            return WidthDecision(self._width_cap(), past_verify_num, False)
        if deltas is None:
            raise ValueError(f"tree_width at depth {depth} needs the host score deltas.")
        base = deltas[0]
        gains = [x - base for x in deltas[1:]]
        self.tier.tiny = batch_size == 1
        width = self._width_cap()
        p = self.params
        gain_per_cost = gains[width - 1] / row.cost_values[-1] * row.model_cost_ratio * p.async_layer_ratio
        bar = self.tree_depth_threshold(depth, row)
        if gain_per_cost < bar:
            return WidthDecision(width, past_verify_num, True)
        if self.tier.size in (SizeClass.MEDIUM, SizeClass.LARGE):
            width = 1 << (width - 1).bit_length()
        return WidthDecision(width, past_verify_num, False)

    # ----------------------------------------------------------- chain depth
    def classify_chain(self, batch_size: int, cost_ratio: Sequence[float] | None = None) -> tuple[ChainTier, float]:
        """The chain tier and its coefficients for a batch size (from the target's cost ratio)."""
        if cost_ratio is None:
            table = self._require_table(batch_size)
            cost_ratio = table.intervals[self.interval_index].target.ratio
        scale_index = (cost_ratio[1] + cost_ratio[3] + cost_ratio[7]) / (3.0 * cost_ratio[0])
        p = self.params
        if scale_index <= p.chain_scale_tiny or batch_size <= p.chain_tiny_max_bs:
            return ChainTier.TINY, scale_index
        if scale_index < p.chain_scale_small:
            return ChainTier.SMALL, scale_index
        if scale_index < p.chain_scale_medium:
            return ChainTier.MEDIUM, scale_index
        return ChainTier.LARGE, scale_index

    def chain_stop(self, depth: int, batch_size: int, deltas: Sequence[float] | None = None) -> bool:
        """Whether a chain draft stops after ``depth`` (odd depths launch deltas)."""
        if depth == 0 or depth == self.tree_depth - 1 or depth % 2 == 1:
            return False
        if deltas is None:
            raise ValueError(f"chain_stop at depth {depth} needs the host score deltas.")
        terms = self.chain_stop_terms(depth, batch_size)
        raw_gain = deltas[1] - deltas[0]
        curr_gain = raw_gain * 2.0 / terms.gain_scale
        stop = curr_gain / terms.cost < terms.threshold
        if stop and terms.cliff_cost > 0.0:
            gain_two = curr_gain + raw_gain * 2.0 / terms.cliff_gain_scale
            stop = gain_two / terms.cliff_cost < terms.threshold
        return stop

    def chain_stop_terms(self, depth: int, batch_size: int) -> ChainStopTerms:
        """Core chain formulas at the selected interval, using the live batch tier.

        Depths without a decision return all zeros and do not need cost data.
        See :class:`ChainStopTerms` for the rounding-preserving evaluation order.
        """
        if depth == 0 or depth == self.tree_depth - 1 or depth % 2 == 1:
            return ChainStopTerms(0.0, 0.0, 0.0, 0.0, 0.0)
        table = self._require_table(batch_size)
        target = table.intervals[self.interval_index].target
        cost_ratio = target.ratio
        cost_llm = cost_ratio[1:]
        if tuple(target.index) != _CHAIN_COST_INDEX or any(type(index) is not int for index in target.index):
            raise ValueError(f"Invalid chain cost index: {list(target.index)}")
        tier, _ = self.classify_chain(batch_size, cost_ratio)
        p = self.params
        coef = p.chain_coef[tier]
        threshold = coef.threshold
        curr_step = depth // 2
        next_step = min(curr_step + 1, len(cost_llm) - 1)
        draft_step_cost = 1.0 / table.model_ratio[self.interval_index]
        curr_cost = draft_step_cost * 2 + max(0, cost_llm[next_step] - cost_llm[curr_step])
        gain_scale = 1.0 + depth * coef.depth_loss
        ref_iter_cost = sum(c + draft_step_cost * (2 * i + 1) for i, c in enumerate(cost_llm)) / len(cost_llm)
        curr_threshold = threshold * (p.chain_ref_cost / ref_iter_cost) ** coef.cost_power
        cliff_gain_scale = 0.0
        cliff_cost = 0.0
        if next_step + 1 < len(cost_llm):
            step_next = cost_llm[next_step] - cost_llm[curr_step]
            step_after = cost_llm[next_step + 1] - cost_llm[next_step]
            if step_next > p.chain_cliff_min_cost and step_next > p.chain_cliff_ratio * max(step_after, 0.0):
                cliff_gain_scale = 1.0 + (depth + 2) * coef.depth_loss
                cliff_cost = draft_step_cost * 4 + max(0, cost_llm[next_step + 1] - cost_llm[curr_step])
        return ChainStopTerms(gain_scale, curr_cost, curr_threshold, cliff_gain_scale, cliff_cost)

    def chain_policy_costs(self) -> ChainPolicyCosts:
        """Export each live batch's nearest measured costs without changing state.

        Each interval contains ``tree_depth`` entries. Validation is specific to
        chains: target ratios above the tree verifier's limit remain valid.
        """
        self._validate_chain_policy_inputs()
        model = replace(self)
        batches = []
        for batch_size in range(1, MAX_BATCH_SIZE_CHAIN + 1):
            table = model._require_table(batch_size)
            intervals = []
            for interval_id in range(len(table.intervals)):
                model.interval_index = interval_id
                try:
                    terms = tuple(model.chain_stop_terms(d, batch_size) for d in range(self.tree_depth))
                except (OverflowError, ZeroDivisionError) as exc:
                    raise ValueError(
                        f"Invalid chain stop terms for batch_size={batch_size}, interval={interval_id}."
                    ) from exc
                for depth, term in enumerate(terms):
                    values = (term.gain_scale, term.cost, term.threshold, term.cliff_gain_scale, term.cliff_cost)
                    decides = depth != 0 and depth != self.tree_depth - 1 and depth % 2 == 0
                    if (
                        any(not math.isfinite(value) or value < 0 for value in values)
                        or (decides and min(term.gain_scale, term.cost, term.threshold) <= 0)
                        or ((term.cliff_gain_scale > 0) != (term.cliff_cost > 0))
                    ):
                        raise ValueError(
                            f"Invalid chain stop terms for batch_size={batch_size}, "
                            f"interval={interval_id}, depth={depth}."
                        )
                intervals.append(terms)
            batches.append(ChainPolicyBatch(table.cache_length_step, tuple(intervals)))
        return ChainPolicyCosts(tuple(batches))

    def _validate_chain_policy_inputs(self) -> None:
        if self.intensity is not ComputeIntensity.CHAIN:
            raise ValueError("Chain policy export requires chain intensity.")
        for name, lower, upper in (
            ("node_top_k", 1, 1),
            ("tree_depth", 2, MAX_CHAIN_DEPTH),
            ("verify_num", 2, MAX_CHAIN_VERIFY_NUM),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name}={value!r} must be an integer in {lower}..{upper} for chain policy export.")
        p = self.params
        if len(p.chain_coef) != len(ChainTier):
            raise ValueError("Chain policy requires four chain_coef entries.")
        for coef in p.chain_coef:
            if (
                not isinstance(coef, ChainCoef)
                or not math.isfinite(coef.threshold)
                or coef.threshold <= 0
                or not math.isfinite(coef.depth_loss)
                or coef.depth_loss < 0
                or not math.isfinite(coef.cost_power)
                or coef.cost_power < 0
            ):
                raise ValueError(f"Invalid chain coefficient {coef!r}.")
        for name in (
            "chain_ref_cost",
            "chain_cliff_min_cost",
            "chain_cliff_ratio",
            "chain_scale_tiny",
            "chain_scale_small",
            "chain_scale_medium",
        ):
            value = getattr(p, name)
            if not math.isfinite(value) or value < 0 or (name != "chain_cliff_min_cost" and value == 0):
                raise ValueError(f"Invalid chain policy parameter {name}={value!r}.")
        if not p.chain_scale_tiny < p.chain_scale_small < p.chain_scale_medium:
            raise ValueError("Chain tier scale thresholds must be strictly increasing.")
        if type(p.chain_tiny_max_bs) is not int or not 0 <= p.chain_tiny_max_bs <= MAX_BATCH_SIZE_CHAIN:
            raise ValueError(f"Invalid chain_tiny_max_bs={p.chain_tiny_max_bs!r}.")
        self._validate_policy_tables(chain=True)

    # ---------------------------------------------------------- verify budget
    def verify_num_paged(
        self,
        scores: Sequence[float],
        requested: int,
        batch_size: int,
        average_cache_len: int,
        *,
        lagged: bool,
    ) -> tuple[int, bool]:
        """Verify tokens per request from cumulative node scores.

        ``scores`` are the cumulative top-k scores: of the previous round when
        ``lagged`` (no sync this round), else of this round. Returns
        ``(verify_num, False)``: the drafter's policy interface keeps a stop flag, never set here.
        """
        last_verify_num = find_majority(self.recent_verify_nums)
        cap = len(scores) if lagged else requested
        self.recent_verify_nums = self.recent_verify_nums[1:] + [requested]
        cost = self._require_verify_cost(batch_size, average_cache_len)
        index = [i for i in cost.index if i < min(cap, requested)]
        if max(index) == 0:
            # Early stopping left fewer candidates than the smallest tree bucket:
            # verify the root alone, a measured and valid target step.
            self.recent_verify_nums[-1] = 1
            return 1, False
        values = cost.value[: len(index)]
        gains = [scores[i] for i in index]
        mask = list(index)
        reset = mask[1]
        thresholds = stable_verify_thresholds(cost.threshold)
        for i in range(len(index) - 1):
            for j in range(i + 1, len(index)):
                stable_threshold = thresholds[0 if last_verify_num <= mask[j] + 1 else 1]
                delta = verify_cost_delta(values[i], values[j])
                if delta > 0.0 and (gains[j] - gains[i]) / delta < stable_threshold:
                    mask[j] = reset
        verify_num = max(mask) + 1
        self.recent_verify_nums[-1] = verify_num
        if verify_num == 1 or requested < verify_num:
            verify_num = min(index[1:]) + 1
        return verify_num, False

    def _require_verify_cost(self, batch_size: int, average_cache_len: int) -> VerifyCost:
        cost = self.verify_cost(batch_size, average_cache_len)
        if cost is None:
            raise ValueError(f"Cost data for batch size {batch_size} not found.")
        return cost


@dataclass
class FixedShapePolicy:
    """The drafter's policy without cost tables: the full ``tree_depth`` every round, ``verify_num`` tokens verified.

    Same calls as :class:`CostModel` (the drafter sees one interface), no host decisions: ``width_step`` keeps
    every depth but the last. ``force_width`` narrows one round's verify (the live-cost seed sweep times each
    captured width through the served round).
    """

    intensity: ComputeIntensity
    node_top_k: int
    tree_depth: int
    verify_num: int
    force_width: int | None = None

    launch_depths: tuple[int, ...] = ()

    @property
    def verifies_whole_draft(self) -> bool:
        """No: the verify budget comes from :meth:`verify_num_paged`."""
        return False

    @property
    def max_history_verify_num(self) -> int:
        """The widest verify budget this policy asks the tree for."""
        return self.verify_num

    def deltas_budget(self, past_verify_num: int) -> int:
        """Never read: this policy has no ``DECIDE`` step."""
        return past_verify_num

    def width_step(self, depth: int) -> WidthStep:
        """Keep drafting through the last depth; never ask for host deltas."""
        return WidthStep.LAST if depth >= self.tree_depth - 1 else WidthStep.KEEP

    def draft_cost_row(self, batch_size: int, average_cache_len: int) -> None:
        """No cost inputs: the width never depends on them."""
        return None

    def history_verify_num(self, batch_size: int) -> int:
        """The verify budget of every round."""
        return self.verify_num

    def verify_count(self, batch_size: int, draft_count: int) -> int:
        """All drafted nodes up to the verify budget."""
        return min(self.verify_num, draft_count)

    def tree_width(self, depth: int, batch_size: int, past_verify_num: int, row, deltas=None) -> WidthDecision:
        """Stop only at the last depth."""
        return WidthDecision(-1, -1, True) if depth >= self.tree_depth - 1 else WidthDecision(self.node_top_k, past_verify_num, False)

    def verify_num_paged(self, scores, requested: int, batch_size: int, average_cache_len: int, *, lagged: bool):
        """The forced width when set, else the full budget."""
        width = self.verify_num if self.force_width is None else self.force_width
        return min(width, requested), False

    def record_verify_num(self, batch_size: int, verify_num: int) -> None:
        """No history."""
