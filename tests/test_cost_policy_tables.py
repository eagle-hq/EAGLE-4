# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The tree-policy export must describe the existing host cost decisions exactly."""

import copy
import json
import math
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from eagle_harness.core.cost_model import (
    CostModel,
    CostModelParams,
    SizeClass,
    depth_loss_ratio,
    ragged_verify_budget,
    stable_verify_thresholds,
    verify_cost_delta,
)
from eagle_harness.core.cost_table import BatchCostTable, CostCurve, IntervalCost, load_cost_tables
from eagle_harness.core.limits import MAX_BATCH_SIZE_TREE, ComputeIntensity
from eagle_harness.tokenspeed.costs import TREE_POLICY_BATCH_CAPACITY, tree_policy_costs


def _curve(ratios, indices):
    return CostCurve(tuple(10 * r for r in ratios), tuple(ratios), tuple(indices), (), ())


def _table(batch_size, states=(1.2, 1.4, 1.8, 2.4)):
    return BatchCostTable(
        batch_size=batch_size,
        cache_length_start=128,
        cache_length_step=128,
        target_max_qo_len_query=64,
        target_max_qo_len_cache=64,
        draft_max_qo_len=8,
        intervals=tuple(
            IntervalCost(
                target=_curve((1.0, state, state + 0.03, state + 0.2, state + 0.6), (0, 7, 15, 31, 63)),
                draft=_curve((1.0, 1.1, 1.3, 1.7), (0, 1, 3, 7)),
            )
            for state in states
        ),
        model_ratio=tuple(8.0 + 6.0 * i for i in range(len(states))),
    )


def _model(**kwargs):
    args = {
        "tables": {bs: _table(bs) for bs in (1, 4, 8, 16, 32)},
        "params": CostModelParams(),
        "intensity": ComputeIntensity.HIGH,
        "node_top_k": 8,
        "tree_depth": 9,
        "verify_num": 64,
    }
    args.update(kwargs)
    return CostModel(**args)


@pytest.mark.parametrize("intensity", [ComputeIntensity.HIGH, ComputeIntensity.LOW])
@pytest.mark.parametrize("target_is_moe", [False, True])
def test_tokenspeed_boundary_preserves_host_rows_within_its_capacity(intensity, target_is_moe):
    model = _model(intensity=intensity, target_is_moe=target_is_moe)
    before = copy.deepcopy(model.__dict__)
    shared = model.tree_policy_costs()
    adapted = tree_policy_costs(model)
    assert len(shared.batches) == MAX_BATCH_SIZE_TREE
    assert len(adapted.batches) == TREE_POLICY_BATCH_CAPACITY
    assert adapted.batches == shared.batches[:TREE_POLICY_BATCH_CAPACITY]
    assert replace(adapted, batches=shared.batches) == shared
    assert model.__dict__ == before


@pytest.mark.parametrize("intensity", list(ComputeIntensity))
@pytest.mark.parametrize("target_is_moe", [False, True])
def test_export_matches_host_cost_rows_exactly(intensity, target_is_moe):
    chain = intensity is ComputeIntensity.CHAIN
    model = _model(
        intensity=intensity,
        node_top_k=1 if chain else 8,
        verify_num=16 if chain else 64,
        target_is_moe=target_is_moe,
        params=CostModelParams(
            depth_loss_ratio=depth_loss_ratio("eagle3", target_is_moe=target_is_moe, intensity=intensity)
        ),
    )
    costs = model.tree_policy_costs()
    assert isinstance(costs.batches, tuple)
    assert len(costs.batches) == MAX_BATCH_SIZE_TREE
    assert costs.history_floors == ((1,) * 5 if chain else (16, 8, 4, 2, 32))
    assert costs.initial_widths == ((1,) * 4 if chain else (8, 8, 4, 4))
    assert costs.async_layer_ratio == model.params.async_layer_ratio
    for bs, batch in enumerate(costs.batches, 1):
        table = model.table(bs)
        assert batch.cost_batch_size == table.batch_size
        assert batch.cache_length_step == table.cache_length_step
        assert len(batch.intervals) == len(table.intervals)
        for interval_id, interval in enumerate(batch.intervals):
            context = (interval_id + 1) * table.cache_length_step
            verify = model.verify_cost(bs, context)
            assert interval.verify_tier == int(model.tier.size)
            assert interval.verify_widths == verify.index
            assert interval.verify_thresholds == (verify.threshold * 0.95, verify.threshold * 1.05)
            assert interval.verify_cost_deltas == tuple(
                tuple(b - a if b - a > 0.04 else 0.0 for b in verify.value) for a in verify.value
            )
            assert len(interval.depth_thresholds) == 5
            for variant in range(5):
                model.tier.size = SizeClass(variant % 4)
                model.tier.tiny = variant == 4
                row = model.draft_cost_row(bs, context)
                assert interval.draft_cost == row.cost_values[-1]
                assert interval.model_cost_ratio == row.model_cost_ratio
                assert interval.depth_thresholds[variant] == tuple(
                    model.tree_depth_threshold(depth, row) for depth in range(model.tree_depth)
                )
    with pytest.raises(FrozenInstanceError):
        costs.async_layer_ratio = 0.0
    with pytest.raises(FrozenInstanceError):
        costs.batches[0].cache_length_step = 0
    with pytest.raises(FrozenInstanceError):
        costs.batches[0].intervals[0].draft_cost = 0.0


@pytest.mark.parametrize("top_k,expected", [(1, (1, 1, 1, 1)), (3, (3, 3, 3, 3)), (8, (8, 8, 4, 4))])
def test_initial_widths_use_host_depth_zero_decision(top_k, expected):
    assert _model(node_top_k=top_k).tree_policy_costs().initial_widths == expected


@pytest.mark.parametrize("intensity", [ComputeIntensity.HIGH, ComputeIntensity.CHAIN])
def test_export_accepts_existing_golden_calibration_tables(tmp_path, intensity):
    golden = json.loads((Path(__file__).parent / "golden" / "fork_golden.json").read_text())
    for name, contents in golden[f"cost_files_{intensity.value}"].items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    model = _model(
        tables=load_cost_tables(str(tmp_path)),
        intensity=intensity,
        node_top_k=1 if intensity is ComputeIntensity.CHAIN else 8,
        verify_num=16 if intensity is ComputeIntensity.CHAIN else 64,
    )
    costs = model.tree_policy_costs()
    for bs, batch in enumerate(costs.batches, 1):
        assert batch.cost_batch_size == model.table(bs).batch_size
        for interval_id, interval in enumerate(batch.intervals):
            context = (interval_id + 1) * batch.cache_length_step
            row = model.draft_cost_row(bs, context)
            assert interval.draft_cost == row.cost_values[-1]
            assert interval.model_cost_ratio == row.model_cost_ratio
            verify = model.verify_cost(bs, context)
            assert interval.verify_widths == verify.index
            assert interval.verify_tier == int(model.tier.size)


def test_nearest_batch_ties_and_cache_interval_boundaries():
    model = _model(tables={bs: _table(bs) for bs in (4, 8, 16, 32)})
    costs = model.tree_policy_costs()
    for bs, measured in ((1, 4), (4, 4), (5, 4), (6, 8), (11, 8), (12, 16), (23, 16), (24, 32), (32, 32)):
        assert costs.batches[bs - 1].cost_batch_size == measured
    # The same measured table must retain live batch 1's special SMALL class.
    assert costs.batches[0].intervals[3].verify_tier == int(SizeClass.SMALL)
    assert costs.batches[1].intervals[3].verify_tier == int(SizeClass.HUGE)
    for bs in (1, 6, 12, 24, 32):
        for context, expected_interval in ((0, 0), (255, 0), (256, 1), (383, 1), (384, 2), (512, 3), (10000, 3)):
            row = model.draft_cost_row(bs, context)
            interval = costs.batches[bs - 1].intervals[expected_interval]
            assert model.interval_index == expected_interval
            assert interval.model_cost_ratio == row.model_cost_ratio
            verify = model.verify_cost(bs, context)
            assert interval.verify_thresholds == (verify.threshold * 0.95, verify.threshold * 1.05)


@pytest.mark.parametrize("target_is_moe", [False, True])
@pytest.mark.parametrize("state", [0.69, 0.70, 0.84, 0.85, 1.25, 1.60, 2.20, 4.0, 4.01])
def test_export_preserves_host_cost_state_checks(target_is_moe, state):
    model = _model(tables={1: _table(1, (state,)), 4: _table(4, (state,))}, target_is_moe=target_is_moe)
    invalid = state < (0.70 if target_is_moe else 0.85) or state > 4.0
    if invalid:
        with pytest.raises(ValueError, match="cost_state"):
            model.tree_policy_costs()
    else:
        costs = model.tree_policy_costs()
        for bs in (1, 4):
            verify = model.verify_cost(bs, 128)
            interval = costs.batches[bs - 1].intervals[0]
            assert interval.verify_tier == int(model.tier.size)
            assert interval.verify_thresholds == (verify.threshold * 0.95, verify.threshold * 1.05)


@pytest.mark.parametrize("intensity", [ComputeIntensity.HIGH, ComputeIntensity.LOW])
@pytest.mark.parametrize("last_budget", [-1, 8, 64])
@pytest.mark.parametrize("lagged", [False, True])
def test_exported_costs_reproduce_paged_verify_decisions(intensity, last_budget, lagged):
    model = _model(intensity=intensity)
    costs = model.tree_policy_costs()
    rng = random.Random(62)
    decisions = set()
    for bs in (1, 3, 8, 17, 32):
        for interval_id, interval in enumerate(costs.batches[bs - 1].intervals):
            for requested in (1, 8, 16, 32, 64):
                scores = [rng.uniform(0.0, 0.1)]
                for _ in range(63):
                    scores.append(scores[-1] + rng.uniform(0.0, 0.12))
                if lagged:
                    scores = scores[:32]
                selected = [i for i, width in enumerate(interval.verify_widths) if width < min(len(scores), requested)]
                mask = [interval.verify_widths[i] for i in selected]
                if len(mask) > 1:
                    reset = mask[1]
                    for i in selected:
                        for j in selected:
                            if j <= i:
                                continue
                            delta = interval.verify_cost_deltas[i][j]
                            threshold = interval.verify_thresholds[0 if last_budget <= mask[j] + 1 else 1]
                            gain = scores[interval.verify_widths[j]] - scores[interval.verify_widths[i]]
                            if delta > 0.0 and gain / delta < threshold:
                                mask[j] = reset
                expected = max(mask) + 1
                model.recent_verify_nums[:] = [last_budget] * 5
                context = (interval_id + 1) * costs.batches[bs - 1].cache_length_step
                assert model.verify_num_paged(scores, requested, bs, context, lagged=lagged) == (expected, False)
                assert model.recent_verify_nums[-1] == expected
                decisions.add(expected)
    assert len(decisions) >= 3


@pytest.mark.parametrize("top_k", [3, 8])
def test_exported_depth_thresholds_reproduce_host_stop_decisions(top_k):
    model = _model(node_top_k=top_k, tree_depth=31)
    costs = model.tree_policy_costs()
    for bs in (1, 6, 32):
        for interval_id, interval in enumerate(costs.batches[bs - 1].intervals):
            context = (interval_id + 1) * costs.batches[bs - 1].cache_length_step
            for variant, thresholds in enumerate(interval.depth_thresholds):
                for depth in (3, 5, 7, 29):
                    for factor in (0.999999999, 1.0, 1.000000001):
                        model.tier.size = SizeClass(variant % 4)
                        model.tier.tiny = variant == 4
                        row = model.draft_cost_row(bs, context)
                        gain = thresholds[depth] * interval.draft_cost / interval.model_cost_ratio
                        gain = gain / costs.async_layer_ratio * factor
                        expected_stop = (
                            gain / interval.draft_cost * interval.model_cost_ratio * costs.async_layer_ratio
                            < thresholds[depth]
                        )
                        result = model.tree_width(depth, bs, 32, row, [0.0] + [gain] * 8)
                        assert result.stop is expected_stop


@pytest.mark.parametrize("delta", [math.nextafter(0.04, 0.0), 0.04, math.nextafter(0.04, math.inf)])
def test_verify_cost_cutoff_is_strict(delta):
    assert verify_cost_delta(0.0, delta) == (delta if delta > 0.04 else 0.0)
    assert verify_cost_delta(delta, 0.0) == 0.0


@pytest.mark.parametrize("last_budget", [1, 64])
@pytest.mark.parametrize("direction", [0.0, math.inf])
def test_verify_stability_boundary_matches_paged_and_ragged(last_budget, direction):
    table = _table(1)
    target = _curve((1.0, 1.2, 1.6), (0, 1, 3))
    table = replace(table, intervals=(replace(table.intervals[0], target=target),), model_ratio=(20.0,))
    model = _model(tables={1: table})
    interval = model.tree_policy_costs().batches[0].intervals[0]
    threshold = interval.verify_thresholds[0 if last_budget <= 4 else 1]
    # A large initial gain prevents root comparisons from hiding the last pair.
    last_score = math.nextafter(10.0 + threshold * (1.6 - 1.2), direction)
    scores = [0.0, 10.0, 10.0, last_score]
    expected = 2 if (scores[3] - scores[1]) / (1.6 - 1.2) < threshold else 4
    assert expected == (2 if direction == 0.0 else 4)
    model.recent_verify_nums[:] = [last_budget] * 5
    assert model.verify_num_paged(scores, 4, 1, 128, lagged=False) == (expected, False)
    assert (
        ragged_verify_budget([scores[1], scores[3]], 4, target.ratio, target.index, 2.4, [last_budget] * 5) == expected
    )
    assert stable_verify_thresholds(2.4) == (2.4 * 0.95, 2.4 * 1.05)


@pytest.mark.parametrize("direction,expected", [(0.0, 4), (math.inf, 2)])
def test_cost_cutoff_matches_export_paged_and_ragged(direction, expected):
    table = _table(1)
    last_cost = math.nextafter(1.06, direction)
    target = _curve((1.0, 1.02, last_cost), (0, 1, 3))
    table = replace(table, intervals=(replace(table.intervals[0], target=target),), model_ratio=(20.0,))
    model = _model(tables={1: table})
    interval = model.tree_policy_costs().batches[0].intervals[0]
    delta = last_cost - 1.02
    assert interval.verify_cost_deltas[1][2] == (delta if delta > 0.04 else 0.0)
    assert model.verify_num_paged([0.0, 10.0, 10.0, 10.001], 4, 1, 128, lagged=False) == (expected, False)
    assert ragged_verify_budget([10.0, 10.001], 4, target.ratio, target.index, 2.4, [-1] * 5) == expected


def test_core_import_does_not_load_engine_dependencies():
    script = """
import importlib.abc
import sys

class RejectEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "tokenspeed"}:
            raise AssertionError(f"Unexpected engine import: {fullname}")

sys.meta_path.insert(0, RejectEngines())
from eagle_harness.core.cost_model import CostModel
assert callable(CostModel.tree_policy_costs)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_export_preserves_all_caller_state_and_returns_detached_records():
    model = _model(eval_batch_size=32)
    model.tier.size = SizeClass.HUGE
    model.tier.tiny = True
    model.interval_index = 3
    model.recent_verify_nums[:] = [8] * 5
    for bs in range(1, 33):
        model.record_verify_num(bs, bs)
    before = copy.deepcopy(model.__dict__)
    tier = model.tier
    histories = (model.recent_verify_nums, model._history)
    costs = model.tree_policy_costs()
    assert model.__dict__ == before
    assert model.tier is tier
    assert all(
        a is b for a, b in zip(histories, (model.recent_verify_nums, model._history))
    )
    assert costs == model.tree_policy_costs()
    model.tier.size = SizeClass.SMALL
    model.interval_index = 0
    model.record_verify_num(1, 64)
    assert costs == model.tree_policy_costs()


@pytest.mark.parametrize(
    "table_change",
    [
        {"intervals": ()},
        {"model_ratio": ()},
        {"model_ratio": (math.nan,) * 4},
        {"model_ratio": (math.inf,) * 4},
        {"model_ratio": (4.99,) * 4},
        {"model_ratio": (100.01,) * 4},
        {"cache_length_step": 0},
        {"cache_length_step": -1},
        {"cache_length_step": 0.5},
        {"batch_size": 2},
        {"batch_size": 1.0},
    ],
)
def test_export_rejects_invalid_table_metadata(table_change):
    table = replace(_table(1), **table_change)
    model = _model(tables={1: table})
    before = copy.deepcopy(model.__dict__)
    with pytest.raises(ValueError):
        model.tree_policy_costs()
    assert model.__dict__ == before


@pytest.mark.parametrize("curve_name", ["target", "draft"])
@pytest.mark.parametrize(
    "curve_change",
    [
        {"ratio": ()},
        {"value": ()},
        {"index": ()},
        {"index": (0, 7, 7, 31, 63)},
        {"index": (-1, 7, 15, 31, 63)},
        {"index": (1, 7, 15, 31, 63)},
        {"index": (0, 7, 15, 31, 64)},
        {"index": (0, 7, 15.5, 31, 63)},
        {"ratio": (1.0, math.nan, 1.3, 1.4, 1.5)},
        {"ratio": (1.0, math.inf, 1.3, 1.4, 1.5)},
        {"ratio": (1.0, 0.0, 1.3, 1.4, 1.5)},
        {"value": (1.0, -1.0, 1.3, 1.4, 1.5)},
    ],
)
def test_export_rejects_invalid_cost_curves(curve_name, curve_change):
    indices = (0, 7, 15, 31, 63) if curve_name == "target" else (0, 1, 3, 5, 7)
    curve = replace(_curve((1.0, 1.1, 1.2, 1.3, 1.4), indices), **curve_change)
    table = _table(1)
    interval = replace(table.intervals[0], **{curve_name: curve})
    with pytest.raises(ValueError):
        _model(tables={1: replace(table, intervals=(interval,), model_ratio=(20.0,))}).tree_policy_costs()


@pytest.mark.parametrize("width_count", [1, 33])
def test_export_rejects_unsupported_target_width_count(width_count):
    table = _table(1)
    target = _curve((1.0,) * width_count, range(width_count))
    interval = replace(table.intervals[0], target=target)
    with pytest.raises(ValueError, match="target"):
        _model(tables={1: replace(table, intervals=(interval,), model_ratio=(20.0,))}).tree_policy_costs()


def test_export_accepts_32_target_widths():
    table = _table(1)
    target = _curve(tuple(1.0 + i * 0.01 for i in range(32)), range(32))
    interval = replace(table.intervals[0], target=target)
    costs = _model(tables={1: replace(table, intervals=(interval,), model_ratio=(20.0,))}).tree_policy_costs()
    assert len(costs.batches[0].intervals[0].verify_widths) == 32


def test_export_rejects_invalid_ratio_even_in_an_unselected_table():
    model = _model()
    model.tables[512] = replace(_table(512), model_ratio=(4.99,) * 4)
    with pytest.raises(ValueError, match="model_cost_ratio"):
        model.tree_policy_costs()


@pytest.mark.parametrize(
    "model_change",
    [
        {"tables": {}},
        {"tree_depth": 0},
        {"tree_depth": 1},
        {"tree_depth": 32},
        {"node_top_k": 0},
        {"node_top_k": 9},
        {"verify_num": 1},
        {"verify_num": 65},
        {"params": CostModelParams(threshold=math.nan)},
        {"params": CostModelParams(async_layer_ratio=math.inf)},
        {"params": CostModelParams(async_layer_ratio=0.0)},
        {"params": CostModelParams(depth_loss_ratio=-0.01)},
        {"params": CostModelParams(depth_loss_ratio=math.inf)},
        {"params": CostModelParams(threshold=1e308)},
    ],
)
def test_export_rejects_invalid_model_bounds(model_change):
    with pytest.raises((ValueError, NotImplementedError)):
        _model(**model_change).tree_policy_costs()
