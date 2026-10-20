# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Chain exports use Eagle's host decisions, including live batch classification."""

import copy
import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from eagle_harness.core import cost_model
from eagle_harness.core.cost_model import ChainCoef, CostModel, CostModelParams, SizeClass
from eagle_harness.core.cost_table import BatchCostTable, CostCurve, IntervalCost, load_cost_tables
from eagle_harness.core.limits import ComputeIntensity

CHAIN_INDEX = (0, 1, 3, 5, 7, 9, 11, 13, 15)


def _curve(ratios, indices=CHAIN_INDEX):
    return CostCurve(tuple(10 * r for r in ratios), tuple(ratios), tuple(indices), (), ())


def _table(batch_size=4, *, step=128, model_ratio=20.0):
    # Includes every tier, a cliff, and decreasing measured costs.
    ratios = (
        (1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40),
        (1.0, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55),
        (1.0, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10),
        (1.0, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50),
        (1.0, 1.10, 1.20, 2.00, 2.05, 2.10, 2.15, 2.20, 2.25),
        (1.0, 1.30, 1.20, 1.10, 1.00, 0.90, 0.80, 0.70, 0.60),
    )
    return BatchCostTable(
        batch_size,
        step,
        step,
        16,
        4096,
        1,
        tuple(IntervalCost(_curve(values), _curve((1.0,), (0,))) for values in ratios),
        (model_ratio,) * len(ratios),
    )


def _model(**changes):
    args = {
        "tables": {4: _table()},
        "params": CostModelParams(),
        "intensity": ComputeIntensity.CHAIN,
        "node_top_k": 1,
        "tree_depth": 15,
        "verify_num": 16,
    }
    args.update(changes)
    return CostModel(**args)


def _stop_from_terms(terms, deltas):
    if terms.cost == 0.0:
        return False
    gain = deltas[1] - deltas[0]
    current = gain * 2.0 / terms.gain_scale
    stop = current / terms.cost < terms.threshold
    if stop and terms.cliff_cost > 0.0:
        lookahead = current + gain * 2.0 / terms.cliff_gain_scale
        stop = lookahead / terms.cliff_cost < terms.threshold
    return stop


def test_chain_export_is_available():
    assert callable(getattr(CostModel, "chain_policy_costs", None))
    assert callable(getattr(CostModel, "chain_stop_terms", None))


@pytest.mark.parametrize("depth", [2, 3, 8, 14, 15])
def test_export_shape_immutable_records_and_no_decision_slots(depth):
    model = _model(tree_depth=depth)
    costs = model.chain_policy_costs()
    assert isinstance(costs, cost_model.ChainPolicyCosts)
    assert isinstance(costs.batches, tuple)
    assert len(costs.batches) == 512
    zero = cost_model.ChainStopTerms(0.0, 0.0, 0.0, 0.0, 0.0)
    for batch in costs.batches:
        assert isinstance(batch, cost_model.ChainPolicyBatch)
        assert batch.cache_length_step == 128
        assert isinstance(batch.intervals, tuple)
        assert len(batch.intervals) == 6
        for interval in batch.intervals:
            assert isinstance(interval, tuple)
            assert len(interval) == depth
            for d, terms in enumerate(interval):
                if d == 0 or d == depth - 1 or d % 2:
                    assert terms == zero
                    assert model.chain_stop(d, 1) is False
                else:
                    assert terms.cost > 0.0
                    assert terms.threshold > 0.0
    for obj, name in (
        (costs, "batches"),
        (costs.batches[0], "cache_length_step"),
        (costs.batches[0].intervals[0][0], "cost"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, name, None)


def test_live_batch_tiers_and_nearest_table_ties():
    model = _model(tables={4: _table(), 8: _table(8, step=256, model_ratio=10)})
    costs = model.chain_policy_costs()
    # All three live batches select measured batch 4; only 1 and 2 are tiny.
    for bs, threshold in ((1, 2.60), (2, 2.60), (3, 2.85)):
        assert costs.batches[bs - 1].intervals[1][2].threshold == threshold
    for bs, step, draft_cost in ((5, 128, 0.1), (6, 256, 0.2), (512, 256, 0.2)):
        batch = costs.batches[bs - 1]
        assert batch.cache_length_step == step
        # The decreasing curve contributes zero extra verify cost.
        assert batch.intervals[5][2].cost == draft_cost


@pytest.fixture(scope="module")
def golden(tmp_path_factory):
    data = json.loads((Path(__file__).parent / "golden" / "fork_golden.json").read_text())
    root = tmp_path_factory.mktemp("chain-costs")
    for name, contents in data["cost_files_chain_intensity"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    return data, load_cost_tables(str(root))


def test_golden_chain_decisions(golden):
    data, tables = golden
    decisions = set()
    count = 0
    for case in data["decisions_chain_intensity"]:
        config = case["case"]
        model = _model(
            tables=tables,
            tree_depth=config["tree_depth"],
            verify_num=config["verify_num"],
            ragged=True,
            eval_batch_size=config["eval_batch_size"],
        )
        costs = model.chain_policy_costs()
        for step in case["script"]:
            if step["op"] == "update_interval":
                model.update_interval(*step["args"])
            elif step["op"] == "chain":
                depth, bs, *deltas = step["args"]
                assert model.chain_stop(*step["args"]) == step["out"]
                terms = costs.batches[bs - 1].intervals[model.interval_index][depth]
                assert terms == model.chain_stop_terms(depth, bs)
                assert _stop_from_terms(terms, deltas[0] if deltas else None) == step["out"]
                decisions.add(step["out"])
                count += 1
    assert count > 100
    assert decisions == {False, True}


def test_custom_parameters_and_ragged_thresholds():
    params = CostModelParams(
        chain_coef=tuple(ChainCoef(t, 0.125, 0.5) for t in (1.0, 2.0, 3.0, 4.0)),
        chain_ref_cost=3.0,
        chain_cliff_min_cost=0.25,
        chain_cliff_ratio=2.0,
        chain_scale_tiny=1.1,
        chain_tiny_max_bs=1,
        chain_scale_small=1.3,
        chain_scale_medium=1.6,
        ragged_draft_relax=(0.5, 0.6, 0.7),
    )
    model = _model(params=params, ragged=True)
    costs = model.chain_policy_costs()
    model.interval_index = 4
    terms = costs.batches[2].intervals[4][2]
    assert terms == model.chain_stop_terms(2, 3)
    assert terms.cost == 0.1 + (2.0 - 1.2)
    assert terms.cliff_cost == 0.2 + (2.05 - 1.2)
    ref_cost = sum(c + 0.05 * (2 * i + 1) for i, c in enumerate((1.1, 1.2, 2, 2.05, 2.1, 2.15, 2.2, 2.25))) / 8
    assert terms.threshold == (4.0 * 0.7) * (3.0 / ref_cost) ** 0.5
    paged = replace(model, ragged=False).chain_stop_terms(2, 3)
    assert paged.threshold == 4.0 * (3.0 / ref_cost) ** 0.5
    tiny = costs.batches[0].intervals[4][2]
    assert tiny.threshold == (3.0 / ref_cost) ** 0.5


@pytest.mark.parametrize(
    "interval,depth,gain,expected",
    [
        (0, 8, 0.5694000000000001, False),
        (0, 10, 0.6630000000000001, True),
        (0, 12, 0.756599999999999, True),
        (1, 2, 0.2886000000000001, False),
    ],
)
def test_preserves_original_eagle_rounding_at_stop_boundary(interval, depth, gain, expected):
    # Original Eagle decisions differ if 2 / gain_scale is precomputed.
    model = _model()
    model.interval_index = interval
    terms = model.chain_stop_terms(depth, 1)
    assert model.chain_stop(depth, 1, [0.0, gain]) is expected
    assert _stop_from_terms(terms, [0.0, gain]) is expected


@pytest.mark.parametrize("gain,expected", [(0.5, True), (1.5, False), (2.0, False)])
def test_cliff_can_rescue_a_stop(gain, expected):
    model = _model()
    model.interval_index = 4
    terms = model.chain_stop_terms(2, 1)
    assert terms.gain_scale == 1.48
    assert terms.cliff_gain_scale == 1.96
    assert terms.cliff_cost == 0.2 + (2.05 - 1.2)
    assert model.chain_stop(2, 1, [0.0, gain]) is expected
    assert _stop_from_terms(terms, [0.0, gain]) is expected
    if gain == 1.5:
        assert gain * 2 / terms.gain_scale / terms.cost < terms.threshold


def test_cliff_preserves_separate_gain_rounding():
    model = _model()
    model.interval_index = 4
    terms = model.chain_stop_terms(2, 3)
    gain = 1.1753189500071877
    assert model.chain_stop(2, 3, [0.0, gain]) is True
    assert _stop_from_terms(terms, [0.0, gain]) is True
    folded_gain = gain * (2.0 / terms.gain_scale + 2.0 / terms.cliff_gain_scale)
    assert folded_gain / terms.cliff_cost >= terms.threshold


@pytest.mark.parametrize(
    "minimum,ratio,enabled",
    [
        (0.5, 1.5, False),
        (math.nextafter(0.5, 0.0), 1.5, True),
        (0.15, 2.0, False),
        (0.15, math.nextafter(2.0, 0.0), True),
    ],
)
def test_cliff_cutoffs_are_strict(minimum, ratio, enabled):
    table = _table()
    target = _curve((1.0, 1.0, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75))
    table = replace(table, intervals=(replace(table.intervals[0], target=target),), model_ratio=(20.0,))
    params = CostModelParams(chain_cliff_min_cost=minimum, chain_cliff_ratio=ratio)
    model = _model(tables={4: table}, params=params)
    terms = model.chain_stop_terms(2, 1)
    assert (terms.cliff_cost > 0.0) is enabled
    assert (terms.cliff_gain_scale > 0.0) is enabled
    if not enabled:
        assert terms.cliff_cost == terms.cliff_gain_scale == 0.0
    last = model.chain_stop_terms(12, 1)
    assert last.cliff_cost == last.cliff_gain_scale == 0.0


@pytest.mark.parametrize("ragged", [False, True])
def test_exported_terms_match_host_across_all_live_batches_and_intervals(ragged):
    model = _model(
        tables={4: _table(), 32: _table(32, step=256, model_ratio=10)},
        ragged=ragged,
    )
    costs = model.chain_policy_costs()
    for bs, batch in enumerate(costs.batches, 1):
        for interval_id, interval in enumerate(batch.intervals):
            model.update_interval(bs, (interval_id + 1) * batch.cache_length_step)
            for depth, terms in enumerate(interval):
                assert model.chain_stop_terms(depth, bs) == terms
                for gain in (0.0, 0.3, 1.5, 10.0):
                    assert model.chain_stop(depth, bs, [0.1, gain]) == _stop_from_terms(terms, [0.1, gain])


def test_export_preserves_values_and_mutable_object_identities():
    model = _model(eval_batch_size=512, ragged=True)
    model.tier.size = SizeClass.HUGE
    model.tier.tiny = True
    model.interval_index = 4
    model.recent_verify_nums[:] = [6] * 5
    for bs in range(1, 513):
        model.record_verify_num(bs, 2 + bs % 15)
    before = copy.deepcopy(model.__dict__)
    objects = dict(model.__dict__)
    histories = tuple(model._history)
    costs = model.chain_policy_costs()
    assert model.__dict__ == before
    assert all(model.__dict__[name] is value for name, value in objects.items())
    assert all(a is b for a, b in zip(histories, model._history))
    model.interval_index = 0
    model.record_verify_num(1, 16)
    assert model.chain_policy_costs() == costs


@pytest.mark.parametrize("model_ratio", [0.5, 4.99, 100.01])
def test_chain_accepts_positive_ratios_outside_tree_adjustment_domain(model_ratio):
    model = _model(tables={4: _table(model_ratio=model_ratio)})
    costs = model.chain_policy_costs()
    assert costs.batches[3].intervals[5][2].cost == 2 / model_ratio


@pytest.mark.parametrize(
    "change",
    [
        {"intensity": ComputeIntensity.HIGH},
        {"intensity": ComputeIntensity.LOW},
        {"node_top_k": 0},
        {"node_top_k": 2},
        {"node_top_k": True},
        {"tree_depth": 1},
        {"tree_depth": 16},
        {"tree_depth": 2.0},
        {"verify_num": 1},
        {"verify_num": 17},
        {"verify_num": 2.0},
        {"tables": {}},
    ],
)
def test_invalid_model_shape(change):
    with pytest.raises((ValueError, NotImplementedError)):
        _model(**change).chain_policy_costs()


@pytest.mark.parametrize(
    "change",
    [
        {"intervals": ()},
        {"model_ratio": ()},
        {"model_ratio": (math.nan,) * 6},
        {"model_ratio": (math.inf,) * 6},
        {"model_ratio": (0.0,) * 6},
        {"model_ratio": (-1.0,) * 6},
        {"cache_length_step": 0},
        {"cache_length_step": -1},
        {"cache_length_step": 1.5},
        {"batch_size": 8},
        {"batch_size": 4.0},
    ],
)
def test_invalid_table_metadata_leaves_state_unchanged(change):
    model = _model(tables={4: replace(_table(), **change)})
    before = copy.deepcopy(model.__dict__)
    with pytest.raises(ValueError):
        model.chain_policy_costs()
    assert model.__dict__ == before


@pytest.mark.parametrize("batch_size", [0, -1, 513, 4.0, True])
def test_invalid_measured_batch_domain(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        _model(tables={batch_size: _table(batch_size)}).chain_policy_costs()


def test_invalid_unselected_table_rejected():
    model = _model()
    model.tables[512] = replace(_table(512), model_ratio=(math.inf,) * 6)
    with pytest.raises(ValueError, match="cost ratio"):
        model.chain_policy_costs()


@pytest.mark.parametrize("curve_name", ["target", "draft"])
@pytest.mark.parametrize(
    "change",
    [
        {"index": ()},
        {"value": ()},
        {"ratio": ()},
        {"index": (0, 1, 2, 5, 7, 9, 11, 13, 15)},
        {"index": (0, 1, 3, 5, 7, 9, 11, 13, 15.0)},
        {"ratio": (1.0, math.nan, 1, 1, 1, 1, 1, 1, 1)},
        {"ratio": (1.0, math.inf, 1, 1, 1, 1, 1, 1, 1)},
        {"ratio": (1.0, 0.0, 1, 1, 1, 1, 1, 1, 1)},
        {"value": (1.0, -1.0, 1, 1, 1, 1, 1, 1, 1)},
    ],
)
def test_invalid_cost_curves(curve_name, change):
    table = _table()
    interval = table.intervals[0]
    curve = replace(getattr(interval, curve_name), **change)
    interval = replace(interval, **{curve_name: curve})
    model = _model(tables={4: replace(table, intervals=(interval,), model_ratio=(20.0,))})
    with pytest.raises(ValueError):
        model.chain_policy_costs()


@pytest.mark.parametrize(
    "change",
    [
        {"chain_coef": ()},
        {"chain_coef": (ChainCoef(0, 0.2, 0),) * 4},
        {"chain_coef": (ChainCoef(2, -0.2, 0),) * 4},
        {"chain_coef": (ChainCoef(2, 0.2, math.nan),) * 4},
        {"chain_ref_cost": 0.0},
        {"chain_ref_cost": math.inf},
        {"chain_cliff_min_cost": -0.1},
        {"chain_cliff_ratio": 0.0},
        {"chain_scale_tiny": math.nan},
        {"chain_scale_small": 1.0},
        {"chain_scale_medium": 1.4},
        {"chain_tiny_max_bs": -1},
        {"chain_tiny_max_bs": 1.5},
        {"ragged_draft_relax": ()},
        {"ragged_draft_relax": (0.75, math.inf, 0.75)},
    ],
)
def test_invalid_chain_parameters(change):
    with pytest.raises(ValueError):
        _model(params=replace(CostModelParams(), **change)).chain_policy_costs()


@pytest.mark.parametrize(
    "change",
    [
        {"chain_coef": (ChainCoef(2, 1e308, 0),) * 4},
        {"chain_coef": (ChainCoef(1e308, 0.24, 1000),) * 4, "chain_ref_cost": 1e308},
    ],
)
def test_nonfinite_derived_terms_rejected_without_state_changes(change):
    model = _model(params=replace(CostModelParams(), **change))
    before = copy.deepcopy(model.__dict__)
    with pytest.raises(ValueError, match="chain stop terms"):
        model.chain_policy_costs()
    assert model.__dict__ == before
