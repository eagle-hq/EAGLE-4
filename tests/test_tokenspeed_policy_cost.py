# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""CPU parity of the packed device policy with the shared core cost model."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from eagle_harness.core.cost_model import CostModel, CostModelParams, SizeClass
from eagle_harness.core.cost_table import BatchCostTable, CostCurve, IntervalCost, load_cost_tables
from eagle_harness.core.limits import ComputeIntensity
from eagle_harness.tokenspeed.costs import tree_policy_costs
from eagle_harness.tokenspeed.tree_round.device_policy import DevicePolicy, write_round_slots
from eagle_harness.tokenspeed.tree_round.policy_cost import build_tables


def cost_fixture():
    """Different measured tiers, context intervals and model ratios."""
    tables = {}
    for bs, slope in ((1, 0.03), (4, 0.07), (8, 0.13), (32, 0.2)):
        intervals = []
        ratios = []
        for interval in range(2):
            index = (0, 1, 3, 7, 15, 31, 63)
            target = CostCurve(
                (100.0,) * len(index),
                tuple(1 + i * slope * (1 + 0.1 * interval) for i in index),
                index,
                (),
                (),
            )
            draft = CostCurve((10.0,) * 4, (1.0, 1.1, 1.2, 1.4), (0, 1, 3, 7), (), ())
            intervals.append(IntervalCost(target, draft))
            ratios.append(10.0 + interval * 10)
        tables[bs] = BatchCostTable(bs, 1024, 1024, 64, 64, 8, tuple(intervals), tuple(ratios))
    return tables


def pack(costs, k=4, depth=8, moe=False, intensity="high_intensity"):
    model = CostModel(
        costs,
        CostModelParams(depth_loss_ratio=0.08 if moe else 0.04),
        ComputeIntensity(intensity),
        k,
        depth,
        32,
        moe,
    )
    return build_tables(tree_policy_costs(model), device=torch.device("cpu"))


@pytest.mark.parametrize("intensity", ["high_intensity", "low_intensity"])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("k", [2, 4, 8])
def test_table_decisions_match_core(k, moe, intensity):
    costs = cost_fixture()
    depth = 8
    packed = pack(costs, k, depth, moe, intensity)
    params = CostModelParams(depth_loss_ratio=0.08 if moe else 0.04)
    model = CostModel(costs, params, ComputeIntensity(intensity), k, depth, 32, moe)
    assert packed.tier_table.tolist() == [16, 8, 4, 2, 32, k, k, min(k, 4), min(k, 4)]

    for bs in (1, 2, 3, 4, 6, 8, 20, 32):
        cost_id = packed.batch_table[bs - 1, 0].item()
        table = model.table(bs)
        for interval in range(2):
            context = (interval + 1) * table.cache_length_step
            verify = model.verify_cost(bs, context)
            assert packed.verify_i32[bs - 1, interval].item() == int(model.tier.size)
            assert packed.verify_f64[bs - 1, interval].tolist() == [verify.threshold * 0.95, verify.threshold * 1.05]
            data = packed.interval_f64[cost_id, interval]
            index = table.intervals[interval].target.index
            assert packed.interval_i32[cost_id, interval].tolist() == [len(index), *index]
            for variant in range(5):
                model.tier.size = SizeClass(variant % 4)
                model.tier.tiny = variant == 4
                row = model.draft_cost_row(bs, context)
                assert data[:2].tolist() == [row.cost_state, row.model_cost_ratio]
                width = packed.tier_table[5 + int(model.tier.size)].item()
                for d in (3, 5):
                    threshold = data[2 + variant * depth + d].item()
                    boundary_gain = threshold / params.async_layer_ratio / row.model_cost_ratio * row.cost_state
                    for factor in (0.9, 1.1):
                        deltas = [1.0] + [1.0 + boundary_gain * factor] * k
                        model.tier.tiny = variant == 4
                        decision = model.tree_width(d, bs, 32, row, deltas)
                        gain = (deltas[width] - deltas[0]) / data[0].item() * data[1].item() * 0.9
                        assert (gain < threshold) == decision.stop
            delta = data[2 + 5 * depth :].view(len(index), len(index))
            values = table.intervals[interval].target.ratio
            for a in range(len(index)):
                for b in range(len(index)):
                    expected = values[b] - values[a] if b > a and values[b] - values[a] > 0.04 else 0.0
                    assert delta[a, b].item() == expected


def test_shared_loader_on_existing_golden_files(tmp_path):
    golden = json.loads((Path(__file__).parent / "golden/fork_golden.json").read_text())
    for name, content in golden["cost_files_high_intensity"].items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    costs = load_cost_tables(str(tmp_path))
    assert costs
    assert all(isinstance(table, BatchCostTable) for table in costs.values())
    packed = pack(costs)
    assert torch.isfinite(packed.interval_f64).all()


@pytest.mark.parametrize(
    "change",
    [
        {"cache_length_step": 0},
        {"cache_length_step": -1},
        {"model_ratio": (float("nan"), 10.0)},
        {"model_ratio": (4.9, 10.0)},
        {"model_ratio": (101.0, 10.0)},
        {"model_ratio": ()},
        {"intervals": ()},
    ],
)
def test_bad_cost_data_rejected_before_cuda(change):
    table = cost_fixture()[1]
    with pytest.raises(ValueError):
        pack({1: replace(table, **change)})


@pytest.mark.parametrize(
    "index,error",
    [
        ((1, 3), "Invalid target cost indices"),
        ((0, 3, 1), "Invalid target cost indices"),
        ((0, 1, 1), "Invalid target cost indices"),
        ((0, 2**31), "Invalid target cost indices"),
        (tuple(range(33)), "Invalid target width count 33"),
    ],
)
def test_bad_widths_rejected(index, error):
    table = cost_fixture()[1]
    interval = table.intervals[0]
    target = replace(interval.target, index=index, value=(100.0,) * len(index), ratio=(1.0,) * len(index))
    with pytest.raises(ValueError, match=error):
        pack({1: replace(table, intervals=(replace(interval, target=target),), model_ratio=(10.0,))})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"node_top_k": 1},
        {"node_top_k": 3},
        {"node_top_k": 16},
        {"tree_depth": 1},
        {"tree_depth": 33},
        {"tree_depth": 18},
        {"verify_num": 1},
        {"verify_num": 65},
        {"compute_intensity": "chain_intensity"},
        {"exec_mode": 1},
        {"exec_mode": 2},
        {"chain_width_table": ((1, 4),)},
        {"device": torch.device("cpu")},
    ],
)
def test_unsupported_policy_rejected_without_build(kwargs):
    config = {
        "verify_num": 32,
        "node_top_k": 4,
        "tree_depth": 8,
        "compute_intensity": "high_intensity",
        "is_moe_target": False,
        "exec_mode": 0,
        "benchmark_config": "",
        "chain_width_table": (),
        "device": torch.device("cuda"),
    }
    config.update(kwargs)
    with pytest.raises(ValueError):
        DevicePolicy({}, **config)


def test_warmup_slots_and_sequence_tagged_polling():
    slots = torch.full((4, 5), -1, dtype=torch.int32)
    count = write_round_slots(slots, 4, [4, 2, 2], lambda d, n: min(n, 16) if d % 2 == 0 else 0)
    assert count == 37
    assert slots.tolist() == [[0, 4, 2, 2, 0], [0, 5, 21, 29, 0], [0, 0, 16, 0, 0], [0, 1, 1, 1, 0]]
    policy = object.__new__(DevicePolicy)
    policy.chain = False
    policy._go_host_view = torch.tensor([0, 15, 14]).numpy()
    assert policy.go_ready(1, 7) is True
    assert policy.go_ready(2, 7) is False
    assert policy.go_ready(1, 8) is None
    with pytest.raises(ValueError, match="live requests"):
        policy.observe_batch_size(33)
