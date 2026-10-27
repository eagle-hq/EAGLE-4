# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Exact CPU packing and validation of caller-owned structural cost records."""

from dataclasses import replace

import pytest
import torch

from eagle_harness.tokenspeed.tree_round.device_policy import (
    DevicePolicy,
    write_round_slots,
)
from eagle_harness.tokenspeed.tree_round.policy_cost import build_tables
from tests.tokenspeed_round.cost_records import records


def test_pack_records_without_policy_recomputation():
    costs = records()
    tables = build_tables(costs, device=torch.device("cpu"))
    assert tables.interval_i32.dtype == torch.int32
    assert tables.interval_f64.dtype == torch.float64
    assert tables.batch_table.dtype == torch.int64
    assert tables.tier_table.dtype == torch.int32
    assert tables.verify_i32.dtype == torch.int32
    assert tables.verify_f64.dtype == torch.float64
    assert tables.interval_i32.shape == (2, 2, 4)
    assert tables.interval_f64.shape == (2, 2, 21)
    assert tables.tier_table.tolist() == [16, 8, 4, 2, 32, 2, 2, 2, 2]
    assert tables.interval_i32[0].tolist() == [[2, 0, 3, 0], [0, 0, 0, 0]]
    assert torch.count_nonzero(tables.interval_f64[0, 1]) == 0
    for live, batch in enumerate(costs.batches):
        cost_id = live % 2
        assert tables.batch_table[live].tolist() == [
            cost_id,
            0,
            len(batch.intervals),
            batch.cache_length_step,
            0,
        ]
        for i, interval in enumerate(batch.intervals):
            n = len(interval.verify_widths)
            packed = tables.interval_f64[cost_id, i]
            assert packed[:2].tolist() == [
                interval.draft_cost,
                interval.model_cost_ratio,
            ]
            assert packed[2:12].view(5, 2).tolist() == [
                list(row) for row in interval.depth_thresholds
            ]
            delta = packed[12:].view(3, 3)
            assert delta[:n, :n].tolist() == [
                list(row) for row in interval.verify_cost_deltas
            ]
            assert torch.count_nonzero(delta[n:]) == 0
            assert torch.count_nonzero(delta[:, n:]) == 0
            assert tables.verify_i32[live, i].item() == interval.verify_tier
            assert tables.verify_f64[live, i].tolist() == list(
                interval.verify_thresholds
            )
        if len(batch.intervals) == 1:
            assert tables.verify_i32[live, 1] == 0
            assert torch.count_nonzero(tables.verify_f64[live, 1]) == 0


@pytest.mark.parametrize(
    "change",
    [
        {"batches": ()},
        {"history_floors": (1,)},
        {"initial_widths": (1,)},
        {"history_floors": (16, 8, 4, 2, 2**31)},
        {"initial_widths": (2, 2, 2, 9)},
        {"async_layer_ratio": float("nan")},
        {"async_layer_ratio": 0.0},
    ],
)
def test_invalid_cost_records_rejected(change):
    with pytest.raises(ValueError):
        build_tables(replace(records(), **change), device=torch.device("cpu"))


@pytest.mark.parametrize(
    "change",
    [
        {"cost_batch_size": 0},
        {"cache_length_step": 0},
        {"cache_length_step": -1},
        {"cache_length_step": 2**63},
        {"intervals": ()},
    ],
)
def test_invalid_batch_rejected(change):
    costs = records()
    costs = replace(
        costs, batches=(replace(costs.batches[0], **change), *costs.batches[1:])
    )
    with pytest.raises(ValueError):
        build_tables(costs, device=torch.device("cpu"))


@pytest.mark.parametrize(
    "change",
    [
        {"verify_widths": (1, 3)},
        {"verify_widths": (0, 3, 1)},
        {"verify_widths": (0, 1, 1)},
        {"verify_widths": (0, 2**31)},
        {"verify_widths": tuple(range(33))},
        {"draft_cost": 0.0},
        {"draft_cost": float("nan")},
        {"model_cost_ratio": 4.9},
        {"model_cost_ratio": 101.0},
        {"model_cost_ratio": float("nan")},
        {"depth_thresholds": ()},
        {"depth_thresholds": ((0.1,),) * 5},
        {"depth_thresholds": ((0.1, float("nan")),) * 5},
        {"verify_cost_deltas": ((0.0, -0.1), (0.0, 0.0))},
        {"verify_cost_deltas": ((0.0,),)},
        {"verify_tier": 4},
        {"verify_thresholds": (0.0, 1.0)},
    ],
)
def test_invalid_interval_rejected(change):
    costs = records()
    batch = costs.batches[0]
    batch = replace(batch, intervals=(replace(batch.intervals[0], **change),))
    with pytest.raises(ValueError):
        build_tables(
            replace(costs, batches=(batch, *costs.batches[1:])),
            device=torch.device("cpu"),
        )


def test_conflicting_measured_rows_cannot_be_silently_discarded():
    costs = records()
    batch = costs.batches[2]
    batch = replace(batch, intervals=(replace(batch.intervals[0], draft_cost=9.0),))
    with pytest.raises(ValueError, match="conflicting costs"):
        build_tables(
            replace(costs, batches=(*costs.batches[:2], batch, *costs.batches[3:])),
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "change",
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
        {"benchmark_config": "unsupported"},
        {"chain_width_table": ((1, 4),)},
        {"device": torch.device("cpu")},
    ],
)
def test_unsupported_policy_rejected_without_build(change):
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
    config.update(change)
    with pytest.raises(ValueError):
        DevicePolicy(records(), **config)


def test_warmup_slots():
    slots = torch.full((4, 5), -1, dtype=torch.int32)
    count = write_round_slots(
        slots, 4, [4, 2, 2], lambda d, n: min(n, 16) if d % 2 == 0 else 0
    )
    assert count == 37
    assert slots.tolist() == [
        [0, 4, 2, 2, 0],
        [0, 5, 21, 29, 0],
        [0, 0, 16, 0, 0],
        [0, 1, 1, 1, 0],
    ]
