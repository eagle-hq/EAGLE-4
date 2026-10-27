# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""CPU packing and startup rejection for structural chain policy records."""

from dataclasses import astuple, replace

import pytest
import torch

from eagle_harness.tokenspeed.tree_round.device_policy import DevicePolicy
from eagle_harness.tokenspeed.tree_round.policy_cost import (
    build_chain_tables,
)
from tests.tokenspeed_round.chain_cost_records import records


def pack(costs, *, depth=7, verify=8, widths=()):
    return build_chain_tables(
        costs,
        tree_depth=depth,
        verify_num=verify,
        chain_width_table=widths,
        device=torch.device("cpu"),
    )


def test_pack_each_live_batch_exactly_and_zero_padding():
    costs = records()
    tables = pack(costs)
    assert tables.chain_f64.shape == (512, 2, 2, 7, 5)
    assert tables.chain_f64.dtype == torch.float64
    assert tables.batch_table.dtype == torch.int64
    for i, batch in enumerate(costs.batches):
        assert tables.batch_table[i].tolist() == [
            i,
            0,
            len(batch.intervals),
            batch.cache_length_step,
            0,
        ]
        for j, interval in enumerate(batch.intervals):
            expected = [list(astuple(terms)) for terms in interval]
            assert tables.chain_f64[i, j, 0].tolist() == expected
            assert tables.chain_f64[i, j, 1].tolist() == expected
        assert torch.count_nonzero(tables.chain_f64[i, len(batch.intervals) :]) == 0
    for unused in (
        tables.interval_i32,
        tables.interval_f64,
        tables.tier_table,
        tables.verify_i32,
        tables.verify_f64,
    ):
        assert unused.numel() == 0


def test_width_table_uses_greatest_covered_live_threshold():
    tables = pack(records(), widths=((4, 8), (16, 4), (512, 2)))
    widths = tables.batch_table[:, 4].tolist()
    assert widths == [0] * 3 + [8] * 12 + [4] * 496 + [2]


@pytest.mark.parametrize(
    "widths",
    [
        ((0, 2),),
        ((4, 2), (4, 4)),
        ((5, 2), (4, 4)),
        ((1, 1),),
        ((1, 9),),
        ((True, 4),),
        ((1, 2.0),),
        ((1, 2, 3),),
    ],
)
def test_invalid_width_table_rejected(widths):
    with pytest.raises(ValueError, match="width table"):
        pack(records(), widths=widths)


@pytest.mark.parametrize("depth,verify", [(1, 8), (16, 8), (7, 1), (7, 17)])
def test_invalid_chain_shape_rejected(depth, verify):
    with pytest.raises(ValueError):
        pack(records(), depth=depth, verify=verify)


@pytest.mark.parametrize(
    "change",
    [{"cache_length_step": 0}, {"cache_length_step": 2**63}, {"intervals": ()}],
)
def test_invalid_chain_batch_rejected(change):
    costs = records()
    with pytest.raises(ValueError):
        pack(
            replace(
                costs, batches=(replace(costs.batches[0], **change), *costs.batches[1:])
            )
        )


def test_chain_batch_count_and_depth_must_match():
    with pytest.raises(ValueError):
        pack(replace(records(), batches=records().batches[:-1]))
    with pytest.raises(ValueError):
        pack(records(depth=5))


@pytest.mark.parametrize(
    "depth,change",
    [
        (2, {"gain_scale": 0.0}),
        (2, {"cost": -1.0}),
        (2, {"threshold": float("nan")}),
        (2, {"cliff_gain_scale": 0.0}),
        (2, {"cliff_cost": 0.0}),
        (2, {"cliff_cost": float("inf")}),
        (1, {"gain_scale": 1.0}),
    ],
)
def test_invalid_stop_terms_rejected(depth, change):
    costs = records()
    batch = costs.batches[0]
    interval = list(batch.intervals[0])
    interval[depth] = replace(interval[depth], **change)
    batch = replace(batch, intervals=(tuple(interval), *batch.intervals[1:]))
    with pytest.raises(ValueError):
        pack(replace(costs, batches=(batch, *costs.batches[1:])))


@pytest.mark.parametrize(
    "change",
    [
        {"node_top_k": 2},
        {"tree_depth": 16},
        {"verify_num": 17},
        {"exec_mode": 1},
        {"benchmark_config": "eval"},
        {"chain_width_table": ((1, 9),)},
        {"device": torch.device("cpu")},
    ],
)
def test_invalid_chain_policy_rejected_before_cuda(change):
    kwargs = {
        "verify_num": 8,
        "node_top_k": 1,
        "tree_depth": 7,
        "compute_intensity": "chain_intensity",
        "is_moe_target": False,
        "exec_mode": 0,
        "benchmark_config": "",
        "chain_width_table": (),
        "device": torch.device("cuda"),
    }
    kwargs.update(change)
    with pytest.raises(ValueError):
        DevicePolicy(records(), **kwargs)


@pytest.mark.parametrize("custom", [False, True])
def test_core_exports_pack_without_recomputing_terms(custom):
    from tests.tokenspeed_round.chain_core_costs import model

    core = model(depth=7, verify=8, custom=custom)
    costs = core.chain_policy_costs()
    tables = pack(costs)
    # Shared measured batch 4 must not merge the live-batch tier distinction.
    assert (
        tables.chain_f64[0, 0, 0, 2].tolist() != tables.chain_f64[2, 0, 0, 2].tolist()
    )
    for bs in (1, 2, 3, 4, 33, 512):
        for interval_id, interval in enumerate(costs.batches[bs - 1].intervals):
            for depth, terms in enumerate(interval):
                assert tables.chain_f64[bs - 1, interval_id, 0, depth].tolist() == list(
                    astuple(terms)
                )
