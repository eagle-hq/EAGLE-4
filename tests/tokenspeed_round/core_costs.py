# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Optional integration fixtures using the installed adapter's shared core."""

import pytest

pytest.importorskip("eagle_harness")

from eagle_harness.core.cost_model import CostModel, CostModelParams, depth_loss_ratio
from eagle_harness.core.cost_table import BatchCostTable, CostCurve, IntervalCost
from eagle_harness.core.limits import ComputeIntensity


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
        tables[bs] = BatchCostTable(
            bs, 1024, 1024, 64, 64, 8, tuple(intervals), tuple(ratios)
        )
    return tables


def policy_costs(
    costs, *, k, depth, verify, moe, intensity="high_intensity", async_layer_ratio=0.9
):
    intensity = ComputeIntensity(intensity)
    model = CostModel(
        costs,
        CostModelParams(
            depth_loss_ratio=depth_loss_ratio(
                "eagle4", target_is_moe=moe, intensity=intensity
            ),
            async_layer_ratio=async_layer_ratio,
        ),
        intensity,
        k,
        depth,
        verify,
        moe,
    )
    from eagle_harness.tokenspeed.costs import tree_policy_costs

    return tree_policy_costs(model)
