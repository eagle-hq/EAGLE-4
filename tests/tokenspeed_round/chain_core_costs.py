# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Integration fixture for the optional installed core cost model."""


def model(depth=15, verify=16, *, custom=False):
    import pytest

    pytest.importorskip("eagle_harness")
    from eagle_harness.core.cost_model import ChainCoef, CostModel, CostModelParams
    from eagle_harness.core.cost_table import BatchCostTable, CostCurve, IntervalCost
    from eagle_harness.core.limits import ComputeIntensity

    index = (0, 1, 3, 5, 7, 9, 11, 13, 15)
    ratios = (
        (1.0, 1.3, 1.4, 2.5, 2.51, 2.6, 2.7, 2.8, 2.9),
        (1.0, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4),
    )
    draft = CostCurve((10.0,), (1.0,), (0,), (), ())
    intervals = tuple(
        IntervalCost(CostCurve(tuple(100 * r for r in row), row, index, (), ()), draft)
        for row in ratios
    )
    # Live batches 1..512 share one measurement, but still have different tiers.
    table = BatchCostTable(4, 1024, 1024, 16, 4096, 1, intervals, (17.0, 29.0))
    params = (
        CostModelParams(
            chain_coef=tuple(ChainCoef(t, 0.125, 0.5) for t in (1.0, 2.0, 3.0, 4.0)),
            chain_ref_cost=3.0,
            chain_cliff_min_cost=0.25,
            chain_cliff_ratio=2.0,
            chain_tiny_max_bs=1,
        )
        if custom
        else CostModelParams()
    )
    return CostModel({4: table}, params, ComputeIntensity.CHAIN, 1, depth, verify)
