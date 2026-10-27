# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""CPU integration: packed records retain the shared core's policy decisions."""

import pytest
import torch

pytest.importorskip("eagle_harness")

from eagle_harness.core.cost_model import CostModel, CostModelParams, SizeClass
from eagle_harness.core.limits import ComputeIntensity
from eagle_harness.tokenspeed.tree_round.policy_cost import build_tables
from tests.tokenspeed_round.core_costs import cost_fixture, policy_costs


def pack(costs, k, depth, moe, intensity):
    return build_tables(
        policy_costs(costs, k=k, depth=depth, verify=32, moe=moe, intensity=intensity),
        device=torch.device("cpu"),
    )


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
            assert packed.verify_f64[bs - 1, interval].tolist() == [
                verify.threshold * 0.95,
                verify.threshold * 1.05,
            ]
            data = packed.interval_f64[cost_id, interval]
            index = table.intervals[interval].target.index
            assert packed.interval_i32[cost_id, interval].tolist() == [
                len(index),
                *index,
            ]
            for variant in range(5):
                model.tier.size = SizeClass(variant % 4)
                model.tier.tiny = variant == 4
                row = model.draft_cost_row(bs, context)
                assert data[:2].tolist() == [row.cost_state, row.model_cost_ratio]
                width = packed.tier_table[5 + int(model.tier.size)].item()
                for d in (3, 5):
                    threshold = data[2 + variant * depth + d].item()
                    boundary_gain = (
                        threshold
                        / params.async_layer_ratio
                        / row.model_cost_ratio
                        * row.cost_state
                    )
                    for factor in (0.9, 1.1):
                        deltas = [1.0] + [1.0 + boundary_gain * factor] * k
                        model.tier.tiny = variant == 4
                        decision = model.tree_width(d, bs, 32, row, deltas)
                        gain = (
                            (deltas[width] - deltas[0])
                            / data[0].item()
                            * data[1].item()
                            * 0.9
                        )
                        assert (gain < threshold) == decision.stop
            delta = data[2 + 5 * depth :].view(len(index), len(index))
            values = table.intervals[interval].target.ratio
            for a in range(len(index)):
                for b in range(len(index)):
                    expected = (
                        values[b] - values[a]
                        if b > a and values[b] - values[a] > 0.04
                        else 0.0
                    )
                    assert delta[a, b].item() == expected
