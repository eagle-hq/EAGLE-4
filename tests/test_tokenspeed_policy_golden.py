# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Golden contract for core costs and the installed engine's DevicePolicy.

The fixture tests individual calls, not complete serving rounds: draft counts,
verify contexts, and lagged scores are independent inputs. The CUDA replay seeds
those inputs at the verify boundary, then runs the production decision kernels.
Only the K=4 serving case is supported by DevicePolicy; K=6 and evaluation mode
are checked as explicit rejections. CPU replay covers all three golden cases.
"""

import copy
import json
from pathlib import Path

import pytest
import torch

from eagle_harness.core.cost_model import CostModel, CostModelParams
from eagle_harness.core.cost_table import load_cost_tables
from eagle_harness.core.limits import ComputeIntensity
from eagle_harness.tokenspeed.costs import tree_policy_costs
from tests.test_cost_golden import _replay


@pytest.fixture(scope="module")
def golden():
    return json.loads((Path(__file__).parent / "golden" / "fork_golden.json").read_text())


@pytest.fixture(scope="module")
def golden_tables(golden, tmp_path_factory):
    root = tmp_path_factory.mktemp("policy-golden")
    for name, contents in golden["cost_files_high_intensity"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    return load_cost_tables(str(root))


@pytest.fixture(scope="module")
def engine_policy():
    pytest.importorskip("tokenspeed")
    from tokenspeed.runtime.execution.drafter.tree_round import device_policy

    return device_policy


def _model(tables, case):
    return CostModel(
        tables=tables,
        params=CostModelParams(depth_loss_ratio=0.04),
        intensity=ComputeIntensity.HIGH,
        **case["case"],
    )


def _device_script(case):
    script = copy.deepcopy(case["script"])
    delta_count = case["case"]["node_top_k"] + 1
    for step in script:
        if step["op"] == "tree_width":
            step["args"][3] = torch.tensor(step["args"][3][:delta_count], dtype=torch.float32).tolist()
        elif step["op"] == "verify_paged":
            step["args"][0] = torch.tensor(step["args"][0], dtype=torch.float32).tolist()
    return script


def _top_scores(cumulative, requested, batch_size):
    return torch.diff(cumulative, prepend=cumulative.new_zeros(1))[:requested].repeat(batch_size, 1)


def _policy_kwargs(case):
    config = case["case"]
    return {
        "verify_num": config["verify_num"],
        "node_top_k": config["node_top_k"],
        "tree_depth": config["tree_depth"],
        "compute_intensity": ComputeIntensity.HIGH.value,
        "is_moe_target": False,
        "exec_mode": int(config["eval_batch_size"] is not None),
        "device": torch.device("cuda"),
    }


@pytest.mark.parametrize("case_index", range(3), ids=["serving-k4", "serving-k6", "evaluation-k8"])
def test_golden_decisions_survive_device_input_precision(golden, golden_tables, case_index):
    case = golden["decisions_high_intensity"][case_index]
    _replay(_model(golden_tables, case), case["script"])
    _replay(_model(golden_tables, case), _device_script(case))


@pytest.mark.parametrize("flat", [False, True], ids=["mean-ranking", "flat-ranking"])
def test_golden_nonlagged_scores_round_trip(golden, golden_tables, flat):
    case = golden["decisions_high_intensity"][0]
    script = _device_script(case)
    for step in script:
        if step["op"] != "verify_paged" or step["args"][4]:
            continue
        scores, requested, batch_size, _, _ = step["args"]
        top = _top_scores(torch.tensor(scores, dtype=torch.float32), requested, batch_size)
        ranked = top.flatten() if flat else top.mean(dim=0)
        step["args"][0] = ranked.cumsum(dim=0).tolist()
    _replay(_model(golden_tables, case), script)


@pytest.mark.parametrize("case_index", range(3), ids=["serving-k4", "serving-k6", "evaluation-k8"])
def test_engine_packs_golden_costs_exactly(golden, golden_tables, engine_policy, case_index):
    from tokenspeed.runtime.execution.drafter.tree_round.policy_cost import build_tables

    case = golden["decisions_high_intensity"][case_index]
    costs = tree_policy_costs(_model(golden_tables, case))
    packed = build_tables(costs, device=torch.device("cpu"))
    assert packed.tier_table.tolist() == [*costs.history_floors, *costs.initial_widths]
    for live, batch in enumerate(costs.batches):
        cost_id, reserved, intervals, step, chain = packed.batch_table[live].tolist()
        assert [reserved, intervals, step, chain] == [0, len(batch.intervals), batch.cache_length_step, 0]
        for interval_id, interval in enumerate(batch.intervals):
            indices = packed.interval_i32[cost_id, interval_id]
            count = len(interval.verify_widths)
            assert indices[: count + 1].tolist() == [count, *interval.verify_widths]
            thresholds = [value for variant in interval.depth_thresholds for value in variant]
            values = packed.interval_f64[cost_id, interval_id]
            assert values[: 2 + len(thresholds)].tolist() == [
                interval.draft_cost,
                interval.model_cost_ratio,
                *thresholds,
            ]
            deltas = values[2 + len(thresholds) :].reshape(len(indices) - 1, len(indices) - 1)
            assert deltas[:count, :count].tolist() == [list(row) for row in interval.verify_cost_deltas]
            assert packed.verify_i32[live, interval_id].item() == interval.verify_tier
            assert packed.verify_f64[live, interval_id].tolist() == list(interval.verify_thresholds)


@pytest.mark.parametrize("case_index,error", [(1, "K=2/4/8"), (2, "serving only")])
def test_unsupported_golden_cases_are_explicit(golden, golden_tables, engine_policy, case_index, error):
    case = golden["decisions_high_intensity"][case_index]
    with pytest.raises(ValueError, match=error):
        engine_policy.DevicePolicy(tree_policy_costs(_model(golden_tables, case)), **_policy_kwargs(case))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="production DevicePolicy requires CUDA")
def test_device_policy_matches_golden_depth_and_width(golden, golden_tables, engine_policy):
    case = golden["decisions_high_intensity"][0]
    assert case["case"] == {"eval_batch_size": None, "node_top_k": 4, "tree_depth": 12, "verify_num": 64}
    script = _device_script(case)
    model = _model(golden_tables, case)
    _replay(model, script)
    policy = engine_policy.DevicePolicy(tree_policy_costs(model), **_policy_kwargs(case))
    live = torch.zeros(1, dtype=torch.int32, device="cuda")
    frontier = torch.zeros(32, dtype=torch.int64, device="cuda")
    sampled = torch.zeros(1, dtype=torch.bool, device="cuda")

    for step_index, step in enumerate(script):
        op, args, expected = step["op"], step["args"], step["out"]
        where = (step_index, op, args)
        if op == "history":
            past = expected
        elif op == "draft_cost_row":
            batch_size, context = args
            policy.observe_batch_size(batch_size)
            live.fill_(batch_size)
            frontier.fill_(context)
            policy.begin_round(live, frontier, sampled)
            assert policy.past_verify_num.item() == past, where
        elif op == "tree_width":
            depth, _, _, deltas = args
            if depth:
                policy.decide(depth, torch.tensor(deltas, dtype=torch.float32, device="cuda"))
            width, _, stop = expected
            assert bool(policy.go[depth + 1].item()) is not stop, where
            if stop:
                assert policy.last_depth.item() == depth, where
                assert policy.stop_width.item() == width, where
            else:
                assert policy.widths[depth + 1].item() == width, where
        elif op == "verify_count":
            # Its arbitrary draft_count is covered by the core replay above;
            # the next fixture call supplies an independent requested count.
            continue
        elif op == "verify_paged":
            scores, requested, batch_size, context, lagged = args
            flat = policy.flat.item()
            frontier.fill_(context)
            policy.begin_round(live, frontier, sampled)
            policy.flat.fill_(flat)
            policy.verify_count.fill_(requested)
            cumulative = torch.tensor(scores, dtype=torch.float32, device="cuda")
            # The fixture supplies arbitrary lagged scores rather than the
            # previous round's output; use the engine's named state offset.
            policy._state[engine_policy._RANKED_COUNT] = len(scores) if lagged else 0
            if lagged:
                policy._ranked[: len(scores)].copy_(cumulative)
                top = torch.zeros((batch_size, requested), dtype=torch.float32, device="cuda")
            else:
                top = _top_scores(cumulative, requested, batch_size)
            policy.select_verify(top)
            assert expected[1] is False, where
            assert policy.verify_num.item() == expected[0], where
        elif op == "record":
            # select_verify records the decision in the production history.
            assert policy.verify_num.item() == args[1], where
        else:
            raise AssertionError(where)
