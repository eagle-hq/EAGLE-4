# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Chain and sampled policy state across graph replays; no model weights needed."""

import math
from dataclasses import replace

import pytest
import torch

pytest.importorskip("tvm_ffi", reason="TokenSpeed round kernels require apache-tvm-ffi")

from eagle_harness.tokenspeed.tree_round.device_policy import DevicePolicy
from tests.tokenspeed_round.chain_cost_records import records
from tests.tokenspeed_round.cost_records import BatchCosts, IntervalCosts, PolicyCosts

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU is required"
)


def chain_policy(*, depth=7, verify=8, widths=(), costs=None):
    return DevicePolicy(
        records(depth) if costs is None else costs,
        verify_num=verify,
        node_top_k=1,
        tree_depth=depth,
        compute_intensity="chain_intensity",
        is_moe_target=False,
        exec_mode=0,
        benchmark_config="",
        chain_width_table=widths,
        device=torch.device("cuda"),
    )


def capture_round(policy):
    capacity = 512 if policy.chain else 32
    live = torch.zeros(1, dtype=torch.int32, device="cuda")
    # Exercise the existing strided frontier contract.
    frontier = torch.zeros((capacity, 2), dtype=torch.int64, device="cuda")[:, 0]
    sampled = torch.zeros(1, dtype=torch.bool, device="cuda")
    deltas = torch.zeros(policy.node_top_k + 1, device="cuda")
    scores = torch.zeros((capacity, policy.verify_num_limit), device="cuda")

    def run_round():
        policy.begin_round(live, frontier, sampled)
        for depth in range(1, policy.tree_depth):
            policy.decide(depth, deltas)
        policy.select_verify(scores)

    # Resolve the FFI launch paths before capture; idle warmups are not rounds.
    run_round()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_round()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, live, frontier, sampled, deltas, scores


def replay(inputs, *, batch, gain, sample):
    graph, live, frontier, sampled, deltas, _ = inputs
    live.fill_(batch)
    frontier.fill_(1024)
    sampled.fill_(sample)
    deltas.zero_()
    deltas[1:].fill_(gain)
    graph.replay()
    torch.cuda.synchronize()


@pytest.mark.parametrize("depth,verify", [(2, 2), (3, 4), (7, 8), (15, 16), (15, 4)])
def test_chain_greedy_stop_and_full_round(depth, verify):
    policy = chain_policy(depth=depth, verify=verify)
    inputs = capture_round(policy)
    for batch in (1, 2, 3, 33, 512):
        policy.observe_batch_size(batch)
        for gain in (0.0, 100000.0):
            replay(inputs, batch=batch, gain=gain, sample=False)
            last = min(2, depth - 1) if gain == 0 else depth - 1
            assert policy.last_depth.item() == last
            assert policy.cumu_count.item() == last + 2
            assert policy.verify_num.item() == min(last + 2, verify)
            assert policy.verify_count.item() == policy.verify_num.item()
            assert policy.past_verify_num.item() == 16
            assert policy.widths[1 : last + 1].tolist() == [1] * last
            assert policy.cumu_starts[1 : last + 1].tolist() == list(range(2, last + 2))
            assert policy.work_sizes[1 : last + 1].tolist() == [
                d + 1 if d % 2 and d < depth - 1 else 0 for d in range(1, last + 1)
            ]
            assert policy.go[last + 1 :].count_nonzero().item() == 0
    with pytest.raises(ValueError):
        policy.observe_batch_size(513)


def test_sampled_chain_updates_next_plan_without_using_current_proposals():
    policy = chain_policy()
    inputs = capture_round(policy)
    # First sampled depth is 0. Low gain holds the next plan at 2; high gain
    # grows it. Current low gain at depth 2 cannot shorten a planned depth 4.
    script = [
        (True, 0.0, 0),
        (True, 0.0, 2),
        (True, 100000.0, 2),
        (True, 0.0, 4),
        (False, 100000.0, 6),
        (True, 100000.0, 2),
        (True, 100000.0, 4),
        (True, 100000.0, 6),
    ]
    for seq, (sampled, gain, last) in enumerate(script, start=1):
        replay(inputs, batch=512, gain=gain, sample=sampled)
        assert policy.last_depth.item() == last
        assert policy.cumu_count.item() == last + 2
        assert policy.verify_num.item() == last + 2
        assert policy.go_ready(last + 1, seq) is False
    replay(inputs, batch=0, gain=0.0, sample=True)
    assert policy.cumu_count.item() == 0
    replay(inputs, batch=33, gain=0.0, sample=True)
    assert policy.last_depth.item() == 6


def test_chain_width_table_overrides_sampled_and_greedy_without_changing_plan():
    policy = chain_policy(widths=((2, 2), (8, 3), (32, 8)))
    inputs = capture_round(policy)
    replay(inputs, batch=1, gain=0.0, sample=True)
    assert policy.last_depth.item() == 0  # next uncovered plan is 2
    for seq, (batch, width, sample) in enumerate(
        [
            (2, 2, True),
            (7, 2, False),
            (8, 3, True),
            (31, 3, False),
            (32, 8, True),
            (512, 8, False),
        ],
        start=2,
    ):
        replay(inputs, batch=batch, gain=0.0, sample=sample)
        assert policy.verify_num.item() == width
        assert policy.cumu_count.item() == width
        assert policy.last_depth.item() == width - 2
        assert policy.work_sizes.count_nonzero().item() == 0
        assert policy.go_ready(width - 1, seq) is False
        assert policy.go[1:].tolist() == [int(d <= width - 2) for d in range(1, 8)]
    replay(inputs, batch=1, gain=0.0, sample=True)
    assert policy.last_depth.item() == 2


def tree_policy(async_layer_ratio):
    interval = IntervalCosts(
        verify_widths=(0, 1, 5),
        verify_cost_deltas=((0.0, 0.1, 0.6), (0.0, 0.0, 0.5), (0.0, 0.0, 0.0)),
        draft_cost=1.0,
        model_cost_ratio=10.0,
        depth_thresholds=((1.0,) * 7,) * 5,
        verify_tier=0,
        verify_thresholds=(2.0, 2.0),
    )
    costs = PolicyCosts(
        (BatchCosts(4, 1024, (interval,)),) * 32,
        (16, 8, 4, 2, 32),
        (2, 2, 2, 2),
        async_layer_ratio,
    )
    return DevicePolicy(
        costs,
        verify_num=8,
        node_top_k=2,
        tree_depth=7,
        compute_intensity="high_intensity",
        is_moe_target=False,
        exec_mode=0,
        benchmark_config="",
        chain_width_table=(),
        device=torch.device("cuda"),
    )


@pytest.mark.parametrize("first_score,lagged_width", [(0.0, 2), (1.0, 6)])
def test_sampled_tree_first_verify_is_unranked_then_uses_previous_scores(
    first_score, lagged_width
):
    policy = tree_policy(0.73)
    inputs = capture_round(policy)
    inputs[-1].fill_(first_score)
    replay(inputs, batch=4, gain=0.0, sample=True)
    assert policy.last_depth.item() == 6
    assert policy.verify_num.item() == 6
    inputs[-1].fill_(1.0 - first_score)
    replay(inputs, batch=4, gain=100.0, sample=True)
    assert policy.last_depth.item() == 1  # prior round's earliest cost stop
    assert policy.verify_num.item() == lagged_width
    replay(inputs, batch=4, gain=0.0, sample=True)
    assert policy.last_depth.item() == 3  # previous high gain grew the plan
    replay(inputs, batch=4, gain=100.0, sample=True)
    assert policy.last_depth.item() == 1


@pytest.mark.parametrize("ratio,last", [(0.73, 1), (0.9, 6)])
def test_tree_retains_custom_async_ratio(ratio, last):
    policy = tree_policy(ratio)
    replay(capture_round(policy), batch=4, gain=0.13, sample=False)
    assert policy.last_depth.item() == last


@pytest.mark.parametrize("custom", [False, True])
def test_chain_rounds_match_core_cost_model(custom):
    from tests.tokenspeed_round.chain_core_costs import model

    core = model(custom=custom)
    policy = chain_policy(depth=15, verify=16, costs=core.chain_policy_costs())
    graph, live, frontier, _sampled, deltas, _ = capture_round(policy)
    generator = torch.Generator().manual_seed(920)
    for bs in (1, 2, 3, 4, 33, 512):
        live.fill_(bs)
        for context in (0, 2047, 2048, 8192):
            frontier.fill_(context)
            core.update_interval(bs, context)
            for gain in [0.0, 10.0, *torch.rand(16, generator=generator).tolist()]:
                # These are the actual float32 inputs the device widens.
                host_deltas = torch.tensor(
                    [0.1, 0.1 + gain], dtype=torch.float32
                ).tolist()
                deltas.copy_(torch.tensor(host_deltas))
                last = next(
                    (
                        d
                        for d in range(1, 15)
                        if d == 14 or core.chain_stop(d, bs, host_deltas)
                    ),
                    14,
                )
                graph.replay()
                torch.cuda.synchronize()
                assert policy.last_depth.item() == last
                assert policy.verify_num.item() == last + 2


@pytest.mark.parametrize("above_boundary", [False, True])
def test_chain_gain_division_preserves_rounding_order(above_boundary):
    costs = records(depth=4)
    delta_values = torch.tensor([0.163, 0.3], dtype=torch.float32).tolist()
    raw_gain = delta_values[1] - delta_values[0]
    denominator = 1.6454386831000494
    threshold = raw_gain * 2.0 / denominator
    # This input distinguishes division from multiplying a precomputed reciprocal.
    assert raw_gain * (2.0 / denominator) < threshold
    if above_boundary:
        threshold = math.nextafter(threshold, math.inf)
    batch = costs.batches[0]
    intervals = []
    for interval in batch.intervals:
        terms = replace(
            interval[2],
            gain_scale=denominator,
            cost=1.0,
            threshold=threshold,
            cliff_gain_scale=0.0,
            cliff_cost=0.0,
        )
        intervals.append((*interval[:2], terms, interval[3]))
    costs = replace(
        costs, batches=(replace(batch, intervals=tuple(intervals)), *costs.batches[1:])
    )
    policy = chain_policy(depth=4, verify=5, costs=costs)
    graph, live, _frontier, _sampled, deltas, _ = capture_round(policy)
    live.fill_(1)
    deltas.copy_(torch.tensor(delta_values))
    graph.replay()
    torch.cuda.synchronize()
    assert policy.last_depth.item() == (2 if above_boundary else 3)


@pytest.mark.parametrize("cliff", [False, True])
def test_chain_cliff_can_rescue_a_stopping_decision(cliff):
    costs = records(depth=4)
    batch = costs.batches[0]
    interval = batch.intervals[0]
    terms = replace(
        interval[2],
        gain_scale=2.0,
        cost=1.0,
        threshold=0.2,
        cliff_gain_scale=1.0 if cliff else 0.0,
        cliff_cost=1.0 if cliff else 0.0,
    )
    interval = (*interval[:2], terms, interval[3])
    costs = replace(
        costs, batches=(replace(batch, intervals=(interval,)), *costs.batches[1:])
    )
    policy = chain_policy(depth=4, verify=5, costs=costs)
    replay(capture_round(policy), batch=1, gain=0.125, sample=False)
    assert policy.last_depth.item() == (3 if cliff else 2)


def test_chain_boundary_rejects_wrong_frontier_and_delta_shapes():
    policy = chain_policy()
    live = torch.ones(1, dtype=torch.int32, device="cuda")
    sampled = torch.zeros(1, dtype=torch.bool, device="cuda")
    with pytest.raises(Exception, match="frontier.ndim"):
        policy.begin_round(
            live, torch.zeros((1, 1), dtype=torch.int64, device="cuda"), sampled
        )
    with pytest.raises(Exception, match="work deltas"):
        policy.decide(2, torch.zeros(3, device="cuda"))


@pytest.mark.parametrize("rows,cols,capacity", [(33, 2, 128), (2, 65, 128), (2, 2, 1)])
def test_ranked_score_capacity_checks_remain_effective(rows, cols, capacity):
    from eagle_harness.tokenspeed.tree_round.ops.policy import (
        eagle_harness_ranked_scores,
    )

    top = torch.zeros((33, 65), device="cuda")
    scalars = torch.tensor([rows, cols, 0, 0], dtype=torch.int32, device="cuda")
    r, c, flat, count = scalars.split(1)
    out = torch.full((capacity,), 123.0, device="cuda")
    eagle_harness_ranked_scores(top, r, c, flat, out, count)
    assert count.item() == -1
    assert torch.all(out == 123.0).item()
