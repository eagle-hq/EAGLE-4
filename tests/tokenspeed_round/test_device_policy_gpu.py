# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Graph-replayed greedy policy decisions against core CostModel."""

import sys

import pytest
import torch

pytest.importorskip("tvm_ffi", reason="TokenSpeed round kernels require apache-tvm-ffi")

pytest.importorskip("eagle_harness")

from eagle_harness.core.cost_model import CostModel, CostModelParams
from eagle_harness.core.limits import ComputeIntensity
from eagle_harness.tokenspeed.tree_round.device_policy import DevicePolicy
from tests.tokenspeed_round.core_costs import cost_fixture, policy_costs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU is required"
)


@pytest.mark.parametrize("async_layer_ratio", [0.9, 0.73])
@pytest.mark.parametrize("k", [2, 4, 8])
@pytest.mark.parametrize("moe", [False, True])
def test_captured_policy_rounds_match_core(k, moe, async_layer_ratio):
    device, depth, verify = torch.device("cuda"), 8, 32
    costs = cost_fixture()
    policy = DevicePolicy(
        policy_costs(
            costs,
            k=k,
            depth=depth,
            verify=verify,
            moe=moe,
            async_layer_ratio=async_layer_ratio,
        ),
        verify_num=verify,
        node_top_k=k,
        tree_depth=depth,
        compute_intensity="high_intensity",
        is_moe_target=moe,
        exec_mode=0,
        benchmark_config="",
        chain_width_table=(),
        device=device,
    )
    model = CostModel(
        costs,
        CostModelParams(
            depth_loss_ratio=0.08 if moe else 0.04, async_layer_ratio=async_layer_ratio
        ),
        ComputeIntensity.HIGH,
        k,
        depth,
        verify,
        moe,
    )
    live = torch.zeros(1, dtype=torch.int32, device=device)
    frontier = torch.zeros(32, dtype=torch.int64, device=device)
    sampled = torch.zeros(1, dtype=torch.bool, device=device)
    deltas = torch.zeros(k + 1, device=device)
    scores = torch.zeros((32, verify), device=device)
    # Load/warm launch paths with no live round before capture.
    policy.begin_round(live, frontier, sampled)
    for d in range(1, depth):
        policy.decide(d, deltas)
    policy.select_verify(scores)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        policy.begin_round(live, frontier, sampled)
        for d in range(1, depth):
            policy.decide(d, deltas)
        policy.select_verify(scores)
    torch.cuda.current_stream().wait_stream(stream)

    previous = None
    generator = torch.Generator().manual_seed(1709 + k)
    for round_id, (bs, gain) in enumerate(
        [(1, 0.02), (32, 0.5), (4, 0.001), (1, 0.9)] * 5, 2
    ):
        context = 1024 if round_id % 2 else 2300
        policy.observe_batch_size(bs)
        live.fill_(bs)
        frontier.fill_(context)
        delta_values = torch.linspace(1.0, 1.0 + gain, k + 1).tolist()
        deltas.copy_(torch.tensor(delta_values))
        top = (
            torch.rand((bs, verify), generator=generator)
            .sort(dim=-1, descending=True)
            .values
        )
        top[:, 0] = 1.0
        scores[:bs].copy_(top)
        past = model.history_verify_num(bs)
        row = model.draft_cost_row(bs, context)
        width = model.tree_width(0, bs, past, row).width
        start, expected_widths = 1 + k, []
        for d in range(1, depth):
            expected_widths.append(width)
            decision = model.tree_width(d, bs, past, row, delta_values)
            start += k * width
            if decision.stop:
                break
            width = decision.width
        count = model.verify_count(bs, start)
        graph.replay()
        torch.cuda.synchronize()
        assert policy.past_verify_num.item() == past
        assert policy.cumu_count.item() == start
        assert policy.last_depth.item() == d
        assert policy.widths[1 : d + 1].tolist() == expected_widths
        assert policy.verify_count.item() == count
        assert policy.go[d + 1].item() == 0
        own = scores[:bs, :count].contiguous()
        ranked = torch.cumsum(
            own.view(-1) if model.tier.tiny else own.mean(dim=0), dim=0
        ).tolist()
        wanted, stop = model.verify_num_paged(
            previous if previous is not None else ranked,
            count,
            bs,
            context,
            lagged=previous is not None,
        )
        assert not stop
        assert policy.verify_num.item() == wanted
        model.record_verify_num(bs, wanted)
        previous = ranked


def test_tiny_two_depth_round_always_finalizes():
    policy = DevicePolicy(
        policy_costs(cost_fixture(), k=2, depth=2, verify=8, moe=False),
        verify_num=8,
        node_top_k=2,
        tree_depth=2,
        compute_intensity="high_intensity",
        is_moe_target=False,
        exec_mode=0,
        benchmark_config="",
        chain_width_table=(),
        device=torch.device("cuda"),
    )
    live = torch.ones(1, dtype=torch.int32, device="cuda")
    frontier = torch.zeros(1, dtype=torch.int64, device="cuda")
    sampled = torch.zeros(1, dtype=torch.bool, device="cuda")
    deltas = torch.zeros(3, device="cuda")
    # Simulate the persistent tiny tier reached after an earlier compared decision.
    policy._state[0] = 1
    policy.begin_round(live, frontier, sampled)
    policy.decide(1, deltas)
    assert policy.go[2].item() == 0
    assert policy.cumu_count.item() == 7
    assert policy.verify_count.item() == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
