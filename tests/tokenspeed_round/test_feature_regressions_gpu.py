# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Regressions at the real policy/round boundary, without model weights."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("tvm_ffi", reason="TokenSpeed round kernels require apache-tvm-ffi")

from tests.tokenspeed_round.test_chain_sampled_policy_gpu import (
    capture_round,
    chain_policy,
    replay,
)

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

pytest.importorskip("tokenspeed", reason="requires the optional TokenSpeed engine")



from tokenspeed.runtime.sampling.sampling_params import SamplingParams

from eagle_harness.tokenspeed.tree_round.device_policy import DevicePolicy
from eagle_harness.tokenspeed.tree_round.dynamic import DynamicDrafter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_fully_scheduled_chain_enters_native_policy_without_costs():
    policy = DevicePolicy(
        None,
        verify_num=8,
        node_top_k=1,
        tree_depth=7,
        compute_intensity="chain_intensity",
        is_moe_target=False,
        exec_mode=0,
        benchmark_config="",
        chain_width_table=((1, 4),),
        device=torch.device("cuda"),
    )
    inputs = capture_round(policy)
    for sample in (False, True):
        for batch in (1, 33, 512):
            replay(inputs, batch=batch, gain=0.0, sample=sample)
            assert policy.verify_num.item() == 4
            assert policy.last_depth.item() == 2


def test_idle_policy_warmups_do_not_advance_the_first_live_sequence():
    policy = chain_policy()
    inputs = capture_round(policy)
    for _ in range(3):
        replay(inputs, batch=0, gain=0.0, sample=False)
    replay(inputs, batch=1, gain=0.0, sample=False)
    assert policy.go_ready(3, 1) is False
    replay(inputs, batch=1, gain=100000.0, sample=False)
    assert policy.go_ready(3, 2) is True


def test_discarded_prefill_is_not_a_live_policy_round():
    # Only the absent model/engine allocation is replaced. Real preparation and
    # policy kernels must use the same round count as finish_round's skip rule.
    drafter = DynamicDrafter.__new__(DynamicDrafter)
    drafter.policy = chain_policy()
    drafter._sampled_q = object()
    drafter._live_rows = 0
    drafter._live_sampled = False
    drafter.input_buffers = SimpleNamespace(all_extends_mid_chunk=True)
    drafter._round = {
        "live_rows": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "sampled": torch.zeros(1, dtype=torch.bool, device="cuda"),
    }
    param = SamplingParams(temperature=1.0, seed=11)
    param.normalize(None)
    drafter.prepare_round_sampling([param])
    assert drafter._round["live_rows"].item() == 0
    drafter.input_buffers.all_extends_mid_chunk = False
    drafter.prepare_round_sampling([param])
    assert drafter._round["live_rows"].item() == 1
    assert drafter._round["sampled"].item()
