# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Real dynamic-tree kernels: eager depths and GPU-tail-launched rounds agree."""

from __future__ import annotations

import contextlib
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("tvm_ffi", reason="TokenSpeed round kernels require apache-tvm-ffi")

# Keep the CPU suites independent of TokenSpeed's GPU-only package imports.
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

pytest.importorskip("tokenspeed", reason="requires the optional TokenSpeed engine")

from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
    TreeDraftInputs,
)

pytest.importorskip("eagle_harness")

from eagle_harness.core.config import HarnessConfig, Mode
from eagle_harness.core.cost_table import BatchCostTable, CostCurve, IntervalCost
from eagle_harness.tokenspeed.tree_round import dynamic as drafter_module
from eagle_harness.tokenspeed.tree_round.device_policy import DevicePolicy
from eagle_harness.tokenspeed.tree_round.dynamic import DynamicDrafter
from tests.tokenspeed_round.core_costs import policy_costs


class EagleHarness(DynamicDrafter):
    @property
    def dynamic(self):
        return True


BS, NODES, WINDOW, VERIFY_COUNT = 2, 9, 16, 4
TREE_K, TREE_DEPTH, HIDDEN = 2, 6, 8


class _OneDepthPolicy:
    """Drafts depth 1 only and verifies two of its ranked nodes."""

    tree_depth = 2
    chain = False

    def __init__(self) -> None:
        int32 = {"dtype": torch.int32, "device": "cuda"}
        self.cumu_count = torch.tensor([NODES], **int32)
        self.verify_count = torch.tensor([VERIFY_COUNT], **int32)
        self.verify_num = torch.tensor([2], **int32)

    def observe_batch_size(self, batch_size: int) -> None:
        pass

    def deltas_depth(self, depth: int) -> bool:
        return False

    def decide(self, depth: int, work_deltas: torch.Tensor) -> None:
        pass

    def go_ready(self, depth: int, round_seq: int) -> bool:
        return depth < self.tree_depth

    def select_verify(self, top_scores: torch.Tensor) -> None:
        pass


def _drafter() -> EagleHarness:
    drafter = EagleHarness.__new__(EagleHarness)
    drafter.policy = _OneDepthPolicy()
    drafter.config = HarnessConfig(mode=Mode.SERVE, data_path="/unused")
    drafter.spec_num_steps = TREE_DEPTH
    drafter.spec_num_tokens = WINDOW
    drafter.input_buffers = SimpleNamespace(all_extends_mid_chunk=False)
    # Every node a child of the root: any ranking is a valid tree.
    prev = torch.zeros((BS, NODES), dtype=torch.int64, device="cuda")
    prev[:, 0] = -1
    gen = torch.Generator(device="cuda").manual_seed(71)
    scores = torch.rand((BS, NODES), device="cuda", generator=gen)
    scores[:, 0] = 2.0
    drafter._pool = {
        "scores": scores,
        "tokens": torch.arange(BS * NODES, device="cuda").view(BS, NODES),
        "prev": prev,
    }
    drafter._round = {
        "root_tokens": torch.zeros(BS, dtype=torch.int32, device="cuda"),
        "rows": torch.arange(BS, device="cuda"),
        "frontier": torch.zeros(BS, dtype=torch.int64, device="cuda"),
        "live_rows": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "sampled": torch.zeros(1, dtype=torch.bool, device="cuda"),
        "work_deltas": torch.zeros((2, VERIFY_COUNT + 1), device="cuda"),
        "top_scores": torch.zeros((BS, WINDOW), device="cuda"),
        "top_index": torch.zeros((BS, WINDOW), dtype=torch.int64, device="cuda"),
    }
    drafter.runtime_states = SimpleNamespace(
        future_input_map=torch.zeros((BS, WINDOW), dtype=torch.int32, device="cuda"),
        future_parent_map=torch.zeros((BS, WINDOW), dtype=torch.int32, device="cuda"),
    )
    drafter._live_rows = 0
    drafter._live_sampled = False
    drafter._round_sampled = False
    drafter._sampled_q = None
    drafter._round_seq = 0
    drafter._depth_done = [torch.cuda.Event() for _ in range(2)]
    drafter._decide_stream = torch.cuda.Stream()
    drafter._round_stream = torch.cuda.Stream()
    drafter._chain = None
    drafter._target_done = torch.cuda.Event()
    drafter._round_done = torch.cuda.Event()
    drafter._launched_rows = None
    drafter._pending = None
    # The depth's forward and pool update: nothing the stream ordering reads.
    drafter._lane_context = lambda *args: None
    drafter._draft_depth = lambda *args, **kwargs: None
    drafter.attn_backend = SimpleNamespace(
        override_num_extends=lambda n: contextlib.nullcontext()
    )
    return drafter


@pytest.mark.parametrize("mode", [Mode.STATIC, Mode.SERVE])
def test_missing_tree_is_rejected_before_first_forward(mode):
    from eagle_harness.tokenspeed.drafter import HarnessDrafter

    drafter = HarnessDrafter.__new__(HarnessDrafter)
    drafter.config = HarnessConfig(mode=mode, data_path="/unused")
    drafter.tree = None
    drafter.tree_lanes = None
    with pytest.raises(NotImplementedError, match="binds explicit draft ancestry"):
        drafter.run(None, None, None, None)


def test_dynamic_binding_has_no_native_sorter_depth_limit():
    drafter = _drafter()
    drafter.input_buffers.max_bs = BS
    drafter.draft_reads_token_history = False
    drafter.device = torch.device("cuda")
    drafter.draft_model_runner = SimpleNamespace(
        model=SimpleNamespace(
            logits_processor=SimpleNamespace(
                do_argmax=True, config=SimpleNamespace(vocab_size=128)
            )
        )
    )
    drafter._allocate = lambda *args: None
    bound = []
    drafter.attn_backend.bind_tree_draft = bound.append
    spec = SimpleNamespace(config=SimpleNamespace(topk=4, num_steps=12, num_nodes=16))
    drafter.bind_dynamic_tree(spec, _OneDepthPolicy())
    assert drafter.tree_spec is spec
    assert bound == [drafter.tree_lanes]
    assert drafter.tree_lanes.num_slots == 44
    assert drafter.lanes_per_request == 4
    assert not drafter.draft_model_runner.model.logits_processor.do_argmax


def _kernel_drafter(
    cost: BatchCostTable, scale: torch.Tensor, sleep: bool
) -> EagleHarness:
    """A drafter of K=2 trees to depth 6 on the device policy and the real
    pool, ranking and verify tree kernels, its depths captured; its draft
    forward is scripted from its inputs and the round's ``scale``, and with
    ``sleep`` runs long."""
    drafter, k = _drafter(), TREE_K
    del drafter._draft_depth
    drafter.config = HarnessConfig(mode=Mode.SERVE, data_path="/unused")
    drafter.spec_num_steps = TREE_DEPTH
    drafter.device = torch.device("cuda")
    drafter.policy = DevicePolicy(
        policy_costs(
            {bs: replace(cost, batch_size=bs) for bs in (1, 2)},
            k=k,
            depth=TREE_DEPTH,
            verify=WINDOW,
            moe=False,
        ),
        verify_num=WINDOW,
        node_top_k=k,
        tree_depth=TREE_DEPTH,
        compute_intensity="high_intensity",
        is_moe_target=False,
        exec_mode=0,
        benchmark_config="",
        chain_width_table=(),
        device=drafter.device,
    )
    drafter.lanes_per_request = k
    drafter.tree_lanes = TreeDraftInputs(k, TREE_DEPTH, BS, drafter.device)
    drafter.token_to_kv_pool = None
    drafter.attn_backend = SimpleNamespace(
        publish_draft_step_locations=lambda **kw: None,
        override_num_extends=lambda n: contextlib.nullcontext(),
        begin_tree_lanes=lambda rows, live_rows: None,
    )

    def forward(input_ids, positions, captured_hidden_states, spec_step_idx, **kw):
        if sleep:
            torch.cuda._sleep(50_000_000)
        base = (input_ids * 31 + positions * 7 + spec_step_idx) % 997
        steps = torch.arange(k, device="cuda")
        probs = torch.sigmoid(scale * torch.sin(base[:, None] + steps).float())
        return SimpleNamespace(
            draft_topk=(
                probs.sort(dim=1, descending=True).values,
                (base[:, None] * 13 + steps * 101) % 5000,
            ),
            hidden_states=captured_hidden_states * 0.5 + 1,
        )

    drafter.draft_model_runner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=HIDDEN)),
        model_config=SimpleNamespace(dtype=torch.float32),
        forward=forward,
    )
    drafter.round_candidates = lambda rows: rows.draft_topk
    drafter._allocate(BS, k, TREE_DEPTH, 1024, 64)
    drafter._depth_graphs, drafter._finalize_graphs = {}, {}
    drafter.capture_round_graphs(BS, lo_bs=0, stream=torch.cuda.Stream(), pool=None)
    return drafter


def _seed_round(drafter: EagleHarness, inputs: dict[str, torch.Tensor]) -> None:
    """What ``draft`` leaves for depth 1 inside the decode graph: the root,
    its children in the pool and depth 1's frontier and lanes."""
    pool, state = drafter._pool, drafter._round
    pool["tokens"][:BS, 0] = inputs["root"]
    drafter_module.eagle_harness_update_node_data(
        pool["scores"][:BS],
        pool["tokens"][:BS],
        pool["work"][:BS],
        pool["sort"][:BS],
        inputs["top_p"],
        inputs["top_tokens"],
    )
    state["root_tokens"][:BS].copy_(inputs["root"])
    state["tokens"][:BS].copy_(inputs["top_tokens"])
    state["scores"][:BS].copy_(inputs["top_p"])
    state["hidden"][:BS].copy_(inputs["hidden"])
    state["frontier"][:BS].copy_(inputs["frontier"])
    state["rows"][:BS].copy_(inputs["rows"])
    drafter._start_lanes(BS, BS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_device_chained_round_drafts_the_host_launched_round() -> None:
    """Rounds whose depths and finalize the GPU chains, launched behind a
    decode graph or by ``finish_round``, grow the same pool, publish the same
    go flags and decisions and write the same verify tree as rounds whose
    depths the host launches one by one, also once their graphs were
    released and captured again; behind a decode graph, the forward's stream
    waits for that whole graph and not for the round."""
    target = CostCurve(
        (1.0,) * 5, (1.0, 1.02, 1.05, 1.1, 1.3), (0, 1, 3, 7, 15), (), ()
    )
    draft = CostCurve((1.0, 1.0), (1.0, 1.2), (0, 1), (), ())
    cost = BatchCostTable(
        1, 1000, 1000, 16, 16384, 2, (IntervalCost(target, draft),), (20.0,)
    )
    scale = torch.ones((), device="cuda")
    host, launched, behind = (
        _kernel_drafter(cost, scale, sleep) for sleep in (False, False, True)
    )
    inputs = {
        "root": torch.zeros(BS, dtype=torch.int64, device="cuda"),
        "top_p": torch.zeros((BS, TREE_K), device="cuda"),
        "top_tokens": torch.zeros((BS, TREE_K), dtype=torch.int64, device="cuda"),
        "hidden": torch.zeros((BS, TREE_K, HIDDEN), device="cuda"),
        "frontier": torch.zeros(BS, dtype=torch.int64, device="cuda"),
        "rows": torch.arange(BS, device="cuda"),
    }
    # The decode graph a chained round follows: the draft's seeding, then a
    # slow tail whose result the forward's stream reads.
    tail = torch.zeros(1, dtype=torch.int64, device="cuda")
    decode = torch.cuda.CUDAGraph()
    with torch.cuda.graph(decode):
        _seed_round(behind, inputs)
        torch.cuda._sleep(20_000_000)
        tail.copy_(inputs["frontier"][:1])
    ctx = SimpleNamespace(bs=BS, num_extends=0, global_num_tokens=None, global_bs=None)
    ctx.all_decode_or_idle = True
    depths = set()
    for i in range(12):
        if i == 10:
            for drafter in (host, launched, behind):
                drafter.release_round_graphs()
                drafter.capture_round_graphs(
                    BS, lo_bs=0, stream=torch.cuda.Stream(), pool=None
                )
        gen = torch.Generator().manual_seed(i)
        scale.fill_(0.5 + 0.4 * i)
        inputs["root"].copy_(torch.randint(0, 5000, (BS,), generator=gen))
        inputs["top_p"].copy_(
            torch.rand((BS, TREE_K), generator=gen).sort(dim=1, descending=True).values
        )
        inputs["top_tokens"].copy_(torch.randint(0, 5000, (BS, TREE_K), generator=gen))
        inputs["hidden"].copy_(torch.rand((BS, TREE_K, HIDDEN), generator=gen))
        inputs["frontier"].copy_(torch.tensor([100 + 37 * i, 900 + 11 * i]))
        for drafter, graph_bs in ((host, None), (launched, BS)):
            _seed_round(drafter, inputs)
            drafter.finish_round(ctx, graph_bs)
            drafter.complete_round()
        behind.replay_decode_graph(decode, bs=BS, graph_bs=BS)
        tail_seen = tail.clone()
        seeded = torch.cuda.Event()
        seeded.record()
        behind.finish_round(ctx, BS)
        seeded.synchronize()
        # The forward's stream ran past the decode graph while the round drafts.
        assert not behind._round_stream.query()
        behind.complete_round()
        assert not torch.cuda.current_stream().query()
        torch.cuda.synchronize()
        assert tail_seen.item() == 100 + 37 * i
        depths.add(host.policy.last_depth.item())
        expected = _round_outputs(host)
        for drafter in (launched, behind):
            outputs = _round_outputs(drafter)
            assert outputs.keys() == expected.keys()
            for name, value in expected.items():
                assert torch.equal(outputs[name], value), (i, name)
    # Rounds stopped at several depths, the deepest at the last.
    assert len(depths) > 2 and TREE_DEPTH - 1 in depths


def _round_outputs(drafter: EagleHarness) -> dict[str, torch.Tensor]:
    """What a round leaves: its pool and round buffers, the policy's state,
    slots (go flags included) and their tagged host mirror, and the verify
    tree."""
    policy = drafter.policy
    return {
        **{f"pool.{name}": value for name, value in drafter._pool.items()},
        **{f"round.{name}": value for name, value in drafter._round.items()},
        "state": policy._state,
        "slots": policy._slots,
        "go_host": policy.go_host,
        "tree": drafter.runtime_states.future_input_map,
        "parents": drafter.runtime_states.future_parent_map,
    }


def test_release_round_graphs_drops_what_named_the_probe_pool() -> None:
    """Graphs recorded on the probe pool must not survive the recapture."""
    harness = EagleHarness.__new__(EagleHarness)
    harness._chain = None
    harness._depth_graphs = {(1, 1): object()}
    harness._finalize_graphs = {(1, False): object()}
    harness.release_round_graphs()
    assert harness._depth_graphs == {} and harness._finalize_graphs == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
