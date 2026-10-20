# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The TokenSpeed adapter on CPU, without TokenSpeed: config, hook check, lane masks, and the verify window.

The window TokenSpeed verifies must be the tree the sglang and vLLM adapters verify for the same draft outputs: the
same nodes, tokens and parents (``test_window_verifies_what_sglang_and_vllm_verify``). The lane masks must give each
draft row exactly its own and its ancestors' window slots.
"""

import dataclasses
import sys
import types

import pytest
import torch

from eagle_harness.core.config import Mode
from eagle_harness.core.draft_tree import DraftTree, TreeShape
from eagle_harness.tokenspeed import hooks
from eagle_harness.tokenspeed.config import parse_algorithm_config
from eagle_harness.tokenspeed.tree_lanes import (
    advance_lane_masks,
    lane_bits,
    parent_lanes,
    seed_lane_masks,
    verify_window,
)

K, STEPS, H, BS, WIDTH = 4, 5, 8, 3, 16


# ------------------------------------------------------------------ config
def test_no_config_is_a_static_tree():
    assert parse_algorithm_config(None).mode is Mode.STATIC


def test_static_config_parses_harness_keys():
    config = parse_algorithm_config({"mode": "static", "draft_model": {"cuda_candidate_kernels": False}})
    assert config.mode is Mode.STATIC and config.draft_model.cuda_candidate_kernels is False


def test_serve_uses_device_policy_with_calibration():
    config = parse_algorithm_config({"mode": "serve", "data_path": "/calibration"})
    assert config.mode is Mode.SERVE and config.cost_dir == "/calibration/cost"


def test_unsupported_cost_model_modes_are_refused():
    with pytest.raises(NotImplementedError, match="static and serve"):
        parse_algorithm_config({"mode": "calibrate", "data_path": "/nonexistent"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"intensity": "chain_intensity"},
        {"draft_family": "eagle3"},
        {"cost_model": {"threshold": 3.0}},
    ],
)
def test_device_policy_preserves_chain_family_and_custom_cost_config(overrides):
    config = parse_algorithm_config({"mode": "serve", "data_path": "/calibration", **overrides})
    if "cost_model" in overrides:
        assert config.cost_model.threshold == 3.0
    else:
        key, value = next(iter(overrides.items()))
        assert getattr(config, key).value == value


def test_chain_width_schedule_is_immutable_and_preserved():
    table = [[1, 16], [64, 4]]
    config = parse_algorithm_config({
        "mode": "serve", "data_path": "/calibration", "intensity": "chain_intensity",
        "chain_width_table": table,
    })
    table[0][1] = 2
    assert config.chain_width_table == ((1, 16), (64, 4))


@pytest.mark.parametrize("table", [None, "1:4", [[1, 1]], [[True, 4]], [[8, 4], [1, 8]], [[1, 4, 8]]])
def test_invalid_chain_width_schedule_is_rejected(table):
    with pytest.raises(ValueError, match="chain_width_table"):
        parse_algorithm_config({
            "mode": "serve", "data_path": "/calibration", "intensity": "chain_intensity",
            "chain_width_table": table,
        })


def test_tree_mode_rejects_chain_width_schedule():
    with pytest.raises(ValueError, match="chain_intensity"):
        parse_algorithm_config({
            "mode": "serve", "data_path": "/calibration", "chain_width_table": [[1, 4]],
        })


def test_complete_chain_schedule_needs_no_calibration():
    config = parse_algorithm_config({
        "mode": "serve", "intensity": "chain_intensity", "chain_width_table": [[1, 8], [64, 4]],
    })
    assert config.mode is Mode.SERVE
    assert not config.data_path
    assert config.chain_width_table == ((1, 8), (64, 4))
    with pytest.raises(ValueError, match="data_path"):
        parse_algorithm_config({
            "mode": "serve", "intensity": "chain_intensity", "chain_width_table": [[64, 4]],
        })


@pytest.mark.parametrize("overrides", [{}, {"intensity": "chain_intensity"}])
def test_serve_without_live_cost_capability_requires_calibration(overrides):
    with pytest.raises(ValueError, match="data_path"):
        parse_algorithm_config({"mode": "serve", **overrides})


def test_unknown_keys_are_refused():
    with pytest.raises(ValueError):
        parse_algorithm_config({"mode": "static", "node_top_k": 4})


# ------------------------------------------------------------------ hook check
@pytest.fixture
def fake_host(monkeypatch):
    """A host module ``fakehost_api`` with a drafter registry and a dataclass field, checked through HOOKS."""
    module = types.ModuleType("fakehost_api")

    def register_drafter(algorithm, drafter_cls, *, model_cls=None, captures_aux_hidden_states=False):
        pass

    @dataclasses.dataclass
    class Context:
        gather_ids: object  # a field without a class-level default

    module.register_drafter, module.Context = register_drafter, Context
    monkeypatch.setitem(sys.modules, "fakehost_api", module)
    return module


def test_hook_check_passes_a_host_with_every_hook(fake_host, monkeypatch):
    monkeypatch.setattr(
        hooks,
        "HOOKS",
        (
            hooks.Hook("fakehost_api:register_drafter", ("algorithm", "captures_aux_hidden_states")),
            hooks.Hook("fakehost_api:Context.gather_ids"),
        ),
    )
    assert [hooks._problem(h) for h in hooks.HOOKS] == [None, None]


def test_hook_check_names_what_is_missing(fake_host):
    assert hooks._problem(hooks.Hook("fakehost_api:register_model")) == "fakehost_api:register_model"
    assert "no parameter override" in hooks._problem(hooks.Hook("fakehost_api:register_drafter", ("override",)))
    assert "not importable" in hooks._problem(hooks.Hook("nosuchhost.api:x"))


# ------------------------------------------------------------------ the tree as TokenSpeed lanes
def _drafted(seed=0):
    """A tree grown as HarnessDrafter grows it, with TokenSpeed's lane masks, plus each node's parent node and the
    window slot each node's draft row wrote (depth-1 lanes at step 1, ...; -1 for nodes never forwarded)."""
    tree = DraftTree(TreeShape(top_k=K, max_depth=STEPS), BS, H, torch.float32, torch.device("cpu"), max_verify=1)
    bits = lane_bits(K, STEPS, torch.device("cpu"))
    lane_mask = torch.zeros(BS * K, dtype=torch.int64)
    gen = torch.Generator().manual_seed(seed)
    tree.stage_root(BS, torch.rand(BS, K, generator=gen).softmax(-1), torch.randint(0, 999, (BS, K), generator=gen),
                    torch.randn(BS, H, generator=gen))  # fmt: skip
    tree.begin(BS)
    seed_lane_masks(lane_mask, BS, bits)
    parent_of = torch.full((BS, tree.shape.capacity), -1, dtype=torch.int64)
    frontier = torch.arange(K).repeat(BS, 1)  # node id of each lane
    slot_of = {}  # (b, node) -> window slot its row wrote
    masks = []  # per step: (frontier nodes, lane masks) the step's rows ran with
    for step in range(1, STEPS):
        for b in range(BS):
            for lane in range(K):
                slot_of[b, int(frontier[b, lane])] = (step - 1) * K + lane
        masks.append((frontier.clone(), lane_mask.view(BS, K).clone()))
        probs = torch.rand(BS * K, K, generator=gen).softmax(-1)
        tokens = torch.randint(0, 999, (BS * K, K), generator=gen)
        tree.step(BS, step, probs, tokens, torch.randn(BS * K, H, generator=gen))
        start = tree.shape.layer_start(step + 1)
        parent_of[:, start : start + K * K] = frontier.repeat_interleave(K, dim=1)
        first = tree.shape.num_parents(step + 1)
        new_frontier = tree.parents[:BS, first : first + K].clone()
        assert torch.equal(torch.gather(frontier, 1, parent_lanes(tree, BS, step)), parent_of.gather(1, new_frontier))
        if step < STEPS - 1:
            advance_lane_masks(tree, lane_mask, BS, step, bits)
        frontier = new_frontier
    return tree, parent_of, slot_of, masks


def _ancestors(parent_of, b, node):
    while node >= 0:
        yield node
        node = int(parent_of[b, node])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lane_masks_show_a_row_its_ancestors_and_itself(seed):
    _, parent_of, slot_of, masks = _drafted(seed)
    for frontier, lane_mask in masks:
        for b in range(BS):
            for lane in range(K):
                want = sum(1 << slot_of[b, n] for n in _ancestors(parent_of, b, int(frontier[b, lane])))
                assert int(lane_mask[b, lane]) & ((1 << 64) - 1) == want


def test_lane_bits_use_bit_63_as_the_sign():
    bits = lane_bits(8, 9, torch.device("cpu"))  # (9 - 1) * 8 = 64 window slots
    assert int(bits[7, 7]) == -(1 << 63) and int(bits[0, 0]) == 1
    with pytest.raises(ValueError, match="64-bit"):
        lane_bits(8, 10, torch.device("cpu"))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_window_verifies_what_sglang_and_vllm_verify(seed):
    tree, parent_of, _, _ = _drafted(seed)
    roots = torch.tensor([11, 22, 33], dtype=torch.int32)
    window = torch.zeros(BS, WIDTH, dtype=torch.int32)
    parents = torch.zeros(BS, WIDTH, dtype=torch.int32)
    verify_window(tree, BS, STEPS, roots, window, parents)
    # sglang's verify input for the same tree: the ranked nodes, their tokens in node order (DraftTree.select)
    ranked, _ = tree.rank(BS, STEPS, WIDTH - 1)
    sglang = tree.select(BS, STEPS, ranked, WIDTH)
    assert torch.equal(window[:, 0], roots)
    assert torch.equal(window[:, 1:].long(), sglang.draft_tokens)
    for b in range(BS):
        nodes = sglang.top_scores_index[b].tolist()
        assert parents[b, 0] == -1
        for row, node in enumerate(nodes, start=1):
            parent = int(parent_of[b, node])
            assert int(parents[b, row]) == (0 if parent < 0 else 1 + nodes.index(parent)) < row
    # vLLM's verify rows for the same tree (TreeDrafter.select = DraftTree.verify_rows)
    tokens, parent_rows = tree.verify_rows_reference(BS, STEPS, WIDTH - 1)
    assert torch.equal(window[:, 1:].long(), tokens) and torch.equal(parents, parent_rows)
