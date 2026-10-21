# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""HarnessConfig parsing and cost-model construction."""

from __future__ import annotations

import json

import pytest

from eagle_harness.core.config import (
    HarnessConfig,
    Mode,
    build_cost_model,
    config_to_dict,
    harness_config_from_dict,
    parse_harness_config,
)
from eagle_harness.core.cost_model import CostModelParams
from eagle_harness.core.limits import ComputeIntensity


def test_inline_json_and_file(tmp_path):
    raw = {
        "data_path": "/cost/qwen",
        "intensity": "chain_intensity",
        "mode": "serve",
        "cost_model": {"threshold": 3, "chain_tiny": [2.7, 0.2, 0.0]},
        "draft_model": {"embed_full": False},
    }
    cfg = parse_harness_config(json.dumps(raw))
    assert cfg.intensity is ComputeIntensity.CHAIN and cfg.mode is Mode.SERVE
    assert cfg.cost_model.threshold == 3.0 and cfg.cost_model.chain_tiny == (2.7, 0.2, 0.0)
    assert cfg.draft_model.embed_full is False
    path = tmp_path / "harness.json"
    path.write_text(json.dumps(raw))
    assert parse_harness_config(str(path)) == cfg
    assert harness_config_from_dict(config_to_dict(cfg)) == cfg


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"data_path": "/x", "thresold": 1}, "unknown keys"),
        ({"data_path": "/x", "cost_model": {"chain_tiny": [1, 2]}}, "list of 3"),
        ({"data_path": "/x", "intensity": "mid"}, "not one of"),
        ({"data_path": "/x", "draft_model": {"embed_full": "yes"}}, "true/false"),
        ({"data_path": "/x", "cost_model": {"chain_tiny_max_bs": 2.5}}, "integer"),
        ({"data_path": "/x", "fast_paths": {"verify_argmax": False}}, "unknown keys"),
    ],
)
def test_rejects_bad_config(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_harness_config(json.dumps(raw))


@pytest.mark.parametrize(
    "raw, key",
    [
        ({"mode": "eval"}, "mode: eval"),
        ({"ragged": {"force_uniform": True}}, "ragged"),
        ({"cost_model": {"ragged_draft_relax": [0.75, 0.75, 0.75]}}, "cost_model.ragged_draft_relax"),
        ({"cost_model": {"ragged_verify_threshold": [1.35, 1.6, 1.4]}}, "cost_model.ragged_verify_threshold"),
        ({"draft_model": {"moe_gate_acc": "fp32"}}, "draft_model.moe_gate_acc"),
    ],
)
def test_options_removed_in_0_1_1_are_refused_by_name(raw, key):
    with pytest.raises(ValueError, match=f"^{key}: removed in eagle-harness 0.1.1"):
        parse_harness_config(json.dumps({"data_path": "/x", **raw}))


def test_absent_config_is_the_default_and_bad_json_is_rejected():
    assert parse_harness_config(None) == parse_harness_config("{}") == HarnessConfig()
    assert parse_harness_config("  ") == HarnessConfig()
    with pytest.raises(ValueError, match="JSON object"):
        parse_harness_config("[1, 2]")
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_harness_config("{nope")


def test_cost_model_params_defaults_match_fork_constants():
    cfg = HarnessConfig(data_path="/x")
    assert cfg.cost_model.params(0.04) == CostModelParams(depth_loss_ratio=0.04)


def test_build_cost_model(tmp_path):
    cfg = HarnessConfig(data_path=str(tmp_path), intensity=ComputeIntensity.CHAIN)
    cm = build_cost_model(cfg, node_top_k=1, tree_depth=15, verify_num=16, target_is_moe=True)
    assert cm.tables == {} and cm.table(1) is None
    # chain: 0.04 doubled for chain / MoE then halved for chain
    assert cm.params.depth_loss_ratio == pytest.approx(0.04)


def test_l2_warm_plan_parses_and_defaults_to_the_fork_gate_up_warm_ahead():
    default = parse_harness_config(json.dumps({"data_path": "/x"})).l2_warm
    assert default.gate_up_gdn == (0.42, 16) and default.gate_up_attention == (0.3, 74)
    assert default.down_proj is None and default.next_input is None
    plan = parse_harness_config(
        json.dumps({"data_path": "/x", "l2_warm": {"down_proj": [0.5, 8], "gate_up_gdn": None}})
    ).l2_warm
    assert plan.down_proj == (0.5, 8) and plan.gate_up_gdn is None
    with pytest.raises(ValueError, match="list of 2"):
        parse_harness_config(json.dumps({"data_path": "/x", "l2_warm": {"down_proj": [0.5]}}))
