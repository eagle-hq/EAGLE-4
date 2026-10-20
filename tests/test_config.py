# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""HarnessConfig parsing and cost-model construction."""

from __future__ import annotations

import dataclasses
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
        "mode": "eval",
        "cost_model": {"threshold": 3, "chain_tiny": [2.7, 0.2, 0.0]},
        "draft_model": {"embed_full": False},
    }
    cfg = parse_harness_config(json.dumps(raw))
    assert cfg.intensity is ComputeIntensity.CHAIN and cfg.mode is Mode.EVAL
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
        ({"data_path": "/x", "ragged": {"enabled": "yes"}}, "true/false"),
        ({"data_path": "/x", "cost_model": {"chain_tiny_max_bs": 2.5}}, "integer"),
        ({"data_path": "/x", "draft_model": {"moe_gate_acc": "fp16"}}, "moe_gate_acc"),
        ({"data_path": "/x", "fast_paths": {"verify_argmax": False}}, "unknown keys"),
    ],
)
def test_rejects_bad_config(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_harness_config(json.dumps(raw))


def test_rejects_missing_and_non_object():
    with pytest.raises(ValueError, match="needs --speculative-algorithm-config"):
        parse_harness_config(None)
    with pytest.raises(ValueError, match="JSON object"):
        parse_harness_config("[1, 2]")
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_harness_config("{nope")


def test_cost_model_params_defaults_match_fork_constants():
    cfg = HarnessConfig(data_path="/x")
    assert cfg.cost_model.params(0.04) == CostModelParams(depth_loss_ratio=0.04)


def test_build_cost_model(tmp_path):
    cfg = HarnessConfig(data_path=str(tmp_path), mode=Mode.EVAL, intensity=ComputeIntensity.CHAIN)
    cm = build_cost_model(
        cfg,
        node_top_k=1,
        tree_depth=15,
        verify_num=16,
        target_is_moe=True,
        ragged_attention_supported=True,
        eval_batch_size=8,
    )
    assert cm.tables == {} and cm.table(1) is None
    assert cm.ragged and cm.eval_batch_size == 8
    # chain: 0.04 doubled for chain / MoE then halved for chain
    assert cm.params.depth_loss_ratio == pytest.approx(0.04)
    serve = build_cost_model(
        dataclasses.replace(cfg, mode=Mode.SERVE),
        node_top_k=1,
        tree_depth=15,
        verify_num=16,
        target_is_moe=False,
        ragged_attention_supported=True,
        eval_batch_size=8,
    )
    assert not serve.ragged and serve.eval_batch_size is None


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
