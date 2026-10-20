# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-4 draft config validation and the low-rank head (checked against golden fixtures)."""

from __future__ import annotations

import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

from eagle_harness.models.eagle4_config import resolve_draft_sliding_window, validate_eagle4_draft_config  # noqa: E402
from eagle_harness.models.low_rank_head import low_rank_factors  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "fork_golden.json")


def _config(**extra):
    base = dict(
        architectures=["LlamaForCausalLMEagle4"],
        num_hidden_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        vocab_size=1000,
    )
    base.update(extra)
    return transformers.LlamaConfig(**base)


def test_plain_draft_config_passes():
    validate_eagle4_draft_config(_config(target_hidden_size=512, draft_vocab_size=1000))


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"mystery_key": 1}, "keys this model does not implement"),
        ({"bias": True}, "bias=True"),
        ({"num_aux_hidden_states": 2}, "num_aux_hidden_states"),
        ({"eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2]}}, "three aux hidden states"),
        ({"efficient_lm_head_config": {"type": "pca"}}, "efficient_lm_head_config"),
        ({"sliding_window": 4096, "num_hidden_layers": 2}, "num_hidden_layers=1"),
    ],
)
def test_bad_draft_configs_rejected(extra, message):
    with pytest.raises(ValueError, match=message):
        validate_eagle4_draft_config(_config(**extra))


def test_sliding_window_resolution():
    assert resolve_draft_sliding_window(_config()) is None
    assert resolve_draft_sliding_window(_config(sliding_window=4096)) == 4096
    assert resolve_draft_sliding_window(_config(sliding_window=4096, use_sliding_window=False)) is None
    assert resolve_draft_sliding_window(_config(sliding_window=64, layer_types=["sliding_attention"])) == 64
    assert resolve_draft_sliding_window(transformers.LlamaConfig(sliding_window=7)) is None  # not EAGLE-4
    with pytest.raises(ValueError, match="positive integer"):
        resolve_draft_sliding_window(_config(sliding_window=0))
    with pytest.raises(ValueError, match="requires a positive"):
        resolve_draft_sliding_window(_config(use_sliding_window=True))


def test_low_rank_head_matches_fork():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        cases = json.load(f)["low_rank_head"]
    assert len(cases) == 3
    for case in cases:
        weight = torch.tensor(case["weight"])
        freq = torch.tensor(case["freq"]) if case["freq"] is not None else None
        factors = low_rank_factors(
            weight,
            case["rank"],
            token_frequency=freq,
            smooth_factor=10,
            shard=tuple(case["shard"]) if case["shard"] else None,
        )
        # A backward-stable fp32 SVD is accurate to about rows * eps * ||W||_2. The CPU LAPACK of
        # GitHub's runners and of the cluster differ within that bound (CI saw 1.7e-5 on scores).
        atol = weight.shape[0] * torch.finfo(torch.float32).eps * torch.linalg.matrix_norm(weight, 2).item()
        torch.testing.assert_close(factors.down_proj, torch.tensor(case["down_proj"]), rtol=0, atol=atol)
        torch.testing.assert_close(factors.scores, torch.tensor(case["scores"]), rtol=0, atol=atol)
        assert 0 < factors.energy_retention <= 1
