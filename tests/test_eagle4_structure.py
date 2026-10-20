# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Engine-independent draft structure decisions and config preparation."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from eagle_harness.models.eagle4_structure import (
    LATENT_MOE_ARCHITECTURE,
    base_layer_overrides,
    draft_moe_architecture,
    draft_uses_rope,
    latent_moe_config,
)


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({}, True),
        ({"use_rope": True, "partial_rotary_factor": 1}, True),
        ({"partial_rotary_factor": None}, True),
        ({"use_rope": False}, False),
        ({"use_rope": True, "position_embedding_type": "none"}, False),
        ({"use_rope": True, "partial_rotary_factor": 0}, False),
    ],
)
def test_rotary_selection(fields, expected):
    assert draft_uses_rope(SimpleNamespace(**fields)) is expected


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({}, None),
        ({"enable_moe": False, "moe_architecture": LATENT_MOE_ARCHITECTURE}, None),
        ({"enable_moe": True}, "sparse_swiglu"),
        ({"enable_moe": True, "moe_architecture": LATENT_MOE_ARCHITECTURE}, LATENT_MOE_ARCHITECTURE),
    ],
)
def test_moe_selection(fields, expected):
    assert draft_moe_architecture(SimpleNamespace(**fields)) == expected


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize(
    "rope", [{}, {"use_rope": False}, {"position_embedding_type": "none"}, {"partial_rotary_factor": 0}]
)
def test_base_layer_overrides_only_replace_temporary_parts(moe, rope):
    config = SimpleNamespace(
        enable_moe=moe,
        moe_architecture=LATENT_MOE_ARCHITECTURE,
        hidden_act="relu2" if moe else "silu",
        max_position_embeddings=131072,
        **rope,
    )
    before = deepcopy(vars(config))
    expected = {"hidden_act": "silu"} if moe else {}
    if rope:
        expected["max_position_embeddings"] = 1
    assert base_layer_overrides(config) == expected
    assert vars(config) == before


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({}, (0, 1, 1, True, 1.0)),
        ({"n_shared_experts": None, "n_group": 0, "topk_group": None}, (0, 1, 1, True, 1.0)),
        ({"n_group": 4}, (0, 4, 4, True, 1.0)),
        (
            {
                "n_shared_experts": 2,
                "n_group": 4,
                "topk_group": 2,
                "norm_topk_prob": False,
                "routed_scaling_factor": 2.5,
            },
            (2, 4, 2, False, 2.5),
        ),
    ],
)
def test_latent_moe_config_preserves_checkpoint_fields_and_normalizes_engine_defaults(fields, expected):
    config = SimpleNamespace(
        hidden_act="relu2",
        hidden_size=256,
        moe_latent_size=64,
        n_routed_experts=8,
        moe_intermediate_size=128,
        num_experts_per_tok=2,
        mlp_bias=True,
        **fields,
    )
    before = deepcopy(vars(config))
    prepared = latent_moe_config(config)
    assert prepared is not config
    assert vars(config) == before
    assert prepared.mlp_hidden_act == "relu2"
    assert prepared.mlp_bias is False
    assert (
        prepared.n_shared_experts,
        prepared.n_group,
        prepared.topk_group,
        prepared.norm_topk_prob,
        prepared.routed_scaling_factor,
    ) == expected
    for field in ("hidden_size", "moe_latent_size", "n_routed_experts", "moe_intermediate_size", "num_experts_per_tok"):
        assert getattr(prepared, field) == getattr(config, field)


def test_latent_moe_config_keeps_config_type_and_class_attributes():
    class Config:
        hidden_act = "relu2"
        moe_latent_size = 64

    prepared = latent_moe_config(Config())
    assert isinstance(prepared, Config)
    assert prepared.mlp_hidden_act == "relu2"
    assert prepared.moe_latent_size == 64
