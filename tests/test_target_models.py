# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Target model properties used by the cost policy, without an engine."""

from types import SimpleNamespace

import pytest

from eagle_harness.core.models import target_is_moe


@pytest.mark.parametrize("field", ["num_experts", "num_local_experts", "n_routed_experts"])
@pytest.mark.parametrize("count, expected", [(None, False), (0, False), (1, True), (8, True)])
@pytest.mark.parametrize("wrapped", [False, True])
def test_moe_comes_from_expert_counts(field, count, expected, wrapped):
    config = SimpleNamespace(**{field: count})
    if wrapped:
        config = SimpleNamespace(text_config=config)
    assert target_is_moe(config) is expected


def test_dense_model_is_not_moe_because_of_its_name():
    assert not target_is_moe(SimpleNamespace(architectures=["MoEForCausalLM"], is_moe=True))


def test_all_expert_fields_are_considered():
    assert target_is_moe(SimpleNamespace(num_experts=0, num_local_experts=None, n_routed_experts=8))


def test_text_config_is_authoritative():
    assert not target_is_moe(SimpleNamespace(num_experts=8, text_config=SimpleNamespace(num_experts=0)))


def test_absent_text_config_uses_outer_fields():
    assert target_is_moe(SimpleNamespace(text_config=None, num_experts=8))
