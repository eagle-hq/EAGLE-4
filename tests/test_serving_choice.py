# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Synthetic costs exercise selection, not a hardware performance claim."""

import pytest

from tools.serving_choice import serving_choice


def test_selects_tree_chain_and_true_no_spec_by_served_rate():
    candidates = {"tree": (4.0, 8.0), "chain": (2.0, 5.0)}
    assert serving_choice(3.0, candidates) == "tree"
    candidates["tree"] = (4.0, 12.0)
    assert serving_choice(3.0, candidates) == "chain"
    assert serving_choice(2.0, candidates) is None


def test_missing_candidates_and_ties_keep_no_spec():
    assert serving_choice(1.0, {}) is None
    assert serving_choice(1.0, {"zero-draft": (1.0, 2.0)}) is None
    assert serving_choice(1.0, {"chain": (2.0, 2.0)}) is None


@pytest.mark.parametrize("cost", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_baseline_or_candidate_time(cost):
    with pytest.raises(ValueError):
        serving_choice(cost, {})
    with pytest.raises(ValueError):
        serving_choice(1, {"tree": (2, cost)})


@pytest.mark.parametrize("tokens", [0, 0.5, float("nan"), float("inf")])
def test_rejects_invalid_expected_tokens(tokens):
    with pytest.raises(ValueError):
        serving_choice(1, {"tree": (tokens, 1)})
