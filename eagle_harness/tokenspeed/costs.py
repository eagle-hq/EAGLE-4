# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Adapt shared cost records to TokenSpeed's fixed tree-policy ABI."""

from dataclasses import replace

from eagle_harness.core.cost_model import CostModel, TreePolicyCosts

# TokenSpeed's tree policy packs exactly this many live-batch rows; its chain ABI is separate.
TREE_POLICY_BATCH_CAPACITY = 32


def tree_policy_costs(model: CostModel) -> TreePolicyCosts:
    """Keep host-computed costs unchanged within the engine's supported tree capacity."""
    costs = model.tree_policy_costs()
    return replace(costs, batches=costs.batches[:TREE_POLICY_BATCH_CAPACITY])
