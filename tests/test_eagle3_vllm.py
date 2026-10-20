# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The harness tree uses native EAGLE3 draft-to-target logits mapping."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")
from eagle_harness.vllm.eagle3 import Eagle3ForHarness


def test_candidates_are_target_ids_with_reduced_vocab_probabilities():
    draft = object.__new__(Eagle3ForHarness)
    torch.nn.Module.__init__(draft)
    draft.config = SimpleNamespace(vocab_size=8, draft_vocab_size=3)
    draft.lm_head = torch.nn.Linear(4, 3, bias=False)
    draft.draft_id_to_target_id = torch.nn.Parameter(torch.tensor([1, 2, 5]), requires_grad=False)
    draft.logits_processor = lambda head, hidden: torch.tensor([[1.0, 3.0, 2.0]])
    probs, ids = draft.candidate_topk(torch.zeros(1, 4), 2)
    assert ids.tolist() == [[3, 7]]
    torch.testing.assert_close(probs, torch.tensor([[1.0, 3.0, 2.0]]).softmax(-1)[:, [1, 2]])
