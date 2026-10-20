# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Dense targets have no recurrent state to commit after tree verification."""

import pytest
import torch

pytest.importorskip("vllm")


def test_empty_linear_state_commit_is_noop():
    # Let kernel tests select their Triton execution mode during collection first.
    from eagle_harness.vllm.gdn_tree_verify import LinearStateCommit

    commit = LinearStateCommit([], [], torch.device("cpu"))
    mapping = torch.tensor([0], dtype=torch.int32)
    path = torch.tensor([[0, 2, 3]], dtype=torch.int32)
    length = torch.tensor([2], dtype=torch.int32)
    commit.commit(mapping, path, length)
    assert path.tolist() == [[0, 2, 3]]
