# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Replay records follow the arch predicate and the state layout; they are inert where the predicate is off."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "replay,can_commit,steps,records",
    [
        (False, True, 16, False),
        (True, False, 16, False),
        (True, True, 128, False),
        (True, True, 16, True),
    ],
)
def test_record_dispatch_is_platform_and_state_gated(monkeypatch, replay, can_commit, steps, records, ssm_dtype):
    # Import at execution, not collection: other CPU kernel suites configure
    # Triton's interpreter before importing decorated kernels.
    pytest.importorskip("sglang")
    from eagle_harness.kernels import arch
    from eagle_harness.sglang.gdn_tree_verify import GDNTreeVerify

    monkeypatch.setattr(arch, "gdn_records_replay", lambda device=None: replay)
    state = SimpleNamespace(
        has_state_snapshots=True,
        conv_state_len=3,
        conv_dtype=torch.bfloat16,
        ssm_dtype=ssm_dtype,
        device=torch.device("cpu"),
        num_v_heads=4,
        snapshot_shape=(2, 3, steps, 4, 128, 128),
        can_commit_state=can_commit,
    )
    from eagle_harness.kernels import gdn_tree_records as records_module

    monkeypatch.setattr(records_module, "_wave_slots", lambda device: 8192)
    extension = GDNTreeVerify.create(state)
    assert extension is not None
    assert (extension.records is not None) is records
    assert extension._uses_path_replay(1, 16) is (ssm_dtype != torch.float32 if records else ssm_dtype == torch.float32)
    if records:
        # A busier GPU must not change the recurrence's rounding semantics.
        for batch, width in ((1, 8), (8, 16), (32, 32), (64, 32)):
            assert extension._uses_path_replay(batch, width) is (ssm_dtype != torch.float32)
        assert extension.records.shape == (2, 3, steps, 4, 260)
    else:
        assert extension.commit_state(None, None, None, None, None) is False


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_records_require_rounded_paths(dtype):
    from eagle_harness.kernels.gdn_tree_records import gdn_tree_records_update

    with pytest.raises(ValueError, match="tree_paths.*low-precision"):
        gdn_tree_records_update(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            state=SimpleNamespace(dtype=dtype),
            state_indices=None,
            query_start_loc=None,
            parent=None,
            tree_paths=None,
            records=None,
            record_indices=None,
            width=16,
        )
