# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Batch admission is independent of the draft shape; widths remain per-request."""

import pytest

from eagle_harness.core.limits import ComputeIntensity, check_shape, cost_batch_sizes, max_batch_size, verify_buckets
from eagle_harness.vllm.config import verify_width_schedule


@pytest.mark.parametrize("intensity", [ComputeIntensity.HIGH, ComputeIntensity.LOW])
@pytest.mark.parametrize("batch_size", [33, 64, 128, 256, 512])
def test_tree_admission_and_schedule_reach_the_shared_batch_limit(intensity, batch_size):
    check_shape(intensity, node_top_k=4, tree_depth=12, verify_num=16, max_running_requests=batch_size)
    assert max_batch_size(intensity) >= batch_size
    assert 1 in verify_buckets(batch_size, intensity)
    schedule = verify_width_schedule(intensity, 16)
    assert next(k for first, last, k in schedule if first <= batch_size <= last) == 15


@pytest.mark.parametrize("intensity", list(ComputeIntensity))
def test_batches_above_the_supported_range_are_rejected(intensity):
    with pytest.raises(ValueError, match="batch limit"):
        check_shape(intensity, node_top_k=1, tree_depth=12, verify_num=16, max_running_requests=513)
    with pytest.raises(NotImplementedError, match="batch limit"):
        verify_buckets(513, intensity)


def test_tree_calibration_covers_large_batches():
    buckets = cost_batch_sizes(max_batch_size(ComputeIntensity.HIGH))
    assert {64, 128, 256, 512}.issubset(buckets)
