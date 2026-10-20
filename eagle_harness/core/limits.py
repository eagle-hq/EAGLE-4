# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Shape limits and measured buckets shared by the cost model and calibration."""

from __future__ import annotations

import enum
from typing import Iterable, List, Tuple


class ComputeIntensity(str, enum.Enum):
    """Draft shape family, serialized as a compute-intensity string."""

    HIGH = "high_intensity"
    LOW = "low_intensity"
    CHAIN = "chain_intensity"


MAX_BATCH_SIZE_CHAIN = 512
# Tree kernels index requests independently; their buffers are sized from the configured batch limit.
MAX_BATCH_SIZE_TREE = MAX_BATCH_SIZE_CHAIN
MAX_VERIFY_TOKENS = 4096
MAX_TREE_DEPTH = 31
MAX_CHAIN_DEPTH = 15
MAX_NODE_TOP_K = 8
TAIL_MASK_BITS = 64  # One uint64 ancestry word per verify row.
MAX_VERIFY_NUM = 64
assert MAX_VERIFY_NUM <= TAIL_MASK_BITS
MAX_CHAIN_VERIFY_NUM = 16

COST_BATCH_SIZES: Tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
)

_CHAIN_VERIFY_BUCKETS = (16, 14, 12, 10, 8, 6, 4, 2, 1)


def max_batch_size(intensity: ComputeIntensity) -> int:
    """The largest batch size served at this compute intensity."""
    if intensity is ComputeIntensity.CHAIN:
        return MAX_BATCH_SIZE_CHAIN
    return MAX_BATCH_SIZE_TREE


def cost_batch_sizes(max_bs: int, start_bs: int = 1) -> List[int]:
    """Batch sizes calibration measures, within ``[start_bs, max_bs]``."""
    return [bs for bs in COST_BATCH_SIZES if start_bs <= bs <= max_bs]


def nearest_cost_batch_size(batch_size: int, measured: Iterable[int]) -> int:
    """The measured batch size closest to ``batch_size``; ties go to the larger."""
    candidates = sorted(measured)
    if not candidates:
        raise ValueError("No measured batch sizes in the cost data.")
    return min(candidates, key=lambda bs: (abs(bs - batch_size), -bs))


def verify_buckets(batch_size: int, intensity: ComputeIntensity) -> List[int]:
    """Verify widths that get a target-verify CUDA graph and a cost curve.

    Returns widths in descending order.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if batch_size > MAX_BATCH_SIZE_CHAIN:
        raise NotImplementedError(f"batch_size={batch_size} exceeds the chain batch limit {MAX_BATCH_SIZE_CHAIN}.")
    if intensity is ComputeIntensity.CHAIN:
        return list(_CHAIN_VERIFY_BUCKETS)
    if batch_size > MAX_BATCH_SIZE_TREE:
        raise NotImplementedError(f"the tree draft batch limit is {MAX_BATCH_SIZE_TREE}, got batch_size={batch_size}.")
    if batch_size <= 2:
        return [64, 32, 16, 8, 1]
    if batch_size <= 8:
        if intensity is ComputeIntensity.HIGH:
            return [64, 32, 16, 8, 1]
        return [32, 16, 8, 4, 1]
    if intensity is ComputeIntensity.HIGH:
        return [32, 16, 8, 4, 1]
    return [16, 8, 4, 2, 1]


def draft_query_lens(batch_size: int, intensity: ComputeIntensity) -> List[int]:
    """Draft-step query lengths (tree widths) calibration measures."""
    if intensity is ComputeIntensity.CHAIN:
        return [1]
    if batch_size <= 32:
        return [1, 2, 4, 8]
    if batch_size <= 64:
        return [1, 2, 4]
    return [1, 2]


def req_to_token_span(node_top_k: int, tree_depth: int, verify_num: int, page_size: int) -> int:
    """Widest block-table span past ``seq_len`` one step writes.

    Draft (``node_top_k * tree_depth`` slots), verify (``verify_num``; a chain
    verifies at most its drafted nodes plus the root) and, for a chain at
    ``page_size > 1``, the shared draft / verify allocation all start at
    ``seq_len``; the span is their maximum.
    """
    draft = node_top_k * tree_depth
    verify = verify_num if node_top_k > 1 else min(verify_num, tree_depth + 1)
    shared = tree_depth + 1 if node_top_k == 1 and page_size > 1 else 0
    return max(draft, verify, shared)


def check_shape(
    intensity: ComputeIntensity, node_top_k: int, tree_depth: int, verify_num: int, max_running_requests: int
) -> None:
    """Reject draft shapes the kernels do not support."""
    chain = intensity is ComputeIntensity.CHAIN
    cap = max_batch_size(intensity)
    if max_running_requests > cap:
        raise ValueError(
            f"max_running_requests={max_running_requests} exceeds the {'chain' if chain else 'tree'} "
            f"draft batch limit {cap}."
        )
    if tree_depth > (MAX_CHAIN_DEPTH if chain else MAX_TREE_DEPTH):
        raise NotImplementedError(f"tree_depth={tree_depth} is not supported for {intensity.value}.")
    if node_top_k > MAX_NODE_TOP_K:
        raise NotImplementedError(f"--speculative-eagle-topk > {MAX_NODE_TOP_K} is not supported.")
    if chain and node_top_k > 1:
        raise NotImplementedError("--speculative-eagle-topk > 1 is not supported for chain_intensity.")
    if verify_num > (MAX_CHAIN_VERIFY_NUM if chain else MAX_VERIFY_NUM):
        raise NotImplementedError(f"verify_num={verify_num} is not supported for {intensity.value}.")
    if verify_num <= 1:
        raise ValueError("verify_num <= 1 is invalid.")
