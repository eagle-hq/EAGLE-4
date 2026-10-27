# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-Harness device policy kernels (EAGLE-4); see
``csrc/eagle_harness_policy.cu`` for the state and slot layouts, and the
runtime's ``tree_round/device_policy.py`` for the tables and call order.

Every kernel is one CTA reading its sizes from device memory, so each launch
can be captured in a CUDA graph once and replayed for any round. Tables and
state are contiguous CUDA tensors; ``frontier`` and the rows of
``top_scores`` may be strided."""

import torch

from eagle_harness.tokenspeed.tree_round.ops.build import load_module

__all__ = [
    "eagle_harness_policy_begin_round",
    "eagle_harness_policy_decide_chain",
    "eagle_harness_policy_decide_tree",
    "eagle_harness_policy_load",
    "eagle_harness_policy_select_verify",
    "eagle_harness_ranked_scores",
]


def eagle_harness_policy_load() -> None:
    """Load the kernel library (no launch), before a graph capture uses it."""
    load_module("policy")


def eagle_harness_policy_begin_round(
    state: torch.Tensor,
    seq: torch.Tensor,
    hist: torch.Tensor,
    slots: torch.Tensor,
    go_host: torch.Tensor,
    batch_table: torch.Tensor,
    tier_table: torch.Tensor,
    cost_f64: torch.Tensor,
    round_f64: torch.Tensor,
    live_rows: torch.Tensor,
    frontier: torch.Tensor,
    sampled: torch.Tensor,
    chain: bool,
    node_top_k: int,
    chain_past_verify_num: int,
    verify_num: int,
    min_verify_count: int,
) -> None:
    """Begin a serving tree or chain round on the current CUDA stream.

    State and slot layouts match device_policy.py. Sampled rounds fix their
    depth before proposals. Invalid live counts trap before any frontier read.
    """
    load_module("policy").eagle_harness_policy_begin_round(
        state,
        seq,
        hist,
        slots,
        go_host,
        batch_table,
        tier_table,
        cost_f64,
        round_f64,
        live_rows,
        frontier,
        sampled,
        chain,
        node_top_k,
        chain_past_verify_num,
        verify_num,
        min_verify_count,
    )


def eagle_harness_policy_decide_tree(
    state: torch.Tensor,
    seq: torch.Tensor,
    slots: torch.Tensor,
    go_host: torch.Tensor,
    work_deltas: torch.Tensor,
    round_f64: torch.Tensor,
    tier_table: torch.Tensor,
    depth: int,
    node_top_k: int,
    exec_mode: int,
    async_layer_ratio: float,
    verify_num: int,
    min_verify_count: int,
) -> None:
    """Decide tree width/depth using device counts; exec_mode must be zero.

    A stop writes verify_count capped by the fixed verify width and pool size.
    """
    load_module("policy").eagle_harness_policy_decide_tree(
        state,
        seq,
        slots,
        go_host,
        work_deltas,
        round_f64,
        tier_table,
        depth,
        node_top_k,
        exec_mode,
        async_layer_ratio,
        verify_num,
        min_verify_count,
    )


def eagle_harness_policy_decide_chain(
    state: torch.Tensor,
    seq: torch.Tensor,
    slots: torch.Tensor,
    go_host: torch.Tensor,
    work_deltas: torch.Tensor,
    round_f64: torch.Tensor,
    depth: int,
    verify_num: int,
) -> None:
    """A chain round's decision after ``depth``, as
    ``eagle_harness_policy_decide_tree`` on contiguous fp32 ``[2]`` work
    deltas, with the round's stop terms per depth in ``round_f64``."""
    load_module("policy").eagle_harness_policy_decide_chain(
        state, seq, slots, go_host, work_deltas, round_f64, depth, verify_num
    )


def eagle_harness_ranked_scores(
    top_scores: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    flat: torch.Tensor,
    out: torch.Tensor,
    out_count: torch.Tensor,
) -> None:
    """Cumulative ranked scores of ``top_scores[:rows[0], :cols[0]]`` (fp32,
    unit column stride; sizes int32 ``[1]`` on the device) into ``out``
    (fp32), bit for bit as
    ``torch.cumsum(t.mean(dim=0), dim=0)``, or ``torch.cumsum(t.view(-1),
    dim=0)`` when ``flat[0]`` (int32) is nonzero, computed on the contiguous
    ``t``. ``out_count[0]`` (int32) gets their number: 0 for no rows or
    columns, -1 for shapes past ``top_scores``, ``out``, 32 rows or 64
    columns (whose torch reduction order is not replicated)."""
    load_module("policy").eagle_harness_ranked_scores(
        top_scores, rows, cols, flat, out, out_count
    )


def eagle_harness_policy_select_verify(
    state: torch.Tensor,
    recent: torch.Tensor,
    hist: torch.Tensor,
    published: torch.Tensor,
    top_scores: torch.Tensor,
    interval_i32: torch.Tensor,
    interval_f64: torch.Tensor,
    verify_i32: torch.Tensor,
    verify_f64: torch.Tensor,
    tree_depth: int,
    exec_mode: int,
) -> None:
    """Publish ranked scores and choose the meaningful count for a fixed verify window.

    Scores follow torch's default CUDA reduction order. History and shapes
    are owned by DevicePolicy; exec_mode must be zero.
    """
    load_module("policy").eagle_harness_policy_select_verify(
        state,
        recent,
        hist,
        published,
        top_scores,
        interval_i32,
        interval_f64,
        verify_i32,
        verify_f64,
        tree_depth,
        exec_mode,
    )
