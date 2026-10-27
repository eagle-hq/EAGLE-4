# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Graph-capturable round decisions; costs are supplied by the shared core model."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch

from eagle_harness.tokenspeed.tree_round.ops.build import load_module
from eagle_harness.tokenspeed.tree_round.ops.policy import (
    eagle_harness_policy_begin_round,
    eagle_harness_policy_decide_chain,
    eagle_harness_policy_decide_tree,
    eagle_harness_policy_load,
    eagle_harness_policy_select_verify,
)
from eagle_harness.tokenspeed.tree_round.policy_cost import (
    ChainPolicyCosts,
    TreePolicyCosts,
    build_chain_tables,
    build_tables,
)

__all__ = ["DevicePolicy", "write_round_slots"]

# Scalars of the int32 state tensor (``State`` in eagle_harness_policy.cu).
(
    _TINY,
    _TIER,
    _SAMPLED_NEXT,
    _SAMPLED_PLAN,
    _SAMPLED_SEEN,
    _BENCH_BATCH_SIZE,
    _RANKED_COUNT,
    _BATCH_SIZE,
    _SAMPLED,
    _COST_ID,
    _INTERVAL,
    _VARIANT,
    _HIST_MAX,
    _PAST_VERIFY_NUM,
    _LAST_DEPTH,
    _STOP_WIDTH,
    _CUMU_COUNT,
    _VERIFY_COUNT,
    _FLAT,
    _VERIFY_NUM,
    _TABLE_WIDTH,
    _NUM_STATE,
) = range(22)
# Rows of the slot tensor (``SlotRow``).
_WIDTH_ROW, _CUMU_START_ROW, _WORK_SIZE_ROW, _GO_ROW, _NUM_SLOT_ROWS = range(5)


def write_round_slots(
    slots: torch.Tensor,
    node_top_k: int,
    widths: list[int],
    work_size: Callable[[int, int], int],
) -> int:
    """Write the ``[4, tree_depth + 1]`` device ``slots`` of a round drafting
    depth ``d >= 1`` at ``widths[d - 1]`` with ``work_size(d, cumu_start)``
    work deltas; returns its pool size."""
    host = torch.zeros(slots.shape, dtype=torch.int32)
    start = 1 + node_top_k
    for depth, width in enumerate(widths, start=1):
        host[:, depth] = torch.tensor([width, start, work_size(depth, start), 1])
        start += node_top_k * width
    slots.copy_(host)
    return start


class DevicePolicy:
    """Device-only serving policy with a fixed verify window.

    ``costs`` supplies host-only structural records computed by the caller.
    The engine packs these records without constructing a cost model. Trees
    support K=2/4/8, 2..32 depths, <=64 lane slots, N=2..64 and 32 requests.
    Chains support K=1, 2..15 depths, N=2..16 and 512 requests. Only serving
    (exec_mode=0) is supported. A chain width table supplies increasing
    ``(min_batch_size, width)`` pairs; uncovered batches use the cost rule.
    ``verify_num`` on the device is the meaningful node count, not the fixed
    window size: write_verify_tree pads the rest with roots (parent -1).

    Call observe_batch_size, begin_round, then decide before each draft
    depth, then select_verify on the ranked pool scores. Counts stay on the
    device; go_ready optionally polls a pinned mirror. Sampled rounds fix
    depth before drawing proposals, observing costs only for the next plan.
    Construct under the desired CUDA device before any graph capture.
    """

    def __init__(
        self,
        costs: TreePolicyCosts | ChainPolicyCosts | None,
        *,
        verify_num: int,
        node_top_k: int,
        tree_depth: int,
        compute_intensity: str,
        is_moe_target: bool,
        exec_mode: int,
        benchmark_config: str = "",
        chain_width_table: tuple[tuple[int, int], ...] = (),
        device: torch.device,
    ) -> None:
        if compute_intensity not in (
            "high_intensity",
            "low_intensity",
            "chain_intensity",
        ):
            raise ValueError("DevicePolicy needs high, low or chain intensity")
        if exec_mode != 0 or benchmark_config:
            raise ValueError(
                "DevicePolicy supports serving only (exec_mode=0, no benchmark config)"
            )
        self.chain = compute_intensity == "chain_intensity"
        self.chain_width_table = chain_width_table
        if self.chain:
            if (
                any(type(v) is not int for v in (node_top_k, tree_depth, verify_num))
                or node_top_k != 1
                or not 2 <= tree_depth <= 15
                or not 2 <= verify_num <= 16
            ):
                raise ValueError(
                    "DevicePolicy chain needs K=1, depth 2..15 and verify 2..16"
                )
        else:
            if chain_width_table:
                raise ValueError("DevicePolicy trees run no chain width table")
            if node_top_k not in (2, 4, 8) or not 2 <= tree_depth <= 32:
                raise ValueError("DevicePolicy needs K=2/4/8 and tree_depth in 2..32")
            if (tree_depth - 1) * node_top_k > 64:
                raise ValueError("DevicePolicy lane window exceeds 64 slots")
            if not 2 <= verify_num <= 64:
                raise ValueError("DevicePolicy verify_num must be in 2..64")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DevicePolicy requires a CUDA device")
        self.chain_past_verify_num = 16 if self.chain else 0
        self.node_top_k = node_top_k
        self.tree_depth = tree_depth
        self.verify_num_limit = verify_num
        self.high_intensity = compute_intensity == "high_intensity"
        self.is_moe_target = is_moe_target
        self.exec_mode = exec_mode
        self.eager_select_verify = False
        max_bs = 512 if self.chain else 32
        self.async_layer_ratio: float | None = None
        if self.chain:
            self._tables = build_chain_tables(
                cast(ChainPolicyCosts, costs),
                tree_depth=tree_depth,
                verify_num=verify_num,
                chain_width_table=chain_width_table,
                device=self.device,
            )
        else:
            tree_costs = cast(TreePolicyCosts, costs)
            self._tables = build_tables(tree_costs, device=self.device)
            packed_depth = (
                self._tables.interval_f64.shape[-1]
                - 2
                - (self._tables.interval_i32.shape[-1] - 1) ** 2
            ) // 5
            if packed_depth != tree_depth:
                raise ValueError("cost records must match the policy tree_depth")
            if any(width > node_top_k for width in tree_costs.initial_widths):
                raise ValueError("cost records exceed the policy node_top_k")
            self.async_layer_ratio = tree_costs.async_layer_ratio
        # Resolve libraries before capture on the device that will run them.
        with torch.cuda.device(self.device):
            eagle_harness_policy_load()
            load_module("tree")
        state = [0] * _NUM_STATE
        state[_SAMPLED_SEEN] = -1
        # The reference starts sampled chains at the root, trees at full depth.
        state[_SAMPLED_NEXT] = 0 if self.chain else tree_depth - 1
        hist = (
            torch.zeros((max_bs, 0), dtype=torch.int32)
            if self.chain
            else torch.full((max_bs, 16), verify_num, dtype=torch.int32)
        )
        recent = (
            torch.zeros(0, dtype=torch.int32)
            if self.chain
            else torch.arange(-5, 0, dtype=torch.int32)
        )
        self._state = torch.tensor(state, dtype=torch.int32, device=device)
        self._hist = hist.to(device=device, dtype=torch.int32)
        self._recent = recent.to(device=device, dtype=torch.int32)
        # The ranked scores the next round sizes its verify on (none before the first).
        self._ranked = torch.zeros(
            (max_bs * verify_num,), dtype=torch.float32, device=device
        )
        self._slots = torch.zeros(
            (_NUM_SLOT_ROWS, self.tree_depth + 1), dtype=torch.int32, device=device
        )
        # The round's cost row, copied by begin_round for its decisions.
        depth = self.tree_depth
        self._round_f64 = torch.zeros(
            (5 * depth if self.chain else 2 + depth,),
            dtype=torch.float64,
            device=device,
        )
        self._round_seq = torch.zeros((1,), dtype=torch.int64, device=device)
        self.go_host = torch.zeros(
            (self.tree_depth + 1,), dtype=torch.int64, pin_memory=True
        )
        self._go_host_view = self.go_host.numpy()

        self.past_verify_num = self._state[_PAST_VERIFY_NUM : _PAST_VERIFY_NUM + 1]
        self.widths = self._slots[_WIDTH_ROW]
        self.cumu_starts = self._slots[_CUMU_START_ROW]
        self.work_sizes = self._slots[_WORK_SIZE_ROW]
        self.go = self._slots[_GO_ROW]
        self.cumu_count = self._state[_CUMU_COUNT : _CUMU_COUNT + 1]
        self.verify_count = self._state[_VERIFY_COUNT : _VERIFY_COUNT + 1]
        self.flat = self._state[_FLAT : _FLAT + 1]
        self.verify_num = self._state[_VERIFY_NUM : _VERIFY_NUM + 1]
        # The depth whose decision stopped the round, and that decision's width.
        self.last_depth = self._state[_LAST_DEPTH : _LAST_DEPTH + 1]
        self.stop_width = self._state[_STOP_WIDTH : _STOP_WIDTH + 1]

    def table_width(self, batch_size: int) -> int:
        """Explicit scheduled width, or zero when this batch uses measured costs."""
        width = 0
        for minimum, candidate in self.chain_width_table:
            if batch_size < minimum:
                break
            width = candidate
        return width

    def observe_batch_size(self, batch_size: int) -> None:
        """Validate the live batch on the host before entering a round."""
        max_bs = 512 if self.chain else 32
        if not 0 <= batch_size <= max_bs:
            raise ValueError(
                f"DevicePolicy needs 0..{max_bs} live requests, got {batch_size}"
            )

    def begin_round(
        self, live_rows: torch.Tensor, frontier: torch.Tensor, sampled: torch.Tensor
    ) -> None:
        """Start a round; sampled is a device bool [1] selecting sampled proposals.

        live_rows is int32 [1] (0..32 tree, 0..512 chain); frontier is int64 lengths.
        Counts and decisions stay on the device, including during graph capture.
        """
        t = self._tables
        eagle_harness_policy_begin_round(
            self._state,
            self._round_seq,
            self._hist,
            self._slots,
            self.go_host,
            t.batch_table,
            t.tier_table,
            t.chain_f64 if self.chain else t.interval_f64,
            self._round_f64,
            live_rows,
            frontier,
            sampled,
            self.chain,
            self.node_top_k,
            self.chain_past_verify_num,
            self.verify_num_limit,
            16,
        )

    def deltas_depth(self, depth: int) -> bool:
        """Whether this non-final depth requests deltas: tree even, chain odd."""
        return depth < self.tree_depth - 1 and depth % 2 == int(self.chain)

    def write_full_round(self) -> None:
        """Slots of a round drafting every depth at full width, which the
        drafter warms its depths up on outside a round (rounds rewrite them).
        No round is live afterwards: decisions launched on these slots (a
        warm-up) write nothing."""
        width = 1 if self.chain else self.node_top_k
        past = self.chain_past_verify_num if self.chain else self.verify_num_limit
        write_round_slots(
            self._slots,
            self.node_top_k,
            [width] * (self.tree_depth - 1),
            lambda depth, start: min(past, start) if self.deltas_depth(depth) else 0,
        )
        self.past_verify_num.fill_(past)
        self._state[_BATCH_SIZE].fill_(0)

    def decide(self, depth: int, work_deltas: torch.Tensor) -> None:
        """Decide before depth d drafts, using depth d-1 work deltas (fp32 [K+1]).

        Writes the next depth slots or stops with cumu_count, verify_count and flat.
        """
        if self.chain:
            eagle_harness_policy_decide_chain(
                self._state,
                self._round_seq,
                self._slots,
                self.go_host,
                work_deltas,
                self._round_f64,
                depth,
                self.verify_num_limit,
            )
            return
        assert self.async_layer_ratio is not None
        eagle_harness_policy_decide_tree(
            self._state,
            self._round_seq,
            self._slots,
            self.go_host,
            work_deltas,
            self._round_f64,
            self._tables.tier_table,
            depth,
            self.node_top_k,
            self.exec_mode,
            self.async_layer_ratio,
            self.verify_num_limit,
            16,
        )

    def select_verify(self, top_scores: torch.Tensor) -> None:
        """Publish cumulative ranked scores and choose the meaningful verify count.

        top_scores is fp32 [rows >= live rows, cols >= verify_count], in rank
        order. The previous round's scores choose this round's count (its own
        scores on the first greedy round; no scores on the first sampled round),
        recorded in the verify history. Chains verify their whole pool and
        update the next sampled plan when their stopping decision runs. Reductions
        follow torch's default CUDA cumsum; deterministic algorithms are refused.
        """
        if self.chain:
            return
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError(
                "the device ranked scores replicate torch's default cumsum, not its deterministic one"
            )
        t = self._tables
        eagle_harness_policy_select_verify(
            self._state,
            self._recent,
            self._hist,
            self._ranked,
            top_scores,
            t.interval_i32,
            t.interval_f64,
            t.verify_i32,
            t.verify_f64,
            self.tree_depth,
            self.exec_mode,
        )

    def go_ready(self, depth: int, round_seq: int) -> bool | None:
        """Whether depth ``depth`` drafts in round ``round_seq`` (the
        ``begin_round`` count, from 1), from the pinned mirror without a
        synchronization: None until the decision wrote it."""
        value = int(self._go_host_view[depth])
        if value >> 1 != round_seq:
            return None
        return bool(value & 1)
