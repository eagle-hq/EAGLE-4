# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""When a round's draft tree stops growing: the stop rule of :class:`~eagle_harness.vllm.tree_drafter.TreeDrafter`.

After draft step ``s`` the tree holds depths ``1 .. s + 1``; step ``s + 1`` would add the children of the depth-``s + 1``
frontier. A node's score is its parent's times a probability, and the verify set ranks nodes by depth-decayed score,
so a child at depth ``s + 2`` ranks at most ``front * DEPTH_DECAY ** (s + 2)`` (``front``: its parent's score); every
later descendant ranks lower still. With ``threshold`` the decayed score of the ``count``-th best node so far (the
last verify row), the next step runs only if

- ``max(front) * decay >= threshold``: some child could still enter the verify set (with no gain asked of the step
  this is the whole rule, and the verify rows are exactly those of the full-depth tree), and
- ``gain = sum(relu(front * decay * ratio - threshold)) >= min_gain + step_share * expected``: what the step is
  expected to add to the verify set's score sum (its expected accepted length) pays for the step. A child scores its
  parent's score times its own probability; ``ratio`` estimates that probability by the last step's: the best score
  of the newest layer over the best score of the layer above it.

The price of a step comes from measured costs (:class:`RoundCosts`): a round that ran ``s`` steps takes
``rest + s * draft_step`` and yields ``expected = 1 + sum(verify set scores)`` accepted tokens by the draft's own
probabilities, so one more step keeps the tokens per time from falling only if it adds at least
``expected * draft_step / (rest + s * draft_step)`` (``step_share = draft_step / (rest + s * draft_step)``). Without
measured costs ``step_share`` is 0 and ``min_gain`` (0 unless the config overrides it) is the whole price.

Across a batch, the next step runs if any request wants it (a request that would stop loses nothing). The decision
stays on the GPU: :func:`keep_growing` writes a scalar bool that a CUDA graph conditional node reads, so a captured
growth skips the steps it does not need without a host read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import torch
import triton
import triton.language as tl

from eagle_harness.core.chain_width import median_cost
from eagle_harness.core.config import Mode
from eagle_harness.core.cost_table import BatchCostTable, load_cost_tables
from eagle_harness.core.limits import MAX_BATCH_SIZE_TREE, nearest_cost_batch_size
from eagle_harness.vllm.config import VllmMethodConfig, verify_width_schedule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoundCosts:
    """What a draft step costs against its round, in microseconds (calibration tables, :func:`round_costs`).

    shortcut: ``rest`` is the verify forward only; the draft prefill, the accept and the state commit are not in the
    tables, which prices a step ~20% high on Qwen3.8-27B at TP4 (inside the flat optimum of the replay). Time them in
    the calibration if a model's round has a larger share outside the verify.
    """

    draft_step: float  # one draft step: the tree step over the frontier
    rest: float  # the round without its draft steps: the target's verify forward

    def step_share(self, steps: int) -> float:
        """One more draft step's cost over a round that ran ``steps`` of them."""
        return self.draft_step / (self.rest + steps * self.draft_step)


def round_costs(
    tables: Mapping[int, BatchCostTable] | None, batch_size: int, top_k: int, count: int
) -> RoundCosts | None:
    """The costs of a round of ``batch_size`` requests with ``count`` verify rows each, from the cost tables of the
    nearest measured batch size: the draft's step of ``top_k`` rows and the target's forward of ``count + 1`` tokens per
    request, each the median over the tables' context intervals. None if the tables did not time these shapes."""
    if not tables:
        return None
    table = tables[nearest_cost_batch_size(batch_size, tables)]
    draft = median_cost([interval.draft for interval in table.intervals], top_k)
    verify = median_cost([interval.target for interval in table.intervals], count + 1)
    if draft is None or verify is None or draft <= 0 or verify <= 0:
        return None
    return RoundCosts(draft_step=draft, rest=verify)


def derived_cost_tables(
    config: VllmMethodConfig,
    key=None,
    *,
    schedule: list[tuple[int, int, int]] | None = None,
    max_batch_size: int = MAX_BATCH_SIZE_TREE,
) -> dict[int, BatchCostTable] | None:
    """Require prices for every automatic serving shape before drafting.

    Static calibration may run without tables; serving must explicitly set
    ``depth_stop_gain`` to opt out of measured pricing.
    """
    cost_dir = config.harness.cost_dir if config.harness.data_path else None
    top_k, width = config.node_top_k, config.max_verify_width
    tables = load_cost_tables(cost_dir, key) if cost_dir else {}
    if config.harness.mode is not Mode.STATIC:
        schedule = schedule if schedule is not None else verify_width_schedule(config.harness.intensity, width)
        for first, last, count in schedule:
            if count == 0:
                continue
            for bs in range(first, min(last, max_batch_size) + 1):
                if round_costs(tables, bs, top_k, count) is None:
                    raise ValueError(
                        f"eagle_harness: no cost tables price batch size {bs}, draft step {top_k}, "
                        f"verify width {count + 1} under {cost_dir}; calibrate this shape or explicitly "
                        "set depth_stop_gain (0 for a lossless stop, null for fixed depth)."
                    )
    costs = round_costs(tables, 1, top_k, width - 1)
    if costs is None and config.harness.mode is not Mode.STATIC:
        logger.info("eagle_harness: all active scheduled shapes are priced under %s", cost_dir)
        return tables
    if costs is None:
        logger.warning(
            "eagle_harness: no cost tables price a draft step of %d rows and a verify of width %d%s; the depth stop is "
            "lossless (set data_path to calibrated cost tables for this round shape, or set depth_stop_gain)",
            top_k,
            width,
            f" under {cost_dir}" if cost_dir else "",
        )
        return None
    logger.info(
        "eagle_harness: depth stop priced by the cost tables under %s: draft step %.1f us, verify %.1f us (batch "
        "size 1, width %d)",
        cost_dir,
        costs.draft_step,
        costs.rest,
        width,
    )
    return tables


def keep_growing_reference(
    scores: torch.Tensor,
    decay: torch.Tensor,
    front: torch.Tensor,
    count: int,
    next_decay: float,
    min_gain: float,
    parent_layer: tuple[int, int],
    step_share: float = 0.0,
) -> torch.Tensor:
    """The rule in torch: ``scores`` ``[bs, nodes]`` (0 where no node was grown), ``decay`` ``[nodes]``, ``front``
    ``[bs, K]`` the frontier's scores (the newest layer's best), ``parent_layer`` the node range of the layer above the
    newest, ``step_share`` :meth:`RoundCosts.step_share`; a scalar bool."""
    ranked = torch.sort(scores * decay, dim=-1, descending=True).values
    threshold = ranked[:, count - 1 : count]
    reach = front * torch.tensor(next_decay, dtype=torch.float32, device=front.device)
    parent_best = scores[:, parent_layer[0] : parent_layer[1]].max(-1, keepdim=True).values
    ratio = front.max(-1, keepdim=True).values / parent_best.clamp(min=1e-30)
    gain = (reach * ratio - threshold).clamp(min=0).sum(-1)
    expected = 1.0 + ranked[:, :count].sum(-1)
    keep = (reach.max(-1).values >= threshold[:, 0]) & (gain >= min_gain + step_share * expected)
    return keep.any()


@triton.jit
def _keep_growing_kernel(
    scores,
    decay,
    front,
    go,
    stats,
    bs,
    stride_scores,
    stride_front,
    count,
    next_decay,
    min_gain,
    step_share,
    parent_lo,
    parent_hi,
    NODES: tl.constexpr,
    N_BLOCK: tl.constexpr,
    K: tl.constexpr,
    K_BLOCK: tl.constexpr,
    STEPS_BEFORE: tl.constexpr,
):
    node = tl.arange(0, N_BLOCK)
    lane = tl.arange(0, K_BLOCK)
    d = tl.load(decay + node, mask=node < NODES, other=0.0)
    keep = tl.full([], 0, tl.int32)
    for b in range(bs):
        raw = tl.load(scores + b * stride_scores + node, mask=node < NODES, other=0.0)
        ranked = tl.sort(raw * d, descending=True)
        threshold = tl.sum(tl.where(node == count - 1, ranked, 0.0))
        f = tl.load(front + b * stride_front + lane, mask=lane < K, other=0.0)
        reach = f * next_decay
        parent_best = tl.max(tl.where((node >= parent_lo) & (node < parent_hi), raw, 0.0))
        ratio = tl.max(f) / tl.maximum(parent_best, 1e-30)
        gain = tl.sum(tl.where(lane < K, tl.maximum(reach * ratio - threshold, 0.0), 0.0))
        expected = 1.0 + tl.sum(tl.where(node < count, ranked, 0.0))
        wants = (tl.max(tl.where(lane < K, reach, -1.0)) >= threshold) & (gain >= min_gain + step_share * expected)
        keep = tl.maximum(keep, wants.to(tl.int32))
    tl.store(go, keep != 0)
    # stats: [rounds, draft steps run]; the round's first decision also counts the steps before it
    if STEPS_BEFORE > 0:
        tl.store(stats, tl.load(stats) + 1)
    tl.store(stats + 1, tl.load(stats + 1) + STEPS_BEFORE + keep)


def keep_growing(
    go: torch.Tensor,
    stats: torch.Tensor,
    scores: torch.Tensor,
    decay: torch.Tensor,
    front: torch.Tensor,
    count: int,
    next_decay: float,
    min_gain: float,
    parent_layer: tuple[int, int],
    *,
    steps_before: int,
    step_share: float = 0.0,
) -> None:
    """Write the rule's decision into ``go`` (a scalar bool) in one launch; ``stats`` (int64 ``[2]``) counts rounds
    and the draft steps they run (``steps_before`` > 0: a round's first decision, after that many steps)."""
    bs, nodes = scores.shape
    k = front.shape[1]
    _keep_growing_kernel[(1,)](
        scores,
        decay,
        front,
        go,
        stats,
        bs,
        scores.stride(0),
        front.stride(0),
        count,
        float(next_decay),
        float(min_gain),
        float(step_share),
        parent_layer[0],
        parent_layer[1],
        NODES=nodes,
        N_BLOCK=triton.next_power_of_2(nodes),
        K=k,
        K_BLOCK=max(2, triton.next_power_of_2(k)),
        STEPS_BEFORE=steps_before,
    )
