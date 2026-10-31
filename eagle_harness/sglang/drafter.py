# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Cost-model-driven draft of one decode round in serve mode.

``DynamicDrafter.draft(batch)`` grows the draft tree one graph-replayed step at
a time, asks the cost model after each step whether another level pays for its
draft cost, picks the verify width from the lagged verify scores, and returns
sglang's ``EagleVerifyInput`` for the best ``width - 1`` nodes plus the root.

Host syncs per round: the work deltas of every second depth (the lagged
width decision: launched at an even depth, read one step later while the next
step runs) and nothing else; the verify scores the width decision reads are
the previous round's, copied asynchronously. With the width known before the
ranking, one graph ranks the tree, copies the scores out and builds the verify
input, including the verify tokens' KV slots and, at batch size 1, the verify
attention's tail-mask words (:mod:`.verify_plan`). The draft steps before the first
decision can be queued ahead, at the end of the previous round
(:meth:`DynamicDrafter.prelaunch`).

A chain with a width per batch size (``chain_width``: the served chain widths, or the calibration's forced
widths) asks the cost model nothing: a round of width ``w`` drafts ``w - 1`` tokens (the draft extend's top token,
then ``w - 2`` draft steps, all queued ahead by the prelaunch) and verifies them with the root; width 1 drafts nothing and verifies the root alone (plain
decoding; the draft extend still runs, so the draft stays current).
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_worker_common import build_eagle_verify_input

from eagle_harness.core.cost_model import CostModel, DraftCostRow, FixedShapePolicy, WidthStep
from eagle_harness.core.cost_table import cost_model_problem
from eagle_harness.core.draft_tree import DraftTree, VerifySelection
from eagle_harness.core.live_costs import DRAFT_WIDTH, LiveCosts
from eagle_harness.kernels import arch
from eagle_harness.sglang.draft_steps import DraftSteps
from eagle_harness.sglang.verify_input import HarnessVerifyInput, build_attn_args
from eagle_harness.sglang.verify_tail import VerifyTail
from eagle_harness.sglang.verify_graphs import round_up_width

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

logger = logging.getLogger(__name__)

# Served rounds between two refreshes of the live cost tables (and their TP broadcast); the cache file is
# rewritten every 16th refresh and when a bucket's seed sweep completes.
LIVE_REFRESH_ROUNDS = 64
LIVE_SAVE_REFRESHES = 16
# Rounds a batch bucket priced from the buckets around it serves before it is seeded itself: a bucket the load only
# passes through (a draining batch) stays interpolated, one it dwells in pays for its own seed sweep.
LIVE_SEED_AFTER_ROUNDS = LIVE_REFRESH_ROUNDS


def spin_wait(event: torch.cuda.Event) -> None:
    """Wait for ``event`` by polling it. A blocking ``Event.synchronize()`` returns well after the event
    completes on these hosts, and every round waits on the GPU a few times."""
    while not event.query():
        pass


class RoundTimer:
    """CUDA-event spans of served work (the draft steps of a round, its verify), fed to :class:`LiveCosts`.

    A span is two timing events on one stream around the work; :meth:`drain` reads the completed ones. The
    draft-step spans are taken only while a bucket seeds (its steps run back to back then, so the span is the
    graphs' time); the verify span is taken every round.
    """

    def __init__(self, live: LiveCosts):
        self.live = live
        self._pending: List[Tuple[torch.cuda.Event, torch.cuda.Event, int, int, int, int]] = []

    @staticmethod
    def start() -> torch.cuda.Event:
        """Record the span's start on the current stream."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop(self, start: torch.cuda.Event, bucket: int, width: int, context: int, count: int = 1) -> None:
        """Record the span's end; ``count`` work units share the span (steps of a draft)."""
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending.append((start, end, bucket, width, context, count))

    def drain(self, *, wait: bool) -> None:
        """Feed the completed spans to the live costs; ``wait`` blocks for the rest (seed rounds, so every
        TP rank records the same rounds and their cells agree). A span of a bucket that is not ready yet is always
        waited for: which buckets are ready decides which ones are priced from their neighbours, so it must
        change on the same round on every rank."""
        left = []
        for span in self._pending:
            start, end, bucket, width, context, count = span
            if wait or not self.live.ready(bucket):
                end.synchronize()
            elif not end.query():
                left.append(span)
                continue
            self.live.record(bucket, width, context, 1000.0 * start.elapsed_time(end) / count)
        self._pending = left


class _HostCopy:
    """A device vector copied to page-locked host memory behind one event."""

    def __init__(self, numel: int):
        self.host = torch.zeros(numel, dtype=torch.float32, pin_memory=True)
        self._values = self.host.numpy()
        self.event = torch.cuda.Event()
        self.size = 0

    def launch(self, values: torch.Tensor) -> None:
        """Copy ``values`` to pinned host memory without waiting; record the event ``read`` waits on."""
        self.copy(values)
        self.queued(values.numel())

    def copy(self, values: torch.Tensor) -> None:
        """The device-to-host copy alone (graph-capturable)."""
        self.host[: values.numel()].copy_(values, non_blocking=True)

    def queued(self, size: int) -> None:
        """A copy of ``size`` values was queued on the current stream; record the event ``read`` waits on."""
        self.size = size
        self.event.record()

    def read(self) -> List[float]:
        """Wait for the copy and return the values."""
        spin_wait(self.event)
        return self._values[: self.size].tolist()


@dataclass(kw_only=True)
class PrelaunchResult(GenerationBatchResult):
    """Fence scheduler GPU mutations after copying this round's independent outputs.

    The non-overlap scheduler copies sequence lengths and then calls ``copy_to_cpu``
    before filtering roots or reusing request slots. Its host wait on ``copy_done``
    must not include the next draft; its subsequent GPU work must include it.
    """

    prelaunch_done: torch.cuda.Event

    def copy_to_cpu(self, return_logprob: bool, return_hidden_states: bool = True):
        super().copy_to_cpu(return_logprob, return_hidden_states)
        torch.cuda.current_stream().wait_event(self.prelaunch_done)


class Prelaunch(NamedTuple):
    """A next-round draft extend and draft queued by :meth:`DynamicDrafter.prelaunch`."""

    key: Tuple[int, ...]  # batch_key of the batch it was queued for
    next_depth: int  # the first depth the round still has to decide at (a fixed-width chain: its depth)
    width: int  # a fixed-width chain's verify width for the round, 0 otherwise
    stream: torch.cuda.Stream  # the stream it was queued on


def batch_key(batch: "ScheduleBatch") -> Tuple[int, ...]:
    """The requests' block-table rows, in batch order: equal keys draft the same requests."""
    return tuple(req.kv.req_pool_idx for req in batch.reqs)


def fit_width(width: int, widths: Sequence[int], available: int) -> int:
    """The captured verify width for a requested ``width`` with ``available`` tree tokens (root included):
    the smallest captured width ``>= width`` if the tree fills it, else the widest one it fills."""
    up = round_up_width(width, widths)
    if up <= available:
        return up
    return max(w for w in widths if w <= available)


def _round_input(verify_input: "EagleVerifyInput", batch: "ScheduleBatch") -> "EagleVerifyInput":
    """This round's copy of a graph's verify input: the graph's is reused every round, and sglang sets per-round
    fields on the copy."""
    verify_input = copy.copy(verify_input)
    if verify_input.prepared_mrope_positions is not None and any(mm is not None for mm in batch.multimodal_inputs):
        verify_input.prepared_mrope_positions = None  # sglang adds the requests' mrope deltas
    return verify_input


class DynamicDrafter:
    """Owns the round's tree, its step graphs and the round policy.

    The policy of a round is the cost model with its tables (offline ``data_path`` tables, or the live tables
    once the batch bucket is seeded), else the fixed shape (``mode: static``, and the live seed sweep, which
    forces each captured verify width for a few rounds).
    """

    def __init__(
        self,
        steps: DraftSteps,
        cost_model: Optional[CostModel],
        *,
        fixed: FixedShapePolicy,
        build_trivial_input: Callable[["ScheduleBatch"], "EagleVerifyInput"],
        target_worker,
        algorithm,
        widths: Callable[[int], Sequence[int]],
        tree_mask_mode,
        device: torch.device,
        chain_width: Optional[Callable[[int], int]] = None,
        live: Optional[LiveCosts] = None,
        bucket: Callable[[int], int] = lambda bs: bs,
        sync_cells: Optional[Callable[[dict], dict]] = None,
        save_cells: Optional[Callable[[], None]] = None,
    ):
        """``chain_width(bs)``: the verify width of the next round of a chain at batch size ``bs`` (called once per
        round, when its first step is queued); None: the cost model sizes every round."""
        tree = steps.tree
        policies = [p for p in (cost_model, fixed) if p is not None]
        missing = set().union(*(set(p.launch_depths) for p in policies)) - tree.delta_slots
        need = max(p.max_history_verify_num for p in policies)
        if tree.max_verify < need or missing:
            raise ValueError(
                f"draft tree keeps work deltas for budgets 1..{tree.max_verify} at depths {sorted(tree.delta_slots)}; "
                f"the cost model reads budgets up to {need} at depths "
                f"{sorted(set().union(*(set(p.launch_depths) for p in policies)))}"
            )
        self.steps = steps
        self.tree: DraftTree = tree
        self.cost = cost_model
        self.fixed = fixed
        self.live = live
        self.timer = None if live is None else RoundTimer(live)
        self._bucket = bucket
        self._sync_cells = sync_cells
        self._save_cells = save_cells
        self._rounds = 0
        self._seeding: Optional[Tuple[int, int]] = None  # (bucket, context) of a seed round's draft steps
        self._interpolated = False  # the last round's bucket was priced from the buckets around it
        self._visits: Counter = Counter()  # rounds served per batch bucket
        self._unpriced: set = set()  # buckets whose live costs the cost model rejects (it keeps the fixed shape)
        self._trivial = build_trivial_input
        self._target_worker = target_worker
        # The process's EAGLE_HARNESS algorithm: its target verify graphs (H7) own the batch-size-1 tail mask.
        self._algorithm = algorithm
        self._widths = widths
        self._tree_mask_mode = tree_mask_mode
        self._device = device
        self._chain_width = chain_width
        self._deltas_ready = torch.cuda.Event()
        self._deltas_slot = 0
        self._verify_scores = _HostCopy(need + 1)
        self._have_lagged = False
        self._pending: Optional[Prelaunch] = None
        self._drafted_row: Optional[torch.Tensor] = None  # batch row of each request slot in a prelaunched draft
        self._bonus: Optional[torch.Tensor] = None
        self._slot_offsets = torch.arange(max(widths(1)), dtype=torch.int64, device=device)
        runner = target_worker.model_runner
        # The verify-input graph also builds what sglang's verify prep derives from the input on the host: the
        # target's mrope positions and, at batch size 1 over untranslated KV ids, the attention's CSR args.
        self._mrope = runner.model_config.model_is_mrope
        translator = runner.kv_index_translator
        self._req_to_token = None if translator.reads_are_translated else translator.req_to_token
        self._kv_indices = (
            None
            if self._req_to_token is None
            else torch.empty(self._req_to_token.shape[1], dtype=torch.int32, device=device)
        )
        self.tail = VerifyTail(tree, self._build_verify_input)
        self._prelaunch_stream: Optional[torch.cuda.Stream] = None
        self.last_depth = 0
        self.last_width = 0
        self._sampled_depth = tree.shape.max_depth

    # ----------------------------------------------------------------- policy
    def warm_up(self, batch_sizes: Sequence[int]) -> None:
        """Compile, before serving, the round tail's Triton kernels at every shape a round can take: the node
        ranking at each tree depth, the verify selection and the batch-size-1 tail-mask words at each verify width
        of ``batch_sizes``. Their shapes are compile-time constants, so a first use inside a served round compiles
        there (hundreds of milliseconds on a cold kernel cache) and stalls every request of the batch."""
        tree = self.tree
        if not tree.scores.is_cuda:
            return

        widths = sorted({w for bs in batch_sizes for w in self._widths(bs)})
        widest = max(widths)
        for depth in range(1, tree.shape.max_depth + 1):
            count = min(tree.shape.num_nodes(depth), widest - 1)
            ranked, _ = tree.rank(1, depth, count)
            for width in widths:
                if 1 < width <= count + 1:
                    tree.select(1, min(depth, width - 1), ranked, width)
        graphs = self._algorithm.target_verify_graphs
        if graphs is not None and graphs.tail_mask is not None:
            from eagle_harness.kernels.verify_prep import MAX_TAIL_WIDTH, tree_tail_words

            committed = torch.zeros(1, dtype=torch.int64, device=tree.scores.device)  # the staged seq_lens' dtype
            for width in self._widths(1):
                if 1 < width <= MAX_TAIL_WIDTH:
                    mask = torch.zeros(width * width, dtype=torch.bool, device=tree.scores.device)
                    tree_tail_words(mask, committed, torch.empty(width, dtype=torch.int64, device=mask.device), width)

    @property
    def seeding(self) -> bool:
        """Whether the last drafted round was forced to seed the live costs of its batch bucket."""
        return self._seeding is not None

    def bucket(self, bs: int) -> int:
        """The cost bucket (captured batch size) a batch of ``bs`` requests runs at."""
        return self._bucket(bs)

    def policy(self, bs: int):
        """This round's policy for ``bs`` requests: the cost model, or the fixed shape (static mode, or the live
        seed sweep of an unseeded bucket, whose forced width it sets)."""
        if self.cost is None:
            return self.fixed
        if self.live is None:
            return self.cost
        bucket = self._bucket(bs)
        if not self._seeds(bucket) and bucket in self.cost.tables and bucket not in self._unpriced:
            return self.cost
        self.fixed.force_width = self.live.seed_width(bucket)
        return self.fixed

    def _seeds(self, bucket: int) -> bool:
        """Whether a round at ``bucket`` belongs to its seed sweep: the bucket is not ready, and either no ready
        buckets around it price it (:meth:`LiveCosts.priced`) or the load has dwelt there long enough
        (:data:`LIVE_SEED_AFTER_ROUNDS`) to pay for measuring it."""
        return not self.live.ready(bucket) and (
            not self.live.priced(bucket) or self._visits[bucket] > LIVE_SEED_AFTER_ROUNDS
        )

    @property
    def round_timer(self) -> Optional[RoundTimer]:
        """The timer of this round's verify span; None for a round priced from neighbouring buckets (only ready
        buckets and seed sweeps record, so the ready buckets change on the same round on every TP rank)."""
        return None if self._interpolated else self.timer

    def _begin_round(self, batch: "ScheduleBatch") -> None:
        """Read the last rounds' timings; refresh (and share) the live tables on the refresh rounds."""
        if self.live is None:
            return
        bs = len(batch.reqs)
        bucket = self._bucket(bs)
        self._visits[bucket] += 1
        seeding = self._seeds(bucket)
        self.timer.drain(wait=seeding)
        self._rounds += 1
        ready = self.live.ready(bucket)
        interpolated = not ready and not self._seeds(bucket)
        # Table refreshes happen on the same rounds on every TP rank (round count and seed state agree), from
        # rank 0's cells, so the ranks' decisions agree.
        newly_priced = interpolated and bucket not in self.cost.tables
        if (ready and (seeding or self._rounds == 1)) or newly_priced or self._rounds % LIVE_REFRESH_ROUNDS == 0:
            if self._sync_cells is not None:
                self.live.cells = self._sync_cells(self.live.cells)
            # Every bucket served so far keeps its table, interpolated ones included (they have no cells).
            self.cost.set_tables(self.live.tables(also=tuple(self._visits)))
            for measured_bucket in self.cost.tables:
                self._check_priced(measured_bucket)
            if self._save_cells is not None and (seeding or self._rounds % (LIVE_REFRESH_ROUNDS * LIVE_SAVE_REFRESHES) == 0):
                self._save_cells()
            if seeding and ready:
                logger.info("eagle-harness: live costs seeded for batch bucket %d: %s", bucket, self._table_summary(bucket))
            elif newly_priced:
                logger.info(
                    "eagle-harness: live costs of batch bucket %d priced from the buckets around it: %s",
                    bucket,
                    self._table_summary(bucket),
                )
        self._interpolated = interpolated
        self._seeding = None if ready or interpolated else (bucket, self.average_context(batch))

    def _check_priced(self, bucket: int) -> None:
        """Keep the fixed shape at a bucket whose measured costs the cost model refuses (its ratio sanity checks)."""
        table = self.cost.tables.get(bucket)
        problem = "no table" if table is None else cost_model_problem(table, target_is_moe=self.cost.target_is_moe)
        if problem is None:
            self._unpriced.discard(bucket)
        elif bucket not in self._unpriced:
            self._unpriced.add(bucket)
            logger.warning("eagle-harness: batch bucket %d keeps the fixed draft shape: %s", bucket, problem)

    def _table_summary(self, bucket: int) -> str:
        table = self.cost.tables.get(bucket)
        if table is None:
            return "no table"
        target, draft = table.intervals[0].target, table.intervals[0].draft
        return (
            f"verify us at widths {[i + 1 for i in target.index]}: {[round(v) for v in target.value]}, "
            f"draft step {round(draft.value[0])} us, ratio {table.model_ratio[0]}"
        )

    def _sample_steps(self) -> bool:
        return self.live is not None and (
            self._seeding is not None or (self._rounds % LIVE_REFRESH_ROUNDS == 0 and not self._interpolated)
        )

    def _timed_steps(self, bs: int, run: Callable[[], int], context: int = 0) -> int:
        """Time seed steps and one queued draft prefix per refresh period, on its executing stream."""
        if not self._sample_steps():
            return run()
        start = self.timer.start()
        steps = run()
        if steps:
            self.timer.stop(start, self._bucket(bs), DRAFT_WIDTH, context, steps)
        return steps

    # --------------------------------------------------------------- prelaunch
    def prelaunch(
        self,
        batch: "ScheduleBatch",
        *,
        extend: Callable[[], None],
        inputs: Iterable[torch.Tensor],
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.cuda.Event:
        """Queue the next draft and return the event fencing its published roots and slot reads.

        ``extend`` writes the next round's roots and staged batch; ``inputs`` are its device inputs.
        Return the fence through ``PrelaunchResult`` so the scheduler's result copies can overlap
        this work, while its later root compaction and slot reuse remain ordered after all of it.
        Waiting only in ``_take_pending`` would be too late for those scheduler mutations.
        An overlap scheduler passes its forward stream (``stream``) so its draft-output relay, queued right after
        this returns, is ordered after the extend; the synchronous scheduler leaves it None (a side stream).
        """
        if not arch.side_stream_prelaunch(self._device):
            extend()
            return torch.cuda.current_stream().record_event()
        if stream is None:
            if self._prelaunch_stream is None:
                self._prelaunch_stream = torch.cuda.Stream()
            stream = self._prelaunch_stream
        caller_stream = torch.cuda.current_stream()
        stream.wait_stream(caller_stream)
        for tensor in (batch.req_pool_indices, *inputs):
            tensor.record_stream(stream)
        bs = len(batch.reqs)
        width = 0 if self._chain_width is None else self._chain_width(bs)
        with torch.cuda.stream(stream):
            extend()
            context = self.average_context(batch) if self._sample_steps() else 0
            depth = (
                self._queue_chain(bs, width) if width else self._queue_until_decision(bs, self.policy(bs), context)
            )
        self._pending = Prelaunch(batch_key(batch), depth, width, stream)
        return stream.record_event()

    def cancel_prelaunch(self) -> None:
        """Drop a prelaunched draft (the next round drafts from scratch)."""
        self._take_pending()

    def _take_pending(self) -> Optional[Prelaunch]:
        """The prelaunched draft, if any; the current stream then waits for its work."""
        pending, self._pending = self._pending, None
        if pending is not None:
            torch.cuda.current_stream().wait_stream(pending.stream)
        return pending

    def _keep_drafted(self, pending: Prelaunch, batch: "ScheduleBatch") -> bool:
        """Whether ``pending`` drafted ``batch``: the batch it was queued for, or that batch less the requests the
        scheduler dropped since (finished ones), whose rows the tree and the staged batch then keep. At large batch
        sizes some request finishes nearly every round, so without the second case each round drafted twice."""
        key = batch_key(batch)
        if pending.key == key:
            return True
        if not set(key) <= set(pending.key):
            return False
        drafted = len(pending.key)  # the staged batch is still the one the prelaunch drafted
        if self._drafted_row is None:
            self._drafted_row = torch.zeros(self.steps.req_to_token.shape[0], dtype=torch.int64, device=self._device)
        self._drafted_row[self.steps.req_pool_indices[:drafted]] = torch.arange(drafted, device=self._device)
        self.tree.keep_rows(self._drafted_row[batch.req_pool_indices])
        self.steps.load_batch(len(key), batch.seq_lens, batch.req_pool_indices)
        return True

    # ------------------------------------------------------------------ round
    def draft(self, batch: "ScheduleBatch") -> "EagleVerifyInput":
        """Grow this round's tree and return the verify input (a trivial one when no tree is drafted)."""
        pending = self._take_pending()
        if self._chain_width is not None:
            return self._draft_chain(batch, pending)
        draft_input = batch.spec_info
        bs = len(batch.seq_lens)
        tree = self.tree
        self._begin_round(batch)
        policy = self.policy(bs)
        context = self.average_context(batch)
        if pending is not None and self._keep_drafted(pending, batch):
            depth = pending.next_depth
        else:
            self.steps.load_batch(bs, batch.seq_lens, batch.req_pool_indices)
            self.steps.begin(bs, draft_input.topk_p, draft_input.topk_index, draft_input.hidden_states)
            depth = self._queue_until_decision(bs, policy, context)
        row = policy.draft_cost_row(bs, context)
        past = policy.history_verify_num(bs)
        depth = self._grow(bs, depth, row, past, policy)
        nodes = tree.shape.num_nodes(depth)
        widths = self._widths(bs)
        verify_count = policy.verify_count(bs, nodes + 1)
        count = min(nodes, max(widths) - 1)
        if self._have_lagged:
            # The width needs only the previous round's scores: rank, hand off and build in one graph.
            width = fit_width(
                self._verify_width(policy, self._verify_scores.read(), verify_count, bs, context), widths, nodes + 1
            )
            self.last_depth, self.last_width = depth, width
            if width == 1:
                self._verify_scores.launch(self.tail.rank(bs, depth, count).scores[:verify_count])
                return self._trivial(batch)
            self._stage_bonus(draft_input.bonus_tokens)
            verify_input = self.tail.ranked_verify_input(
                bs, depth, count, width, lambda ranking: self._verify_scores.copy(ranking.scores)
            )
            self._verify_scores.queued(min(verify_count, count + 1))
        else:
            ranking = self.tail.rank(bs, depth, count)
            scores = ranking.scores[:verify_count]
            self._verify_scores.launch(scores)
            self._have_lagged = True
            sampled = getattr(self.steps, "sampled", None)
            if sampled is not None and sampled.active:
                # No history: use the allowed budget, never inspect this round's draws to decide whether to verify them.
                width = fit_width(verify_count, widths, nodes + 1)
            else:
                width = fit_width(
                    self._verify_width(policy, scores.tolist(), verify_count, bs, context, lagged=False), widths, nodes + 1
                )
            self.last_depth, self.last_width = depth, width
            if width == 1:
                return self._trivial(batch)
            self._stage_bonus(draft_input.bonus_tokens)
            verify_input = self.tail.verify_input(bs, depth, width, ranking)
        return _round_input(verify_input, batch)

    def _draft_chain(self, batch: "ScheduleBatch", pending: Optional[Prelaunch]) -> "EagleVerifyInput":
        """A fixed-width chain round: ``width - 1`` drafted tokens (queued by the prelaunch when it drafted this
        batch), verified with the root; width 1 verifies the root alone."""
        bs = len(batch.seq_lens)
        if pending is not None and pending.key == batch_key(batch):
            width = pending.width
        else:
            width = self._chain_width(bs)
        if pending is None or pending.width != width or not self._keep_drafted(pending, batch):
            # Staged at every width: the draft extend after the verify reads the batch from the steps.
            self.steps.load_batch(bs, batch.seq_lens, batch.req_pool_indices)
            if width > 1:
                draft_input = batch.spec_info
                self.steps.begin(bs, draft_input.topk_p, draft_input.topk_index, draft_input.hidden_states)
                self._queue_chain(bs, width)
        self.last_depth, self.last_width = width - 1, width
        if width == 1:
            return self._trivial(batch)
        self._stage_bonus(batch.spec_info.bonus_tokens)
        steps = width - 1
        return _round_input(self.tail.ranked_verify_input(bs, steps, steps, width, lambda ranking: None), batch)

    def _queue_chain(self, bs: int, width: int) -> int:
        """Queue the draft steps of a fixed-width chain after its begin; returns the chain's depth.

        The begin drafts depth 1 and step ``d`` drafts depth ``d + 1``, so ``width - 1`` drafted tokens take steps
        ``1 .. width - 2`` (the widest chain, ``--speculative-num-draft-tokens``, ends at the last step).
        """
        for step in range(1, width - 1):
            self.steps.run(bs, step)
        return width - 1

    def _stage_bonus(self, bonus_tokens: torch.Tensor) -> None:
        """Copy the round's bonus tokens where the verify-input graphs read them."""
        if self._bonus is None:
            self._bonus = torch.zeros(self.tree.max_bs, dtype=bonus_tokens.dtype, device=self._device)
        self._bonus[: bonus_tokens.shape[0]].copy_(bonus_tokens)

    def _build_verify_input(self, bs: int, selection: VerifySelection, steps: int, width: int) -> HarnessVerifyInput:
        """The verify input of a selection, from the static ``seq_lens`` and bonus tokens (graph-captured):
        sglang's, plus the verify tokens' KV slots and, when the target runs the batch-size-1 tail-mask
        verify, its mask words."""
        seq_lens, req_pool_indices = self.steps.seq_lens[:bs], self.steps.req_pool_indices[:bs]
        verify_input = self._build_eagle_verify_input(bs, selection, steps, width)
        slots = self.steps.req_to_token[req_pool_indices[:, None], seq_lens[:, None] + self._slot_offsets[:width]]
        verify_input.prepared_out_cache_loc = slots.reshape(-1).to(torch.int64)
        graphs = self._algorithm.target_verify_graphs
        tail_mask = None if graphs is None else graphs.tail_mask
        if tail_mask is not None and bs == 1:
            tail_mask.write(verify_input.custom_mask, seq_lens[:1], width)
        if self._mrope:
            # sglang's text-only spec mrope positions (ForwardBatch.compute_spec_mrope_positions).
            verify_input.prepared_mrope_positions = (
                verify_input.positions.to(torch.int64).flatten().unsqueeze(0).repeat(3, 1)
            )
        harness_input = HarnessVerifyInput.from_eagle(verify_input, proposal_nodes=selection.top_scores_index)
        if bs == 1 and self._kv_indices is not None:
            harness_input.prepared_attn_args = build_attn_args(
                self._req_to_token, req_pool_indices, seq_lens, width, self._kv_indices
            )
        return harness_input

    def _build_eagle_verify_input(
        self, bs: int, selection: VerifySelection, steps: int, width: int
    ) -> "EagleVerifyInput":
        """sglang's verify input of a selection."""
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, seq_lens=self.steps.seq_lens[:bs], seq_lens_sum=None)
        draft_input = SimpleNamespace(bonus_tokens=self._bonus[:bs])
        return build_eagle_verify_input(
            batch,
            draft_input,
            selection.parent_list,
            selection.top_scores_index,
            selection.draft_tokens,
            None,
            target_worker=self._target_worker,
            topk=self.tree.shape.top_k,
            num_steps=steps,
            num_draft_tokens=width,
            tree_mask_mode=self._tree_mask_mode,
            device=self._device,
        )

    # ------------------------------------------------------------------ depth
    def _queue_until_decision(self, bs: int, policy, context: int = 0) -> int:
        """Run the draft steps whose depths need no host decision; returns the first depth that does."""

        def run() -> int:
            depth = 0
            limit = self.steps.max_step
            if getattr(self.steps, "sampled", None) is not None and self.steps.sampled.active:
                limit = self._sampled_depth - 1
            while depth < limit:
                kind = policy.width_step(depth)
                if kind is WidthStep.LAUNCH:
                    self._launch_deltas(depth)
                elif kind is not WidthStep.KEEP:
                    break
                depth += 1
                self.steps.run(bs, depth)
            return depth

        return self._timed_steps(bs, run, context)

    def _grow(self, bs: int, depth: int, row: Optional[DraftCostRow], past: int, policy) -> int:
        """Decide at ``depth`` (its step already queued), then step and decide until the policy
        stops; returns the tree's depth levels."""
        if getattr(self.steps, "sampled", None) is not None and self.steps.sampled.active:
            planned = self._sampled_depth
            next_depth = None
            while True:
                stop = self._decide(depth, bs, row, past, policy)
                if stop and next_depth is None:
                    next_depth = depth + 1
                if depth >= planned - 1:
                    break
                depth += 1
                self.steps.run(bs, depth)
            self._sampled_depth = next_depth or min(planned + 2, self.tree.shape.max_depth)
            return planned
        while not self._decide(depth, bs, row, past, policy):
            if depth == self.steps.max_step:
                return self.tree.shape.max_depth
            depth += 1
            self.steps.run(bs, depth)
        return depth + 1

    def _launch_deltas(self, depth: int) -> None:
        """The work deltas of ``depth`` are on their way to the host (its graph copies them)."""
        self._deltas_ready.record()
        self._deltas_slot = depth

    def _decide(self, depth: int, bs: int, row: Optional[DraftCostRow], past: int, policy) -> bool:
        """Decide whether to stop growing after the layer of ``depth`` (stop = True); chain rounds decide with
        :meth:`~eagle_harness.core.cost_model.CostModel.chain_stop`.

        The work deltas of a LAUNCH depth are already on their way to the host
        (the step's graph copies them); the next depth reads them.
        """
        kind = policy.width_step(depth)
        if kind is WidthStep.KEEP:
            return False
        if kind is WidthStep.LAST:
            return True
        if kind is WidthStep.LAUNCH:
            self._launch_deltas(depth)
            return False
        spin_wait(self._deltas_ready)
        # Row budget - 1: deltas for the policy's verify budget (sized by the cost model's bound, not the verify width).
        budget = policy.deltas_budget(past)
        deltas = self.steps.host_deltas.numpy()[self._deltas_slot, budget - 1].tolist()
        return policy.tree_width(depth, bs, past, row, deltas).stop

    # ------------------------------------------------------------------ width
    def _verify_width(
        self, policy, scores: List[float], verify_count: int, bs: int, context: int, *, lagged: bool = True
    ) -> int:
        """The verify width from the policy, fed cumulative verify scores: the previous round's
        (``lagged``), or this round's on the first round. A chain verifies every drafted token."""
        if policy.verifies_whole_draft:
            return verify_count
        width, _ = policy.verify_num_paged(scores, verify_count, bs, context, lagged=lagged)
        policy.record_verify_num(bs, width)
        return min(width, verify_count)

    @staticmethod
    def average_context(batch: "ScheduleBatch") -> int:
        """Mean committed length of the batch (host side, no sync)."""
        seq_lens_cpu: Optional[torch.Tensor] = getattr(batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None:
            return sum(seq_lens_cpu.tolist()) // len(seq_lens_cpu)
        if batch.seq_lens_sum is not None:
            return int(batch.seq_lens_sum) // len(batch.seq_lens)
        # The overlap scheduler drops both host mirrors; the requests still hold their lengths (lagging one round).
        return sum(req.seqlen for req in batch.reqs) // len(batch.reqs)
