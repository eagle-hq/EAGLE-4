# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Device-decided tree rounds on TokenSpeed's draft lanes.

The decode graph owns target verification and draft extend. Each following
frontier has its own graph; the GPU tail-launches the next depth or finalize.
The host joins the pending round before any next-forward state mutation.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ

import torch
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.eagle import (
    AcceptedPrefixPublisher,
    Eagle,
    EagleDraftInput,
)
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends.base import (
    SamplingBackend,
    SpeculativeSamplingPools,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils.common import maybe_inference_mode
from typing_extensions import override

from eagle_harness.tokenspeed.tree_round.chain import DeviceChain
from eagle_harness.tokenspeed.tree_round.device_policy import DevicePolicy
from eagle_harness.tokenspeed.tree_round.ops.tree import (
    eagle_harness_draft_depth_update,
    eagle_harness_publish_draft_q,
    eagle_harness_sample_children,
    eagle_harness_stable_topk_desc,
    eagle_harness_update_node_data,
    eagle_harness_work_deltas,
    eagle_harness_write_verify_tree,
)

_ROOT_SCORE = 1.001


@dataclasses.dataclass
class _PendingRound:
    ctx: ForwardContext
    stream: torch.cuda.Stream
    bs: int
    rows: int
    graphs: bool
    seq: int
    sampled: bool = False
    chained: bool = False
    depth: int = 1


class DynamicDrafter(Eagle):
    """Deferred device-policy rounds; static mode retains Eagle's whole graph."""

    supports_speculative_sampling = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = None
        self._pool = {}
        self._round = {}
        self._pending = None
        self._chain = None
        self._launched_rows = None
        self._round_seq = 0
        self._depth_graphs = {}
        self._finalize_graphs = {}
        self._live_rows = 0
        self._live_sampled = False
        self._root_children = None
        self._depth_done = []
        self._decide_stream = None
        self._round_stream = None
        self._target_done = None
        self._round_done = None
        self._sampling_backend: SamplingBackend | None = None
        self._pools: SpeculativeSamplingPools | None = None
        self._sampled: dict[str, torch.Tensor] = {}
        self._sampled_q: DraftTreeQ | None = None
        self._sampled_seed: int = 0
        self._sampled_num_candidates: int = 0
        self._round_sampled = False

    @property
    def dynamic(self) -> bool:
        """Adapters opt into deferred rounds after supplying their policy."""
        return False

    @property
    def sampled_trees(self) -> bool:
        """Whether this drafter records q for sampled tree proposals."""
        return self._sampled_q is not None

    @override
    def bind_sampling_backend(self, sampling_backend: SamplingBackend) -> None:
        self._sampling_backend = sampling_backend
        if sampling_backend.supports_tree_sampler:
            from eagle_harness.tokenspeed.tree_rejection import TreeRejectionSampler

            sampling_backend.bind_tree_sampler(TreeRejectionSampler)
        if not self.dynamic:
            super().bind_sampling_backend(sampling_backend)

    def enable_sampled_trees(self, *, num_candidates: int, seed: int) -> None:
        """Enable reference sampled proposals after binding the dynamic tree.

        Call before graph capture, after ``bind_sampling_backend``. The draft
        head must provide fp32 ``next_token_logits`` and int64 ``candidate_ids``
        of width ``num_candidates`` alongside the deterministic top-K scores.
        ``seed`` selects the proposal Philox stream; the verifier owns its coins.
        """
        if not self.dynamic or self.tree_spec is None or self.policy is None:
            raise ValueError("sampled trees require a bound dynamic tree")
        if self._depth_graphs or self._pending is not None or self.sampled_trees:
            raise RuntimeError("enable sampled trees once, before capture or serving")
        from eagle_harness.tokenspeed.hooks import sampled_hook_problems

        problems = sampled_hook_problems()
        if problems:
            raise ValueError("sampled trees require engine hooks: " + "; ".join(problems))
        backend = self._sampling_backend
        if backend is None or not backend.verifies_draft_tree_q:
            raise ValueError("sampled trees require a verifier of recorded tree q")
        config = self.tree_spec.config
        if not config.topk <= num_candidates <= 1024:
            raise ValueError(
                f"sampled trees need {config.topk} <= num_candidates <= 1024, "
                f"got {num_candidates}"
            )
        self._pools = backend.speculative_sampling_pools()
        self._sampled_seed = seed
        self._sampled_num_candidates = num_candidates
        self._allocate_sampled(
            self.input_buffers.max_bs,
            config.topk,
            config.num_steps,
            config.num_nodes,
            self._pool["tokens"].shape[1],
            num_candidates,
        )

    def prepare_round_sampling(
        self, sampling_params_list: list[SamplingParams]
    ) -> None:
        """Prepare the live request mix before eager forward or graph replay.

        The caller must first complete any pending round. Sampling requests use
        the policy's depth planned from earlier rounds, before this round draws.
        """
        live = [] if self.input_buffers.all_extends_mid_chunk else sampling_params_list
        self._round_sampled = self.sampled_trees and any(
            params.top_k != 1 for params in live
        )
        if self.sampled_trees:
            self._open_round(len(live))

    def bind_dynamic_tree(self, tree_spec, policy: DevicePolicy) -> None:
        """Bind tree lanes and an already constructed device policy before capture."""
        config = tree_spec.config
        allowed_topk = (1,) if policy.chain else (2, 4, 8)
        max_bs = 512 if policy.chain else 32
        if config.topk not in allowed_topk or config.num_steps < 2:
            raise ValueError(
                f"device-policy rounds need top-k {allowed_topk} and at least two steps"
            )
        if self.input_buffers.max_bs > max_bs:
            raise ValueError(f"device-policy rounds support at most {max_bs} requests")
        self.bind_tree_lanes(tree_spec)
        logits_processor = self.draft_model_runner.model.logits_processor
        logits_processor.do_argmax = False
        self.tree_vocab_size = logits_processor.config.vocab_size
        self.policy = policy
        self._allocate(
            self.input_buffers.max_bs,
            config.topk,
            config.num_steps,
            max(1024, config.topk * config.topk * config.num_steps),
            max(64, config.topk * (config.num_steps + 1) + 1),
        )

    def round_candidates(self, logits_output):
        """Rank native EAGLE-3 logits and map reduced-vocabulary ids once."""
        from tokenspeed_kernel.ops.sampling.triton import logprob_topk

        log_probs, tokens = logprob_topk(
            logits_output.next_token_logits, self.lanes_per_request
        )
        if self.sampled_trees:
            logits, ids = torch.topk(
                logits_output.next_token_logits, self._sampled_num_candidates, dim=-1
            )
            logits_output.next_token_logits = logits.float()
            logits_output.candidate_ids = self._map_hot(ids)
        return log_probs.exp(), self._map_hot(tokens)

    def _allocate_sampled(
        self,
        max_bs: int,
        topk: int,
        depth: int,
        num_nodes: int,
        pool_nodes: int,
        num_candidates: int,
    ) -> None:
        """Sampled trees' round buffers and the next round's q tables: one q
        slot for the root and one per lane row of every later depth."""
        from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ

        device = self.device
        num_slots = 1 + (depth - 1) * topk
        self._sampled_q = DraftTreeQ.zeros(
            max_bs, num_slots, num_candidates, pool_nodes, device
        )
        self._sampled = {
            "round": torch.zeros((1,), dtype=torch.int64, device=device),
            "temperature": torch.zeros((max_bs,), dtype=torch.float32, device=device),
            "top_k": torch.ones((max_bs,), dtype=torch.int32, device=device),
            "top_p": torch.ones((max_bs,), dtype=torch.float32, device=device),
        }
        self.runtime_states.init_draft_tree_q(num_nodes, num_slots, num_candidates)

    def _sample_children(
        self,
        logits_output: LogitsProcessorOutput,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        *,
        width: torch.Tensor | None,
        slot_base: int,
        cumu_start: torch.Tensor,
        launch: int,
    ) -> None:
        """Redraw the sampling requests' children of their expanded rows
        (``tokens`` / ``scores``: ``[bs, rows, K]``, each request's first
        ``rows`` forwarded rows, of which it expands the first ``width``; None:
        all) from their draft q, in place; greedy requests keep their top-K."""
        sampled = self._sampled
        bs, rows = tokens.shape[:2]
        if logits_output.candidate_ids is None:
            raise ValueError("sampled trees require draft candidate_ids")
        logits, candidates = (
            output.view(bs, -1, output.shape[-1])[:, :rows]
            for output in (logits_output.next_token_logits, logits_output.candidate_ids)
        )
        eagle_harness_sample_children(
            logits,
            candidates,
            tokens,
            scores,
            sampled["temperature"],
            sampled["top_k"],
            sampled["top_p"],
            self._sampled_q.q,
            self._sampled_q.q_ids,
            self._sampled_q.slot,
            self._sampled_q.pos,
            width,
            slot_base,
            cumu_start,
            self.tree_vocab_size,
            self._sampled_seed,
            sampled["round"],
            launch,
        )

    def _load_round_sampling(self, bs: int) -> None:
        """The round's per-request sampling scalars (greedy requests at
        temperature 0) and a fresh draw counter; tensor-only, in the graph."""
        pools, sampled = self._pools, self._sampled
        pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        top_k = sampled["top_k"][:bs]
        torch.index_select(pools.top_k, 0, pool_indices, out=top_k)
        torch.index_select(pools.top_p, 0, pool_indices, out=sampled["top_p"][:bs])
        temperature = sampled["temperature"][:bs]
        torch.index_select(pools.temperature, 0, pool_indices, out=temperature)
        temperature.masked_fill_(top_k == 1, 0.0)
        sampled["round"].add_(1)

    def round_depth(self, bs):
        # A padded graph bucket can replay any smaller live batch, including
        # batches whose schedule allows a longer chain.
        widths = [self.policy.table_width(live) for live in range(1, bs + 1)]
        return max(1, max(widths) - 2) if all(widths) else self.spec_num_steps - 1

    def _allocate(
        self, max_bs: int, topk: int, depth: int, nodes: int, verify: int
    ) -> None:
        """The node pool and lists (``nodes`` nodes, work lists for ``verify``),
        initialized like the reference's, and the round buffers ``draft`` and
        ``finish_round`` share."""
        device = self.device
        scores = torch.ones((max_bs, nodes), dtype=torch.float32, device=device)
        scores[:, 0] = _ROOT_SCORE
        depths = torch.zeros((max_bs, nodes), dtype=torch.int64, device=device)
        depths[:, 1 : topk + 1] = 1
        prev = torch.full((max_bs, nodes), -1, dtype=torch.int64, device=device)
        prev[:, 1 : topk + 1] = 0
        nxt = torch.full((max_bs, nodes), -1, dtype=torch.int64, device=device)
        nxt[:, 0] = 1
        side = torch.full((max_bs, nodes), -1, dtype=torch.int64, device=device)
        side[:, 1:topk] = torch.arange(2, topk + 1, device=device)
        work = torch.zeros((max_bs, verify + topk), dtype=torch.float32, device=device)
        work[:, 0] = _ROOT_SCORE
        sort = torch.zeros((max_bs, verify), dtype=torch.float32, device=device)
        sort[:, 0] = _ROOT_SCORE
        self._pool = {
            "tokens": torch.zeros((max_bs, nodes), dtype=torch.int64, device=device),
            "scores": scores,
            "depths": depths,
            "prev": prev,
            "next": nxt,
            "side": side,
            "work": work,
            "sort": sort,
        }

        hidden_size = self.draft_model_runner.model.config.hidden_size
        dtype = self.draft_model_runner.model_config.dtype
        int64 = {"dtype": torch.int64, "device": device}
        float32 = {"dtype": torch.float32, "device": device}
        self._round = {
            # The frontier: its rows' tokens, path scores and input hidden rows.
            "tokens": torch.zeros((max_bs, topk), **int64),
            "scores": torch.zeros((max_bs, topk), **float32),
            "hidden": torch.zeros(
                (max_bs, topk, hidden_size), dtype=dtype, device=device
            ),
            # The accepted frontier (lane slot 0) and the lanes' positions and slots.
            "frontier": torch.zeros((max_bs,), **int64),
            "positions": torch.zeros((max_bs * topk,), **int64),
            "slot_starts": torch.zeros((max_bs,), **int64),
            "root_tokens": torch.zeros((max_bs,), dtype=torch.int32, device=device),
            # Frontier pool indices, ping-ponged between consecutive depths.
            "frontier_index": torch.zeros((2, max_bs, topk), **int64),
            # The round's requests' rows of the next round's inputs.
            "rows": torch.zeros((max_bs,), **int64),
            # Live requests, which a graph's batch statistics read past its padding.
            "live_rows": torch.zeros((1,), dtype=torch.int32, device=device),
            "sampled": torch.zeros((1,), dtype=torch.bool, device=device),
            # The root's first child's pool index, where depth 0's children start.
            "first_child": torch.ones((1,), dtype=torch.int32, device=device),
            # By depth parity: a depth's decision reads the previous depth's deltas.
            "work_deltas": torch.zeros((2, topk + 1), **float32),
            # The ranked pool: each request's best nodes and their scores.
            "top_scores": torch.zeros(
                (max_bs, self.policy.verify_num_limit), **float32
            ),
            "top_index": torch.zeros((max_bs, self.policy.verify_num_limit), **int64),
        }
        self._live_rows = 0
        self._live_sampled = False
        # Depth 1's frontier: the root's children, pool nodes 1..K.
        self._root_children = (torch.arange(topk, device=device) + 1).repeat(max_bs, 1)
        # Depth 1's lane masks: lane r sees its own slot r (bit r of word 0).
        seed = torch.zeros((topk, 1), **int64)
        seed[:, :1] = (1 << torch.arange(topk, device=device))[:, None]
        self._tree_seed_lane_mask = seed
        self._depth_done = [torch.cuda.Event() for _ in range(depth)]
        self._decide_stream = torch.cuda.Stream(device=device)
        self._round_stream = torch.cuda.Stream(device=device)
        self._target_done = torch.cuda.Event()
        self._round_done = torch.cuda.Event()
        if isinstance(self.policy, DevicePolicy):
            self._chain = DeviceChain(max_bs, depth - 1, device)

    @override
    def draft(self, draft_input: EagleDraftInput) -> torch.Tensor:
        """The round's first step, the root's children and depth 1's lanes; the
        next round's inputs are the root alone until ``finish_round`` writes
        the tree."""
        if not self.dynamic:
            return super().draft(draft_input)
        bs = draft_input.accept_lengths.shape[0]
        k = self.lanes_per_request
        pool, state = self._pool, self._round

        next_tokens = self._last_verified_window(bs, draft_input)
        self.tree_spec.draft_parent_buf[:bs].copy_(self.tree_spec.chain_parent)

        frontier = self._accepted_frontier(bs, draft_input)
        narrowing = AcceptedPrefixPublisher(self.attn_backend, frontier)
        logits_output, _ = self._run_first_step(bs, draft_input, narrowing)

        # Depth 1: the root's K best children (head softmax, best first).
        top_p, top_tokens = self.round_candidates(logits_output)
        # Policy frontiers use int64; input-buffer cache lengths use int32.
        state["frontier"][:bs].copy_(frontier)
        if self.sampled_trees:
            # The sampled plan must precede the current round's random proposals.
            self.policy.begin_round(
                state["live_rows"], state["frontier"][:bs], state["sampled"]
            )
            self._load_round_sampling(bs)
            self._sample_children(
                logits_output,
                top_tokens[:, None],
                top_p[:, None],
                width=None,
                slot_base=0,
                cumu_start=state["first_child"],
                launch=0,
            )
        pool["tokens"][:bs, 0] = next_tokens[:, 0]
        eagle_harness_update_node_data(
            pool["scores"][:bs],
            pool["tokens"][:bs],
            pool["work"][:bs],
            pool["sort"][:bs],
            top_p,
            top_tokens,
        )
        state["root_tokens"][:bs].copy_(next_tokens[:, 0])
        state["tokens"][:bs].copy_(top_tokens)
        state["scores"][:bs].copy_(top_p)
        state["hidden"][:bs].copy_(
            logits_output.hidden_states[:, None, :].expand(-1, k, -1)
        )
        # The next forward refills the input buffers before the round completes.
        state["rows"][:bs].copy_(
            self.input_buffers.state_write_req_pool_indices_buf[:bs]
        )
        self._start_lanes(bs, bs)
        return next_tokens

    @override
    def replay_decode_graph(
        self, graph: torch.cuda.CUDAGraph, *, bs: int, graph_bs: int
    ) -> None:
        """With device-chained rounds, replay the decode graph on the round
        stream and launch its round's depth 1 behind it: the forward's stream
        waits for the decode graph alone, so its outputs and the next
        forward's staging run beside the round."""
        if self._chain is None or bs == 0:
            graph.replay()
            return
        if self._launched_rows is not None:
            raise RuntimeError(
                "EAGLE-Harness decode graph replayed before finish_round booked the round the last replay launched"
            )
        self._check_round_graphs(bs, graph_bs)
        self._open_round(bs)
        stream, round_stream = torch.cuda.current_stream(), self._round_stream
        round_stream.wait_stream(stream)
        with torch.cuda.stream(round_stream):
            graph.replay()
            self._target_done.record()
            self._launch_chain(graph_bs)
        stream.wait_event(self._target_done)
        self._launched_rows = graph_bs

    @maybe_inference_mode()
    def finish_round(self, ctx, graph_bs):
        if not self.dynamic:
            return
        bs = ctx.bs
        launched, self._launched_rows = self._launched_rows, None
        if bs == 0 or self.input_buffers.all_extends_mid_chunk:
            return
        chained = self._chain is not None and graph_bs is not None
        if chained and launched is None:
            self._check_round_graphs(bs, graph_bs)
        self._round_seq += 1
        self._pending = pending = _PendingRound(
            ctx=ctx,
            stream=torch.cuda.current_stream(),
            bs=bs,
            rows=bs if graph_bs is None else graph_bs,
            graphs=graph_bs is not None,
            seq=self._round_seq,
            sampled=self._round_sampled,
            chained=chained,
        )
        if launched is not None:
            if launched != pending.rows:
                raise RuntimeError("decode and draft graph batch sizes differ")
            return
        self._open_round(bs)
        if ctx.num_extends > 0 and graph_bs is not None and graph_bs > bs:
            self._pad_round_rows(bs, graph_bs)
        self._round_stream.wait_stream(pending.stream)
        with torch.cuda.stream(self._round_stream):
            if chained:
                self._launch_chain(pending.rows)
            else:
                self._launch_depth(pending, 1)
                self._advance(pending, pause_before_wait=True)

    def _check_round_graphs(self, bs: int, rows: int) -> None:
        """Refuse a device-chained round of ``bs`` requests on padded batch
        ``rows`` that could reach a depth the batch did not capture."""
        depth = self.round_depth(bs)
        if (rows, depth) not in self._depth_graphs:
            raise RuntimeError(
                f"EAGLE-Harness round of {bs} requests can draft depth {depth}, "
                f"which its padded batch {rows} did not capture (round_depth)"
            )

    def _launch_chain(self, rows: int) -> None:
        """Launch a device-chained round's depth 1 on the current stream,
        which orders its later work after the round."""
        self._depth_graphs[(rows, 1)].replay()
        self._round_done.record()

    def _open_round(self, bs: int) -> None:
        """Before the round's depth 1, on the forward's stream: its live
        requests and sampled flag (written on change), and the policy's batch
        size."""
        state = self._round
        if bs != self._live_rows:
            state["live_rows"].fill_(bs)
            self._live_rows = bs
        if self._round_sampled != self._live_sampled:
            state["sampled"].fill_(self._round_sampled)
            self._live_sampled = self._round_sampled
        self.policy.observe_batch_size(bs)

    def _pad_round_rows(self, bs: int, rows: int) -> None:
        """Rows ``bs .. rows`` of an extend's round that replays graphs at
        ``rows``: no children, one cached key and the null page, so their
        lanes read and write only the dummy page, as a padded decode graph's
        padding rows do."""
        pool, state = self._pool, self._round
        pad = slice(bs, rows)
        if self.sampled_trees:
            self._sampled["temperature"][pad].zero_()
            self._sampled["top_k"][pad].fill_(1)
            self._sampled["top_p"][pad].fill_(1.0)
        state["tokens"][pad].zero_()
        state["scores"][pad].zero_()
        state["hidden"][pad].zero_()
        state["root_tokens"][pad].zero_()
        pool["tokens"][pad, 0].zero_()
        eagle_harness_update_node_data(
            pool["scores"][pad],
            pool["tokens"][pad],
            pool["work"][pad],
            pool["sort"][pad],
            state["scores"][pad],
            state["tokens"][pad],
        )
        state["frontier"][pad].fill_(1)
        state["rows"][pad].fill_(self.input_buffers.state_write_padding_pool_index)
        self._start_lanes(rows, bs)

    @override
    def has_round_graphs(self, graph_bs: int) -> bool:
        # Exec mode 2 drafts extends eagerly: it measures only decode rounds.
        return self.dynamic and (graph_bs, False) in self._finalize_graphs

    @override
    @maybe_inference_mode()
    def complete_round(self) -> None:
        """Draft the rest of the pending round, then rank its pool and write
        its verify tree into the next round's inputs (a device-chained round
        does all this on the GPU), and order the round's forward stream after
        it.

        The rest runs on the round stream, so the outputs and the next
        forward's inputs and attention metadata queued on the forward's stream
        since ``finish_round`` run beside the depths instead of between them.
        """
        pending = self._pending
        if pending is None:
            return
        self._pending = None
        if not pending.chained:
            with torch.cuda.stream(self._round_stream):
                self._advance(pending, pause_before_wait=False)
                self._finalize(pending)
        pending.stream.wait_stream(self._round_stream)
        # Non-forward device tasks can enter on another stream before reusing a slot.
        torch.cuda.current_stream().wait_stream(self._round_stream)

    def _advance(self, pending: _PendingRound, *, pause_before_wait: bool) -> None:
        """Launch each next depth the policy decides to draft, until the round
        stops (or, with ``pause_before_wait``, until a decision has not landed
        yet). A depth's decision runs at its start beside its forward, so the
        host learns whether to launch the next depth while the GPU drafts it;
        a width-table round's depths are known up front and launch at once."""
        policy = self.policy
        while True:
            depth = pending.depth
            go = policy.go_ready(depth + 1, pending.seq)
            if go is None:
                if pause_before_wait:
                    return
                go = self._wait_decision(pending, depth)
            if not go:
                return
            pending.depth = depth + 1
            self._launch_depth(pending, depth + 1)

    def _wait_decision(self, pending: _PendingRound, depth: int) -> bool:
        """Whether depth ``depth + 1`` drafts: once ``depth`` has started, its
        decision lands within microseconds, so poll the go flag until then."""
        self._depth_done[depth - 1].synchronize()
        done = self._depth_done[depth]
        while True:
            # Queried first: a finished depth has published its decision.
            finished = done.query()
            go = self.policy.go_ready(depth + 1, pending.seq)
            if go is not None:
                return go
            if finished:
                raise RuntimeError(
                    f"EAGLE-Harness round {pending.seq} has no decision after depth {depth}"
                )

    def _launch_depth(self, pending: _PendingRound, depth: int) -> None:
        """Draft ``depth`` and its decision (its captured graph, or eagerly),
        and mark where it ends."""
        if depth == 1:
            self._depth_done[0].record()
        if pending.graphs:
            graph = self._depth_graphs.get((pending.rows, depth))
            if graph is None:
                raise RuntimeError(
                    f"EAGLE-Harness round of {pending.bs} requests drafts depth "
                    f"{depth}, which its padded batch {pending.rows} did not "
                    "capture (round_depth)"
                )
            graph.replay()
        else:
            ctx = pending.ctx
            lanes = self._lane_context(
                pending.rows,
                ctx.global_num_tokens,
                ctx.global_bs,
                ctx.all_decode_or_idle,
            )
            with self.attn_backend.override_num_extends(0):
                self._draft_depth(pending.rows, lanes, depth, decide=True, chain=False)
        self._depth_done[depth].record()

    def _start_lanes(self, rows: int, live_rows: int) -> None:
        """Depth 1's lane positions and slots (the accepted frontier) and lane
        masks; rows past ``live_rows`` are padding (null page)."""
        state, lanes = self._round, self.tree_lanes
        self.attn_backend.begin_tree_lanes(rows, live_rows)
        frontier = state["frontier"][:rows]
        lanes.set_frontier(rows, frontier)
        k, words = self.lanes_per_request, 1
        state["positions"][: rows * k].view(rows, k).copy_(
            frontier[:, None].expand(-1, k)
        )
        state["slot_starts"][:rows].copy_(frontier)
        lanes.lane_mask[: rows * k * words].view(rows, k, words).copy_(
            self._tree_seed_lane_mask
        )

    def _queue_work_deltas(self, rows: int, depth: int) -> None:
        """Depth ``depth``'s work deltas over the live rows among ``rows``, at
        the depths whose slot can ask for them (its work size; 0: none)."""
        policy = self.policy
        if not policy.deltas_depth(depth):
            return
        eagle_harness_work_deltas(
            self._pool["work"][:rows],
            self._round["work_deltas"][depth % 2],
            self._round["live_rows"],
            policy.work_sizes[depth : depth + 1],
            self.lanes_per_request,
            policy.past_verify_num,
        )

    @override
    @maybe_inference_mode()
    def capture_round_graphs(
        self, graph_bs: int, *, lo_bs: int, stream: torch.cuda.Stream, pool
    ) -> None:
        """Capture each draft depth at the decode graph's padded batch: its
        widths, pool sizes and work list sizes are device slots, so one graph
        per depth serves every round. Only the depths the rounds of ``lo_bs +
        1`` to ``graph_bs`` live requests (those the batch pads) can reach.
        A device chain's depths launch the round's next graph after their
        decision; depths 2 and on and the finalize are instantiated for device
        launch."""
        if not self.dynamic:
            return
        depths = max(self.round_depth(bs) for bs in range(lo_bs + 1, graph_bs + 1))
        lanes = self._lane_context(graph_bs, None, None, True)
        chain = self._chain
        with torch.cuda.stream(stream), self.attn_backend.override_num_extends(0):
            # Warm up on a full round, without the decisions whose state the rounds own.
            self.policy.write_full_round()
            self._start_lanes(graph_bs, graph_bs)
            for depth in range(1, depths + 1):
                self._draft_depth(graph_bs, lanes, depth, decide=False, chain=False)
                torch.cuda.synchronize()
                # The host launches depth 1 (an ordinary graph); the GPU, the rest.
                graph = torch.cuda.CUDAGraph(keep_graph=chain is not None and depth > 1)
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._draft_depth(
                        graph_bs, lanes, depth, decide=True, chain=chain is not None
                    )
                self._depth_graphs[(graph_bs, depth)] = graph
            samples = (False, True) if self.sampled_trees else (False,)
            for sampled in samples:
                graph = torch.cuda.CUDAGraph(keep_graph=chain is not None)
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._write_verify_tree(graph_bs, sampled)
                self._finalize_graphs[(graph_bs, sampled)] = graph
            if chain is not None:
                chain.instantiate(
                    graph_bs,
                    [self._depth_graphs[(graph_bs, d)] for d in range(2, depths + 1)],
                    [self._finalize_graphs[(graph_bs, sampled)] for sampled in samples],
                    stream,
                )
            if self.sampled_trees:
                self._sampled["round"].zero_()

    @override
    def release_round_graphs(self) -> None:
        if self._chain is not None:
            self._chain.release()
        self._depth_graphs.clear()
        self._finalize_graphs.clear()

    def _lane_context(
        self,
        bs: int,
        global_num_tokens: list[int] | None,
        global_bs: list[int] | None,
        all_decode_or_idle: bool,
    ) -> ForwardContext:
        """The forward context of a depth's ``K`` lanes per request."""
        k = self.lanes_per_request
        return ForwardContext(
            bs=bs,
            num_extends=0,
            output_layout=ForwardOutputLayout(0, 0, bs, k),
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            input_num_tokens=bs * k,
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            global_num_tokens=global_num_tokens,
            global_bs=global_bs,
            all_decode_or_idle=all_decode_or_idle,
        )

    def _draft_depth(
        self, bs: int, lanes: ForwardContext, depth: int, *, decide: bool, chain: bool
    ) -> None:
        """Forward depth ``depth``'s frontier (lane slots ``(depth - 1) * K``
        on) and append the ``K`` best children of each row's ``widths[depth]``
        best frontier nodes to the pool from ``cumu_starts[depth]`` on; leaves
        the next frontier in the round buffers. With ``decide``, the decision
        after ``depth`` (which reads only the previous depth's work deltas)
        runs beside the forward, after the round's start at depth 1; with
        ``chain`` (a captured depth of a device chain), the launch of the
        round's next graph follows that decision there, off the forward's
        path, and starts once this depth's graph has ended.

        All ``K`` lanes forward; past the width their outputs are dropped, and
        their K/V (in their own lane slots) is no node's ancestor. The lane
        masks this depth's forward reads come from the previous depth's
        update (depth 1's from ``_start_lanes``).
        """
        k, policy = self.lanes_per_request, self.policy
        pool, state, tree_lanes = self._pool, self._round, self.tree_lanes
        width = policy.widths[depth : depth + 1]
        cumu_start = policy.cumu_starts[depth : depth + 1]
        stream, decide_stream = torch.cuda.current_stream(), self._decide_stream
        if decide:
            decide_stream.wait_stream(stream)
            with torch.cuda.stream(decide_stream):
                if depth == 1:
                    if not self.sampled_trees:
                        policy.begin_round(
                            state["live_rows"], state["frontier"][:bs], state["sampled"]
                        )
                    # Depth 0 weighs the root (one node) against its K children.
                    self._queue_work_deltas(bs, 0)
                policy.decide(depth, state["work_deltas"][(depth - 1) % 2])
                if chain:
                    self._chain.end_depth(bs, depth, policy.go, state["sampled"])
        tree_lanes.active = True
        tree_lanes.step = depth
        self.attn_backend.publish_draft_step_locations(
            cache_start=state["slot_starts"][:bs], num_tokens=k
        )
        logits_output = self.draft_model_runner.forward(
            ctx=lanes,
            input_ids=state["tokens"][:bs].view(-1),
            positions=state["positions"][: bs * k],
            captured_hidden_states=state["hidden"][:bs].view(bs * k, -1),
            spec_step_idx=depth,
        )
        tree_lanes.active = False
        if decide and depth == 1:
            # Depth 1's counts come from the round's start.
            stream.wait_stream(decide_stream)
        # The lanes' children; each request expands its first ``width`` rows.
        child_p, child_tokens = (
            output.view(bs, k, k) for output in self.round_candidates(logits_output)
        )
        if self.sampled_trees:
            self._sample_children(
                logits_output,
                child_tokens,
                child_p,
                width=width,
                slot_base=1 + (depth - 1) * k,
                cumu_start=cumu_start,
                launch=depth,
            )
        # Children sit one depth below the frontier the forward expanded.
        expanded = (
            self._root_children[:bs]
            if depth == 1
            else state["frontier_index"][(depth + 1) % 2][:bs]
        )
        words = 1
        eagle_harness_draft_depth_update(
            child_p,
            child_tokens,
            expanded,
            logits_output.hidden_states.view(bs, k, -1),
            state["tokens"][:bs],
            state["scores"][:bs],
            state["hidden"][:bs],
            state["frontier_index"][depth % 2][:bs],
            pool["tokens"][:bs],
            pool["scores"][:bs],
            pool["depths"][:bs],
            pool["prev"][:bs],
            pool["next"][:bs],
            pool["side"][:bs],
            pool["work"][:bs],
            pool["sort"][:bs],
            state["positions"][: bs * k].view(bs, k),
            state["slot_starts"][:bs],
            tree_lanes.lane_mask[: bs * k * words].view(bs, k, words),
            width,
            depth,
            cumu_start,
            policy.past_verify_num,
            tree_lanes.num_slots,
        )
        self._queue_work_deltas(bs, depth)
        if decide and depth > 1:
            stream.wait_stream(decide_stream)

    def _finalize(self, pending: _PendingRound) -> None:
        """Rank the pool and write the verify tree (its captured graph, or
        eagerly, as when the policy's select_verify waits on the host)."""
        if pending.graphs and not self.policy.eager_select_verify:
            self._finalize_graphs[(pending.rows, pending.sampled)].replay()
        else:
            self._write_verify_tree(pending.rows, pending.sampled)

    def _write_verify_tree(self, bs: int, sampled: bool) -> None:
        """Rank the pool of the live rows among ``bs``, size the verify, and
        write the verify tree as the next round's inputs: the best nodes in
        ancestor-first score order, root first; unused slots repeat the root
        with parent -1 and are never accepted.

        All on the device: the verify size reads the previous round's ranked
        scores (the first round its own), and this round's wait there for the
        next.
        """
        pool, policy, state = self._pool, self.policy, self._round
        live_rows = state["live_rows"]
        top_scores, top_index = state["top_scores"][:bs], state["top_index"][:bs]
        eagle_harness_stable_topk_desc(
            pool["scores"][:bs],
            live_rows,
            policy.cumu_count,
            policy.verify_count,
            top_scores,
            top_index,
        )
        policy.select_verify(top_scores)
        eagle_harness_write_verify_tree(
            top_index,
            pool["tokens"][:bs],
            pool["prev"][:bs],
            state["root_tokens"][:bs],
            state["rows"][:bs],
            self.runtime_states.future_input_map,
            self.runtime_states.future_parent_map,
            live_rows,
            policy.verify_num,
        )
        if sampled:
            draft_q, round_q = self.runtime_states.future_draft_q, self._sampled_q
            eagle_harness_publish_draft_q(
                top_index,
                round_q.slot[:bs],
                round_q.pos[:bs],
                round_q.q[:bs],
                round_q.q_ids[:bs],
                state["rows"][:bs],
                draft_q.q,
                draft_q.q_ids,
                draft_q.slot,
                draft_q.pos,
                live_rows,
                policy.verify_num,
            )

    def _last_verified_window(
        self, bs: int, draft_input: EagleDraftInput
    ) -> torch.Tensor:
        """The next verify window, every column the request's last verified id."""
        next_tokens = torch.empty(
            (bs, self.spec_num_tokens),
            dtype=torch.int32,
            device=self.device,
        )

        # Last verified id per request → next_tokens[:, 0].
        num_extends = draft_input.num_extends
        num_decodes = bs - num_extends
        if num_extends > 0:
            next_tokens[:num_extends, 0] = draft_input.base_model_output[:num_extends]
        if num_decodes > 0:
            indices = (
                self.padded_gather_ids_offsets_buf[:num_decodes]
                + draft_input.accept_lengths[num_extends:]
            )
            if num_extends > 0:
                indices.add_(num_extends)
            torch.index_select(
                draft_input.base_model_output,
                0,
                indices,
                out=next_tokens[num_extends:, 0],
            )
        if self.spec_num_tokens > 1:
            next_tokens[:, 1:] = next_tokens[:, :1]
        return next_tokens
