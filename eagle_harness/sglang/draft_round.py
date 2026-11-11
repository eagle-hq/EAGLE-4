# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""A decode round's whole draft as one launch: the draft-round contract of the draftmk megakernel.

The launch runs the draft extend after verify, the tree's begin, the draft steps with the stop rule decided per depth
from a :class:`RoundPlan` the host builds once a round, and the ranking of the tree's best nodes; the harness keeps the
verify width (from the lagged scores), the selection and sglang's verify-input build, and the policy's history.

``EAGLE_DRAFT_ROUND`` picks the backend:

- unset: ``draftmk`` where :func:`eagle_harness.kernels.draftmk.library` finds the megakernel, else the plain drafter.
- ``draftmk``: :class:`DraftmkRound`, the megakernel's ctypes binding; an error where there is no megakernel.
- ``graphs``: :class:`GraphRound`, the contract on the harness's own graphs (extend, begin, steps, rank), every stop
  taken from the plan on the host: the reference the megakernel is checked against.
- ``off``: the plain drafter.

Rounds the plan does not cover (chains, sampled trees, the live costs' timed and seeding rounds, batches the backend
does not take, a batch that changed since the prelaunch) run the parent :class:`DynamicDrafter` unchanged.
``EAGLE_DRAFT_ROUND_CHECK=1`` also asks the policy's own ``tree_width`` at every decided depth of the stand-in and
counts the rounds where it disagrees with the plan.
"""

from __future__ import annotations

import atexit
import ctypes
from dataclasses import dataclass
import json
import logging
import os
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from eagle_harness.core.cost_model import WidthStep
from eagle_harness.kernels import arch, draftmk
from eagle_harness.sglang.drafter import DynamicDrafter, LIVE_REFRESH_ROUNDS, Prelaunch, _round_input, batch_key, fit_width, spin_wait
from eagle_harness.sglang.verify_tail import Ranking, VerifyTail

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

logger = logging.getLogger(__name__)

KEEP, LAUNCH, DECIDE, LAST = 0, 1, 2, 3
_KINDS = {WidthStep.KEEP: KEEP, WidthStep.LAUNCH: LAUNCH, WidthStep.DECIDE: DECIDE, WidthStep.LAST: LAST}
ROUND = -1  # Prelaunch.next_depth of a round queued by RoundDrafter


@dataclass
class RoundPlan:
    """The stop rule of one round, per depth ``d`` (begin is 0, step ``d`` drafts depth ``d + 1``)."""

    kind: List[int]
    bar: List[float]  # DECIDE depths: CostModel.tree_depth_threshold(d, row)
    cost: float  # row.cost_values[-1]
    ratio: float  # row.model_cost_ratio
    async_ratio: float  # params.async_layer_ratio
    budget: int  # the deltas row read: deltas_budget(history) - 1
    lane: int  # the deltas column against column 0: CostModel._width_cap()

    @property
    def fixed_depth(self) -> Optional[int]:
        """The round's depth levels when no depth is decided (the fixed shape): the first LAST depth + 1."""
        if DECIDE in self.kind:
            return None
        return next((d + 1 for d, kind in enumerate(self.kind) if kind == LAST), len(self.kind))

    @property
    def min_depth(self) -> int:
        """The fewest depth levels the round can stop at (its first DECIDE or LAST depth + 1)."""
        return next((d + 1 for d, kind in enumerate(self.kind) if kind in (DECIDE, LAST)), len(self.kind))

    def stop(self, depth: int, deltas: List[float]) -> bool:
        """``CostModel.tree_width``'s stop at a DECIDE depth, in its evaluation order (Python floats = fp64)."""
        return (deltas[self.lane] - deltas[0]) / self.cost * self.ratio * self.async_ratio < self.bar[depth]


def round_plan(policy, bs: int, context: int, max_depth: int) -> RoundPlan:
    """The plan :meth:`DynamicDrafter._grow` follows this round, from the policy's own calls (and their side effect:
    ``tree_width`` sets ``tier.tiny`` at a DECIDE before it reads the width cap)."""
    row = policy.draft_cost_row(bs, context)
    kinds, bars, lane = [], [], 0
    for depth in range(max_depth):
        step = policy.width_step(depth)
        kinds.append(_KINDS[step])
        bar = 0.0
        if step is WidthStep.DECIDE:
            policy.tier.tiny = bs == 1
            lane = policy._width_cap()
            bar = policy.tree_depth_threshold(depth, row)
        bars.append(bar)
    budget = policy.deltas_budget(policy.history_verify_num(bs)) - 1
    if row is None:  # the fixed shape: no DECIDE depth reads these
        return RoundPlan(kinds, bars, 1.0, 1.0, 1.0, budget, lane)
    return RoundPlan(kinds, bars, row.cost_values[-1], row.model_cost_ratio, policy.params.async_layer_ratio, budget, lane)


class GraphRound:
    """The round contract on the harness's graphs: the plan's stops decided on the host from the steps' host deltas."""

    def __init__(self, drafter: "RoundDrafter"):
        self.drafter = drafter
        self.max_bs = drafter.tree.max_bs
        self._depth = 0
        self._slot = 0
        self._deltas_ready = torch.cuda.Event()

    def start(self, bs: int, plan: RoundPlan, width: int) -> None:
        """The extend of a ``width``-token verify (0: none, the tree's root buffers are staged) and the begin, then
        the steps up to the plan's first DECIDE depth."""
        steps = self.drafter.steps
        if width:
            self.drafter.extend.run(bs, width)
        else:
            steps.run(bs, 0)
        depth = 0
        while depth < steps.max_step and plan.kind[depth] in (KEEP, LAUNCH):
            self._launched(plan, depth)
            depth += 1
            steps.run(bs, depth)
        self._depth = depth

    def finish(self, bs: int, plan: RoundPlan, check: Optional[callable] = None) -> int:
        """Decide and step until the plan stops; returns the tree's depth levels."""
        steps, depth = self.drafter.steps, self._depth
        while True:
            kind = plan.kind[depth]
            if kind == LAST:
                return depth + 1
            self._launched(plan, depth)
            if kind == DECIDE:
                spin_wait(self._deltas_ready)
                deltas = steps.host_deltas.numpy()[self._slot, plan.budget].tolist()
                stop = plan.stop(depth, deltas)
                if check is not None:
                    check(depth, deltas, stop)
                if stop:
                    return depth + 1
            if depth == steps.max_step:
                return self.drafter.tree.shape.max_depth
            depth += 1
            steps.run(bs, depth)

    def _launched(self, plan: RoundPlan, depth: int) -> None:
        if plan.kind[depth] == LAUNCH:  # its step's graph copies the deltas to the host
            self._deltas_ready.record()
            self._slot = depth

    def ranking(self, bs: int, depth: int, count: int) -> Ranking:
        return self.drafter.tail.rank(bs, depth, count)


# ctypes mirrors of the draftmk C ABI (its capi.h); pointers are device pointers
# unless noted.
_ptr = ctypes.c_void_p


class DraftmkWeights(ctypes.Structure):
    _fields_ = [(name, _ptr) for name in ("embed", "lm_head", "fc")] + [("fc_norm", _ptr * 3)] + [
        (name, _ptr) for name in ("input_norm", "hidden_norm", "qkv", "o", "post_norm", "gate_up", "down",
                                  "final_norm", "head_down", "head_up", "cos_sin")
    ] + [("max_positions", ctypes.c_int64)] + [
        (name, ctypes.c_int32) for name in ("h", "heads", "inter", "vocab", "rank", "window")
    ] + [("eps", ctypes.c_float), ("pad", ctypes.c_int32)]


class DraftmkPlan(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_uint8 * 12), ("bar", ctypes.c_double * 12), ("cost", ctypes.c_double),
                ("ratio", ctypes.c_double), ("async_ratio", ctypes.c_double), ("budget", ctypes.c_int32),
                ("lane", ctypes.c_int32)]


class DraftmkRoundArgs(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int32) for name in ("flags", "bs", "width", "req_to_token_stride")] + [
        (name, _ptr) for name in ("accepted_tokens", "aux_hidden", "accept_lens", "seq_lens", "req_pool_indices",
                                  "req_to_token", "k_cache", "v_cache", "root_probs", "root_tokens", "root_hidden")
    ] + [("plan", ctypes.POINTER(DraftmkPlan))] + [
        (name, _ptr) for name in ("tokens", "parents", "ranked", "verify_scores", "depth", "deltas", "scores")
    ] + [("context", ctypes.c_int32)]


class DraftmkLimits(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int32) for name in ("max_bs", "max_width", "top_k", "max_depth", "verify_tokens", "window")]


class DraftmkTp(ctypes.Structure):
    _fields_ = [("rank", ctypes.c_int32), ("ranks", ctypes.c_int32), ("region", _ptr), ("region_mc", _ptr),
                ("bytes", ctypes.c_int64)]


DRAFTMK_UNSUPPORTED = -1  # capi.h: a round or a setup the library does not take (the graphs run it)
DRAFTMK_EXTEND = 1
DRAFTMK_CONTEXT = 2  # the round's context picks its plan (libraries without per-context plans ignore it)


SERVED_MAX_BS = 4  # draftmk ahead of the graphs through 4 requests at TP1-8, behind from 5


class LaunchLog:
    """Every served draftmk launch as one JSON line in ``PATH.RANK`` (``EAGLE_DRAFT_ROUND_LOG=PATH``), for the
    frozen-cost calibration (draftmk's own price per draft step) and the replay bench (the round's shape at its depth):
    wall time, batch, extend width, the policy's average context, each request's committed length and accepted tokens, the plan,
    the depth the launch drew and its GPU microseconds; in check mode also the graphs' depth and microseconds on the
    same inputs. Lines are written once a launch's events complete, so logging adds one small copy per launch."""

    def __init__(self, path: str):
        self.file = open(path, "a", buffering=1)
        self.pending: List[dict] = []
        atexit.register(self.close)

    def close(self) -> None:
        """The launches still pending, once their events complete (at exit: the last launches' lines)."""
        if self.file.closed:
            return
        if self.pending:
            torch.cuda.synchronize()
            self.drain()
        self.file.close()

    @staticmethod
    def event() -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def begin(self, launch: "DraftmkRound", bs: int, width: int, plan: RoundPlan) -> dict:
        """Before the launch (it advances seq_lens): the lengths' copy, then the start event."""
        d = launch.drafter
        lengths = torch.cat([d.steps.seq_lens[:bs].long(), d.extend.accept_lens[:bs].long()])
        return {"t": time.time(), "bs": bs, "width": width, "context": launch.context, "kind": list(plan.kind), "bar": list(plan.bar),
                "cost": plan.cost, "ratio": plan.ratio, "async_ratio": plan.async_ratio, "budget": plan.budget,
                "lane": plan.lane, "host": lengths.to("cpu", non_blocking=True), "events": [self.event()]}

    def end(self, entry: dict, depth: torch.Tensor) -> None:
        entry["events"].append(self.event())
        entry["depth"] = depth.to("cpu", non_blocking=True)
        entry["events"].append(self.event())  # the copies are done once this completes
        self.pending.append(entry)

    def drain(self) -> None:
        left = []
        for entry in self.pending:
            events = entry["events"]
            if not all(e.query() for e in events):
                left.append(entry)
                continue
            lengths, bs = entry.pop("host").tolist(), entry["bs"]
            entry.pop("events")
            entry.update(depth=int(entry["depth"][0]), seq_lens=lengths[:bs], accept_lens=lengths[bs:],
                         us=1000.0 * events[0].elapsed_time(events[1]))
            if len(events) == 5:  # check mode: the graphs' round on the same inputs
                entry.update(graphs_us=1000.0 * events[3].elapsed_time(events[4]))
            self.file.write(json.dumps(entry) + "\n")
        self.pending = left


def _gathered(weight: torch.Tensor, dim: int, group) -> torch.Tensor:
    """The whole tensor from every rank's shard along ``dim`` (once, at init)."""
    parts = [torch.empty_like(weight) for _ in range(torch.distributed.get_world_size(group))]
    torch.distributed.all_gather(parts, weight.contiguous(), group=group)
    return torch.cat(parts, dim=dim).contiguous()


def draftmk_weights(model, group=None, keep: Optional[list] = None) -> DraftmkWeights:
    """The draft's parameters as sglang holds them (they must stay alive and unmoved: the launch reads them).

    Under TP (``group``, the TP device group) the layer's GEMMs are this rank's shards as sglang holds them, and the
    tensors draftmk reads whole are gathered once into ``keep``: the vocab-parallel embedding (unless the harness
    replicates it already), lm_head and efficient_lm_head, and fc, which sglang shards along its input."""
    body, layer = model.model, model.model.layers[0]
    attn, mlp, config = layer.self_attn, layer.mlp, model.config
    ranks = 1 if group is None else torch.distributed.get_world_size(group)
    cos_sin = attn.rotary_emb.cos_sin_cache
    if cos_sin.dtype != torch.float32 or not cos_sin.is_contiguous():
        raise ValueError(f"draftmk reads an f32 contiguous cos_sin_cache, the draft has {cos_sin.dtype}")
    whole = {"embed": body.embed_tokens.weight, "lm_head": model.lm_head.weight, "fc": body.fc.weight,
             "head_up": model.efficient_lm_head.weight}
    if ranks > 1:
        for name in ("embed", "lm_head", "head_up"):
            if whole[name].shape[0] * ranks != config.vocab_size:
                raise NotImplementedError(f"draftmk at TP{ranks}: {name} has padded vocab shards {tuple(whole[name].shape)}")
        whole = {"embed": body.embed_full if body.embed_full is not None else _gathered(whole["embed"], 0, group),
                 "lm_head": _gathered(whole["lm_head"], 0, group), "fc": _gathered(whole["fc"], 1, group),
                 "head_up": _gathered(whole["head_up"], 0, group)}
        keep.extend(whole.values())
    tensors = {**whole,
               "input_norm": layer.input_layernorm.weight, "hidden_norm": layer.hidden_norm.weight,
               "qkv": attn.qkv_proj.weight, "o": attn.o_proj.weight, "post_norm": layer.post_attention_layernorm.weight,
               "gate_up": mlp.gate_up_proj.weight, "down": mlp.down_proj.weight, "final_norm": body.norm.weight,
               "head_down": model.efficient_lm_head_down_proj.weight, "cos_sin": cos_sin}
    w = DraftmkWeights(**{name: tensor.data_ptr() for name, tensor in tensors.items()})
    for i, norm in enumerate(body.fc_norm or ()):
        w.fc_norm[i] = norm.weight.data_ptr()
    w.max_positions = cos_sin.shape[0]
    w.h, w.vocab = config.hidden_size, config.vocab_size
    w.heads, w.inter = config.num_attention_heads // ranks, config.intermediate_size // ranks
    w.rank, w.window = model.efficient_lm_head_down_proj.out_features, model.window or 0
    w.eps = config.rms_norm_eps
    return w


class DraftmkUnsupported(RuntimeError):
    """draftmk_init refused the draft's shapes (UNSUPPORTED, e.g. a TP share its packer cannot tile): every rank's are
    the same, so every rank falls back to the graphs."""


class DraftmkRound:
    """The megakernel's binding: one launch per round on the current stream, the depth read back after it. A round
    the launch refuses (< 0) runs on the graphs (:class:`GraphRound`) and its ranking is copied to this one's buffers.

    ``check``: every launched round also runs on the graphs from the same inputs (``seq_lens`` and, without an
    extend, the roots restored after the launch), the graphs' round is the one served, and :class:`RoundCheck`
    compares the two.
    """

    def __init__(self, drafter: "RoundDrafter", lib_path: str, check: bool = False):
        self.drafter = drafter
        self.fallback = GraphRound(drafter)
        self.lib = ctypes.CDLL(lib_path)
        self.lib.draftmk_init.argtypes = [ctypes.POINTER(DraftmkWeights), ctypes.c_int]
        self.lib.draftmk_round.argtypes = [ctypes.POINTER(DraftmkRoundArgs), ctypes.c_void_p]
        self.lib.draftmk_limits.argtypes = [ctypes.POINTER(DraftmkLimits)]
        tree, device = drafter.tree, drafter.tree.scores.device
        from sglang.srt.distributed import get_tp_group

        tp_group = get_tp_group()
        limits = DraftmkLimits()
        self.lib.draftmk_limits(ctypes.byref(limits))
        if (limits.top_k, limits.max_depth) != (tree.shape.top_k, tree.shape.max_depth):
            raise DraftmkUnsupported(f"draftmk is built for top_k {limits.top_k}, depth {limits.max_depth}; the tree is {tree.shape}")
        self._keep: list = []  # gathered tensors and the symmetric region the launches read
        if tp_group.world_size == 1:
            self.weights = draftmk_weights(drafter.steps.model)
            status = self.lib.draftmk_init(ctypes.byref(self.weights), device.index or 0)
        else:
            status = self._init_tp(drafter.steps.model, tp_group, device)
        if status == DRAFTMK_UNSUPPORTED:
            raise DraftmkUnsupported(f"draftmk_init: the draft's shapes at TP {tp_group.world_size} are unsupported")
        if status != 0:
            raise RuntimeError(f"draftmk_init failed ({status})")
        # DRAFTMK_MAX_BS: the most requests a round runs as draftmk, the rest as the graphs. The default is the served
        # crossover, 4 at every TP; a library's limit above it only serves when asked, e.g. a check-mode record past 4
        cap = int(os.environ.get("DRAFTMK_MAX_BS", SERVED_MAX_BS))
        self.max_bs, self.max_width = min(limits.max_bs, tree.max_bs, cap), limits.max_width
        rows = limits.verify_tokens - 1
        self.ranked = torch.zeros(tree.max_bs, rows, dtype=torch.int64, device=device)
        self.verify_scores = torch.zeros(rows + 1, dtype=torch.float32, device=device)
        self.depth = torch.zeros(1, dtype=torch.int32, device=device)
        self._host_depth = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._depth_ready = torch.cuda.Event()
        self._plan = DraftmkPlan()
        self._fill_static()
        self._graph_round = False
        self.refused = 0
        self.launches, self.host_us = 0, 0.0  # draftmk_round calls and their host time
        self.check = RoundCheck(tree, self) if check else None
        log = os.environ.get("EAGLE_DRAFT_ROUND_LOG")
        self.log = LaunchLog(f"{log}.{tp_group.rank_in_group}") if log else None
        self.context = 0  # the policy's average context of the round being launched (RoundDrafter.prelaunch)
        self._entry: Optional[dict] = None  # this round's LaunchLog line
        logger.info("eagle-harness: draftmk round backend %s (bs <= %d, width <= %d)", lib_path, self.max_bs, self.max_width)

    def _init_tp(self, model, tp_group, device: torch.device) -> int:
        """draftmk at TP: this rank's shards, the whole tables gathered once, and a
        symmetric region whose multicast address carries the o / down sums; every rank sets it up before any launches."""
        import torch.distributed._symmetric_memory as symm

        group, ranks = tp_group.device_group, tp_group.world_size
        self.lib.draftmk_tp_bytes.restype = ctypes.c_int64
        self.lib.draftmk_tp_bytes.argtypes = [ctypes.c_int32]
        self.lib.draftmk_init_tp.argtypes = [ctypes.POINTER(DraftmkWeights), ctypes.c_int, ctypes.POINTER(DraftmkTp)]
        self.weights = draftmk_weights(model, group, self._keep)
        size = self.lib.draftmk_tp_bytes(ranks)
        if size < 0:  # every rank's answer is the same: all fall back before any collective
            raise DraftmkUnsupported(f"draftmk_init: the draft's shapes at TP {ranks} are unsupported")
        if not symm.is_symm_mem_enabled_for_group(group.group_name):
            symm.enable_symm_mem_for_group(group.group_name)
        region = symm.empty(size, dtype=torch.uint8, device=device)
        handle = symm.rendezvous(region, group)
        if not handle.multicast_ptr:
            raise RuntimeError("draftmk at TP needs NVLink multicast: symmetric memory has no multicast address here")
        self._keep += [region, handle]
        self._tp = DraftmkTp(rank=tp_group.rank_in_group, ranks=ranks, region=region.data_ptr(),
                             region_mc=handle.multicast_ptr, bytes=size)
        status = self.lib.draftmk_init_tp(ctypes.byref(self.weights), device.index or 0, ctypes.byref(self._tp))
        torch.distributed.barrier(group=group)
        return status

    def _fill_static(self) -> None:
        d = self.drafter
        tree, extend, steps = d.tree, d.extend, d.steps
        pool = steps.runner.token_to_kv_pool
        tensors = {"accepted_tokens": extend.tokens, "aux_hidden": extend.input_hidden, "accept_lens": extend.accept_lens,
                   "seq_lens": steps.seq_lens, "req_pool_indices": steps.req_pool_indices, "req_to_token": steps.req_to_token,
                   "k_cache": pool.get_key_buffer(0), "v_cache": pool.get_value_buffer(0), "root_probs": tree.root_probs,
                   "root_tokens": tree.root_tokens, "root_hidden": tree.root_hidden, "tokens": tree.tokens,
                   "parents": tree.parents, "ranked": self.ranked, "verify_scores": self.verify_scores, "depth": self.depth}
        self.args = DraftmkRoundArgs(
            **{name: tensor.data_ptr() for name, tensor in tensors.items()},
            req_to_token_stride=steps.req_to_token.stride(0),
            plan=ctypes.pointer(self._plan),
        )

    def start(self, bs: int, plan: RoundPlan, width: int) -> None:
        a, p = self.args, self._plan
        a.flags = (DRAFTMK_EXTEND if width else 0) | DRAFTMK_CONTEXT  # always: a library keys its plans on the flags
        a.bs, a.width, a.context = bs, width, self.context
        for i in range(12):
            p.kind[i], p.bar[i] = plan.kind[i], plan.bar[i]
        p.cost, p.ratio, p.async_ratio, p.budget, p.lane = plan.cost, plan.ratio, plan.async_ratio, plan.budget, plan.lane
        inputs = None if self.check is None else self.check.save_inputs(bs, width)
        status, entry = -1, None
        if width <= self.max_width:
            if self.log is not None:
                self.log.drain()
                entry = self.log.begin(self, bs, width, plan)
            started = time.perf_counter()
            torch.cuda.nvtx.range_push("draftmk.round")
            status = self.lib.draftmk_round(ctypes.byref(a), torch.cuda.current_stream().cuda_stream)
            torch.cuda.nvtx.range_pop()
            self.host_us += 1e6 * (time.perf_counter() - started)
            self.launches += 1
        if status not in (0, DRAFTMK_UNSUPPORTED):  # not initialized, a CUDA error or a caught panic: never silent
            raise RuntimeError(f"draftmk_round failed ({status})")
        self._graph_round = status < 0 or inputs is not None
        self._entry = None
        if status < 0:
            self.refused += 1
        else:
            if entry is not None:
                self.log.end(entry, self.depth)
                self._entry = entry
            self._host_depth.copy_(self.depth, non_blocking=True)
            self._depth_ready.record()
            if inputs is not None:
                self.check.save_outputs(bs, inputs)
        if self._graph_round:
            if self._entry is not None:
                self._entry["events"].append(LaunchLog.event())
            self.fallback.start(bs, plan, width)

    def finish(self, bs: int, plan: RoundPlan, check=None) -> int:
        if not self._graph_round:
            if plan.fixed_depth is not None:  # nothing to wait for: the host builds the verify input behind the launch
                return plan.fixed_depth
            if self.drafter.tree.shape.num_nodes(plan.min_depth) >= self.ranked.shape[1]:
                return None  # unknown to the host, which does not need it (RoundDrafter.draft)
            spin_wait(self._depth_ready)
            return int(self._host_depth[0])
        depth = self.fallback.finish(bs, plan, check)
        if self._entry is not None:
            self._entry["graphs_depth"] = depth
        if self.check is not None and self.check.pending:
            spin_wait(self._depth_ready)
            self.check.compare(bs, int(self._host_depth[0]), depth)
        return depth

    def ranking(self, bs: int, depth: int, count: int) -> Ranking:
        if self._graph_round:
            ranking = self.fallback.ranking(bs, depth, count)
            self.ranked[:bs, :count].copy_(ranking.ranked)
            self.verify_scores[: count + 1].copy_(ranking.scores)
            if self._entry is not None:
                self._entry["events"].append(LaunchLog.event())
        return Ranking(self.ranked[:bs, :count], self.verify_scores[: count + 1])


class RoundCheck:
    """draftmk's round against the graphs' on the same inputs, as a tolerance check (bitwise is out of reach: cuBLAS
    and the megakernel accumulate in different orders, which moves single bf16 outputs by an ulp).

    A round is *exact*, *near*, or *outside*. Near means every difference is a reordering among values within
    ``TOL`` (relative) of each other, judged on the graphs' values: the ranked nodes against the graphs' decayed node
    scores, the extend's root top-k against the graphs' root probabilities, and the root hidden state within
    ``TOL``. A depth difference is counted apart (the stop rule). Each of the first ``DUMPS`` outside rounds is saved
    to ``EAGLE_DRAFT_ROUND_DUMP`` (a directory) with the inputs to replay it.
    """

    TOL = 2e-2
    DUMPS = 5

    def __init__(self, tree, launch: DraftmkRound):
        self.tree, self.launch = tree, launch
        self.pending = False
        self.rounds = self.exact = self.near = self.outside = self.depth_diffs = self.stale = 0
        self.max_score_err = self.max_root_prob_err = self.max_root_hidden_err = 0.0
        self.dump_dir = os.environ.get("EAGLE_DRAFT_ROUND_DUMP")
        self.dumps = 0
        self._saved = {}

    def save_inputs(self, bs: int, width: int):
        """What the launch overwrites: the committed lengths (an extend adds the accept lengths), the roots it reads
        without an extend."""
        tree, steps = self.tree, self.launch.drafter.steps
        inputs = {"seq_lens": steps.seq_lens[:bs].clone(), "width": width}
        if not width:
            inputs["roots_in"] = [t[:bs].clone() for t in (tree.root_probs, tree.root_tokens, tree.root_hidden)]
        return inputs

    def save_outputs(self, bs: int, inputs) -> None:
        """Set draftmk's outputs aside and restore its inputs for the graphs' round."""
        tree, launch = self.tree, self.launch
        self._saved = {"ranked": launch.ranked[:bs].clone(), "scores": launch.verify_scores.clone(),
                       "tokens": tree.tokens[:bs].clone(), "parents": tree.parents[:bs].clone(),
                       "roots": [t[:bs].clone() for t in (tree.root_probs, tree.root_tokens, tree.root_hidden)], "bs": bs,
                       **inputs}
        launch.drafter.steps.seq_lens[:bs].copy_(inputs["seq_lens"])
        if "roots_in" in inputs:
            for dst, src in zip((tree.root_probs, tree.root_tokens, tree.root_hidden), inputs["roots_in"]):
                dst[:bs].copy_(src)
        self.pending = True

    @classmethod
    def _reordered_within(cls, ours: List[int], ref: List[int], value) -> bool:
        """Whether ``ours`` is ``ref``'s top list up to swaps of near-equal ``value``s: each entry is within ``TOL`` of
        the reference entry at its position."""
        return all(abs(value(a) - value(b)) <= cls.TOL * max(abs(value(a)), abs(value(b)), 1e-30) for a, b in zip(ours, ref))

    def compare(self, bs: int, mk_depth: int, depth: int) -> None:
        self.pending = False
        tree, saved = self.tree, self._saved
        if saved["bs"] != bs:  # outputs set aside for another batch: not this round's
            self.stale += 1
            logger.warning("eagle-harness: draftmk check skipped a stale round (saved at bs %d, compared at bs %d; %d so far)",
                           saved["bs"], bs, self.stale)
            return
        self.rounds += 1
        notes, outside, exact = [], False, True
        if saved["width"]:
            probs, tokens, hidden = saved["roots"]
            ref_probs, ref_tokens = tree.root_probs[:bs], tree.root_tokens[:bs]
            ref = tree.root_hidden[:bs].float()
            hidden_err = ((hidden.float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
            self.max_root_hidden_err = max(self.max_root_hidden_err, hidden_err)
            if not torch.equal(tokens, ref_tokens):
                exact = False
                for r in range(bs):
                    ours, theirs = tokens[r].tolist(), ref_tokens[r].tolist()
                    # A token outside the graphs' top-k counts at draftmk's own probability.
                    prob_of = dict(zip(theirs, ref_probs[r].tolist())) | {
                        t: p for t, p in zip(ours, probs[r].tolist()) if t not in theirs}
                    if not self._reordered_within(ours, theirs, prob_of.__getitem__):
                        outside = True
                        notes.append(f"request {r} root tokens {ours} vs {theirs} (probs {ref_probs[r].tolist()})")
            else:
                self.max_root_prob_err = max(self.max_root_prob_err, (probs - ref_probs).abs().max().item())
            if hidden_err > self.TOL:
                outside = True
                notes.append(f"root hidden rel err {hidden_err:.3g}")
        if mk_depth != depth:
            self.depth_diffs += 1
            exact = False
            notes.append(f"depth {mk_depth} vs {depth}")
        else:
            count = min(tree.shape.num_nodes(depth), saved["ranked"].shape[1])
            ranking = self.launch.fallback.ranking(bs, depth, count)
            nodes = tree.shape.num_nodes(depth)
            same = True
            for r in range(bs):
                ours, ref = saved["ranked"][r, :count].tolist(), ranking.ranked[r].tolist()
                decayed = (tree.scores[r, :nodes] * tree.decay[:nodes]).tolist()
                same &= ours == ref
                if ours != ref and not self._reordered_within(ours, ref, decayed.__getitem__):
                    outside = True
                    notes.append(f"request {r} ranked {ours} vs {ref}")
            if not same or not torch.equal(saved["tokens"][:, :nodes], tree.tokens[:bs, :nodes]):
                exact = False
            if same:
                self.max_score_err = max(self.max_score_err, (saved["scores"][: count + 1] - ranking.scores).abs().max().item())
        if exact:
            self.exact += 1
        elif outside:
            self.outside += 1
            self._dump(bs, notes)
        else:
            self.near += 1
        if notes and outside and self.outside <= 20:
            logger.warning("eagle-harness: draftmk check round %d outside tolerance: %s", self.rounds, "; ".join(notes))

    def _dump(self, bs: int, notes: List[str]) -> None:
        """The round's inputs (accepted tokens, aux hidden, lengths, block-table row, the window's draft KV), both
        rounds' outputs, for a replay against the f32 reference."""
        if self.dump_dir is None or self.dumps >= self.DUMPS:
            return
        self.dumps += 1
        d, saved, tree = self.launch.drafter, self._saved, self.tree
        steps, extend, pool = d.steps, d.extend, d.steps.runner.token_to_kv_pool
        width, seq_len = saved["width"], int(saved["seq_lens"][0])
        row = steps.req_to_token[steps.req_pool_indices[0]]
        window = row[max(seq_len - 4096, 0) : seq_len + max(width, 1)].long()
        record = {
            "notes": notes, "width": width, "seq_lens": saved["seq_lens"].cpu(),
            "accept_lens": extend.accept_lens[:bs].cpu(), "accepted_tokens": extend.tokens[: bs * width].cpu(),
            "aux_hidden": extend.input_hidden[: bs * width].cpu(), "req_to_token_row": row[: seq_len + 64].cpu(),
            "window_positions": (max(seq_len - 4096, 0), seq_len + max(width, 1)),
            "k_window": pool.get_key_buffer(0)[window].cpu(), "v_window": pool.get_value_buffer(0)[window].cpu(),
            "kv_note": "positions >= seq_len hold the graphs' extend KV (written after draftmk's)",
            "roots_in": [t.cpu() for t in saved.get("roots_in", [])],
            "draftmk": {"ranked": saved["ranked"].cpu(), "verify_scores": saved["scores"].cpu(),
                        "tokens": saved["tokens"].cpu(), "parents": saved["parents"].cpu(),
                        "roots": [t.cpu() for t in saved["roots"]]},
            "graphs": {"tokens": tree.tokens[:bs].cpu(), "parents": tree.parents[:bs].cpu(), "scores": tree.scores[:bs].cpu(),
                       "roots": [t[:bs].cpu() for t in (tree.root_probs, tree.root_tokens, tree.root_hidden)]},
        }
        os.makedirs(self.dump_dir, exist_ok=True)
        path = os.path.join(self.dump_dir, f"round{self.rounds}.pt")
        torch.save(record, path)
        logger.warning("eagle-harness: draftmk check dumped round %d to %s", self.rounds, path)

    def summary(self) -> str:
        return (f"rounds {self.rounds} (stale, skipped: {self.stale}): exact {self.exact}, near {self.near}, outside {self.outside}, depth diffs "
                f"{self.depth_diffs}; max verify score err {self.max_score_err:.3g}, max root prob err "
                f"{self.max_root_prob_err:.3g}, max root hidden rel err {self.max_root_hidden_err:.3g}; "
                f"refused {self.launch.refused}")


class RoundDrafter(DynamicDrafter):
    """:class:`DynamicDrafter` whose tree rounds run as one round launch (``backend``: ``graphs`` or ``draftmk``)."""

    def __init__(self, *args, extend, backend: str, check: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.extend = extend
        if backend == "draftmk":
            try:
                # Its own select/build graphs: the parent's are captured over the tail's ranking buffers.
                self.round = DraftmkRound(self, draftmk.library(), check=check)
                self.round_tail = VerifyTail(self.tree, self._build_verify_input)
            except (DraftmkUnsupported, OSError, RuntimeError, ValueError, NotImplementedError) as e:
                # chosen by default (EAGLE_DRAFT_ROUND unset), any setup failure runs the graphs; asked for, only an
                # unsupported setup does
                if os.environ.get("EAGLE_DRAFT_ROUND") == "draftmk" and not isinstance(e, DraftmkUnsupported):
                    raise
                logger.warning("eagle-harness: draftmk: %s; the rounds run on the graphs", e)
                backend = "graphs"
        if backend == "graphs":
            self.round, self.round_tail = GraphRound(self), self.tail
        self._plan: Optional[RoundPlan] = None
        self._check = check
        self.check_rounds = self.check_mismatches = 0
        self.launched_rounds = self.parent_rounds = 0
        self.select_checks = self.select_mismatches = 0
        self._select_tail: Optional[VerifyTail] = None
        self._select_ranking: Optional[Ranking] = None

    def _round_plan(self, bs: int, context: int) -> Optional[RoundPlan]:
        """The next round's plan, or None when the parent drafter runs it."""
        sampled = self.steps.sampled
        if self._chain_width is not None or (sampled is not None and sampled.active) or bs > self.round.max_bs:
            return None
        # The live costs' timed and seeding rounds time the parent's step graphs; static mode (no cost model) never
        # reads those costs.
        timed = self.cost is not None and self.live is not None
        if timed and (
            self._sample_steps()
            or (self._rounds + 1) % LIVE_REFRESH_ROUNDS == 0
            or not self.live.ready(self._bucket(bs))
        ):
            return None
        policy = self.policy(bs)
        if getattr(policy, "chain_rounds", False):
            return None
        return round_plan(policy, bs, context, self.tree.shape.max_depth)

    def prelaunch(self, batch: "ScheduleBatch", *, extend, inputs, stream=None) -> torch.cuda.Event:
        bs = len(batch.reqs)
        plan = self._round_plan(bs, self.average_context(batch))
        if plan is None or not arch.side_stream_prelaunch(self._device):
            return super().prelaunch(batch, extend=extend, inputs=inputs, stream=stream)
        if stream is None:
            if self._prelaunch_stream is None:
                self._prelaunch_stream = torch.cuda.Stream()
            stream = self._prelaunch_stream
        stream.wait_stream(torch.cuda.current_stream())
        for tensor in (batch.req_pool_indices, *inputs):
            tensor.record_stream(stream)
        tree = self.tree
        self.round.context = self.average_context(batch)

        def run(bs: int, width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self.round.start(bs, plan, width)
            return tree.root_probs[:bs], tree.root_tokens[:bs], tree.root_hidden[:bs]

        with torch.cuda.stream(stream):
            extend(run=run)
        self._plan = plan
        self._pending = Prelaunch(batch_key(batch), ROUND, 0, stream)
        return stream.record_event()

    def draft(self, batch: "ScheduleBatch") -> "EagleVerifyInput":
        pending = self._pending
        if pending is None or pending.next_depth != ROUND or pending.key != batch_key(batch):
            if pending is not None and pending.next_depth == ROUND:
                self._take_pending()
            self._count(launched=False)
            return super().draft(batch)
        self._take_pending()
        self._count(launched=True)
        bs = len(batch.seq_lens)
        self._begin_round(batch)
        policy = self.policy(bs)
        context = self.average_context(batch)
        check = None
        if self._check and isinstance(self.round, GraphRound):
            row, past = policy.draft_cost_row(bs, context), policy.history_verify_num(bs)
            self.check_rounds += 1

            def check(depth, deltas, stop):
                if policy.tree_width(depth, bs, past, row, deltas).stop != stop:
                    self.check_mismatches += 1
                    logger.warning("eagle-harness: round plan disagrees with tree_width at depth %d (%d of %d rounds)",
                                   depth, self.check_mismatches, self.check_rounds)

        depth = self.round.finish(bs, self._plan, check)
        # A depth the device decided and the host does not wait for: every verify width fills from the shallowest
        # tree the plan allows, and the verify input is built at the deepest key, whose wider parent list and
        # steps = width - 1 give sglang's tree build the same input (checked by _deepest_key_input in check mode).
        select_depth = self.tree.shape.max_depth if depth is None else depth
        nodes = self.tree.shape.num_nodes(self._plan.min_depth if depth is None else depth)
        widths = self._widths(bs)
        verify_count = policy.verify_count(bs, nodes + 1)
        ranking = self.round.ranking(bs, depth, min(nodes, max(widths) - 1))
        scores = ranking.scores[:verify_count]
        if self._have_lagged:
            lagged = self._verify_scores.read()
            requested = self._verify_width(policy, lagged, verify_count, bs, context)
        else:
            requested = self._verify_width(policy, scores.tolist(), verify_count, bs, context, lagged=False)
            self._have_lagged = True
        self._verify_scores.launch(scores)
        width = fit_width(requested, widths, nodes + 1)
        self.last_depth, self.last_width = select_depth, width
        if width == 1:
            return self._trivial(batch)
        self._stage_bonus(batch.spec_info.bonus_tokens)
        deep = None
        if check is not None and depth < self.tree.shape.max_depth:
            deep = self._deepest_key_input(bs, width, ranking)
        # draftmk's ranking is its own static buffers (DraftmkRound.ranking); the graphs' is the tail's.
        static_ranking = isinstance(self.round, DraftmkRound)
        verify_input = self.round_tail.verify_input(bs, select_depth, width, ranking, static_ranking)
        if deep is not None:
            self._compare_select(depth, width, deep, verify_input)
        return _round_input(verify_input, batch)

    _SELECT_FIELDS = ("draft_token", "positions", "retrieve_index", "retrieve_next_token", "retrieve_next_sibling",
                      "custom_mask")

    def _deepest_key_input(self, bs: int, width: int, ranking: Ranking) -> dict:
        """Check mode: the verify input of the deepest key on this round's ranking (its own tail and ranking
        buffers: a captured build reads the ranking it was captured with). Built before the served one, since both
        write the target's one tree-mask buffer."""
        if self._select_tail is None:
            self._select_tail = VerifyTail(self.tree, self._build_verify_input)
            count = max(self._widths(self.tree.max_bs)) - 1
            device = self.tree.scores.device
            self._select_ranking = Ranking(torch.zeros(self.tree.max_bs, count, dtype=torch.int64, device=device),
                                           torch.zeros(count + 1, dtype=torch.float32, device=device))
        count = ranking.ranked.shape[1]
        self._select_ranking.ranked[:bs, :count].copy_(ranking.ranked)
        self._select_ranking.scores[: count + 1].copy_(ranking.scores)
        fixed = Ranking(self._select_ranking.ranked[:bs, :count], self._select_ranking.scores[: count + 1])
        deep = self._select_tail.verify_input(bs, self.tree.shape.max_depth, width, fixed)
        return {f: getattr(deep, f).clone() for f in self._SELECT_FIELDS}

    def _compare_select(self, depth: int, width: int, deep: dict, served) -> None:
        differ = [f for f in self._SELECT_FIELDS if not torch.equal(deep[f], getattr(served, f))]
        self.select_checks += 1
        if differ:
            self.select_mismatches += 1
            logger.warning("eagle-harness: deepest-key verify input differs at depth %d width %d: %s (%d of %d)",
                           depth, width, differ, self.select_mismatches, self.select_checks)

    def _count(self, *, launched: bool) -> None:
        if launched:
            self.launched_rounds += 1
        else:
            self.parent_rounds += 1
        if (self.launched_rounds + self.parent_rounds) % 500 == 0:
            check = getattr(self.round, "check", None)
            launches = getattr(self.round, "launches", 0)
            refused = getattr(self.round, "refused", 0)
            host = (f"; draftmk_round host {self.round.host_us / launches:.1f} us/call over {launches}, refused {refused}"
                    if launches else "")
            select = f"; select-depth checks {self.select_mismatches}/{self.select_checks} differ" if self.select_checks else ""
            logger.info("eagle-harness: draft rounds: %d by the round backend, %d by the parent drafter%s%s%s",
                        self.launched_rounds, self.parent_rounds, host, select,
                        "" if check is None else f"; draftmk check {check.summary()}")


def round_backend() -> Optional[str]:
    """``EAGLE_DRAFT_ROUND`` (``draftmk`` / ``graphs`` / ``off``), or unset: ``draftmk`` where the megakernel is
    there; None is the plain drafter."""
    backend = os.environ.get("EAGLE_DRAFT_ROUND") or None
    if backend not in (None, "draftmk", "graphs", "off"):
        raise ValueError(f"EAGLE_DRAFT_ROUND={backend!r}: expected draftmk, graphs or off")
    if backend == "draftmk" and draftmk.library() is None:
        raise RuntimeError("EAGLE_DRAFT_ROUND=draftmk, but no draft-round megakernel for this GPU (eagle-harness-draftmk)")
    if backend is None:
        return "draftmk" if draftmk.library() is not None else None
    return None if backend == "off" else backend
