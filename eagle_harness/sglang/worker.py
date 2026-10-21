# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EagleHarnessWorker: cost-model dynamic-tree serving on sglang's V2 schema.

One decode round:

1. draft: :class:`DynamicDrafter` grows the tree with the harness's per-step
   graphs (tree attention, candidate top-k) until the cost model stops it, and
   picks the verify width;
2. verify: sglang's EAGLE verify (``run_eagle_verify``: target TARGET_VERIFY
   forward, tree acceptance, accepted-path KV compaction, GDN state commit) at
   that width, replaying the width's target graph (hook H7); a greedy batch-size-1
   round plugs the fused accept step into sglang's ``accept_step`` hook
   (:mod:`eagle_harness.sglang.verify_step`);
3. draft extend: :class:`DraftExtend` over the accepted tokens (graph-replayed,
   the draft's sliding window), which also seeds the next round's depth-1
   candidates;
4. prelaunch: the draft extend and then the next round's draft up to its first
   host decision (:meth:`DynamicDrafter.prelaunch`) run on a side stream, so
   the scheduler's read of this round's results does not wait for them and
   they run on the GPU while the scheduler processes this round; the next draft
   resumes them if the batch holds the same requests, else drafts from scratch.

A chain (top-k 1, ``chain_intensity``) with served round times under ``data_path`` verifies a fixed width per
batch size instead (:mod:`eagle_harness.sglang.chain_rounds`), and ``"mode": "calibrate"`` forces and times every
chain width while it serves.

Prefill (target extend with aux hidden capture, then the draft extend) is
sglang's EAGLE V2 path unchanged. With overlap scheduling, the worker publishes
lengths after verify and prelaunches on the forward stream for the scheduler's
FutureMap relay. Synchronous scheduling instead prelaunches the
next draft on a side stream before waiting for verify.
"""

from __future__ import annotations

import contextlib
from functools import partial
import logging
import os
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.layers.moe.utils import speculative_moe_a2a_backend_context, speculative_moe_backend_context
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, check_cuda_graph_backend
from sglang.srt.model_executor.runner import get_batch_sizes_to_capture
from sglang.srt.runtime_context import get_exec, get_model, get_parallel, get_schedule, get_spec
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_utils import default_tree_mask_mode
from sglang.srt.speculative.eagle_worker_common import run_eagle_verify
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_utils import draft_tp_context, spec_stage_span
from sglang.version import __version__ as sglang_version

from eagle_harness.core.chain_width import chain_widths
from eagle_harness.core.config import HarnessConfig, Mode, build_cost_model, parse_harness_config
from eagle_harness.core.cost_model import CostModel, FixedShapePolicy
from eagle_harness.core.limits import ComputeIntensity
from eagle_harness.core.draft_tree import DraftTree, TreeShape
from eagle_harness.core.models import target_is_moe
from eagle_harness.sglang.chain_rounds import ChainCalibration, ChainRounds
from eagle_harness.core.live_costs import CostKey, LiveCosts, config_digest
from eagle_harness.core.gemm_solutions import load_gemm_solutions
from eagle_harness.sglang.draft_extend import DraftExtend
from eagle_harness.sglang.draft_steps import DraftSteps
from eagle_harness.sglang.drafter import DynamicDrafter, PrelaunchResult, spin_wait
from eagle_harness.sglang.l2_warm import MlpL2WarmAhead
from eagle_harness.sglang.plain_decode import NativeDecode
from eagle_harness.sglang.row_attention import DraftRowAttention
from eagle_harness.sglang.verify_step import AcceptTailGraphs, fused_accept_step, fused_accept_supported
from eagle_harness.sglang.verify_graphs import verify_widths
from eagle_harness.sglang.hooks import parameter_names
from eagle_harness.sglang.verify_input import HarnessVerifyInput

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

logger = logging.getLogger(__name__)

Policy = Union[CostModel, FixedShapePolicy]


def cost_key(runner) -> CostKey:
    """The setup this server's costs belong to (:class:`~eagle_harness.core.live_costs.CostKey`)."""
    return CostKey(
        gpu=torch.cuda.get_device_name(runner.device),
        gpu_count=get_parallel().tp_size,
        model=config_digest(get_model().model_path),
        draft=config_digest(get_spec().speculative_draft_model_path),
        engine=f"sglang {sglang_version}",
        dtype=str(runner.dtype),
    )


def _sampled_tree(worker):
    """The draft's recorded-q sampled tree, or None: engines without sglang's verify sampler hook build none, and a
    worker built without sglang's __init__ (or serving natively, OFF) may have no draft steps to ask."""
    return getattr(getattr(getattr(worker, "_draft_worker", None), "steps", None), "sampled", None)


class HarnessDraftWorker(EagleDraftWorker):
    """sglang's EAGLE draft worker with the harness's decode-time draft forwards.

    Kept from sglang: the draft TpModelWorker, its KV pool, the shared target
    embedding and the prefill draft extend with its attention backend.
    Replaced: the fixed multi-step draft (by :class:`DraftSteps`) and the
    decode draft extend (by :class:`DraftExtend`), both on
    :class:`DraftRowAttention` and in the harness's own CUDA graphs.
    """

    tree: DraftTree
    steps: DraftSteps
    extend: DraftExtend

    def __init__(self, server_args, gpu_id, nccl_port, target_worker, *, policy: Policy):
        # Read by init_attention_backend, which sglang's __init__ calls.
        self._intensity = policy.intensity
        self._verify_budget = policy.max_history_verify_num
        self._delta_slots = policy.launch_depths
        super().__init__(server_args, gpu_id, nccl_port, target_worker)

    def init_attention_backend(self) -> None:
        """Create the draft-extend attention backend and make it the draft runner's."""
        factory = DraftBackendFactory(self.draft_runner, self.topk, self.speculative_num_steps)
        self.draft_extend_attn_backend = factory.create_draft_extend_backend()
        if self.draft_extend_attn_backend is not None:
            self.draft_runner.attn_backend = self.draft_extend_attn_backend
        model = self.draft_runner.model
        max_bs = get_schedule().max_running_requests
        width = self.speculative_num_draft_tokens
        shape = TreeShape(top_k=self.topk, max_depth=self.speculative_num_steps)
        device = torch.device(self.device)
        self.tree = DraftTree(
            shape,
            max_bs,
            model.config.hidden_size,
            self.draft_runner.dtype,
            device,
            max_verify=self._verify_budget,
            delta_slots=self._delta_slots,
        )
        self.draft_attn_backend = DraftRowAttention(
            self.draft_runner, max_bs=max_bs, max_rows_per_request=max(self.topk, width), window=getattr(model, "window", model.get_attention_sliding_window_size())
        )
        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        # The sampled tree needs sglang's verify tree-sampler hook; engines without it keep 0.1.0's T=1 path (greedy
        # draft, target-only verify). Its draws are keyed by sglang's TP-synchronized seed: every rank draws alike.
        sampled_seed = self.target_worker.random_seed if "sampling_step" in parameter_names(run_eagle_verify) else None
        self.steps = DraftSteps(self.draft_runner, self.tree, self.draft_attn_backend, sampled_seed=sampled_seed)
        intensity = self._intensity
        self.extend = DraftExtend(
            self.draft_runner,
            self.draft_attn_backend,
            self.steps,
            width=width,
            input_hidden_size=(
                model.model.fc.input_size if hasattr(model.model.fc, "input_size") else model.model.fc.in_features
            ),
            widths=lambda bs: verify_widths(bs, intensity, width),
        )
        self.tree_mask_mode = default_tree_mask_mode()

    def _capture_cuda_graphs(self) -> None:
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None
        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return
        capture_bs, _ = get_batch_sizes_to_capture(self.draft_runner)
        self.steps.capture(capture_bs)
        self.extend.capture(capture_bs)

    def extend_after_verify(self, batch: "ScheduleBatch", result, width: int, staged: bool) -> None:
        """The decode draft extend (its input staged first unless the verify ``staged`` it); installs the next
        round's depth-1 candidates in ``result.next_draft_input``."""
        if not staged:
            self.extend.stage(result.next_token_ids, result.logits_output.hidden_states, result.accept_lens)
        draft_input = result.next_draft_input
        draft_input.topk_p, draft_input.topk_index, draft_input.hidden_states = self.extend.run(
            len(batch.seq_lens), width
        )


class EagleHarnessWorker(EAGLEWorkerV2):
    """The EAGLE_HARNESS speculative worker: the harness round in every mode.

    ``mode: static`` runs the fixed shape (full depth, ``--speculative-num-draft-tokens`` verified) with no cost
    model. The cost-model modes read offline tables from ``data_path`` (their ``key.json`` must name this GPU,
    TP size, model pair, engine and dtype) or, without ``data_path``, measure the costs on the served rounds
    (:class:`~eagle_harness.core.live_costs.LiveCosts`) and cache them under ``cost_cache``.
    """

    def __init__(self, server_args: "ServerArgs", gpu_id: int, nccl_port: int, target_worker: "TpModelWorker"):
        spec = get_spec()
        self.config: HarnessConfig = parse_harness_config(spec.speculative_algorithm_config)
        shape = dict(
            node_top_k=spec.speculative_eagle_topk,
            tree_depth=spec.speculative_num_steps,
            verify_num=spec.speculative_num_draft_tokens,
        )
        self._chain_schedule = self._fixed_chain_widths(spec)
        native_decode = self.config.mode is Mode.CALIBRATE or any(
            width == 0 for _, _, width in (self._chain_schedule or ())
        )
        if native_decode and (get_exec().mamba.enable_mamba_extra_buffer or get_parallel().dp_size != 1):
            raise ValueError("Native no-spec fallback requires DP1 and disabled Mamba prefix-cache tracking.")
        self._native_decode_enabled = native_decode
        self.fixed = FixedShapePolicy(self.config.intensity, **shape)
        # Built before sglang's __init__, which creates the draft worker through create_draft_worker.
        self.cost_model: Optional[CostModel] = None
        self.live: Optional[LiveCosts] = None
        if self.config.mode.uses_cost_model and self._chain_schedule is None:
            runner = target_worker.model_runner
            key = cost_key(runner)
            if self.config.data_path:
                tables = None  # loaded below, keyed
            else:
                tables = {}
                self.live = LiveCosts(
                    widths=lambda bs: verify_widths(bs, self.config.intensity, shape["verify_num"]), key=key,
                    draft_shape=(shape["node_top_k"], shape["tree_depth"]),
                )
                self._cache_path = os.path.join(self.config.cost_cache_dir, f"{key.digest}.json")
                if self.live.load(self._cache_path):
                    logger.info("eagle-harness: live costs loaded from %s", self._cache_path)
                    tables = self.live.tables()
            self.cost_model = build_cost_model(
                self.config, target_is_moe=target_is_moe(runner.model_config.hf_config), tables=tables, key=key, **shape
            )
            if not self.cost_model.tables and self.live is None:
                raise ValueError(f"EAGLE_HARNESS: no cost tables under {self.config.cost_dir}.")
        load_gemm_solutions(self.config.data_path)  # before sglang captures any graph
        super().__init__(server_args, gpu_id, nccl_port, target_worker)
        # Hooked before the target's verify graphs are captured (sglang captures every worker's graphs after
        # all workers exist): the graphs record the prefetch's fork and join.
        runner = target_worker.model_runner
        MlpL2WarmAhead.install(runner.model, self.config.l2_warm, runner.device)
        self._drafter: Optional[DynamicDrafter] = None
        self._tail_graphs = AcceptTailGraphs()
        self._verified = torch.cuda.Event()

    def _fixed_chain_widths(self, spec) -> Optional[list]:
        """The served chain widths per batch size range for a chain with round times under ``data_path``; None for
        trees, the calibration, and chains without them (the cost model then sizes each round)."""
        config = self.config
        chain = config.intensity is ComputeIntensity.CHAIN and spec.speculative_eagle_topk == 1
        if config.mode is Mode.CALIBRATE:
            if not chain or config.mode is Mode.STATIC or not config.data_path:
                raise ValueError(
                    'EAGLE_HARNESS "mode": "calibrate" times chains: set --speculative-eagle-topk 1 and '
                    '"intensity": "chain_intensity".'
                )
            return None
        if not chain or config.mode is Mode.STATIC or not config.data_path:
            return None
        max_bs = get_schedule().max_running_requests
        schedule = chain_widths(config.data_path, spec.speculative_num_draft_tokens, max_bs, allow_no_spec=True)
        if schedule is None:
            logger.warning(
                "EAGLE_HARNESS: no served chain round times with an acceptance profile under %s; the cost model "
                "sizes each chain round (the chain-depth rule)",
                config.data_path,
            )
            return None
        for first, last, width in schedule:
            if width != 0 and width not in verify_widths(first, config.intensity, spec.speculative_num_draft_tokens):
                raise ValueError(
                    f"EAGLE_HARNESS: chain width {width} (batch sizes {first}-{last}) has no verify graph."
                )
        logger.info("EAGLE_HARNESS: chain widths per batch size from %s: %s", config.data_path, schedule)
        return schedule
    def _build_trivial_verify_input(self, batch: "ScheduleBatch") -> "EagleVerifyInput":
        """sglang's one-token verify input (plain decoding), with the root token as int64.

        sglang's builder passes the int32 bonus tokens as ``draft_token``; its greedy verify kernel takes int64
        candidates and rejects them (``Dtype value [int32] not in the allowed options: [int64]``) at every batch
        size the fused accept step does not take.
        """
        verify_input = EAGLEWorkerV2._build_trivial_verify_input(self, batch)
        verify_input.draft_token = verify_input.draft_token.to(torch.int64)
        return verify_input

    def create_draft_worker(self, server_args, gpu_id, nccl_port, target_worker) -> HarnessDraftWorker:
        """sglang's EAGLE draft worker with the harness's decode-time draft forwards."""
        return HarnessDraftWorker(
            server_args, gpu_id, nccl_port, target_worker, policy=self.cost_model or self.fixed
        )

    def init_attention_backends(self) -> None:
        """Target and draft backends, then the drafter that owns the per-round tree."""
        super().init_attention_backends()
        intensity, full = self.config.intensity, self.speculative_num_draft_tokens
        calibration = None
        if self.config.mode is Mode.CALIBRATE:
            writer = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            calibration = ChainCalibration(
                self.config.data_path,
                verify_widths(1, intensity, full),
                torch.device(self.device),
                writer=writer,
                native_decode=True,
            )
        self._chain_rounds = ChainRounds(self._chain_schedule, calibration, DynamicDrafter.average_context)
        capture_bs, _ = get_batch_sizes_to_capture(self.target_worker.model_runner)
        tp_group = get_tp_group()
        self._drafter = DynamicDrafter(
            self._draft_worker.steps,
            self.cost_model,
            fixed=self.fixed,
            build_trivial_input=self._build_trivial_verify_input,
            target_worker=self.target_worker,
            algorithm=self.speculative_algorithm,
            widths=lambda bs: verify_widths(bs, intensity, full),
            tree_mask_mode=self._draft_worker.tree_mask_mode,
            device=self.device,
            live=self.live,
            bucket=lambda bs: min((b for b in capture_bs if b >= bs), default=max(capture_bs, default=bs)),
            sync_cells=lambda cells: tp_group.broadcast_object(cells, src=0),
            save_cells=self._save_live_costs if tp_group.rank_in_group == 0 else None,
            chain_width=(
                self._chain_rounds.draft_width if calibration is not None or self._chain_schedule is not None else None
            ),
        )
        self._native_decode = NativeDecode(
            self.target_worker,
            self.req_to_token_pool.req_to_token,
            self._chain_rounds,
            enabled=self._native_decode_enabled,
        )

    def _save_live_costs(self) -> None:
        try:
            self.live.save(self._cache_path)
        except OSError as e:
            logger.warning("eagle-harness: could not write the live cost cache %s: %s", self._cache_path, e)

    def forward_batch_generation(
        self, batch: "ScheduleBatch", on_publish=None, grammar_barrier=None, pp_proxy_tensors=None
    ):
        """One scheduler step: prefill through sglang, else draft, verify and prelaunch the next round."""
        rounds = self._chain_rounds
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._drafter.cancel_prelaunch()
            rounds.interrupt()
            sampled = _sampled_tree(self)
            if sampled is not None:
                # Prefilled requests start from sglang's deterministic top-k roots: no recorded proposal distribution.
                sampled.invalidate(batch.req_pool_indices)
            return super().forward_batch_generation(batch, on_publish, grammar_barrier, pp_proxy_tensors)
        native_decode = self._native_decode.select(batch)
        if native_decode:
            self._drafter.cancel_prelaunch()
            result = self._native_decode.run(batch, grammar_barrier)
            self._verified.record()
        else:
            sampled = _sampled_tree(self)
            if sampled is not None:
                active = (
                    self.topk > 1
                    and not batch.forward_mode.is_idle()
                    and not batch.sampling_info.is_all_greedy
                    and all(req.sampling_params.temperature == 1.0 for req in batch.reqs)
                    and not batch.has_grammar
                    and batch.sampling_info.acc_additive_penalties is None
                    and batch.sampling_info.acc_scaling_penalties is None
                    and not batch.sampling_info.has_watermark_candidates
                )
                if active != sampled.active:
                    self._drafter.cancel_prelaunch()
                    sampled.root_valid.zero_()  # this round's roots were drafted without recorded proposals
                    logger.debug("eagle-harness: sampled tree %s", "on" if active else "off")
                sampled.active = active
            draft = self._draft_worker
            timer = self._drafter.timer
            if batch.forward_mode.is_idle():
                self._drafter.cancel_prelaunch()
                batch.spec_info, width = self._build_trivial_verify_input(batch), 1
                rounds.interrupt()
                timer = None
            else:
                with self._draft_context(), spec_stage_span("draft"):
                    batch.spec_info = self._drafter.draft(batch)
                width = self._drafter.last_width
                rounds.round_start(len(batch.seq_lens), width, batch)
            with spec_stage_span("verify"):
                started = None if timer is None else timer.start()
                result, staged = self._verify(batch, width, grammar_barrier)
                self._verified.record()
                if started is not None:
                    timer.stop(started, self._drafter.bucket(len(batch.reqs)), width, self._drafter.average_context(batch))
            if not batch.forward_mode.is_idle():
                rounds.accepted(width, result.accept_lens)
        if on_publish is not None:
            if native_decode:
                self._native_decode.publish(on_publish, result)
            else:
                on_publish(result.new_seq_lens)
        if not native_decode and not batch.forward_mode.is_idle():
            with self._draft_context(), spec_stage_span("draft_extend"):
                prelaunch_done = self._drafter.prelaunch(
                    batch,
                    extend=lambda: draft.extend_after_verify(batch, result, width, staged),
                    inputs=(
                        batch.seq_lens,
                        result.next_token_ids,
                        result.accept_lens,
                        result.logits_output.hidden_states,
                    ),
                    # FutureMap relays draft outputs immediately after return on this stream.
                    stream=torch.cuda.current_stream() if on_publish is not None else None,
                )
            result = PrelaunchResult(**vars(result), prelaunch_done=prelaunch_done)
        # Only the synchronous scheduler reads the results right away; poll here rather than block in its copy.
        if on_publish is None:
            spin_wait(self._verified)
        return result

    def _verify(self, batch: "ScheduleBatch", width: int, grammar_barrier):
        """sglang's EAGLE verify at ``width`` tokens per request (the width's target graph, via H7).

        A greedy batch-size-1 round asks the target forward for the rows' greedy tokens and accepts with the
        fused accept step (:func:`~eagle_harness.sglang.verify_step.fused_accept_step`, sglang's
        ``accept_step`` hook), replayed as a graph once its inputs are the draft's and the target graph's
        static buffers; that step also stages the draft extend's input.
        Returns the result and whether the draft extend's input is staged.
        """
        accept_step = None
        if fused_accept_supported(batch, self.topk, self.token_to_kv_pool_allocator.get_kvcache()):
            if isinstance(batch.spec_info, HarnessVerifyInput):
                batch.spec_info.greedy_tokens_only = True
            steps = self._draft_worker.steps  # the batch's lengths and rows, staged in static buffers
            accept_step = fused_accept_step(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                num_draft_tokens=width,
                tail_graphs=self._tail_graphs,
                staged=(steps.seq_lens[:1], steps.req_pool_indices[:1]),
                extend=self._draft_worker.extend,
            )
        sampled = _sampled_tree(self)
        hook = {}  # never passed to an engine without the hook (sampled is None there)
        if sampled is not None and sampled.active and width > 1:
            hook["sampling_step"] = partial(sampled.sample, batch.spec_info)
        result = run_eagle_verify(
            batch,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_draft_tokens=width,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            grammar_barrier=grammar_barrier,
            accept_step=accept_step,
            **hook,
        )
        if accept_step is not None and batch.enable_overlap:
            # The scheduler reads on a leaf stream; the next tail replay may overwrite its graph buffers.
            result.next_token_ids = result.next_token_ids.clone()
            result.accept_lens = result.accept_lens.clone()
            if batch.return_hidden_states and result.logits_output.hidden_states is not None:
                result.logits_output.hidden_states = result.logits_output.hidden_states.clone()
        return result, accept_step is not None

    def _draft_context(self) -> contextlib.ExitStack:
        """sglang's draft-side contexts (draft TP group, speculative MoE backends)."""
        stack = contextlib.ExitStack()
        stack.enter_context(draft_tp_context(self._draft_worker.draft_owns_attention))
        stack.enter_context(speculative_moe_backend_context())
        stack.enter_context(speculative_moe_a2a_backend_context())
        return stack
