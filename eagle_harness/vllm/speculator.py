# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""``HarnessSpeculator``: EAGLE-Harness as a vLLM Model Runner V2 speculator.

Chain mode (``node_top_k == 1``) drafts with vLLM's own EAGLE loop on the EAGLE-4 draft model and is
verified by vLLM's chain rejection sampler; only the method registry is involved.

Tree mode (``node_top_k > 1``) runs one round as follows (vLLM's hooks in brackets):

1. [``prepare_verify_inputs``] the tree ``propose`` stored for each request becomes the batch's
   :class:`~eagle_harness.vllm.verify_layout.VerifyStage` and the forward's positions become depths. The
   target forward runs the ``CUSTOM`` tree attention and the linear-attention (GDN / Mamba2) tree verify extension.
2. [``make_rejection_sampler``] the tree walk (:mod:`~eagle_harness.vllm.tree_accept`) accepts a path, greedy or
   sampled (lossless). A batch whose logits processing reads each row's prefix (penalties, bad words, a thinking
   budget) is verified as chains instead.
3. [``post_accept``] the target's KV and linear-attention states of the accepted rows move to the chain slots.
4. [``propose``] the draft prefill runs over the accepted rows in path order (:func:`chainize`) and yields
   the next depth-1 candidates; :class:`~eagle_harness.vllm.tree_drafter.TreeDrafter` grows the tree and
   writes the next verify rows (``draft_tokens`` in node order, their parents in the :class:`TreeStore`).

vLLM's scheduler fixes the verify width of a step from the batch size (``num_speculative_tokens``), so the
width is the batch-size schedule's, not a per-round choice. The draft depth is per round: a captured growth skips
the steps whose nodes could not enter the verify set (:mod:`~eagle_harness.vllm.depth_stop`).
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention import Attention
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.cudagraph_utils import SpeculatorCudaGraphManager

from eagle_harness.vllm.config import DERIVED, VllmMethodConfig, parse_method_config
from eagle_harness.vllm.depth_stop import derived_cost_tables
from eagle_harness.vllm.gdn_tree_verify import LinearStateCommit, check_tree_verifiable
from eagle_harness.vllm.round_commit import KvCompactor, chainize
from eagle_harness.vllm.round_recorder import ServedRoundRecorder
from eagle_harness.vllm.tree_accept import TreeRejectionSampler
from eagle_harness.vllm.tree_drafter import TreeDrafter
from eagle_harness.vllm.verify_layout import TreeStore, VerifyStage, set_stage

logger = logging.getLogger(__name__)
STATS_LOG_INTERVAL_S = 60.0  # how often the draft steps per round are logged (one host read each)


class HarnessSpeculator(EagleSpeculator):
    def __init__(self, vllm_config: Any, device: Any):
        super().__init__(vllm_config, device)
        self.method_config: VllmMethodConfig = parse_method_config(self.speculative_config.method_config)
        self.tree_mode = self.method_config.node_top_k > 1
        if self.tree_mode:
            width = self.num_speculative_steps + 1
            self.store = TreeStore(self.max_num_reqs, width, device)
            self.stage = VerifyStage(self.max_num_reqs, self.max_num_tokens, width, device)
            set_stage(self.stage)
            self.sampled_tokens: torch.Tensor | None = None  # the last verify's tokens, by batch row
            self.rejection_sampler: TreeRejectionSampler | None = None
            self.rows = torch.arange(self.max_num_tokens, dtype=torch.int64, device=device)
            self.src_rows = self.rows.clone()  # the row each draft input takes its hidden state from
            self._bound: SimpleNamespace | None = None
            self.recorder: ServedRoundRecorder | None = None
            if self.method_config.record_rounds:
                from vllm.distributed import get_tensor_model_parallel_rank

                self.recorder = ServedRoundRecorder(
                    self.method_config.harness.data_path,
                    writer=get_tensor_model_parallel_rank() == 0,
                    sync_allowed=gpu_sync_allowed,
                )

    def load_model(self, target_model: Any) -> None:
        super().load_model(target_model)
        if self.tree_mode:
            (self.draft_layer_name,) = self.draft_attn_layer_names
            layer = self.vllm_config.compilation_config.static_forward_context[self.draft_layer_name]
            gain = self.method_config.depth_stop_gain
            key = None
            if gain == DERIVED and self.method_config.harness.data_path:
                from eagle_harness.vllm.calibrate import cost_key

                vc = self.vllm_config
                key = cost_key(
                    vc.model_config.model,
                    vc.speculative_config.draft_model_config.model,
                    vc.parallel_config.tensor_parallel_size,
                    vc.model_config.dtype,
                )
            self.drafter = TreeDrafter(
                self.model,
                layer,
                top_k=self.method_config.node_top_k,
                depth=self.method_config.tree_depth,
                max_reqs=self.max_num_reqs,
                device=self.device,
                min_gain=0.0 if gain == DERIVED else gain,
                cost_tables=(
                    derived_cost_tables(
                        self.method_config, key,
                        schedule=self.speculative_config.num_speculative_tokens_per_batch_size,
                        max_batch_size=self.max_num_reqs,
                    )
                    if gain == DERIVED
                    else None
                ),
            )
            self._next_stats_log = time.monotonic() + STATS_LOG_INTERVAL_S
            if self.drafter.tree.shape.num_nodes(self.drafter.depth) < self.num_speculative_steps:
                raise ValueError(
                    "the draft tree has fewer nodes than the verify width needs; raise tree_depth or node_top_k"
                )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        if self.tree_mode:
            linear = [layers for _, _, layers in self._groups() if layers and not isinstance(layers[0], Attention)]
            check_tree_verifiable([layer for layers in linear for layer in layers])
            for layers in linear:  # one linear layer of each group publishes the group's state slots
                self.stage.publish_slots_of(layers[0].prefix)
            self.drafter.graphs = cudagraph_mode != CUDAGraphMode.NONE
            # the draft steps are the drafter's (one graph of its own per batch size); only the draft prefill is a vLLM graph
            self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
                self.vllm_config, self.device, CUDAGraphMode.NONE, decode_query_len=1
            )

    # ------------------------------------------------------------------ KV caches and state of the target
    def _groups(self) -> list[tuple[int, int, list[Any]]]:
        """``(group id, block size, layers)`` of the target's KV cache groups (the draft's layer is not one of them)."""
        context = self.vllm_config.compilation_config.static_forward_context
        return [
            (
                gid,
                group.kv_cache_spec.block_size,
                [context[name] for name in group.layer_names if name != self.draft_layer_name],
            )
            for gid, group in enumerate(self.kv_cache_config.kv_cache_groups)
        ]

    def _bind(self) -> SimpleNamespace:
        """The layers' caches in one launch each (KV compaction, linear-state commit); needs the KV caches allocated."""
        if self._bound is None:
            tables = self.block_tables.input_block_tables
            kv_layers, kv_tables, kv_blocks, linear_layers, linear_slots = [], [], [], [], []
            for gid, block_size, layers in self._groups():
                if layers and all(isinstance(layer, Attention) for layer in layers):
                    kv_layers += layers
                    kv_tables += [tables[gid]] * len(layers)
                    kv_blocks += [block_size] * len(layers)
                elif layers:
                    linear_layers += layers
                    linear_slots += [self.stage.state_slots[layers[0].prefix]] * len(layers)
            groups = self.kv_cache_config.kv_cache_groups
            draft = next(gid for gid, group in enumerate(groups) if self.draft_layer_name in group.layer_names)
            draft_layer = self.vllm_config.compilation_config.static_forward_context[self.draft_layer_name]
            self.drafter.bind_cache(draft_layer.kv_cache, groups[draft].kv_cache_spec.block_size)
            self._bound = SimpleNamespace(
                kv=KvCompactor([layer.kv_cache for layer in kv_layers], kv_tables, kv_blocks, self.device),
                state=LinearStateCommit(linear_layers, linear_slots, self.device),
                draft_group=draft,
            )
        return self._bound

    # ------------------------------------------------------------------ verify hooks
    def prepare_verify_inputs(self, input_batch: Any, model_inputs: dict[str, Any], *, dummy_run: bool = False) -> None:
        if not self.tree_mode:
            return
        if dummy_run:  # a capture or warm-up batch: a uniform multi-token decode shape is a verify batch
            per_req = input_batch.num_tokens // input_batch.num_reqs
            tree = 1 < per_req <= self.stage.width and input_batch.num_tokens == per_req * input_batch.num_reqs
        else:
            drafts = input_batch.num_draft_tokens_per_req
            tree = drafts is not None and not input_batch.has_prefill and bool((drafts >= 1).all())
            # Logits processing that reads each row's prefix needs chains (exact, fewer accepted tokens).
            tree = tree and not self.rejection_sampler.reads_draft_prefix(input_batch.idx_mapping_np)
        self.stage.stage(
            self.store,
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.cu_num_logits,
            input_batch.positions,
            input_batch.num_reqs,
            input_batch.num_tokens,
            tree=tree,
            model_positions=None if model_inputs["positions"] is input_batch.positions else model_inputs["positions"],
        )

    def make_rejection_sampler(self, sampler: Any) -> TreeRejectionSampler | None:
        if not self.tree_mode:
            return None
        self.rejection_sampler = TreeRejectionSampler(
            sampler, self.speculative_config, self.device, self.store, self.stage
        )
        return self.rejection_sampler

    def post_accept(
        self, input_batch: Any, sampler_output: Any, num_sampled: torch.Tensor, num_rejected: torch.Tensor
    ) -> None:
        if not self.tree_mode:
            return
        self.sampled_tokens = sampler_output.sampled_token_ids
        if not self.stage.active:
            return
        bound, store = self._bind(), self.store
        bound.kv.compact(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.cu_num_logits,
            input_batch.seq_lens,
            store.path,
            store.path_len,
        )
        bound.state.commit(input_batch.idx_mapping, store.path, store.path_len)

    # ------------------------------------------------------------------ propose
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync_state=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
        num_speculative_tokens=None,
    ):
        def draft_prefill(hidden, aux, steps):
            return super(HarnessSpeculator, self).propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                hidden,
                aux,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                dp_sync_state,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile,
                steps,
            )

        if not self.tree_mode:
            return draft_prefill(last_hidden_states, aux_hidden_states, num_speculative_tokens)
        count = self.num_speculative_steps if num_speculative_tokens is None else num_speculative_tokens
        num_reqs = input_batch.num_reqs
        if self.recorder is not None and not dummy_run:
            self.recorder.round(num_reqs, count, bool(input_batch.has_prefill), num_sampled)
        if not dummy_run and self.sampled_tokens is not None:
            # Rows in path order: the draft prefill then sees the accepted chain, a prefix of the rows.
            self.src_rows.copy_(self.rows)  # rows that are not a verified tree keep their place
            chainize(
                input_batch.input_ids,
                input_batch.positions,
                self.src_rows,
                idx_mapping=input_batch.idx_mapping,
                query_start_loc=input_batch.query_start_loc,
                cu_num_logits=input_batch.cu_num_logits,
                seq_lens=input_batch.seq_lens,
                path=self.store.path,
                path_len=self.store.path_len,
                sampled=self.sampled_tokens,
                width=self.stage.width,
            )
            src = self.src_rows[: last_hidden_states.shape[0]]
            last_hidden_states = last_hidden_states[src]
            aux_hidden_states = [aux[src] for aux in aux_hidden_states] if aux_hidden_states else aux_hidden_states
        draft_prefill(last_hidden_states, aux_hidden_states, 1)
        drafts = self.draft_tokens[:num_reqs]
        drafts.fill_(-1)
        if dummy_run or count == 0:
            return drafts
        bound = self._bind()
        committed = (input_batch.seq_lens[:num_reqs] - num_rejected[:num_reqs]).long()
        table = self.block_tables.input_block_tables[bound.draft_group][:num_reqs]
        tokens, parent_rows = self.drafter.grow_and_select(num_reqs, count, committed, table)
        if self.drafter.min_gain is not None and time.monotonic() > self._next_stats_log:
            self._next_stats_log = time.monotonic() + STATS_LOG_INTERVAL_S
            with gpu_sync_allowed():  # one deliberate host read a minute (VLLM_GPU_SYNC_CHECK)
                steps = self.drafter.mean_steps()
            if steps is not None:
                logger.info("eagle_harness: %.2f draft steps per round (tree_depth %d)", steps, self.drafter.depth)
        drafts[:, :count] = tokens
        self.store.parents[input_batch.idx_mapping, : count + 1] = parent_rows
        return drafts

    def _prefill(
        self,
        num_reqs,
        num_tokens,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        mm_inputs=None,
    ):
        """The draft prefill; in tree mode it ends in the depth-1 candidates instead of a sampled draft token."""
        if not self.tree_mode:
            return super()._prefill(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                mm_inputs,
            )
        last = self.last_token_indices[:num_reqs]
        normed, hidden = self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            mm_inputs=mm_inputs,
        )
        probs, tokens = self.model.candidate_topk(normed[last], self.method_config.node_top_k)
        self.drafter.seed(num_reqs, probs, tokens, hidden[last])
