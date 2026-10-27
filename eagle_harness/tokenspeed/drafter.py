# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""``HarnessDrafter``: EAGLE-Harness as a TokenSpeed drafter (algorithm ``EAGLE_HARNESS``).

A round is TokenSpeed's EAGLE tree round (``Eagle``: the draft extend over the accepted rows, then ``S - 1`` lane
forwards of ``K`` rows per request, all inside TokenSpeed's decode graph) with the harness's own tree policy in place
of TokenSpeed's:

- the EAGLE-4 candidate head ranks each row's next tokens (:meth:`LlamaForCausalLMEagle4.candidate_topk`; no
  whole-vocab LM head per draft step);
- the core :class:`~eagle_harness.core.draft_tree.DraftTree` (shared with the sglang and vLLM adapters) scores the
  nodes, keeps the frontier and picks the ``N - 1`` verify nodes (:meth:`DraftTree.verify_rows`), so the tree verified
  for the same draft outputs is the one the other engines verify;
- :mod:`~eagle_harness.tokenspeed.tree_lanes` turns the frontier choices into TokenSpeed's lane masks and the verify
  window and parents TokenSpeed verifies with its own tree attention and GDN tree verify.

Static rounds use Eagle's complete graph. Serve rounds delegate chain/tree execution, fixed-ahead sampled-depth
planning and recorded-q rejection to the harness. Cost decisions come only from eagle_harness.core; native EAGLE-3
and reduced-vocabulary maps stay owned by TokenSpeed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch
from tokenspeed.runtime.models.llama_eagle3 import LlamaForCausalLMEagle3
from typing_extensions import override

from eagle_harness.core.config import DraftFamily, Mode, build_cost_model
from eagle_harness.core.draft_tree import DraftTree, TreeShape
from eagle_harness.core.limits import ComputeIntensity
from eagle_harness.core.models import target_is_moe
from eagle_harness.tokenspeed.config import ALGORITHM_NAME, TokenSpeedConfig, parse_algorithm_config
from eagle_harness.tokenspeed.costs import tree_policy_costs
from eagle_harness.tokenspeed.models import CandidateRows, LlamaForCausalLMEagle4
from eagle_harness.tokenspeed.tree_lanes import advance_lane_masks, lane_bits, seed_lane_masks
from eagle_harness.tokenspeed.tree_lanes import verify_window as fill_verify_window
from eagle_harness.tokenspeed.tree_round.device_policy import DevicePolicy
from eagle_harness.tokenspeed.tree_round.dynamic import DynamicDrafter

# The harness round executor ranks ordinary EAGLE-3 logits.
_ENGINE_CANDIDATES = DynamicDrafter.round_candidates


class HarnessDrafter(DynamicDrafter):
    supports_speculative_sampling = True

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        model = self.draft_model_runner.model
        if not isinstance(model, LlamaForCausalLMEagle3):
            raise TypeError(f"{ALGORITHM_NAME} requires an EAGLE-3 or EAGLE-4 draft, got {type(model)}")
        self.draft_model = model
        self.config: TokenSpeedConfig | None = None  # set by configure_algorithm, else resolved in bind_tree
        self.tree: DraftTree | None = None
        self._lane_bits: torch.Tensor | None = None
        self.measurement = None

    def configure_algorithm(self, algorithm_config: Mapping[str, Any]) -> None:
        self.config = parse_algorithm_config(algorithm_config)
        if isinstance(self.draft_model, LlamaForCausalLMEagle4):
            self.draft_model.cuda_candidate_kernels = self.config.draft_model.cuda_candidate_kernels
        elif "draft_family" not in algorithm_config:
            self.config = replace(self.config, draft_family=DraftFamily.EAGLE3)

    @override
    def wire_target(self, target: Any) -> None:
        super().wire_target(target)
        self._target_is_moe = target_is_moe(target.config)

    @property
    def dynamic(self) -> bool:
        return self.config is not None and self.config.mode in (Mode.SERVE, Mode.CALIBRATE)

    def _make_policy(self, verify_num: int, topk: int, depth: int) -> DevicePolicy:
        if self.config.mode is Mode.CALIBRATE:
            from tokenspeed.runtime.distributed.process_group_manager import process_group_manager

            from eagle_harness.tokenspeed.measurement import CostMeasurement, MeasurePolicy

            mapping = self.draft_model_runner.mapping
            self.measurement = CostMeasurement(
                run_dir=self.config.data_path,
                tree_depth=depth,
                tp_group=(process_group_manager.get_device_process_group(mapping.attn.tp_group)
                          if mapping.attn.tp_size > 1 else None),
                rank=mapping.rank,
                device=self.device,
            )
            return MeasurePolicy(
                verify_num=verify_num, node_top_k=topk, tree_depth=depth,
                compute_intensity=self.config.intensity.value, device=self.device,
            )
        model = None if self.config.fully_scheduled else build_cost_model(
            self.config,
            node_top_k=topk,
            tree_depth=depth,
            verify_num=verify_num,
            target_is_moe=self._target_is_moe,
        )
        costs = None
        if model is not None:
            costs = model.chain_policy_costs() if self.config.intensity is ComputeIntensity.CHAIN else tree_policy_costs(model)
        return DevicePolicy(
            costs,
            verify_num=verify_num,
            node_top_k=topk,
            tree_depth=depth,
            compute_intensity=self.config.intensity.value,
            is_moe_target=self._target_is_moe,
            exec_mode=0,
            chain_width_table=self.config.chain_width_table,
            device=self.device,
        )

    @override
    def bind_tree(self, tree_spec: Any) -> None:
        if self.config is None:  # launched without --speculative-algorithm-config: the defaults, or their refusal
            self.config = parse_algorithm_config(None)
        if self._sampling_backend.config.enable_speculative_sampling:
            raise ValueError("Use sampled_proposals in --speculative-algorithm-config, not --enable-speculative-sampling")
        if self.dynamic:
            shape = tree_spec.config
            self.bind_dynamic_tree(tree_spec, self._make_policy(shape.num_nodes, shape.topk, shape.num_steps))
            if self.config.sampled_proposals:
                self.enable_sampled_trees(
                    num_candidates=(self.draft_model.num_candidates if isinstance(self.draft_model, LlamaForCausalLMEagle4)
                                    else min(1024, getattr(self.draft_model.config, "draft_vocab_size", None) or self.draft_model.config.vocab_size)),
                    seed=self._sampling_backend.config.random_seed,
                )
            return
        super().bind_tree(tree_spec)
        config = tree_spec.config
        if config.num_steps < 2:
            raise ValueError(f"{ALGORITHM_NAME} needs --speculative-num-steps >= 2, got {config.num_steps}")
        shape = TreeShape(top_k=config.topk, max_depth=config.num_steps)
        if config.num_nodes - 1 > shape.num_nodes(config.num_steps):
            raise ValueError(
                f"--speculative-num-draft-tokens {config.num_nodes} verifies more nodes than a tree of depth "
                f"{config.num_steps} and top-k {config.topk} holds ({shape.num_nodes(config.num_steps)} + the root)"
            )
        self.tree = DraftTree(
            shape,
            self.input_buffers.max_bs,
            self.draft_model.config.hidden_size,
            self.draft_model_runner.model_config.dtype,
            self.device,
            max_verify=1,
            delta_slots=(),  # the work deltas are the cost model's input; the static tree does not read them
        )
        self._lane_bits = lane_bits(config.topk, config.num_steps, self.device)

    @override
    def run(self, base_ctx: Any, logits_output: Any, output_tokens: torch.Tensor, accept_lengths: torch.Tensor):
        if self.tree_lanes is None:
            raise NotImplementedError(
                f"{ALGORITHM_NAME} requires an engine that binds explicit draft ancestry, including chains"
            )
        return super().run(base_ctx, logits_output, output_tokens, accept_lengths)

    # ------------------------------------------------------------------ the tree policy
    def prepare_target_forward(self, ctx: Any) -> None:
        super().prepare_target_forward(ctx)
        if self.measurement is not None and ctx.num_extends == 0:
            self.measurement.verify_start.record()

    def rewrite_verify_outputs(self, ctx: Any, output_tokens: torch.Tensor, accept_lengths: torch.Tensor) -> None:
        """Calibration commits one fixed token so context grows by one each round."""
        if self.measurement is None or ctx.num_extends > 0:
            return
        self.measurement.verify_end.record()
        rows, width = accept_lengths.shape[0], self.spec_num_tokens
        accept_lengths.fill_(1)
        output_tokens.view(rows, width)[:, 0].copy_(self.measurement.commit_token.expand(rows))
        path = self._sampling_backend.accepted_path(rows, width)
        path[:, 0].zero_()
        path[:, 1:].fill_(-1)

    def has_round_graphs(self, graph_bs: int) -> bool:
        return self.measurement is None and super().has_round_graphs(graph_bs)

    def finish_round(self, ctx: Any, graph_bs: int | None) -> None:
        measurement = self.measurement
        if measurement is None or ctx.bs == 0 or self.input_buffers.all_extends_mid_chunk:
            return super().finish_round(ctx, graph_bs)
        from tokenspeed.runtime.utils.host_sync import allow_host_sync

        with allow_host_sync("calibration measures between drained GPUs"):
            if ctx.num_extends > 0:
                measurement.on_extend()
                self.policy.write_full_round()
                return super().finish_round(ctx, graph_bs)
            measurement.record_verify(ctx.bs)
            cache_len = int(self._round["frontier"][:ctx.bs].sum()) // ctx.bs
            if measurement.measured_batch_size is not None:
                self.policy.next_widths(ctx.bs, measurement.measured_batch_size)
            self.policy.write_round()
            measurement.begin_draft()
            super().finish_round(ctx, graph_bs)
            with torch.cuda.stream(self._round_stream):
                measurement.end_draft(ctx.bs, self.policy.draft_width, self.policy.verify_width_now, cache_len)
            if measurement.complete():
                measurement.write_model_records()

    # Defined under TokenSpeed's public names (hooks.PUBLIC_POLICY) and aliased to the release engine's private ones
    # at the end of the class; each engine calls only its own names.
    def round_candidates(self, rows: CandidateRows) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.draft_model, LlamaForCausalLMEagle4):
            if self.sampled_trees:
                rows.next_token_logits, rows.candidate_ids, topk = self.draft_model.candidate_head(
                    rows.normed, self.lanes_per_request
                )
                return topk
            return self.draft_model.candidate_topk(rows.normed, self.lanes_per_request)
        return _ENGINE_CANDIDATES(self, rows)

    def seed_tree(self, bs: int, logits_output: CandidateRows, frontier: torch.Tensor):
        """Depth 1 from the draft extend's candidates; returns the lanes' ``[bs, K]`` tokens and ``[bs * K, H]`` rows."""
        tree, k = self.tree, self.lanes_per_request
        probs, tokens = self.round_candidates(logits_output)
        tree.stage_root(bs, probs, tokens, logits_output.hidden_states)
        tree.begin(bs)
        seed_lane_masks(self.tree_lanes.lane_mask, bs, self._lane_bits)
        self.tree_lanes.set_frontier(bs, frontier)
        return tree.front_tokens[: bs * k].view(bs, k), tree.front_hidden[: bs * k]

    def expand_tree(self, bs: int, step: int, logits_output: CandidateRows):
        """Children of the depth-``step`` lanes; returns the next lanes' tokens and rows (``None`` after the last step)."""
        tree, k = self.tree, self.lanes_per_request
        probs, tokens = self.round_candidates(logits_output)
        tree.step(bs, step, probs, tokens, logits_output.hidden_states)
        lanes = tree.front_tokens[: bs * k].view(bs, k)
        if step == self.spec_num_steps - 1:
            return lanes, None
        advance_lane_masks(tree, self.tree_lanes.lane_mask, bs, step, self._lane_bits)
        return lanes, tree.front_hidden[: bs * k]

    def verify_window(self, bs: int, next_tokens: torch.Tensor) -> torch.Tensor:
        """The best ``N - 1`` nodes and the root as next round's verify window; their parents into the tree spec."""
        fill_verify_window(
            self.tree, bs, self.spec_num_steps, next_tokens[:, 0], next_tokens, self.tree_spec.draft_parent_buf[:bs]
        )
        return next_tokens

    # The release engine's names for the same policy (hooks.RELEASE_POLICY).
    _candidates = round_candidates
    _seed_tree_lanes = seed_tree
    _expand_tree_lanes = expand_tree
    _draft_window = verify_window
