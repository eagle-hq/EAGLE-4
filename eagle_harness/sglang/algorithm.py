# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The EAGLE_HARNESS speculative algorithm as sglang sees it.

Registered by :func:`eagle_harness.sglang.plugin.register` through
``SpeculativeAlgorithm.register`` with ``supports_overlap=True``. Both overlap
and synchronous scheduling use sglang's V2 speculative schema.

It reports ``is_eagle()``, the documented way for a plugin to join sglang's
EAGLE-family branches (the draft gets its own KV pool and the target
embedding, the relay carries the draft top-k), and ``uses_aux_hidden_states()``,
so the target captures the draft's aux hidden states without the other
EAGLE-3 branches. Hooks of branch ``eagle-hooks`` it implements:

- H2 ``supports_mm_inputs``: text only (sglang's ``validate_request`` refuses multimodal requests).
- H7 ``create_decode_graph_variants``: one target-verify graph per verify
  width (at batch size 1 also the greedy graphs, whose
  forward returns the rows' tokens: ``SpecInput.greedy_tokens_only``), with the
  tree-tail mask attention at batch size 1 (``SpecInput.flashinfer_verify_variant``).
  The algorithm keeps the provider (:attr:`EagleHarnessSpecAlgo.target_verify_graphs`)
  for the drafter, which writes the tail masks.
- H10 ``create_linear_attn_verify_extension``: GDN / Mamba2 tree verify kernels on the target.
- H5 ``req_to_token_extra_len``: block-table headroom for draft / verify slots.
- H11 ``speculative_algorithm_config``: parsed into :class:`HarnessConfig`.

Decode preparation (KV slots for the rounds' draft / verify tokens) is sglang's
EAGLE preparation, which allocates them in chunks.
"""

from __future__ import annotations

import json
import logging
import os
from functools import cached_property
from typing import TYPE_CHECKING, Optional

from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

from eagle_harness.core.config import HarnessConfig, Mode, config_to_dict, parse_harness_config
from eagle_harness.core.cost_profiles import packaged_cost_profile
from eagle_harness.core.gemm_solutions import AITER_GEMM_TABLE, AITER_GEMM_TABLE_ENV, serve_bf16_gemms_through_torch
from eagle_harness.core.limits import ComputeIntensity, check_shape, req_to_token_span
from eagle_harness.kernels import arch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.linear.verify_extension import LinearAttnVerifyExtension
    from sglang.srt.model_executor.runner.decode_graph_variants import DecodeGraphRunnerInfo
    from sglang.srt.server_args import ServerArgs

    from eagle_harness.sglang.verify_graphs import VerifyWidthGraphs

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "EAGLE_HARNESS"
# The draft attention backends the harness plans tree attention for.
DRAFT_ATTENTION_BACKENDS = ("flashinfer", "fa3", "fa4")
# sglang's ROCm switches the measured AMD recipe depends on (read by sglang at import, never set here; the harness sets
# only aiter's GEMM table, eagle_harness.core.gemm_solutions.serve_bf16_gemms_through_torch).
ROCM_ENVIRONMENT = {
    "SGLANG_USE_AITER": ("0", "aiter kernels for the GEMMs, norms, RoPE and the aiter attention backend"),
    "SGLANG_SPLITKV_VERIFY_FLYDSL": ("1", "FlyDSL prefix stage of the split-KV tree verify (faster from batch 4)"),
    "SGLANG_SPLITKV_VERIFY_FLYDSL_COMBINE": ("1", "FlyDSL combine stage of the split-KV tree verify (<= 16 tokens)"),
}


def _check_rocm_environment(server_args: "ServerArgs") -> None:
    """On AMD, log the effective sglang ROCm switches the AMD recipe depends on and warn about the ones that make
    the measured configuration: an aiter target backend without SGLANG_USE_AITER runs torch GEMMs and norms."""
    g = arch.gpu()
    if g is None or g.nvidia:
        return
    effective = {name: os.environ.get(name, default) for name, (default, _) in ROCM_ENVIRONMENT.items()}
    logger.info(
        "EAGLE_HARNESS on %s: %s",
        g.name,
        ", ".join(f"{name}={value} ({ROCM_ENVIRONMENT[name][1]})" for name, value in effective.items()),
    )
    if server_args.attention_backend == "aiter" and effective["SGLANG_USE_AITER"] != "1":
        logger.warning(
            "EAGLE_HARNESS: --attention-backend aiter without SGLANG_USE_AITER=1; the measured MI355X recipe sets it "
            "(sglang then also routes GEMMs, norms and RoPE through aiter)."
        )


def harness_config() -> HarnessConfig:
    """The harness config of this process (``--speculative-algorithm-config``)."""
    return parse_harness_config(get_spec().speculative_algorithm_config)


class EagleHarnessSpecAlgo(CustomSpecAlgo):
    """Algorithm-level hooks. One instance per process (the registry's)."""

    # The target's per-width verify graphs (H7), once sglang asked for them; None without target CUDA graphs.
    target_verify_graphs: Optional["VerifyWidthGraphs"] = None

    @cached_property
    def config(self) -> HarnessConfig:
        """This process's harness config (``--speculative-algorithm-config``)."""
        return parse_harness_config(get_spec().speculative_algorithm_config)

    # ---- EAGLE-family membership (draft KV pool, top-k relay) and aux hidden capture
    def is_eagle(self) -> bool:
        """Member of the EAGLE family (draft KV pool, target embedding, top-k relay)."""
        return True

    def is_eagle3(self) -> bool:
        """Use native learned-head initialization and checkpoint token maps."""
        return True

    def uses_aux_hidden_states(self) -> bool:
        """The target captures the aux hidden states the EAGLE-4 draft reads (its ``eagle_config`` layers)."""
        return True

    # ---- H2: request admission
    def supports_mm_inputs(self) -> bool:
        """The draft embeds text tokens only."""
        # The draft embeds text tokens only.
        return False

    # ---- H7: per-width target-verify graphs
    def create_decode_graph_variants(self, info: "DecodeGraphRunnerInfo") -> Optional["VerifyWidthGraphs"]:
        """H7: one target-verify graph per verify width (and per tree width the greedy graph of a
        batch-size-1 round) on the target runner, kept as :attr:`target_verify_graphs`."""
        if info.is_draft_worker:
            return None
        from eagle_harness.sglang.verify_graphs import VerifyWidthGraphs
        from eagle_harness.sglang.verify_plan import TreeTailMask

        max_width = get_spec().speculative_num_draft_tokens
        from eagle_harness.core.chain_width import load_round_times

        rounds = (
            load_round_times(self.config.data_path)
            if self.config.mode is Mode.SERVE and self.config.data_path else None
        )
        plain = get_spec().speculative_eagle_topk == 1 and (
            self.config.mode is Mode.CALIBRATE or any(0 in row for row in (rounds or {}).values())
        )
        self.target_verify_graphs = VerifyWidthGraphs(
            self.config.intensity, max_width, TreeTailMask(info.device), plain_decode=plain
        )
        return self.target_verify_graphs

    # ---- H10: GDN / Mamba2 tree verify on hybrid targets
    def create_linear_attn_verify_extension(self, state) -> Optional[LinearAttnVerifyExtension]:
        """H10: the tree verify of a hybrid target's GDN or Mamba2 layers."""
        # An sglang whose layout has no ``kind`` asks GDN backends only.
        if getattr(state, "kind", "gdn") == "mamba2":
            from eagle_harness.sglang.mamba2_tree_verify import Mamba2TreeVerify

            return Mamba2TreeVerify.create(state)
        from eagle_harness.sglang.gdn_tree_verify import GDNTreeVerify

        return GDNTreeVerify.create(state)

    # ---- H5: block-table headroom
    def req_to_token_extra_len(self, page_size: int) -> int:
        """H5: block-table headroom for the largest tree."""
        spec = get_spec()
        return req_to_token_span(
            spec.speculative_eagle_topk or 1,
            spec.speculative_num_steps or 1,
            spec.speculative_num_draft_tokens or 1,
            page_size,
        )

    # ---- H11 + server-argument resolution
    def handle_server_args(self, server_args: "ServerArgs") -> None:
        """Parse the harness config and apply what the harness needs from sglang.

        Runs during argument resolution, on a stand-in whose writes sglang
        records as this algorithm's resolutions.
        """
        gpu = arch.gpu()
        if gpu is not None:  # a launcher without a GPU resolves the config too
            arch.require_supported()  # the kernels cover NVIDIA sm_80+ and AMD CDNA2-4: refuse with the reason
        default_data_path = None
        if gpu is not None and not gpu.nvidia:
            default_data_path = packaged_cost_profile(
                device_name=gpu.name,
                model_path=server_args.model_path,
                draft_model_path=server_args.speculative_draft_model_path,
                tp_size=server_args.tp_size,
                dtype=server_args.dtype,
                quantization=server_args.quantization,
            )
        config = parse_harness_config(server_args.speculative_algorithm_config, default_data_path=default_data_path)
        if default_data_path is not None and config.data_path == default_data_path:
            # Every spawned worker must see the same resolved path, without an environment overlay.
            server_args.speculative_algorithm_config = json.dumps(config_to_dict(config))
            logger.info("EAGLE_HARNESS: selected packaged MI355X cost profile %s", default_data_path)
        if server_args.speculative_num_steps is None or server_args.speculative_eagle_topk is None:
            raise ValueError(
                "EAGLE_HARNESS needs --speculative-num-steps, --speculative-eagle-topk and "
                "--speculative-num-draft-tokens (the maximum tree shape)."
            )
        if server_args.speculative_num_draft_tokens is None:
            raise ValueError("EAGLE_HARNESS needs --speculative-num-draft-tokens (the maximum verify width).")
        if server_args.pp_size > 1:
            raise NotImplementedError("pp_size > 1 is not supported for EAGLE_HARNESS.")
        if server_args.speculative_token_map is not None:
            raise NotImplementedError(
                "--speculative-token-map does not apply: EAGLE-4 drafts score the full vocabulary."
            )
        if server_args.speculative_adaptive:
            raise NotImplementedError("--speculative-adaptive does not apply: the cost model picks every round's tree.")
        if server_args.scheduler_recv_interval <= 1:
            server_args.scheduler_recv_interval = 4
        if server_args.enable_mixed_chunk:
            server_args.enable_mixed_chunk = False
            logger.warning("Mixed chunked prefill is disabled: EAGLE_HARNESS does not take mixed steps.")
        if server_args.max_running_requests is None:
            # Keep the memory-conservative default separate from the supported batch limit.
            server_args.max_running_requests = 48 if config.intensity is ComputeIntensity.CHAIN else 32
        topk = server_args.speculative_eagle_topk
        check_shape(
            config.intensity,
            topk,
            server_args.speculative_num_steps,
            server_args.speculative_num_draft_tokens,
            server_args.max_running_requests,
        )
        draft_backend = server_args.speculative_draft_attention_backend or server_args.attention_backend
        if (
            server_args.page_size is not None
            and server_args.page_size > 1
            and draft_backend not in DRAFT_ATTENTION_BACKENDS
        ):
            raise NotImplementedError(
                "page_size > 1 needs --speculative-draft-attention-backend flashinfer, fa3 or fa4 for EAGLE_HARNESS."
            )
        if server_args.attention_backend == "trtllm_mha" and topk > 1:
            raise NotImplementedError("A trtllm_mha target verifies causally only; use a chain (topk 1).")
        if serve_bf16_gemms_through_torch():
            logger.info(
                "EAGLE_HARNESS: %s=%s (every bf16 GEMM on torch, repeatable)", AITER_GEMM_TABLE_ENV, AITER_GEMM_TABLE
            )
        _check_rocm_environment(server_args)
