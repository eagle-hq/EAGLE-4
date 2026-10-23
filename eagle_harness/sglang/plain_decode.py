# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Native decode handoff for a request that will not resume drafting."""

import copy
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from eagle_harness.sglang.hooks import parameter_names

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.managers.utils import GenerationBatchResult

    from eagle_harness.sglang.chain_rounds import ChainRounds


def native_decode_available() -> bool:
    """Whether native decode can serve requests that stop drafting: the engine captures native DECODE graphs among
    the speculative graph variants, and neither Mamba prefix-cache tracking nor DP attention needs the draft's state."""
    from sglang.srt.runtime_context import get_exec, get_parallel

    from eagle_harness.sglang.verify_graphs import native_decode_graphs

    return native_decode_graphs() and not get_exec().mamba.enable_mamba_extra_buffer and get_parallel().dp_size == 1


def native_decode_blocked() -> bool:
    """Whether the engine has native DECODE graphs but this server's settings (Mamba prefix-cache tracking, DP
    attention) keep requests from switching to them."""
    from eagle_harness.sglang.verify_graphs import native_decode_graphs

    return native_decode_graphs() and not native_decode_available()


class NoSpecRequests:
    """Keep a request off until it finishes; skipped draft KV is never reused."""

    def __init__(self) -> None:
        self._disabled: weakref.WeakSet[Req] = weakref.WeakSet()
        # Newer hosts take the one-token commit's host lengths and skip the relay's device pull.
        self.publishes_host_lengths = "new_seq_lens_cpu" in parameter_names(FutureMap.publish)

    def select(self, requests: Sequence["Req"], width: int) -> bool:
        disabled = width == 0 or any(req in self._disabled for req in requests)
        if disabled:
            self._disabled.update(requests)
        return disabled

    def publish(self, on_publish, result: "GenerationBatchResult") -> None:
        """Relay a plain decode's lengths, on host too when the host relay reads them there."""
        if self.publishes_host_lengths and result.new_seq_lens_cpu is not None:
            on_publish(result.new_seq_lens, new_seq_lens_cpu=result.new_seq_lens_cpu)
        else:
            on_publish(result.new_seq_lens)


class NativeDecode:
    """Sticky native-decode routing and its grammar/calibration boundary.

    The caller cancels pending draft work before ``run`` and fences/publishes the
    returned result. Request lifetime tracking survives calibration interruptions.
    """

    def __init__(
        self,
        target_worker: "TpModelWorker",
        req_to_token: torch.Tensor,
        rounds: "ChainRounds",
        *,
        enabled: bool,
    ):
        self._target_worker = target_worker
        self._req_to_token = req_to_token
        self._rounds = rounds
        self._requests = NoSpecRequests() if enabled else None

    def select(self, batch: "ScheduleBatch") -> bool:
        if self._requests is None or batch.forward_mode.is_idle():
            return False
        return self._requests.select(batch.reqs, self._rounds.plain_width(len(batch.reqs)))

    def publish(self, on_publish, result: "GenerationBatchResult") -> None:
        """Publish native lengths through the host-compatible relay."""
        assert self._requests is not None
        self._requests.publish(on_publish, result)

    def run(self, batch: "ScheduleBatch", grammar_barrier=None) -> "GenerationBatchResult":
        if grammar_barrier is not None:
            grammar_barrier()
        self._rounds.round_start(len(batch.reqs), 0, batch)
        return plain_decode(batch, self._target_worker, self._req_to_token)


def plain_decode(
    batch: "ScheduleBatch", target_worker: "TpModelWorker", req_to_token: torch.Tensor
) -> "GenerationBatchResult":
    """Consume the bonus token in the already-reserved EAGLE KV slot.

    The scheduler has already reserved KV, accumulated penalties, and advanced
    request decode counters. Repeating its ordinary decode preparation would
    allocate twice. The caller must never resume drafting from the carried state.
    """
    draft_input = batch.spec_info
    algorithm = batch.spec_algorithm
    batch.input_ids = draft_input.bonus_tokens.to(torch.int64)
    batch.out_cache_loc = req_to_token[batch.req_pool_indices.long(), batch.seq_lens.long()]
    batch.seq_lens = batch.seq_lens + 1
    if batch.seq_lens_cpu is not None:
        batch.seq_lens_cpu = batch.seq_lens_cpu + 1
    batch.seq_lens_sum = None
    batch.spec_info = None
    batch.spec_algorithm = SpeculativeAlgorithm.NONE
    batch.forward_mode = ForwardMode.DECODE
    try:
        result = target_worker.forward_batch_generation(batch, capture_hidden_mode=CaptureHiddenMode.FULL)
    finally:
        batch.spec_algorithm = algorithm
        batch.spec_info = draft_input
    if batch.return_logprob:
        # The overlap result consumer expects a padded accepted-token dimension.
        result.logits_output.next_token_logprobs = result.logits_output.next_token_logprobs[:, None]
    result.new_seq_lens = batch.seq_lens
    # Exactly one token per row: the next lengths are known on host, so the relay needs no device pull.
    result.new_seq_lens_cpu = batch.seq_lens_cpu
    result.speculative_num_draft_tokens = 1
    result.accept_lens = torch.ones_like(batch.seq_lens, dtype=torch.int32)
    result.accept_index = torch.arange(len(batch.seq_lens), dtype=torch.int32, device=batch.seq_lens.device)[:, None]
    result.next_draft_input = copy.copy(draft_input)
    result.next_draft_input.bonus_tokens = result.next_token_ids
    return result
