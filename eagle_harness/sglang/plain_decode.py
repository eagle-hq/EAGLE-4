# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Native decode handoff for a request that will not resume drafting."""

import copy
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

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


def native_decode_tracks_mamba() -> bool:
    """Whether the engine's native DECODE graphs save Mamba prefix-cache states, as its no-spec decode does."""
    from sglang.srt.model_executor.runner.decode_graph_variants import DecodeGraphRunnerInfo

    return "mamba_track" in DecodeGraphRunnerInfo._fields


def verify_limit_supported() -> bool:
    """Whether the engine keeps verify states for a verify batch limit (``max_verify_batch_size``) and can decode
    the larger batches natively, radix cache or not."""
    from sglang.srt.speculative.spec_registry import CustomSpecAlgo

    from eagle_harness.sglang.verify_graphs import native_decode_graphs

    return hasattr(CustomSpecAlgo, "max_verify_batch_size") and native_decode_graphs() and native_decode_tracks_mamba()


def native_decode_available() -> bool:
    """Whether native decode can serve requests that stop drafting: the engine captures native DECODE graphs among
    the speculative graph variants, they track Mamba prefix-cache states when the radix cache does, and DP attention
    is off."""
    from sglang.srt.runtime_context import get_exec, get_parallel

    from eagle_harness.sglang.verify_graphs import native_decode_graphs

    return (
        native_decode_graphs()
        and (native_decode_tracks_mamba() or not get_exec().mamba.enable_mamba_extra_buffer)
        and get_parallel().dp_size == 1
    )


def native_decode_blocked() -> bool:
    """Whether the engine has native DECODE graphs but this server's settings (Mamba prefix-cache tracking on an
    engine whose native graphs do not track, DP attention) keep requests from switching to them."""
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

    def disable(self, requests: Sequence["Req"]) -> None:
        self._disabled.update(requests)

    def all_disabled(self, requests: Sequence["Req"]) -> bool:
        return all(req in self._disabled for req in requests)

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
        track_mamba: bool = False,
        verify_limit: Optional[int] = None,
    ):
        self._target_worker = target_worker
        self._req_to_token = req_to_token
        self._rounds = rounds
        self._requests = NoSpecRequests() if enabled else None
        self._track_mamba = track_mamba
        self._verify_limit = verify_limit  # larger batches decode natively
        # The running batch the next prefill's requests join: the last decode's rows plus the requests prefilled since.
        self._decoding: tuple = ()
        self._since = 0
        self._admitted: weakref.WeakSet[Req] = weakref.WeakSet()

    def skips_draft(self, batch: "ScheduleBatch") -> bool:
        """Whether a prefill's requests join a running batch above the verify limit (or are all off already): they
        decode natively for life, so they are turned off here and their prompt pass needs only the draft's KV. A
        chunked request counts once."""
        if self._requests is None or self._verify_limit is None:
            return False
        new = [req for req in batch.reqs if req not in self._admitted]
        self._admitted.update(new)
        self._since += len(new)
        running = len(self._decoding) + self._since
        if running > self._verify_limit:
            # Rows that finished in the last decode leave before this prefill joins: after a burst whose last step
            # finishes 40 rows, the next lone request would otherwise decode natively for life.
            running = sum(not req.finished() for req in self._decoding) + self._since
        skip = running > self._verify_limit or self._requests.all_disabled(batch.reqs)
        if skip:
            self._requests.disable(batch.reqs)
        return skip

    def select(self, batch: "ScheduleBatch") -> bool:
        if self._requests is None or batch.forward_mode.is_idle():
            return False
        bs = len(batch.reqs)
        self._decoding, self._since = tuple(batch.reqs), 0
        if self._verify_limit is not None and bs > self._verify_limit:
            return self._requests.select(batch.reqs, 0)
        return self._requests.select(batch.reqs, self._rounds.plain_width(bs))

    def publish(self, on_publish, result: "GenerationBatchResult") -> None:
        """Publish native lengths through the host-compatible relay."""
        assert self._requests is not None
        self._requests.publish(on_publish, result)

    def run(self, batch: "ScheduleBatch", grammar_barrier=None) -> "GenerationBatchResult":
        if grammar_barrier is not None:
            grammar_barrier()
        self._rounds.round_start(len(batch.reqs), 0, batch)
        return plain_decode(batch, self._target_worker, self._req_to_token, track_mamba=self._track_mamba)


def plain_decode(
    batch: "ScheduleBatch", target_worker: "TpModelWorker", req_to_token: torch.Tensor, *, track_mamba: bool = False
) -> "GenerationBatchResult":
    """Consume the bonus token in the already-reserved EAGLE KV slot.

    The scheduler has already reserved KV, accumulated penalties, and advanced
    request decode counters. Repeating its ordinary decode preparation would
    allocate twice. The caller must never resume drafting from the carried state. ``track_mamba``: the server
    tracks Mamba prefix-cache states (the radix cache on a hybrid target).
    """
    draft_input = batch.spec_info
    algorithm = batch.spec_algorithm
    batch.input_ids = draft_input.bonus_tokens.to(torch.int64)
    batch.out_cache_loc = req_to_token[batch.req_pool_indices.long(), batch.seq_lens.long()]
    batch.seq_lens = batch.seq_lens + 1
    if batch.seq_lens_cpu is not None:
        batch.seq_lens_cpu = batch.seq_lens_cpu + 1
    batch.seq_lens_sum = None
    if track_mamba:
        track_mamba_states(batch)
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


def track_mamba_states(batch: "ScheduleBatch") -> None:
    """Save the Mamba state of every row whose new length lands on the prefix-cache track grid, as a no-spec decode
    does (``ScheduleBatch.prepare_for_decode``). The track slots are the ones the speculative round planned, and the
    result processor records a one-token round's grid crossing like a verify's."""
    from sglang.srt.runtime_context import mamba_track_grid
    from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify

    prepare_mamba_track_for_verify(batch)
    batch.mamba_track_mask = batch.seq_lens % mamba_track_grid(batch.tree_cache.page_size) == 0
