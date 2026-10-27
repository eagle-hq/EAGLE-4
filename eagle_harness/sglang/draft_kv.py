# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Prompt prefills that write only the draft's KV, and which KV slots lack the draft's KV.

A request admitted into a running batch above the verify limit decodes natively for life
(:class:`~eagle_harness.sglang.plain_decode.NativeDecode`), so of its prompt pass the draft needs only the KV, for
later requests that reuse the prompt through the prefix cache: :func:`kv_only_prefill` writes it without the
draft's attention, MLP, head and top-k. :class:`StaleDraftKV` records the slots written by the target alone (native
decode steps) and counts the drafted prefills that reuse any of them, so drafting over them is known.
"""

import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker

logger = logging.getLogger(__name__)


def kv_only_prefill_supported() -> bool:
    """Whether the engine binds a draft forward batch to the target prefill's KV write plan (``kv_loc_plan``) and
    writes KV through ``KVWriteLoc``: the draft KV then lands where its attention would write it."""
    import inspect

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    try:
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc  # noqa: F401
    except ImportError:
        return False
    return "kv_loc_plan" in inspect.signature(ForwardBatch.init_new).parameters


def kv_only_prefill(
    batch: "ScheduleBatch",
    target_worker: "TpModelWorker",
    draft_runner,
    *,
    topk: int,
    hidden: "tuple[int, torch.dtype]",
    on_publish=None,
    pp_proxy_tensors=None,
):
    """The target's prefill, then only the draft's KV of the prompt (the engine's draft prompt pass up to its
    attention: the same token shift, hidden states and write plan). The next draft input is a placeholder that native
    decode does not read (it consumes ``bonus_tokens`` only), shaped as the overlap relay stores it. ``hidden``: the
    draft's recurrent hidden state (size, dtype). The caller enters the draft's TP context."""
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
    from sglang.srt.speculative.eagle_info import EagleDraftExtendInput, EagleDraftInput
    from sglang.srt.speculative.eagle_utils import _eagle_prefill_tail_tokens

    out = target_worker.forward_batch_generation(
        batch, pp_proxy_tensors=pp_proxy_tensors, capture_hidden_mode=CaptureHiddenMode.FULL, return_kv_loc_plan=True
    )
    plan, out.kv_loc_plan = out.kv_loc_plan, None
    out.new_seq_lens = batch.seq_lens  # spec v2: the length before this iteration's tokens, as a drafted prefill
    if on_publish is not None:
        on_publish(out.new_seq_lens)
    tail = _eagle_prefill_tail_tokens(batch, out.next_token_ids)
    batch.spec_info = EagleDraftExtendInput(
        hidden_states=out.logits_output.hidden_states, num_tokens_per_req=1, num_tokens_for_logprob_per_req=1
    )
    forward_batch = ForwardBatch.init_new(
        batch, draft_runner, capture_hidden_mode=CaptureHiddenMode.NULL, return_hidden_states_before_norm=False,
        kv_loc_plan=plan,
    )
    # Each position's draft input is the next token: shift every request's ids left, its last one the tail token.
    ids = torch.roll(forward_batch.input_ids, -1)
    ids[torch.cumsum(forward_batch.extend_seq_lens, 0) - 1] = tail.to(ids.dtype)
    forward_batch.input_ids = ids
    draft_runner.model.write_prefill_kv(forward_batch, draft_runner.token_to_kv_pool)
    rows, device = len(batch.reqs), out.next_token_ids.device
    size, dtype = hidden
    out.next_draft_input = EagleDraftInput(
        topk_p=torch.zeros((rows, topk), device=device, dtype=torch.float32),
        topk_index=torch.zeros((rows, topk), device=device, dtype=torch.int64),
        hidden_states=torch.zeros((rows, size), device=device, dtype=dtype),
        bonus_tokens=out.next_token_ids,
        num_tokens_per_req=1,
        num_tokens_for_logprob_per_req=1,
    )
    return out


class StaleDraftKV:
    """Per KV slot: written by the target without the draft (True: native decode) or with it.

    ``window``: the draft's attention window (None: the whole context); a reused slot older than the window of the
    request's first drafted position is never read and is not counted.
    """

    LOG_EVERY = 64  # drafted prefills with a reused prefix between two reads of the device counters

    def __init__(self, slots: int, device, window: Optional[int] = None):
        self.stale = torch.zeros(slots, dtype=torch.bool, device=device)
        self.window = window
        self._reused = torch.zeros(2, dtype=torch.int64, device=device)  # prefills, stale slots read
        self._checked = 0
        self._logged = 0

    # index_fill_, not ``stale[slots] = value``: the item assignment copies the Python value to the device and
    # waits for the stream (once per decode step).
    def target_only(self, slots: torch.Tensor) -> None:
        self.stale.index_fill_(0, slots.long(), True)

    def drafted(self, slots: torch.Tensor) -> None:
        self.stale.index_fill_(0, slots.long(), False)

    def reused(self, req_to_token: torch.Tensor, rows, prefix_lens, seq_lens) -> None:
        """Count the stale slots a drafted prefill reads in its requests' cached prefixes (``rows`` / ``prefix_lens``
        / ``seq_lens`` per request, host values)."""
        for row, prefix, total in zip(rows, prefix_lens, seq_lens):
            if prefix <= 0:
                continue
            first = 0 if self.window is None else max(0, total - self.window)
            if first >= prefix:
                continue
            stale = self.stale[req_to_token[row, first:prefix].long()].sum()
            self._reused[0] += (stale > 0).long()
            self._reused[1] += stale
            self._checked += 1
            if self._checked % self.LOG_EVERY == 0:
                self.log()

    def log(self) -> None:
        prefills, slots = self._reused.tolist()
        if prefills != self._logged:
            self._logged = prefills
            logger.info(
                "eagle-harness: %d of %d drafted prefills reused prefix-cache KV without the draft's KV (%d slots: "
                "prompts prefilled or tokens decoded above the verify limit); their drafts read stale draft KV",
                prefills, self._checked, slots,
            )
