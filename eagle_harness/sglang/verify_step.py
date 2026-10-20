# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The accept step of a batch-size-1 greedy tree round, fused.

sglang's ``run_eagle_verify`` ends in about thirty tiny kernels: the tree walk,
accept lengths, the bonus token, the KV slots of the accepted path, and the
compaction of the accepted path to the front. :func:`fused_accept_step` replaces
them with :func:`eagle_harness.kernels.accept_tail.accept_tail` and
:func:`...compact_rows` through sglang's ``accept_step`` hook of
``run_eagle_verify``; the verify preparation, the target forward, the state
commit of the accepted path and the result stay sglang's. Rounds it does not cover (other batch sizes, sampling,
grammar, logprobs, chains) run sglang's accept code
(:func:`fused_accept_supported`).

When the verify input came from the draft's graphs and the target forward
replayed a graph, every input of the step (target tokens, hidden states, the
verify input's tables, the staged lengths) sits in a static buffer, so the step
(acceptance, KV move, row compaction) replays as one CUDA graph per set of
buffers (:class:`AcceptTailGraphs`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Tuple

import torch

from sglang.srt.speculative.eagle_worker_common import VerifyAcceptInput, VerifyAcceptResult, VerifyAcceptStep

from eagle_harness.kernels.accept_tail import Accepted, accept_tail, compact_rows
from eagle_harness.sglang.graph_pool import graph_memory_pool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    from eagle_harness.sglang.draft_extend import DraftExtend

logger = logging.getLogger(__name__)

# Captured tails at most; a key past it runs eagerly (a guard against inputs that are not static).
MAX_TAIL_GRAPHS = 256


def fused_accept_supported(batch: "ScheduleBatch", topk: int, kv_pool: Any) -> bool:
    """Whether the fused accept tail serves this round: one greedy request of a tree (``topk > 1``) without
    grammar, logprobs, penalties or logit bias."""
    sampling = batch.sampling_info
    return (
        topk > 1
        and len(batch.seq_lens) == 1
        and not batch.forward_mode.is_idle()
        and not batch.has_grammar
        and not batch.return_logprob
        and sampling.is_all_greedy
        and sampling.acc_additive_penalties is None
        and sampling.acc_scaling_penalties is None
        and sampling.logit_bias is None
        and getattr(kv_pool, "clear_unaccepted_c128_draft_states", None) is None
    )


class AcceptTailGraphs:
    """The accept tail of a batch-size-1 verify, one CUDA graph per set of input buffers.

    A key's first round runs eagerly (it compiles the kernels, and the tail is not idempotent: its KV
    move must run once per round); the second captures and replays. Keys past :data:`MAX_TAIL_GRAPHS`
    stay eager.
    """

    def __init__(self) -> None:
        self._graphs: Dict[Tuple, Tuple[torch.cuda.CUDAGraph, Any]] = {}
        self._seen: Set[Tuple] = set()

    def run(self, key: Tuple, body: Callable[[], Any]) -> Any:
        """``body()``'s outputs, replayed from the key's graph once it has one."""
        entry = self._graphs.get(key)
        if entry is None:
            if key not in self._seen or len(self._graphs) >= MAX_TAIL_GRAPHS:
                if len(self._seen) < 4 * MAX_TAIL_GRAPHS:
                    self._seen.add(key)
                return body()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=graph_memory_pool(), capture_error_mode="thread_local"):
                outputs = body()
            entry = self._graphs[key] = (graph, outputs)
            if len(self._graphs) == MAX_TAIL_GRAPHS:
                logger.warning("eagle-harness: %d accept-tail graphs; new verify layouts run eagerly", MAX_TAIL_GRAPHS)
        graph, outputs = entry
        graph.replay()
        return outputs


def _accept(
    verify_input: Any,
    target_tokens: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    out_cache_loc: torch.Tensor,
    req_to_token: torch.Tensor,
    kv_cache: Any,
    extend: Optional["DraftExtend"] = None,
) -> Tuple[Accepted, Optional[torch.Tensor]]:
    """Accept, move the KV of the accepted path and compact the hidden states (into the draft extend's input,
    which this then stages, when ``extend`` is given)."""
    if target_tokens.dim() == 2:  # logits
        target_tokens = torch.argmax(target_tokens, dim=-1)
    accepted = accept_tail(
        target_tokens,
        verify_input.draft_token,
        verify_input.retrieve_index,
        verify_input.retrieve_next_token,
        verify_input.retrieve_next_sibling,
        seq_lens,
        out_cache_loc,
        req_to_token,
        req_pool_indices,
        verify_input.max_tree_depth,
    )
    kv_cache.move_kv_cache(accepted.committed_slots, accepted.accepted_slots)
    if hidden_states is None:
        return accepted, None
    out = None if extend is None else extend.hidden_rows(hidden_states.shape[0])
    compacted = compact_rows(hidden_states, accepted.accept_index, out=out)
    if extend is not None:
        extend.stage(accepted.predict, compacted, accepted.accept_lens)
    return accepted, compacted


def fused_accept_step(
    *,
    req_to_token_pool: "ReqToTokenPool",
    token_to_kv_pool_allocator: Any,
    num_draft_tokens: int,
    tail_graphs: Optional[AcceptTailGraphs] = None,
    staged: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    extend: Optional["DraftExtend"] = None,
) -> VerifyAcceptStep:
    """The ``accept_step`` of ``run_eagle_verify`` for a round :func:`fused_accept_supported` accepts.

    ``staged`` holds static copies of the batch's ``(seq_lens, req_pool_indices)`` (the draft's); with
    ``tail_graphs`` they let the step replay as a graph. With ``extend`` the step stages the draft extend's
    input (:meth:`~eagle_harness.sglang.draft_extend.DraftExtend.stage`).
    """

    def accept_step(inp: VerifyAcceptInput) -> VerifyAcceptResult:
        verify_input, hidden_states = inp.verify_input, inp.hidden_states
        # A greedy input gets the rows' tokens; sglang returns logits where it cannot compute them that way.
        target_tokens = inp.next_token_ids if inp.next_token_ids is not None else inp.next_token_logits
        kv_cache = token_to_kv_pool_allocator.get_kvcache()
        req_to_token = req_to_token_pool.req_to_token
        slots = verify_input.prepared_out_cache_loc
        static_tail = tail_graphs is not None and staged is not None and inp.can_run_cuda_graph and slots is not None
        if static_tail:
            # Every input is a static buffer: the target graph's outputs, the draft graph's verify input, the
            # draft's staged lengths.
            seq_lens, req_pool_indices = staged
            key = (
                num_draft_tokens,
                verify_input.max_tree_depth,
                target_tokens.data_ptr(),
                None if hidden_states is None else hidden_states.data_ptr(),
                verify_input.draft_token.data_ptr(),
                slots.data_ptr(),
                seq_lens.data_ptr(),
            )
            accepted, compacted = tail_graphs.run(
                key,
                lambda: _accept(
                    verify_input,
                    target_tokens,
                    hidden_states,
                    seq_lens,
                    req_pool_indices,
                    slots,
                    req_to_token,
                    kv_cache,
                    extend,
                ),
            )
        else:
            accepted, compacted = _accept(
                verify_input,
                target_tokens,
                hidden_states,
                inp.seq_lens,
                inp.req_pool_indices,
                inp.out_cache_loc,
                req_to_token,
                kv_cache,
                extend,
            )
        new_seq_lens = accepted.new_seq_lens
        if static_tail:
            # A replayed tail's lengths are a static buffer, which the scheduler makes the next round's
            # ``batch.seq_lens``: hand it a copy, or the next replay would rewrite the batch's lengths.
            new_seq_lens = new_seq_lens.clone()
        return VerifyAcceptResult(
            next_token_ids=accepted.predict,
            accept_lens=accepted.accept_lens,
            accept_index=accepted.accept_index,
            new_seq_lens=new_seq_lens,
            bonus_tokens=accepted.bonus_tokens,
            hidden_states=compacted,
        )

    return accept_step
