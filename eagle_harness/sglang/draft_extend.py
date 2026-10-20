# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The decode draft extend after verify, replayed as one CUDA graph per (batch size, verify width).

Request ``b`` runs ``width`` (the round's verify width) causal rows at positions
``seq_len[b] + j``: row ``j`` takes the target's accepted token ``j`` (the bonus
token last) and the target's aux hidden state of verify row ``j``, writes the
draft KV of its position and attends to the draft's sliding window
(:class:`DraftRowAttention`). Rows past the round's accept length write draft
KV beyond the committed length, which the next round overwrites. Row
``accept_len[b] - 1`` yields the next round's depth-1 candidates (the EAGLE-4
candidate head's top-``K``, no full-vocab softmax) and the root hidden state.
The same graph then advances the batch staged on :class:`DraftSteps` to the
next round (committed lengths ``seq_len + accept_len``) and seeds its tree
(:meth:`DraftSteps.begin_body`), so the next round's draft steps follow it
directly.

The batch (``seq_lens``, ``req_pool_indices``) is the one :class:`DraftSteps`
staged for the round's draft and verify input; the verify's results come from
static buffers that :meth:`DraftExtend.stage` fills. The fused accept tail stages
them inside its own graph (writing the compacted hidden states in place,
:meth:`DraftExtend.hidden_rows`).

This replaces sglang's decode draft extend (``_draft_extend_for_decode``),
whose sliding-window FlashInfer planning syncs the host several times per
round; the prefill draft extend stays sglang's.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Sequence, Tuple

import torch

from eagle_harness.sglang.draft_head import draft_topk

from eagle_harness.sglang.draft_steps import DraftSteps, capture_graphs, forward_draft_rows
from eagle_harness.sglang.row_attention import DraftRowAttention

logger = logging.getLogger(__name__)


class DraftExtend:
    """Static buffers and graphs of the decode draft extend."""

    def __init__(
        self,
        draft_runner,
        attention: DraftRowAttention,
        steps: DraftSteps,
        *,
        width: int,
        input_hidden_size: int,
        widths: Callable[[int], Sequence[int]],
    ):
        """``width``: the widest verify; ``widths(bs)``: the verify widths that get a graph at ``bs``. Outputs
        go to the tree's root buffers and ``steps``' staged batch (this round's on entry), which the next round's
        draft reads in place."""
        self.model = draft_runner.model
        self.attention = attention
        self.steps = steps
        self.tree = steps.tree
        self.req_to_token = draft_runner.req_to_token_pool.req_to_token
        self.width = width
        self._widths = widths
        max_bs = self.tree.max_bs
        device = self.req_to_token.device
        i64 = torch.int64
        self.tokens = torch.zeros(max_bs * width, dtype=i64, device=device)
        self.input_hidden = torch.zeros(max_bs * width, input_hidden_size, dtype=draft_runner.dtype, device=device)
        self.accept_lens = torch.ones(max_bs, dtype=i64, device=device)
        self.slots = torch.zeros(max_bs * width, dtype=i64, device=device)
        self.positions = torch.zeros(max_bs * width, dtype=i64, device=device)
        self.last = torch.zeros(max_bs, dtype=i64, device=device)  # each request's row of its last accepted token
        self._graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}

    def hidden_rows(self, rows: int) -> torch.Tensor:
        """The input hidden-state buffer of ``rows`` (= batch size x verify width) rows, to write in place."""
        return self.input_hidden[:rows]

    def stage(self, next_token_ids: torch.Tensor, input_hidden: torch.Tensor, accept_lens: torch.Tensor) -> None:
        """Copy a round's verify results where the extend reads them (device copies, graph-capturable).

        ``next_token_ids`` / ``input_hidden``: the verify's accept-compacted ``[bs * width, ...]`` outputs
        (``input_hidden`` may already be :meth:`hidden_rows`); ``accept_lens``: ``[bs]``.
        """
        rows = next_token_ids.numel()
        self.accept_lens[: accept_lens.shape[0]].copy_(accept_lens)
        self.tokens[:rows].copy_(next_token_ids.reshape(rows))
        hidden = input_hidden.reshape(rows, -1)
        if hidden.data_ptr() != self.input_hidden.data_ptr():
            self.input_hidden[:rows].copy_(hidden)

    def run(self, bs: int, width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the staged round's extend, stage the next round and seed its tree; returns views
        ``(topk_p, topk_index, hidden)`` of ``[bs, ...]`` (valid until the next call)."""
        graph = self._graphs.get((bs, width))
        if graph is not None:
            graph.replay()
        else:
            self._forward((bs, width))
        tree = self.tree
        return tree.root_probs[:bs], tree.root_tokens[:bs], tree.root_hidden[:bs]

    def capture(self, batch_sizes: Sequence[int]) -> None:
        """Capture one draft-extend graph per batch size and verify width."""
        keys = [(bs, w) for bs in sorted(batch_sizes, reverse=True) for w in self._widths(bs) if w <= self.width]
        self._graphs = capture_graphs(keys, self._forward)
        logger.info("eagle-harness: captured %d draft-extend graphs", len(self._graphs))

    def _forward(self, key: Tuple[int, int]) -> None:
        bs, width = key
        rows = bs * width
        seq_lens = self.steps.seq_lens[:bs]  # the round's committed lengths, the draft's staged batch
        req = self.steps.req_pool_indices[:bs]
        accept_lens = self.accept_lens[:bs]
        slots, positions, last = self.slots[:rows], self.positions[:rows], self.last[:bs]
        self.attention.build_extend_rows(seq_lens, req, accept_lens, slots, positions, last, bs=bs, width=width)
        normed, hidden = forward_draft_rows(
            self.model,
            self.attention,
            input_ids=self.tokens[:rows],
            positions=positions,
            out_cache_loc=slots,
            hidden_states=self.input_hidden[:rows],
        )
        tree = self.tree
        probs, tokens = draft_topk(self.model, normed.index_select(0, last), tree.shape.top_k)
        tree.root_probs[:bs] = probs
        tree.root_tokens[:bs] = tokens
        torch.index_select(hidden, 0, last, out=tree.root_hidden[:bs])
        # The next round: its committed lengths (same requests), then the tree's begin from the root just written.
        seq_lens += accept_lens
        self.steps.begin_body(bs)
