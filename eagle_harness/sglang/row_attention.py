# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""sglang attention backend of the decode-time draft forwards (draft steps and the draft extend after verify).

The rows of a request share most of their keys; :mod:`eagle_harness.kernels.row_attention` attends them, reading
the keys through sglang's ``req_to_token`` (one token per page, rows picked by ``req_pool_indices``):

- draft step ``i`` (``K`` frontier rows per request at position ``seq_len[b] + i - 1``): the committed draft KV of the
  window frozen at the round's last committed position, ``[seq_len[b] - window, seq_len[b] - 1]``, plus the row's
  ``i`` tree ancestors, whose KV slots :attr:`DraftTree.ancestors` keeps
  (:meth:`DraftRowAttention.build_tree_rows` also allocates the step's slots and records them as the new ancestors);
- draft extend (``W`` causal rows per request at ``seq_len[b] + j``): the positions ``[pos - window + 1, pos]``.

The prefill draft extend keeps sglang's attention backend.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.runtime_context import get_parallel

from eagle_harness.kernels.row_attention import RowsLaunch, rows_attention, rows_padded

# Per mode, the fastest of a sweep at the TP4 draft shapes on B300 (10 heads x 128, 148 SMs, window 4096, bs 1,
# contexts 1k-4k; attention + merge in a CUDA graph with a cold L2); neighbouring split counts are within ~10%.
# The sweep's winners put 4 waves of (request, head, split) programs on the SMs for a step (64 splits) and 1.5 for
# the extend (24); other GPUs, TP sizes and batch sizes keep those waves (draft_launches). Fixed per captured batch
# size, not autotuned, so the merge order of the splits, and so the outputs, stay the same run to run.
STEP_WAVES, EXTEND_WAVES = 4.0, 1.5
STEP_LAUNCH = RowsLaunch(splits=64, block_n=64)  # a draft step's top-K rows, at the reference shape
EXTEND_LAUNCH = RowsLaunch(splits=24, block_n=128)  # the draft extend's causal rows, at the reference shape
MAX_SPLITS = max(STEP_LAUNCH.splits, EXTEND_LAUNCH.splits)


def draft_launches(programs: int, device: torch.device) -> Tuple[RowsLaunch, RowsLaunch]:
    """The (step, extend) launches for ``programs`` (request, head) programs per key split (batch size x draft heads
    per rank) on ``device``: the key splits that keep the reference sweep's program waves (a step's splits a power of
    two, the extend's a multiple of 8, below 8 a power of two; at most 128)."""
    if device.type != "cuda":
        return STEP_LAUNCH, EXTEND_LAUNCH
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    step = min(triton.next_power_of_2(triton.cdiv(int(STEP_WAVES * sms), programs)), 128)
    extend = triton.cdiv(int(EXTEND_WAVES * sms), programs)
    extend = min(triton.next_power_of_2(extend), triton.cdiv(extend, 8) * 8, 128)
    return RowsLaunch(splits=step, block_n=STEP_LAUNCH.block_n), RowsLaunch(splits=extend, block_n=EXTEND_LAUNCH.block_n)


@triton.jit
def _tree_rows_kernel(
    req_to_token,
    req_to_token_stride,
    req_pool_indices,
    seq_lens,
    ancestors,
    slots_out,
    positions_out,
    step,
    max_depth: tl.constexpr,
    top_k: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // top_k
    lane = row % top_k
    seq_len = tl.load(seq_lens + b).to(tl.int32)
    req = tl.load(req_pool_indices + b).to(tl.int64)
    # This step's draft KV slot of the row: the block table entry after the committed length and the
    # earlier steps' slots (K per step), one per lane.
    own = tl.load(req_to_token + req * req_to_token_stride + seq_len + (step - 1) * top_k + lane)
    tl.store(ancestors + row * max_depth + step - 1, own)
    tl.store(slots_out + row, own)
    tl.store(positions_out + row, seq_len + step - 1)


@triton.jit
def _extend_rows_kernel(
    req_to_token,
    req_to_token_stride,
    req_pool_indices,
    seq_lens,
    accept_lens,
    slots_out,
    positions_out,
    last_out,
    width: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // width
    j = row % width
    seq_len = tl.load(seq_lens + b).to(tl.int32)
    req = tl.load(req_pool_indices + b).to(tl.int64)
    tl.store(slots_out + row, tl.load(req_to_token + req * req_to_token_stride + seq_len + j))
    tl.store(positions_out + row, seq_len + j)
    if j == 0:  # the row whose output seeds the next round: the last accepted token's
        tl.store(last_out + b, b * width + tl.load(accept_lens + b) - 1)


class DraftRowAttention(AttentionBackend):
    """Attention backend of the draft steps and the decode draft extend (decode mode only)."""

    needs_cpu_seq_lens = False

    def __init__(self, draft_runner, *, max_bs: int, max_rows_per_request: int, window: Optional[int]):
        """``max_rows_per_request``: the widest row group (the tree's top-``K`` or the extend's width)."""
        super().__init__()
        self.token_to_kv_pool = draft_runner.token_to_kv_pool
        self.req_to_token = draft_runner.req_to_token_pool.req_to_token
        self.window = int(window or 0)
        device = self.req_to_token.device
        config = draft_runner.model_config
        self.num_heads = config.num_attention_heads // get_parallel().attn_tp_size
        self.head_dim = config.head_dim
        self._device = device
        # The splits shrink as the batch grows (draft_launches); a batch of one has the most per request.
        splits = max(launch.splits for launch in draft_launches(self.num_heads, device))
        parts = max_bs * self.num_heads * (splits + 1) * rows_padded(max_rows_per_request)
        self.part_out = torch.empty(parts, self.head_dim, dtype=torch.float32, device=device)
        self.part_lse = torch.empty(parts, dtype=torch.float32, device=device)
        # The row forward's ForwardBatch needs per-row request indices and lengths, which nothing reads
        # (the rows' keys come from the group build_*_rows sets); one static buffer stands in for both.
        self.unread_rows = torch.zeros(max_bs * max_rows_per_request, dtype=torch.int64, device=device)
        # (rows per request, step (0: extend), seq_lens, req_pool_indices, ancestors) of the next forward.
        self._group = None

    def build_tree_rows(
        self,
        ancestors: torch.Tensor,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        slots: torch.Tensor,
        positions: torch.Tensor,
        *,
        bs: int,
        step: int,
    ) -> None:
        """Rows of draft step ``step``, ``K`` per request (``ancestors``: ``[bs, K, max_depth]`` int32).

        Allocates the step's KV slots (``slots`` and ``positions``: ``[bs * K]`` int64, written) and
        records them as column ``step - 1`` of ``ancestors``.
        """
        _, top_k, max_depth = ancestors.shape
        if not ancestors.is_contiguous():
            raise ValueError("ancestors must be contiguous")
        _tree_rows_kernel[(bs * top_k,)](
            self.req_to_token,
            self.req_to_token.stride(0),
            req_pool_indices,
            seq_lens,
            ancestors,
            slots,
            positions,
            step,
            max_depth=max_depth,
            top_k=top_k,
        )
        self._group = (top_k, step, seq_lens, req_pool_indices, ancestors)

    def build_extend_rows(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        accept_lens: torch.Tensor,
        slots: torch.Tensor,
        positions: torch.Tensor,
        last: torch.Tensor,
        *,
        bs: int,
        width: int,
    ) -> None:
        """Rows of a causal extend, ``width`` per request, the first at position ``seq_lens``.

        Writes the rows' KV slots and positions (``slots``, ``positions``: ``[bs * width]`` int64, the block
        table entries from the committed length on) and ``last`` (``[bs]`` int64): each request's row of its
        last accepted token, ``b * width + accept_lens[b] - 1``.
        """
        _extend_rows_kernel[(bs * width,)](
            self.req_to_token,
            self.req_to_token.stride(0),
            req_pool_indices,
            seq_lens,
            accept_lens,
            slots,
            positions,
            last,
            width=width,
        )
        self._group = (width, 0, seq_lens, req_pool_indices, None)

    def init_forward_metadata(self, forward_batch) -> None:
        """Metadata comes from :meth:`build_tree_rows` / :meth:`build_extend_rows`."""

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs) -> torch.Tensor:
        """Attention of the rows over the committed window and, in a draft step, their tree ancestors."""
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        rows = q.shape[0]
        q = q.view(rows, self.num_heads, self.head_dim)
        out = torch.empty_like(q)
        rows_per_request, step, seq_lens, req_pool_indices, ancestors = self._group
        step_launch, extend_launch = draft_launches(rows // rows_per_request * self.num_heads, self._device)
        rows_attention(
            q,
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            out,
            self.part_out,
            self.part_lse,
            self.req_to_token,
            seq_lens,
            ancestors,
            rows_per_request=rows_per_request,
            step=step,
            window=self.window,
            scale=layer.scaling,
            launch=extend_launch if step == 0 else step_launch,
            table_rows=req_pool_indices,
        )
        return out.view(rows, self.num_heads * self.head_dim)
