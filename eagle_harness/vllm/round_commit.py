# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""What a tree round leaves behind, turned back into what vLLM expects after a chain round.

After the tree walk the accepted rows are not a prefix of the verify rows. Two things put them in order,
so that everything downstream of the verify (the next forward's KV cache, the draft's re-seeding) sees the
chain it knows:

- :class:`KvCompactor` (``post_accept``) copies the KV of the accepted rows of the target's full-attention
  layers to the chain slots ``context + j``. A row's KV was written at ``context + row`` and is ready to
  move because the sampler ran after the forward.
- :func:`chainize` (before the draft's prefill) rewrites the target's inputs of the round the same way:
  the draft sees the accepted rows in path order, at chain positions, with the sampled tokens.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _is_tree(query_start_loc_ptr, cu_num_logits_ptr, b):
    width = tl.load(query_start_loc_ptr + b + 1) - tl.load(query_start_loc_ptr + b)
    return (width > 1) & (width == tl.load(cu_num_logits_ptr + b + 1) - tl.load(cu_num_logits_ptr + b))


@triton.jit
def _compact_kv_kernel(
    layer_ptrs,  # [L] uint64: base address of each layer's KV cache
    table_ptrs,  # [L] uint64: the layer's KV group's block table, [num_reqs, max_blocks] int32: the batch's rows
    table_strides,  # [L] int64: row stride of that table
    table_block_sizes,  # [L] int32: tokens per block of that table
    idx_mapping_ptr,
    query_start_loc_ptr,
    cu_num_logits_ptr,
    seq_lens_ptr,
    path_ptr,  # [slots, W] int32
    stride_path,
    path_len_ptr,  # [slots] int32
    stride_block,  # KV cache strides in elements
    stride_head,
    stride_slot,
    block_size,  # tokens per block of the cache as the attention backend views it
    DIM: tl.constexpr,  # K and V of one token and head, adjacent
    DTYPE: tl.constexpr,
):
    b, layer, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if not _is_tree(query_start_loc_ptr, cu_num_logits_ptr, b):
        return
    slot = tl.load(idx_mapping_ptr + b)
    context = tl.load(seq_lens_ptr + b) - (tl.load(query_start_loc_ptr + b + 1) - tl.load(query_start_loc_ptr + b))
    base = tl.load(layer_ptrs + layer).to(tl.pointer_type(DTYPE))
    block_table_ptr = tl.load(table_ptrs + layer).to(tl.pointer_type(tl.int32))
    stride_table = tl.load(table_strides + layer)
    table_block_size = tl.load(table_block_sizes + layer).to(tl.int64)
    dims = tl.arange(0, DIM)
    # One program per (request, layer, head) moves the rows in path order: a row's destination can be another
    # accepted row's source, which was read earlier in the path.
    for j in range(1, tl.load(path_len_ptr + slot)):
        src = context + tl.load(path_ptr + slot * stride_path + j)
        dst = context + j
        if src != dst:
            src_token = (
                tl.load(block_table_ptr + b * stride_table + src // table_block_size).to(tl.int64) * table_block_size
                + src % table_block_size
            )
            dst_token = (
                tl.load(block_table_ptr + b * stride_table + dst // table_block_size).to(tl.int64) * table_block_size
                + dst % table_block_size
            )
            src_at = (
                (src_token // block_size) * stride_block + head * stride_head + (src_token % block_size) * stride_slot
            )
            dst_at = (
                (dst_token // block_size) * stride_block + head * stride_head + (dst_token % block_size) * stride_slot
            )
            tl.store(base + dst_at + dims, tl.load(base + src_at + dims))
            tl.debug_barrier()


class KvCompactor:
    """Moves the KV of the accepted rows of a tree round to the chain slots, in every given layer.

    ``block_tables[i]`` and ``table_block_sizes[i]`` are layer ``i``'s KV cache group's persistent block table
    (the batch's rows, as the forward got them) and its tokens per block; all layers go in one launch.
    """

    def __init__(
        self,
        kv_caches: Sequence[torch.Tensor],
        block_tables: Sequence[torch.Tensor],
        table_block_sizes: Sequence[int],
        device: torch.device,
    ):
        first = kv_caches[0]
        # (blocks, slots in a block, heads, 2 * head_dim): K then V of a token are adjacent, as the backend views it
        view = first.transpose(1, 2)
        if any(c.transpose(1, 2).stride() != view.stride() or c.dtype != first.dtype for c in kv_caches):
            raise ValueError("the target's full-attention layers must share one KV cache layout")
        if view.stride(3) != 1:
            raise ValueError("KV cache rows must be contiguous")
        self.strides = (view.stride(0), view.stride(2), view.stride(1))  # block, head, slot
        self.num_heads, self.dim, self.dtype = view.shape[2], view.shape[3], first.dtype
        self.block_size = view.shape[1]
        self.layer_ptrs = torch.tensor([c.data_ptr() for c in kv_caches], dtype=torch.uint64, device=device)
        self.table_ptrs = torch.tensor([t.data_ptr() for t in block_tables], dtype=torch.uint64, device=device)
        self.table_strides = torch.tensor([t.stride(0) for t in block_tables], dtype=torch.int64, device=device)
        self.table_block_sizes = torch.tensor(list(table_block_sizes), dtype=torch.int32, device=device)
        if any(t.dtype != torch.int32 for t in block_tables):
            raise ValueError("block tables must be int32")

    def compact(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        cu_num_logits: torch.Tensor,
        seq_lens: torch.Tensor,
        path: torch.Tensor,
        path_len: torch.Tensor,
    ) -> None:
        _compact_kv_kernel[(idx_mapping.shape[0], self.layer_ptrs.shape[0], self.num_heads)](
            self.layer_ptrs,
            self.table_ptrs,
            self.table_strides,
            self.table_block_sizes,
            idx_mapping,
            query_start_loc,
            cu_num_logits,
            seq_lens,
            path,
            path.stride(0),
            path_len,
            *self.strides,
            self.block_size,
            DIM=self.dim,
            DTYPE={torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[self.dtype],
            num_warps=2,
        )


@triton.jit
def _chainize_kernel(
    idx_mapping_ptr,
    query_start_loc_ptr,
    cu_num_logits_ptr,
    seq_lens_ptr,
    path_ptr,
    stride_path,
    path_len_ptr,
    sampled_ptr,  # [num_reqs, W] int64: the sampled tokens, in path order
    stride_sampled,
    input_ids_ptr,  # [tokens] in/out
    positions_ptr,  # [tokens] in/out
    src_rows_ptr,  # [tokens] out: the row whose hidden state each row takes (rows of a tree only)
    W: tl.constexpr,
    WP: tl.constexpr,
):
    b = tl.program_id(0)
    start = tl.load(query_start_loc_ptr + b)
    width = tl.load(query_start_loc_ptr + b + 1) - start
    rows = tl.arange(0, WP)
    in_query = rows < width
    if not _is_tree(query_start_loc_ptr, cu_num_logits_ptr, b):
        return
    slot = tl.load(idx_mapping_ptr + b)
    accepted = tl.minimum(tl.load(path_len_ptr + slot), width)
    context = tl.load(seq_lens_ptr + b) - width
    path = tl.load(path_ptr + slot * stride_path + rows, mask=rows < accepted, other=0)
    # Row j of the round is the j-th accepted row; rows past the path are padding the draft ignores.
    tl.store(src_rows_ptr + start + rows, start + tl.where(rows < accepted, path, rows), mask=in_query)
    tl.store(positions_ptr + start + rows, context + rows, mask=in_query)
    # Row j holds the token the draft is given with it: the sampled token j - 1 (row 0 keeps the root).
    sampled = tl.load(sampled_ptr + b * stride_sampled + rows - 1, mask=(rows >= 1) & (rows < accepted), other=0)
    tl.store(input_ids_ptr + start + rows, sampled, mask=(rows >= 1) & (rows < accepted))


def chainize(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    src_rows: torch.Tensor,
    *,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    cu_num_logits: torch.Tensor,
    seq_lens: torch.Tensor,
    path: torch.Tensor,
    path_len: torch.Tensor,
    sampled: torch.Tensor,
    width: int,
) -> None:
    """Rewrite the round's ``input_ids`` / ``positions`` in place for the draft's prefill and fill
    ``src_rows`` (preset to the identity): row ``i`` of the draft's input takes the hidden state of target row
    ``src_rows[i]``. Only the rows of a verified tree change."""
    _chainize_kernel[(idx_mapping.shape[0],)](
        idx_mapping,
        query_start_loc,
        cu_num_logits,
        seq_lens,
        path,
        path.stride(0),
        path_len,
        sampled,
        sampled.stride(0),
        input_ids,
        positions,
        src_rows,
        W=width,
        WP=triton.next_power_of_2(width),
        num_warps=1,
    )
