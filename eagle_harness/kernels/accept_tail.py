# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The greedy tree acceptance and everything that follows it, for one request, in two launches.

After a tree verify sglang runs about thirty tiny kernels (tree walk, accept
lengths, bonus token, KV slot lists, compaction of the accepted path to the
front of the block), launch-bound at a few microseconds each. :func:`accept_tail`
is the walk and all of the index and token bookkeeping in one program;
:func:`compact_rows` moves the accepted rows of a node-indexed tensor (the
verify's hidden states) to the front. Both reproduce sglang's results exactly
(``verify_tree_greedy``, ``fill_bonus_tokens``, ``assign_extend_cache_locs``,
``fill_accept_out_cache_loc`` and ``_compact_accept_to_front``).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl


class Accepted(NamedTuple):
    """What a greedy tree verify of one request accepted."""

    predict: torch.Tensor  # int32 [width]: target token of each accepted node, the accepted path moved to the front
    accept_index: torch.Tensor  # int32 [1, depth]: node of each accepted token, -1 padded
    accept_lens: torch.Tensor  # int32 [1]: accepted drafts + the bonus token
    new_seq_lens: torch.Tensor  # [1], seq_lens + accept_lens
    bonus_tokens: torch.Tensor  # int32 [1]: the last accepted token
    committed_slots: torch.Tensor  # int64 [depth]: KV slots the accepted tokens move to (0 past accept_lens)
    accepted_slots: torch.Tensor  # int64 [depth]: KV slots they were written to by the verify (0 past accept_lens)


@triton.jit
def _accept_tail_kernel(
    target_ptr,
    candidates_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    seq_lens_ptr,
    out_cache_loc_ptr,
    req_to_token_ptr,
    req_stride,
    req_pool_idx_ptr,
    predict_ptr,
    accept_index_ptr,
    accept_lens_ptr,
    new_seq_lens_ptr,
    bonus_ptr,
    committed_slots_ptr,
    accepted_slots_ptr,
    WIDTH: tl.constexpr,
    DEPTH: tl.constexpr,
    WIDTH_P2: tl.constexpr,
    DEPTH_P2: tl.constexpr,
):
    ds = tl.arange(0, DEPTH_P2)
    tl.store(accept_index_ptr + ds, -1, mask=ds < DEPTH)
    tl.debug_barrier()

    # The tree walk: at each level take the first child of the last accepted node whose token is the
    # target's prediction at that node (children are siblings of the first child).
    last = tl.load(retrieve_index_ptr)
    tl.store(accept_index_ptr, last.to(tl.int32))
    cur = last * 0
    num = last * 0
    active = last * 0 + 1
    for level in range(1, DEPTH):
        if active != 0:
            cur = tl.load(retrieve_next_token_ptr + cur)
            target_token = tl.load(target_ptr + last)
            found = last * 0
            for sibling in range(WIDTH):
                if found == 0:
                    valid = cur != -1
                    safe = tl.where(valid, cur, 0)
                    node = tl.load(retrieve_index_ptr + safe)
                    match = valid & (tl.load(candidates_ptr + safe) == target_token)
                    tl.store(accept_index_ptr + num + 1, node.to(tl.int32), mask=match)
                    num = num + match.to(tl.int64)
                    last = tl.where(match, node, last)
                    cur = tl.where(valid & ~match, tl.load(retrieve_next_sibling_ptr + safe), cur)
                    found = tl.where(match, 1, tl.where(valid, 0, -1)).to(tl.int64)
            active = (found == 1).to(tl.int64)
    tl.debug_barrier()

    # Accept lengths, the bonus token, the new length.
    seq_len = tl.load(seq_lens_ptr)
    tl.store(accept_lens_ptr, (num + 1).to(tl.int32))
    tl.store(new_seq_lens_ptr, (seq_len + num + 1).to(new_seq_lens_ptr.dtype.element_ty))
    tl.store(bonus_ptr, tl.load(target_ptr + last).to(tl.int32))

    # The predictions, the accepted path moved to the front (padding: node 0, which is always accepted).
    nodes = tl.arange(0, WIDTH_P2)
    head = nodes < DEPTH
    path = tl.load(accept_index_ptr + nodes, mask=head, other=-1).to(tl.int64)
    head_value = tl.load(target_ptr + tl.maximum(path, 0), mask=head, other=0)
    accepted = tl.load(accept_index_ptr + ds, mask=(ds <= num) & (ds < DEPTH), other=-1).to(tl.int64)
    on_path = tl.sum((nodes[:, None] == accepted[None, :]).to(tl.int32), axis=1) > 0
    tail_value = tl.where(on_path, tl.load(target_ptr + nodes, mask=nodes < WIDTH, other=0), 0)
    tl.store(predict_ptr + nodes, tl.where(head, head_value, tail_value).to(tl.int32), mask=nodes < WIDTH)

    # The KV slots to move: the request's next free slots, and where the verify wrote the accepted tokens.
    row = tl.load(req_pool_idx_ptr).to(tl.int64)
    keep = (ds <= num) & (ds < DEPTH)
    path = tl.load(accept_index_ptr + ds, mask=ds < DEPTH, other=-1).to(tl.int64)
    committed = tl.load(req_to_token_ptr + row * req_stride + seq_len + ds, mask=keep, other=0)
    tl.store(committed_slots_ptr + ds, committed.to(tl.int64), mask=ds < DEPTH)
    accepted_slot = tl.load(out_cache_loc_ptr + path, mask=keep, other=0)
    tl.store(accepted_slots_ptr + ds, accepted_slot.to(tl.int64), mask=ds < DEPTH)


def accept_tail(
    target_predict: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    depth: int,
) -> Accepted:
    """Greedy-accept one request's verify tree against ``target_predict`` (the target's argmax per node).

    ``depth`` is the width of ``accept_index`` (the verify input's ``max_tree_depth``). Tables are the
    verify input's; ``seq_lens``, ``out_cache_loc`` and ``req_pool_indices`` the batch's.
    """
    width = target_predict.numel()
    device = target_predict.device
    out = Accepted(
        torch.empty(width, dtype=torch.int32, device=device),
        torch.empty((1, depth), dtype=torch.int32, device=device),
        torch.empty(1, dtype=torch.int32, device=device),
        torch.empty(1, dtype=seq_lens.dtype, device=device),
        torch.empty(1, dtype=torch.int32, device=device),
        torch.empty(depth, dtype=torch.int64, device=device),
        torch.empty(depth, dtype=torch.int64, device=device),
    )
    _accept_tail_kernel[(1,)](
        target_predict,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        seq_lens,
        out_cache_loc,
        req_to_token,
        req_to_token.stride(0),
        req_pool_indices,
        out.predict,
        out.accept_index,
        out.accept_lens,
        out.new_seq_lens,
        out.bonus_tokens,
        out.committed_slots,
        out.accepted_slots,
        WIDTH=width,
        DEPTH=depth,
        WIDTH_P2=triton.next_power_of_2(width),
        DEPTH_P2=triton.next_power_of_2(depth),
        num_warps=1,
    )
    return out


@triton.jit
def _compact_rows_kernel(
    src_ptr,
    dst_ptr,
    accept_index_ptr,
    src_stride,
    dst_stride,
    cols,
    DEPTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    path = tl.load(accept_index_ptr + tl.minimum(row, DEPTH - 1)).to(tl.int64)
    source = tl.where(row < DEPTH, tl.maximum(path, 0), row)
    values = tl.load(src_ptr + source * src_stride + offsets, mask=offsets < cols)
    tl.store(dst_ptr + row.to(tl.int64) * dst_stride + offsets, values, mask=offsets < cols)


def compact_rows(
    rows: torch.Tensor, accept_index: torch.Tensor, block: int = 2048, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """A copy of the node-indexed ``rows`` [width, cols] whose first ``accept_index.shape[1]`` rows are the
    accepted path (padding: row 0), the other rows unchanged; into ``out`` (same shape, not ``rows``) if given."""
    width, cols = rows.shape
    if rows.stride(1) != 1:
        raise ValueError("compact_rows needs rows with a unit column stride")
    if out is None:
        out = torch.empty_like(rows, memory_format=torch.contiguous_format)
    elif out.shape != rows.shape or out.stride(1) != 1 or out.data_ptr() == rows.data_ptr():
        raise ValueError("compact_rows needs an out of the rows' shape, unit column stride, not the rows")
    _compact_rows_kernel[(width, triton.cdiv(cols, block))](
        rows, out, accept_index, rows.stride(0), out.stride(0), cols, DEPTH=accept_index.shape[1], BLOCK=block
    )
    return out
