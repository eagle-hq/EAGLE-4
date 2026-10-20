# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Index math and KV writes of the draft steps, one launch each (they replay inside the growth graph).

:func:`draft_step_slots` gives every step's draft KV slots and positions at once (was ~16 small torch kernels per
round); :func:`store_rows_kv` writes a step's keys and values into the draft KV cache (was a ``torch.cat`` and an
indexed store, ~10 us per step at TP4); :func:`stack_layer_input` puts a step's token embeddings and input hidden states side by
side for the draft layer's two input norms (one norm launch then writes the qkv projection's concatenated input).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _draft_step_slots_kernel(
    block_table,
    committed,
    slots,
    blocks,
    offsets,
    positions,
    stride_table,
    num_reqs,
    table_block,
    cache_block,
    STEPS: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    i = tl.arange(0, BLOCK)  # step-major: (step - 1) * K + lane
    ok = i < STEPS * K
    step, lane = i // K, i % K
    base = tl.load(committed + req)
    scratch = base + i
    block = tl.load(block_table + req * stride_table + scratch // table_block, mask=ok, other=0).to(tl.int64)
    slot = block * table_block + scratch % table_block
    tl.store(slots + req * STEPS * K + i, slot, mask=ok)
    row = step * (num_reqs * K) + req * K + lane  # a step's rows: request-major, then lane
    tl.store(blocks + row, slot // cache_block, mask=ok)
    tl.store(offsets + row, slot % cache_block, mask=ok)
    tl.store(positions + row, base + step, mask=ok)


def draft_step_slots(
    block_table: torch.Tensor, committed: torch.Tensor, *, steps: int, top_k: int, table_block: int, cache_block: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step ``s`` (1-based) writes lane ``j`` of request ``b`` at token ``committed[b] + (s - 1) * K + j`` and runs it
    at position ``committed[b] + s - 1``. Returns ``slots [bs, steps, K]`` (flat token slots of the block table),
    ``blocks`` / ``offsets`` ``[steps, bs * K]`` (in the cache's own blocks) and ``positions [steps, bs * K]``, int64."""
    bs = committed.shape[0]
    slots = torch.empty(bs, steps, top_k, dtype=torch.int64, device=committed.device)
    blocks = torch.empty(steps, bs * top_k, dtype=torch.int64, device=committed.device)
    offsets, positions = torch.empty_like(blocks), torch.empty_like(blocks)
    _draft_step_slots_kernel[(bs,)](
        block_table,
        committed,
        slots,
        blocks,
        offsets,
        positions,
        block_table.stride(0),
        bs,
        table_block,
        cache_block,
        STEPS=steps,
        K=top_k,
        BLOCK=triton.next_power_of_2(steps * top_k),
    )
    return slots, blocks, offsets, positions


@triton.jit
def _store_rows_kv_kernel(
    cache,
    key,
    value,
    blocks,
    offsets,
    stride_block,
    stride_slot,
    stride_head,
    stride_key,
    stride_value,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    ok = d < D
    dst = cache + tl.load(blocks + row) * stride_block + tl.load(offsets + row) * stride_slot + head * stride_head
    tl.store(dst + d, tl.load(key + row * stride_key + head * D + d, mask=ok), mask=ok)
    tl.store(dst + D + d, tl.load(value + row * stride_value + head * D + d, mask=ok), mask=ok)


def store_rows_kv(
    cache: torch.Tensor, key: torch.Tensor, value: torch.Tensor, blocks: torch.Tensor, offsets: torch.Tensor
) -> None:
    """``cache[blocks, offsets] = cat(key, value)`` per head: ``cache`` ``[blocks, slots, heads, 2 * D]`` (any strides
    with a unit last one), ``key`` / ``value`` ``[rows, heads * D]`` rows (unit last stride)."""
    rows = key.shape[0]
    heads, dim = cache.shape[2], cache.shape[3] // 2
    assert cache.stride(3) == 1 and key.stride(1) == 1 and value.stride(1) == 1
    _store_rows_kv_kernel[(rows, heads)](
        cache,
        key,
        value,
        blocks,
        offsets,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        key.stride(0),
        value.stride(0),
        D=dim,
        BLOCK_D=triton.next_power_of_2(dim),
    )


@triton.jit
def _stack_layer_input_kernel(table, ids, hidden, out, stride_table, stride_hidden, H, BLOCK: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    ok = cols < H
    token = tl.load(ids + row).to(tl.int64)
    tl.store(out + row * 2 * H + cols, tl.load(table + token * stride_table + cols, mask=ok), mask=ok)
    tl.store(out + row * 2 * H + H + cols, tl.load(hidden + row * stride_hidden + cols, mask=ok), mask=ok)


def stack_layer_input(table: torch.Tensor, ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """``[rows, 2, H]``: row ``r`` holds ``table[ids[r]]`` (the full, unsharded embedding table) then ``hidden[r]``.

    Replaces the vocab-parallel embedding (a masked lookup and a TP all-reduce of the shards' rows, which adds zeros:
    the same values) and lets one RMSNorm launch with a per-row weight normalise both halves into ``cat`` layout."""
    rows, hidden_size = hidden.shape
    out = torch.empty(rows, 2, hidden_size, dtype=hidden.dtype, device=hidden.device)
    block = 1024
    _stack_layer_input_kernel[(rows, triton.cdiv(hidden_size, block))](
        table, ids, hidden, out, table.stride(0), hidden.stride(0), hidden_size, BLOCK=block
    )
    return out
