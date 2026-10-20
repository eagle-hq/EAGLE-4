# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Attention of the draft tree's steps on vLLM.

Draft step ``s`` runs ``K`` frontier rows per request, each at position ``c + s - 1`` (``c``: the committed
length), writing its draft KV to a slot past ``c`` and attending to

- the request's committed draft KV inside the draft's sliding window (the same keys for all rows), and
- the row's own ``s`` tree ancestors, itself last, whose slots :attr:`DraftTree.ancestors` keeps.

:func:`tree_step_attention` reads those keys through the block table and loads each committed key once for all the
rows of a request (:mod:`eagle_harness.kernels.row_attention`, the kernel sglang's draft uses).
"""

from __future__ import annotations

import torch

from eagle_harness.kernels.row_attention import RowsLaunch, program_splits, rows_attention, rows_padded


# Keys per iteration of :func:`tree_step_attention` (a [rows, 128] x [128, D] tile per step).
STEP_BLOCK = 128


def tree_step_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    table_block: int,
    committed: torch.Tensor,
    ancestors: torch.Tensor,
    scale: float,
    *,
    step: int,
    window: int,
) -> torch.Tensor:
    """Attention of draft step ``step``'s ``bs * K`` rows ``q`` ``[bs * K, H, D]`` (no key lists).

    ``key_cache``: the layer's KV cache viewed ``[blocks, slots, heads, 2 * D]``; ``block_table[b]`` maps request
    ``b``'s positions to blocks of ``table_block`` tokens; ``ancestors`` ``[bs, K, depth]`` holds flat token slots.
    The window is cut into about one program per SM in all.
    """
    _, heads, head_dim = q.shape
    bs, top_k, _ = ancestors.shape
    launch = RowsLaunch(splits=program_splits(bs * heads, key_cache.device), block_n=STEP_BLOCK)
    rows_p = rows_padded(top_k)
    part = torch.empty(bs, heads, launch.splits + 1, rows_p, head_dim, dtype=torch.float32, device=q.device)
    part_lse = torch.empty(bs, heads, launch.splits + 1, rows_p, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    rows_attention(
        q,
        key_cache,
        None,
        out,
        part,
        part_lse,
        block_table,
        committed,
        ancestors,
        rows_per_request=top_k,
        step=step,
        window=window,
        scale=scale,
        launch=launch,
        table_page=table_block,
    )
    return out
