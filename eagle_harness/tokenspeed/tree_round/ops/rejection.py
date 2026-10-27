# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Recursive tree rejection and residual-token sampling."""

import functools
from eagle_harness.tokenspeed.tree_round.ops.build import load_module

import torch

__all__ = ["residual_sample", "tree_reject_sample"]


@functools.cache
def _module():
    return load_module("rejection")


def tree_reject_sample(
    probas: torch.Tensor,
    tokens: torch.Tensor,
    parents: torch.Tensor,
    q_rows: torch.Tensor | None,
    q_slot: torch.Tensor,
    q_pos: torch.Tensor,
    q_store: torch.Tensor,
    qid_store: torch.Tensor,
    coins: torch.Tensor,
    top_ks: torch.Tensor,
    argmax: torch.Tensor,
    path: torch.Tensor,
    predicts: torch.Tensor,
    accept_length: torch.Tensor,
    last_row: torch.Tensor,
) -> None:
    """Accept each request's draft tree against the target by rejection sampling.

    ``probas`` (``[bs * N, V]`` fp32) holds the target's distribution at every
    node row; the rows of tried nodes become their residuals. A node's children
    (nodes whose ``parents`` entry it is, in index order) are tried in turn:
    sampled ones (``q_slot`` >= 0 at the request's ``q_rows`` row of the
    ``[R, N]`` tables, with q in ``q_store`` / ``qid_store`` ``[R, S, NC]``) by
    recursive rejection without replacement, the others by their target mass
    over the mass left; ``coins`` (``[bs, N]``) hold one uniform per child.
    ``q_rows`` None: every child is deterministic. A sampled row whose residual
    vanishes takes the target's argmax (``torch.argmax``'s) as its bonus.

    Greedy rows (``top_ks``, ``[bs * N]`` int32, equal to 1) have a one-hot
    target at ``argmax`` (``[bs * N]`` int64, read for them only): the first
    child holding it is accepted, the others are rejected, and their
    ``probas`` rows are never read (they may be unwritten).

    Writes ``path`` (``[bs, N]`` int32, accepted nodes root first, -1 past the
    path), ``predicts`` (``[bs * N]`` int32; the accepted children's tokens, the
    bonus slot left for ``residual_sample``), ``accept_length``
    (``[bs]``, accepted drafts plus the bonus) and ``last_row`` (``[bs]`` int64,
    the last accepted node's row of ``probas``).
    """
    _module().tree_reject_sample(
        probas,
        tokens,
        parents,
        q_rows,
        q_slot,
        q_pos,
        q_store,
        qid_store,
        coins,
        top_ks,
        argmax,
        path,
        predicts,
        accept_length,
        last_row,
    )


def residual_sample(
    probas: torch.Tensor,
    last_row: torch.Tensor,
    coins: torch.Tensor,
    top_ks: torch.Tensor,
    argmax: torch.Tensor,
    accept_length: torch.Tensor,
    predicts: torch.Tensor,
    num_nodes: int,
) -> None:
    """Draw each request's bonus token from row ``last_row[b]`` of ``probas``
    (normalized or not) by inverse CDF with ``coins[b]``, into the end of its
    accepted path in ``predicts`` (``[bs * num_nodes]``). A greedy last row
    (``top_ks`` 1, per ``probas`` row) gives its ``argmax`` (int64, per
    ``probas`` row) instead, its ``probas`` row unread."""
    _module().residual_sample(probas, last_row, coins, top_ks, argmax, accept_length, predicts, num_nodes)
