# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-4 efficient (low-rank) LM head: factors from the full LM head by SVD.

The draft scores the vocab with ``scores @ down_proj`` (rank r), takes the
top candidates and re-scores only those with the full head.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LowRankFactors:
    """Factors of a low-rank LM head: ``scores @ down_proj(hidden)`` approximates the full logits."""

    down_proj: torch.Tensor  # [rank, hidden]: weight of nn.Linear(hidden, rank)
    scores: torch.Tensor  # [vocab (shard), rank]
    energy_retention: float


def low_rank_scores(
    hidden: torch.Tensor, down_proj: torch.Tensor, score_head: torch.Tensor, vocab_rows: int
) -> torch.Tensor:
    """``[rows, shard vocab]`` rank-r scores of one vocab shard (``down_proj`` [rank, H], ``score_head`` [shard vocab,
    rank]), ``-inf`` past the shard's ``vocab_rows`` real rows."""
    return shard_scores(hidden @ down_proj.t(), score_head, vocab_rows)


def shard_scores(low_rank: torch.Tensor, score_head: torch.Tensor, vocab_rows: int) -> torch.Tensor:
    """:func:`low_rank_scores` from the rank-r projection ``low_rank`` [rows, rank] the caller computed."""
    scores = low_rank.to(score_head.dtype) @ score_head.t()
    if scores.shape[-1] > vocab_rows:
        scores[:, vocab_rows:] = float("-inf")
    return scores


def load_token_frequency(model_dir: str, vocab_size: int) -> torch.Tensor:
    """Per-token counts (``token_frequency.npy`` or raw int64 ``.bin``) as float32 [vocab]."""
    npy_path = os.path.join(model_dir, "token_frequency.npy")
    bin_path = os.path.join(model_dir, "token_frequency.bin")
    if os.path.exists(npy_path):
        freq, src = np.load(npy_path), npy_path
    elif os.path.exists(bin_path):
        freq, src = np.fromfile(bin_path, dtype=np.int64), bin_path
    else:
        raise FileNotFoundError(
            f"weighted_low_rank lm_head needs {npy_path} or {bin_path}: int64 per-token counts of a reference corpus."
        )
    freq = torch.from_numpy(np.ascontiguousarray(freq)).flatten()
    if freq.shape[0] != vocab_size:
        raise ValueError(f"token_frequency length {freq.shape[0]} != vocab_size {vocab_size} ({src}).")
    logger.info("Loaded token frequency from %s", src)
    return freq.to(dtype=torch.float32)


def low_rank_factors(
    weight: torch.Tensor,
    rank: int,
    *,
    token_frequency: Optional[torch.Tensor] = None,
    smooth_factor: float = 10,
    shard: Optional[tuple] = None,
) -> LowRankFactors:
    """SVD factors of an LM head (or its TP vocab shard).

    With ``token_frequency`` (full vocab) the SVD is frequency weighted:
    ``D^(1/2) W`` with ``D = freq / mean(freq) + smooth_factor``, so frequent
    tokens keep more energy; the row weighting is undone in ``scores`` so
    ``scores @ down_proj`` still approximates ``W``. ``shard`` is the
    ``(org_vocab_start, org_vocab_end)`` of a TP vocab shard; its padding rows
    get the smoothing floor.
    """
    w = weight.to(dtype=torch.float32, copy=True)
    sqrt_freq = None
    if token_frequency is not None:
        freq = token_frequency.to(device=w.device, dtype=torch.float32)
        freq = freq / freq.mean() + smooth_factor
        if freq.shape[0] != w.shape[0]:
            start, end = shard if shard is not None else (0, w.shape[0])
            rows = freq[start:end]
            freq = torch.full((w.shape[0],), float(smooth_factor), device=w.device, dtype=torch.float32)
            freq[: rows.shape[0]] = rows
        sqrt_freq = torch.sqrt(freq).unsqueeze(1)
        w = w * sqrt_freq
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    scores = u[:, :rank] * s[:rank]
    if sqrt_freq is not None:
        scores = scores / sqrt_freq
    energy = float((s[:rank] ** 2).sum() / (s**2).sum())
    logger.info("Low-rank LM head: rank=%d, energy retention=%.4f", rank, energy)
    return LowRankFactors(down_proj=vh[:rank, :], scores=scores, energy_retention=energy)
