# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
"""Ordered proposals and recursive rejection without replacement.

Proposal probabilities are recorded before drawing; after each rejection q
is conditioned on the earlier draws. Subtract the *whole* conditional q from
p, not just the rejected token. This implements the probability recursion of
Recursive Speculative Decoding (Jeon et al., arXiv:2402.14160), independently
of any engine implementation.
"""

from typing import NamedTuple, Optional

import torch


class ChildProposals(NamedTuple):
    """One expanded parent's sparse proposal distribution and ordered draws.

    ``scores`` are descending order statistics of q, NOT the probabilities of
    the sampled tokens. A planner may use them without inspecting a child's
    own draw. Selection must retain a prefix of each parent's draw order.
    Candidate ids must be distinct, valid vocabulary ids; zero q is allowed.
    """

    probs: torch.Tensor  # [rows, candidates]
    token_ids: torch.Tensor  # [rows, candidates]
    positions: torch.Tensor  # [rows, children], indices into the candidate row
    tokens: torch.Tensor  # [rows, children], global vocabulary ids
    scores: torch.Tensor  # [rows, children], sample-independent rank scores


class RejectionResult(NamedTuple):
    tokens: torch.Tensor  # [rows], accepted child or residual draw
    accepted_position: torch.Tensor  # [rows], draw index, or -1 for a residual draw


def filter_proposal_logits(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    """Apply the request's temperature and filter settings to the draft candidate logits.

    Top-k and top-p cut the unfiltered draft probabilities, and min-p drops what is below ``min_p`` times the
    most likely. Removed candidates get ``-inf`` (q = 0). The draft and target can retain different tokens under
    the same settings. Recursive rejection uses the recorded q and the target p, including their differing support.
    Per-row settings are ``[rows]`` tensors; ties rank by candidate index.
    """
    scaled = logits.float() / temperature.float()[:, None]
    probs = torch.softmax(scaled, dim=-1)
    order = torch.sort(-probs, dim=-1, stable=True).indices
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(logits.shape[1], device=logits.device).expand_as(order).contiguous())
    above = (probs.gather(1, order).cumsum(dim=-1) - probs.gather(1, order)).gather(1, rank)
    keep = (rank < top_k[:, None]) & (above < top_p[:, None])
    keep &= probs >= min_p[:, None] * probs.amax(dim=-1, keepdim=True)
    return scaled.masked_fill(~keep, float("-inf"))


def draw_children(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    count: int,
    *,
    generator: Optional[torch.Generator] = None,
    uniforms: Optional[torch.Tensor] = None,
) -> ChildProposals:
    """Gumbel-top-k ordered draws from finite, nonempty candidate distributions.

    ``logits`` already include the chosen proposal temperature/filters. The
    target need not have the same support or distribution. Slots after the
    nonzero support are inert (score zero), never valid proposals.
    """
    if logits.ndim != 2 or token_ids.shape != logits.shape or not 1 <= count <= logits.shape[1]:
        raise ValueError("expected matching [rows, candidates] logits/ids and count within the candidate width")
    probs = torch.softmax(logits.float(), dim=-1)
    if uniforms is None:
        uniforms = torch.rand(probs.shape, dtype=probs.dtype, device=probs.device, generator=generator)
    # rand is [0, 1); clamping only its possible zero avoids log(0).
    keys = probs.log() - (-uniforms.clamp_min(torch.finfo(probs.dtype).tiny).log()).log()
    positions = keys.topk(count, dim=-1).indices
    return ChildProposals(probs, token_ids, positions, token_ids.gather(1, positions), probs.topk(count, dim=-1).values)


def reject_children(
    target_probs: torch.Tensor,
    proposals: ChildProposals,
    uniforms: torch.Tensor,
    final_uniforms: torch.Tensor,
) -> RejectionResult:
    """Reference batched recursive rejection for one parent's retained prefix.

    Target rows are normalized, and uniforms are independent of the proposals
    and each other, in [0, 1). Float64 arithmetic makes this a numerical oracle
    for the GPU implementation. It is not the serving implementation.
    """
    rows, vocab = target_probs.shape
    count = proposals.positions.shape[1]
    if uniforms.shape != (rows, count) or final_uniforms.shape != (rows,):
        raise ValueError("accept coins must match the draws, with one final coin per target row")
    residual = target_probs.double().clone()
    q = proposals.probs.double().clone()
    chosen = torch.full((rows,), -1, dtype=torch.int64, device=target_probs.device)
    tokens = torch.zeros(rows, dtype=torch.int64, device=target_probs.device)
    for child in range(count):
        index = proposals.positions[:, child : child + 1]
        token = proposals.tokens[:, child : child + 1]
        mass = residual.sum(dim=-1, keepdim=True)
        q_mass = q.sum(dim=-1, keepdim=True)
        conditional = q / q_mass.clamp_min(torch.finfo(q.dtype).tiny)
        q_y = conditional.gather(1, index).flatten()
        p_y = residual.gather(1, token).flatten()
        accept = (chosen == -1) & (q_y > 0) & (uniforms[:, child] * mass.flatten() * q_y < p_y)
        chosen = torch.where(accept, child, chosen)
        tokens = torch.where(accept, token.flatten(), tokens)
        full_q = torch.zeros((rows, vocab), dtype=residual.dtype, device=residual.device)
        full_q.scatter_(1, proposals.token_ids, conditional)
        updated = (residual - mass * full_q).clamp_min(0)
        residual = torch.where((chosen == -1)[:, None], updated, residual)
        q.scatter_(1, index, 0)
    cdf = residual.cumsum(dim=-1)
    threshold = final_uniforms.double() * cdf[:, -1]
    bonus = torch.searchsorted(cdf.contiguous(), threshold[:, None].contiguous(), right=True).flatten()
    # Accepted rows do not consume a residual draw (it can be all zero there).
    return RejectionResult(torch.where(chosen == -1, bonus.clamp_max(vocab - 1), tokens), chosen)
