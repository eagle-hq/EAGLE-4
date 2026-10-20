# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The harness's EAGLE verify input: sglang's, plus the tree-tail mask attention of its batch-size-1 verify
graphs and the CSR attention args the draft's verify-input graph builds for them.

With :attr:`HarnessVerifyInput.greedy_tokens_only` set (a greedy round), the target forward returns the
verify rows' argmax (``LogitsProcessorOutput.next_token_ids``) instead of gathering their logits; the
target-verify graphs of a greedy batch-size-1 round are captured with such an input
(:class:`.verify_graphs.VerifyWidthGraphs`).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind
from sglang.srt.speculative.eagle_info import EagleVerifyInput

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
    from sglang.srt.mem_cache.kv_loc_plan import KVLocPlan
    from sglang.srt.speculative.spec_info import FlashInferVerifyVariant

    from eagle_harness.sglang.verify_plan import TreeTailMask


def build_attn_args(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    width: int,
    kv_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """sglang's CSR args of a one-request verify of ``width`` tokens over untranslated KV ids
    (``EagleVerifyInput.generate_attn_arg_prefill``), from device lengths, so the launches can sit in a
    CUDA graph: ``(kv_indices, kv_indptr, qo_indptr)``, kv_indices written into ``kv_indices`` (room for
    the request's row)."""
    kv_lens = seq_lens[:1] + width
    kv_indptr = torch.zeros(2, dtype=torch.int32, device=kv_indices.device)
    kv_indptr[1:] = kv_lens
    create_flashinfer_kv_indices_triton[(1,)](
        req_to_token, req_pool_indices, kv_lens, kv_indptr, None, kv_indices, req_to_token.stride(0)
    )
    qo_indptr = torch.arange(0, 2 * width, width, dtype=torch.int32, device=kv_indices.device)
    return kv_indices, kv_indptr, qo_indptr


@dataclasses.dataclass
class HarnessVerifyInput(EagleVerifyInput):
    """``EagleVerifyInput`` with the greedy-ids flag and the tail mask."""

    @classmethod
    def from_eagle(cls, verify_input: EagleVerifyInput, **extra):
        """This class's verify input with the fields of sglang's (and ``extra`` fields of this class)."""
        fields = (f.name for f in dataclasses.fields(EagleVerifyInput) if f.init)
        return cls(**{name: getattr(verify_input, name) for name in fields}, **extra)

    # A greedy round: the target returns the rows' argmax, not their logits.
    greedy_tokens_only: bool = False
    # The tail-mask attention of the batch-size-1 verify graphs (set on the inputs they are captured with).
    tail_mask: Optional["TreeTailMask"] = dataclasses.field(default=None, repr=False)
    # (kv_indices, kv_indptr, qo_indptr) of a one-request verify over untranslated KV ids, built with the rest
    # of the input in the draft's verify-input graph; kv_indices has room for the request's whole row.
    prepared_attn_args: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = dataclasses.field(
        default=None, repr=False
    )

    def generate_attn_arg_prefill(
        self,
        *,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        translator: "KVIndexTranslator",
        plan: "KVLocPlan",
        kind: IdSpaceKind = IdSpaceKind.FULL,
    ):
        """sglang's CSR verify args; the prepared ones when this input carries them for these reads (one
        request, ids read untranslated), so the replay launches nothing for them."""
        args = self.prepared_attn_args
        if (
            args is None
            or req_pool_indices.numel() != 1
            or kind is not IdSpaceKind.FULL
            or translator.reads_are_translated
        ):
            return super().generate_attn_arg_prefill(
                req_pool_indices=req_pool_indices,
                paged_kernel_lens=paged_kernel_lens,
                paged_kernel_lens_sum=paged_kernel_lens_sum,
                translator=translator,
                plan=plan,
                kind=kind,
            )
        kv_indices, kv_indptr, qo_indptr = args
        return kv_indices[: paged_kernel_lens_sum + self.draft_token_num], kv_indptr, qo_indptr, self.custom_mask

    def flashinfer_verify_variant(
        self, bs: int, num_tokens: int, *, q_dtype: torch.dtype, kv_dtype: torch.dtype, head_dim: int
    ) -> Optional["FlashInferVerifyVariant"]:
        """The tail-mask variant of a batch-size-1 verify graph under capture, else None."""
        if self.tail_mask is None:
            return None
        return self.tail_mask.variant(bs, num_tokens, q_dtype, kv_dtype, head_dim)
