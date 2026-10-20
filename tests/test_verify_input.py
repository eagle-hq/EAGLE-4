# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""HarnessVerifyInput: sglang's verify input plus the greedy flag and tail-mask variant (CPU)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang")
from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind  # noqa: E402
from sglang.srt.speculative.eagle_info import EagleVerifyInput  # noqa: E402

from eagle_harness.sglang.verify_input import HarnessVerifyInput  # noqa: E402


def _eagle():
    return EagleVerifyInput.create_idle_input(topk=4, spec_steps=3, num_verify_tokens=8, device="cpu")


def test_from_eagle_keeps_sglangs_fields():
    eagle = _eagle()
    verify = HarnessVerifyInput.from_eagle(eagle, greedy_tokens_only=True)
    assert verify.custom_mask is eagle.custom_mask and verify.draft_token_num == 8
    assert verify.greedy_tokens_only and verify.tail_mask is None
    assert not HarnessVerifyInput.from_eagle(eagle).greedy_tokens_only  # default: logits


def test_variant_comes_from_the_tail_mask():
    dims = dict(q_dtype=torch.bfloat16, kv_dtype=torch.bfloat16, head_dim=256)
    assert HarnessVerifyInput.from_eagle(_eagle()).flashinfer_verify_variant(1, 8, **dims) is None
    asked = []
    tail_mask = SimpleNamespace(variant=lambda *args: asked.append(args) or "variant")
    verify = HarnessVerifyInput.from_eagle(_eagle(), tail_mask=tail_mask)
    assert verify.flashinfer_verify_variant(1, 8, **dims) == "variant"
    assert asked == [(1, 8, torch.bfloat16, torch.bfloat16, 256)]


class _Translator:
    """Counts sglang's own CSR builds (``EagleVerifyInput.generate_attn_arg_prefill`` packs through it)."""

    def __init__(self, reads_are_translated: bool):
        self.reads_are_translated = reads_are_translated
        self.packs = 0

    def pack_read_stream(self, plan, **kwargs) -> bool:
        self.packs += 1
        return False


def _args(verify, translator, bs=1, kind=IdSpaceKind.FULL, committed=37):
    return verify.generate_attn_arg_prefill(
        req_pool_indices=torch.arange(bs),
        paged_kernel_lens=torch.full((bs,), committed),
        paged_kernel_lens_sum=committed * bs,
        translator=translator,
        plan=None,
        kind=kind,
    )


def _prepared(width=8, committed=37):
    return (
        torch.arange(64, dtype=torch.int32),
        torch.tensor([0, committed + width], dtype=torch.int32),
        torch.tensor([0, width], dtype=torch.int32),
    )


def test_prepared_attn_args_serve_a_one_request_untranslated_verify():
    """The draft graph's CSR args replace sglang's per-replay build (no pack), kv_indices cut to kv_len."""
    verify = HarnessVerifyInput.from_eagle(_eagle())
    verify.prepared_attn_args = prepared = _prepared()
    translator = _Translator(reads_are_translated=False)
    kv_indices, kv_indptr, qo_indptr, mask = _args(verify, translator)
    assert translator.packs == 0
    assert kv_indices.data_ptr() == prepared[0].data_ptr() and kv_indices.numel() == 37 + 8
    assert kv_indptr is prepared[1] and qo_indptr is prepared[2] and mask is verify.custom_mask


@pytest.mark.parametrize(
    "translated, kind, bs",
    [(True, IdSpaceKind.FULL, 1), (False, IdSpaceKind.SLIDING_WINDOW, 1), (False, IdSpaceKind.FULL, 2)],
)
def test_other_reads_build_sglangs_attn_args(translated, kind, bs):
    """Translated ids, another id space or more requests than the prepared args cover: sglang builds them."""
    verify = HarnessVerifyInput.from_eagle(_eagle())
    verify.prepared_attn_args = _prepared()
    translator = _Translator(reads_are_translated=translated)
    _args(verify, translator, bs=bs, kind=kind)
    assert translator.packs == 1
