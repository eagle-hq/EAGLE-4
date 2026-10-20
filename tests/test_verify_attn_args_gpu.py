# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The draft graph's CSR verify args equal sglang's per-replay build (GPU)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA (Triton kernel)", allow_module_level=True)

from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator  # noqa: E402
from sglang.srt.speculative.eagle_info import EagleVerifyInput  # noqa: E402

from eagle_harness.sglang.verify_input import build_attn_args  # noqa: E402


@pytest.mark.parametrize("committed, width", [(1, 2), (37, 8), (3000, 16)])
def test_built_args_equal_sglangs(committed, width):
    gen = torch.Generator().manual_seed(committed + width)
    rows, row_len, row = 4, 4096, 2
    req_to_token = torch.randperm(rows * row_len, generator=gen).to(torch.int32).view(rows, row_len).cuda()
    req_pool_indices = torch.tensor([row], device="cuda")
    seq_lens = torch.tensor([committed], device="cuda")
    translator = KVIndexTranslator(
        req_to_token=req_to_token,
        token_to_kv_pool_allocator=SimpleNamespace(),
        token_to_kv_pool=SimpleNamespace(),
        page_size=1,
        device="cuda",
    )
    assert not translator.reads_are_translated
    reference = EagleVerifyInput.create_idle_input(topk=4, spec_steps=3, num_verify_tokens=width, device="cuda")
    ref_indices, ref_indptr, ref_qo_indptr, _ = reference.generate_attn_arg_prefill(
        req_pool_indices=req_pool_indices,
        paged_kernel_lens=seq_lens,
        paged_kernel_lens_sum=committed,
        translator=translator,
        plan=None,
    )
    buffer = torch.full((row_len,), -1, dtype=torch.int32, device="cuda")
    kv_indices, kv_indptr, qo_indptr = build_attn_args(req_to_token, req_pool_indices, seq_lens, width, buffer)
    torch.testing.assert_close(kv_indices[: committed + width], ref_indices, rtol=0, atol=0)
    torch.testing.assert_close(kv_indptr, ref_indptr, rtol=0, atol=0)
    torch.testing.assert_close(qo_indptr, ref_qo_indptr, rtol=0, atol=0)
    assert kv_indptr.dtype == ref_indptr.dtype and qo_indptr.dtype == ref_qo_indptr.dtype
