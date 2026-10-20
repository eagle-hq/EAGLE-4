# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The tree-tail mask variant of the batch-size-1 verify graphs (public FlashInfer jit_args), CPU only."""

import sys
import types
from unittest import mock

import pytest
import torch

pytest.importorskip("sglang")
from eagle_harness.sglang import tail_mask, verify_plan  # noqa: E402

BF16 = torch.bfloat16


@pytest.fixture
def fake_uri(monkeypatch):
    """flashinfer.jit.attention.get_batch_prefill_uri without FlashInfer installed."""
    module = types.ModuleType("flashinfer.jit.attention")
    module.get_batch_prefill_uri = lambda *args: "stock_uri"
    for name in ("flashinfer", "flashinfer.jit"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "flashinfer.jit.attention", module)
    tail_mask.jit_args.cache_clear()
    yield
    tail_mask.jit_args.cache_clear()


def test_variant_for_batch_size_1_trees_with_16_bit_dtypes(fake_uri):
    mask = verify_plan.TreeTailMask(torch.device("cpu"))
    variant = mask.variant(1, 16, BF16, BF16, 128)
    assert variant.mask_buf.dtype == torch.uint8 and variant.mask_buf.data_ptr() == mask.tail_bits.data_ptr()
    uri, *_, tensor_names, _, scalar_names, _, name, decl = variant.jit_args
    assert uri == "tree_tail_mask_stock_uri" and name == tail_mask.VARIANT_NAME and name in decl
    # FlashInfer reads the additional parameter names at jit_args[7] and [9] and fills them by name.
    assert variant.jit_args[7] is tensor_names and "maybe_custom_mask" in tensor_names
    assert variant.jit_args[9] is scalar_names and "sm_scale" in scalar_names
    for args in ((2, 16, BF16, BF16), (1, 1, BF16, BF16), (1, 65, BF16, BF16), (1, 16, BF16, torch.float8_e4m3fn)):
        assert mask.variant(*args, 128) is None
    assert mask.widths == {16}


def test_write_fills_only_the_variant_widths(fake_uri):
    mask = verify_plan.TreeTailMask(torch.device("cpu"))
    mask.variant(1, 8, BF16, BF16, 128)
    custom_mask, committed = torch.ones(8 * 20, dtype=torch.bool), torch.tensor([12])
    with mock.patch.object(verify_plan, "tree_tail_words") as words:
        mask.write(custom_mask, committed, 8)
        mask.write(custom_mask, committed, 4)  # captured without the variant
    words.assert_called_once_with(custom_mask, committed, mask.tail_bits, 8)
