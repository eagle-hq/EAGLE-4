# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The tail-mask word kernel against a torch reference (GPU)."""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from eagle_harness.kernels.verify_prep import tree_tail_words  # noqa: E402


def _signed(word: int) -> int:
    return word - (1 << 64) if word >= 1 << 63 else word


@pytest.mark.parametrize("width,kv_len", [(1, 7), (5, 40), (16, 3000), (64, 5000), (16, 70000)])
def test_matches_reference(width, kv_len):
    torch.manual_seed(width)
    mask = torch.randint(0, 2, (width * kv_len,), dtype=torch.uint8, device="cuda").bool()
    committed_len = torch.tensor([kv_len - width], dtype=torch.int64, device="cuda")
    tail_bits = torch.full((width + 1,), 7, dtype=torch.int64, device="cuda")

    tree_tail_words(mask, committed_len, tail_bits, width)

    rows = mask.view(width, kv_len)[:, kv_len - width :].int().cpu().tolist()
    expected = [_signed(sum(bit << j for j, bit in enumerate(row))) for row in rows]
    assert tail_bits[:width].tolist() == expected
    assert tail_bits[width].item() == 7  # no word past the last row


def test_width_over_a_word_is_rejected():
    with pytest.raises(ValueError, match="verify width 65"):
        tree_tail_words(None, None, None, width=65)
