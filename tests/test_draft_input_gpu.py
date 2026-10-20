# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The draft embedding on CUDA: sglang's Triton lookup gives the same bits as ``F.embedding`` (GPU)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from types import SimpleNamespace  # noqa: E402

from eagle_harness.sglang.models.llama_eagle4 import Eagle4Model  # noqa: E402


@pytest.mark.parametrize("rows", [1, 4, 16, 300])
def test_triton_embedding_is_the_table_rows(rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    table = torch.randn(1000, 5120, device="cuda", generator=g).bfloat16()
    ids = torch.randint(0, 1000, (rows,), device="cuda", generator=g)
    model = SimpleNamespace(embed_full=table, embed_tokens=None, triton_embed=True)
    assert torch.equal(Eagle4Model._embed(model, ids), torch.nn.functional.embedding(ids, table))
