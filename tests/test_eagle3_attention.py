# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Grouped-query EAGLE3 draft attention in the cost-model worker."""

import pytest
import torch
from tests.triton_probe import needs_triton

pytest.importorskip("sglang")


@needs_triton
@pytest.mark.parametrize("step", [0, 2])
@pytest.mark.parametrize("window", [0, 8])
def test_native_grouped_query_draft_rows(step, window):
    from eagle_harness.sglang.row_attention import EXTEND_LAUNCH, MAX_SPLITS, STEP_LAUNCH, rows_attention

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    gen = torch.Generator().manual_seed(732)
    q = torch.randn(4, 4, 64, generator=gen).to(device=device, dtype=dtype)
    k, v = [torch.randn(64, 1, 64, generator=gen).to(device=device, dtype=dtype) for _ in range(2)]
    table = torch.arange(64, dtype=torch.int32, device=device)[None]
    req = torch.zeros(1, dtype=torch.int64, device=device)
    lengths = torch.tensor([16], device=device)
    ancestors = torch.arange(32, 40, device=device, dtype=torch.int32).view(1, 4, 2)
    out = torch.empty_like(q)
    parts = 4 * (MAX_SPLITS + 1) * 16
    partial = torch.empty(parts, 64, dtype=torch.float32, device=device)
    lse = torch.empty(parts, dtype=torch.float32, device=device)
    rows_attention(
        q,
        k,
        v,
        out,
        partial,
        lse,
        table,
        lengths,
        ancestors,
        rows_per_request=4,
        step=step,
        window=window,
        scale=0.125,
        launch=EXTEND_LAUNCH if step == 0 else STEP_LAUNCH,
        table_rows=req,
    )
    for row in range(4):
        position = 16 + (row if step == 0 else step - 1)
        start = max(0, position - window + 1) if window else 0
        keys = list(range(start, position + 1 if step == 0 else 16))
        if step:
            keys += ancestors[0, row].tolist()
        kr, vr = k[keys, 0].float(), v[keys, 0].float()
        want = (q[row].float() @ kr.T * 0.125).softmax(-1) @ vr
        torch.testing.assert_close(out[row].float(), want, atol=2e-2, rtol=2e-2)
