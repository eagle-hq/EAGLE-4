# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-Harness device policy kernels: the cumulative ranked scores against
torch's, bit for bit, at every shape the policy ranks."""

import sys

import pytest
import torch

pytest.importorskip("tvm_ffi", reason="TokenSpeed round kernels require apache-tvm-ffi")

from eagle_harness.tokenspeed.tree_round.ops.policy import (
    eagle_harness_ranked_scores,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU is required"
)

_DEVICE = "cuda"
_MAX_ROWS, _MAX_COLS = 32, 64


def _ranked_rows(
    rows: int, cols: int, binades: int, generator: torch.Generator
) -> torch.Tensor:
    """Each row's scores in rank order: descending scores in (0, 1) spread
    over ``binades`` binades (many: sums in another order round apart)."""
    scores = torch.rand((rows, cols), generator=generator, dtype=torch.float64)
    scale = torch.exp2(-torch.randint(0, binades, (rows, cols), generator=generator))
    return (scores * scale).float().sort(dim=-1, descending=True).values


@pytest.mark.parametrize("flat", [False, True])
def test_ranked_scores_match_torch(flat: bool) -> None:
    generator = torch.Generator().manual_seed(int(flat))
    # The kernel reads the live corner of a padded buffer.
    buffer = torch.full((_MAX_ROWS + 3, _MAX_COLS + 5), float("nan"), device=_DEVICE)
    out = torch.empty(_MAX_ROWS * _MAX_COLS, device=_DEVICE)
    sizes = torch.empty(4, dtype=torch.int32, device=_DEVICE)
    rows_t, cols_t, flat_t, count = sizes.split(1)
    flat_t.fill_(int(flat))
    shapes, mismatches = [], []
    for binades in (1, 24):
        for rows in range(1, _MAX_ROWS + 1):
            for cols in range(1, _MAX_COLS + 1):
                shapes.append((binades, rows, cols))
                top = _ranked_rows(rows, cols, binades, generator).to(_DEVICE)
                expected = torch.cumsum(
                    top.view(-1) if flat else top.mean(dim=0), dim=0
                )
                buffer[:rows, :cols] = top
                rows_t.fill_(rows)
                cols_t.fill_(cols)
                eagle_harness_ranked_scores(buffer, rows_t, cols_t, flat_t, out, count)
                n = expected.numel()
                mismatches.append(
                    (count[0] != n)
                    | (out[:n].view(torch.int32) != expected.view(torch.int32)).any()
                )
    failed = torch.stack(mismatches).nonzero().flatten().tolist()
    assert not failed, f"(binades, rows, cols) {[shapes[i] for i in failed[:5]]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
