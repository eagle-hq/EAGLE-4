# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Whether this host can run a Triton kernel at all (a GPU, or the CPU interpreter of a Triton that handles ``tl.zeros``)."""

from __future__ import annotations

import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _probe_kernel(out, N: tl.constexpr):
    tl.store(out + tl.arange(0, N), tl.sum(tl.zeros((N, N), dtype=tl.float32), axis=0))


def triton_runs() -> bool:
    """Launch a minimal kernel; False where the interpreter cannot execute ``tl.zeros`` (older and newer Triton CPU builds)."""
    out = torch.ones(4, device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        _probe_kernel[(1,)](out, N=4)
    except Exception:
        return False
    return bool(out.eq(0).all())


needs_triton = pytest.mark.skipif(not triton_runs(), reason="this Triton cannot run kernels on this host")
