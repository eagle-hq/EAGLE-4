# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Per-GPU GEMM solutions: a PyTorch TunableOp results file under ``data_path``, loaded before graph capture.

The BLAS library's default heuristic can pick a poor kernel for the skinny GEMMs of a speculative round: on
MI355X hipBLASLt streams the target's MLP gate_up weight (5120 x 34816) at 4.4 TB/s for 32 verify rows while
another of its own solutions reaches 5.8 TB/s (81 -> 62 us), and the draft, output and down projections lose
10-30% the same way. ``torch.cuda.tunable`` (TunableOp) is PyTorch's autotuner for exactly this: it times every
available solution of a GEMM signature once, offline, and serves the saved pick. The harness keeps the saved
picks with the GPU's other calibration data as ``<data_path>/gemm_solutions.csv``
(``tools/tune_gemm_solutions.py`` writes it) and loads them with tuning disabled, so nothing is timed at server
startup and every process and rank runs the same kernels. TunableOp validates the file against the PyTorch,
ROCm/CUDA and BLAS library versions and the GPU architecture that tuned it; a file from another stack is
ignored with a warning and GEMMs without a saved pick keep the default solution.

On ROCm the harness also keeps sglang's aiter GEMM dispatch on that torch path for every bf16 GEMM
(:func:`serve_bf16_gemms_through_torch`): aiter's own split-K picks are not repeatable run to run.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

GEMM_SOLUTIONS_FILE = "gemm_solutions.csv"
# aiter's bf16 GEMM dispatch (sglang with SGLANG_USE_AITER=1 routes every bf16 linear through it) reads its tuned table
# from this variable, before the first GEMM of the process.
AITER_GEMM_TABLE_ENV = "AITER_CONFIG_GEMM_BF16"
# The harness's table: empty, so aiter keeps every bf16 GEMM on its torch (hipBLASLt) path, where the saved solutions
# above apply, instead of its own picks.
AITER_GEMM_TABLE = os.path.join(os.path.dirname(__file__), "aiter_gemm_bf16.csv")


def serve_bf16_gemms_through_torch() -> bool:
    """On ROCm, point aiter's bf16 GEMM dispatch at the harness's empty table (``AITER_CONFIG_GEMM_BF16``), unless
    the caller set the variable itself; returns whether it was set.

    aiter's shipped tables pick split-K kernels (asm, FlyDSL) for some shapes: on MI355X the EAGLE-4 draft's
    5120 x 5120 projections at <= 32 rows. Those reduce the K splits in a varying order and round them to bf16
    (twice the error of one rounding), so the same draft step gives different hidden states from one run to the
    next, the draft tree changes, and greedy outputs flip at bf16 near ties. Every bf16 GEMM then runs torch's
    ``F.linear`` (hipBLASLt; TunableOp serves the saved solutions), which is bitwise repeatable.

    Call in the process that launches the server before any model code imports aiter: the dispatch reads the
    table once, at import, and the scheduler processes inherit the environment.
    """
    if torch.version.hip is None or os.environ.get(AITER_GEMM_TABLE_ENV):
        return False
    os.environ[AITER_GEMM_TABLE_ENV] = AITER_GEMM_TABLE
    return True


def gemm_solutions_path(data_path: Optional[str]) -> Optional[str]:
    """``<data_path>/gemm_solutions.csv`` when it exists, else None."""
    if data_path is None:
        return None
    path = os.path.join(data_path, GEMM_SOLUTIONS_FILE)
    return path if os.path.isfile(path) else None


def load_gemm_solutions(data_path: Optional[str]) -> int:
    """Serve the GEMM solutions saved under ``data_path`` (tuning off); returns how many were loaded.

    Call before the first forward, so captured graphs record the tuned kernels. Without the file nothing
    changes; a file this stack cannot validate leaves TunableOp disabled and logs how to re-tune it.
    """
    path = gemm_solutions_path(data_path)
    if path is None:
        return 0
    tunable = torch.cuda.tunable
    tunable.enable(True)
    tunable.tuning_enable(False)
    loaded = len(tunable.get_results()) if tunable.read_file(path) else 0
    if loaded == 0:
        tunable.enable(False)
        logger.warning(
            "EAGLE-Harness: no GEMM solutions usable from %s (tuned on another PyTorch/ROCm/BLAS/GPU stack?); "
            "re-tune with tools/tune_gemm_solutions.py or remove the file. Default GEMM kernels stay in use.",
            path,
        )
        return 0
    logger.info("EAGLE-Harness: %d saved GEMM solutions from %s", loaded, path)
    return loaded
