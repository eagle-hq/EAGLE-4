# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-Harness: dynamic-tree speculative decoding driven by a measured cost model.

Layout:
    core/     cost model, calibration, draft-tree bookkeeping, HarnessConfig (engine-agnostic, CPU-testable)
    kernels/  CUDA / Triton kernels (candidate head, GDN tree verify), built lazily
    models/   EAGLE-4 draft config validation and the low-rank LM head
    sglang/   sglang plugin: algorithm (hooks), workers, draft steps, tree attention, draft model
"""

__version__ = "0.1.2"  # x-release-please-version
