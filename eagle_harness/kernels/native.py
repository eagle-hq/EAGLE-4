# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Load the native CUDA library: ``$EAGLE_HARNESS_NATIVE_LIB`` if set, else the prebuilt binary of this CUDA major,
else an nvcc build cached on disk.

:func:`library` returns the bound :class:`ctypes.CDLL`, or None (logged once) where there is no nvcc, no prebuilt binary
and no JIT build for this machine; callers then run the Triton kernels, which give the same bits.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import torch

from eagle_harness.kernels import arch, build

logger = logging.getLogger(__name__)

# C parameter types of the library's entries: i = int, q = int64_t, P = pointer (see csrc/candidate.cu).
SIGNATURES = {
    "eagle4_shard_topk": "iiPqiiiPqPqqP",
    "eagle4_candidate_tail": "iiiPqqPqqPqPqPqPqiiiiiP",
}
_CTYPES = {"i": ctypes.c_int, "q": ctypes.c_int64, "P": ctypes.c_void_p}


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    for name, signature in SIGNATURES.items():
        function = getattr(lib, name)
        function.argtypes = [_CTYPES[c] for c in signature]
        function.restype = ctypes.c_int
    return lib


def _cache_dir() -> Path:
    return Path(os.environ.get("EAGLE_HARNESS_CACHE") or Path.home() / ".cache" / "eagle_harness")


def _jit() -> ctypes.CDLL:
    """Build for the visible GPUs only (fast), keyed by source, toolkit and architectures."""
    nvcc = build.find_nvcc()
    if nvcc is None:
        raise RuntimeError("no nvcc (set CUDA_HOME)")
    sass = sorted({g.sm for i in range(torch.cuda.device_count()) if (g := arch.gpu(i)) and g.nvidia})
    version = build.toolkit_version(nvcc)
    key = hashlib.sha256(f"{build.SOURCE.read_bytes().hex()}{version}{sass}".encode()).hexdigest()[:16]
    path = _cache_dir() / key / build.LIBRARY
    if not path.exists():
        logger.info("eagle-harness: building the native kernels for sm_%s with %s", sass, nvcc)
        build.compile_library(path, sass, max(sass) if sass else None, nvcc)
    return ctypes.CDLL(str(path))


def _explicit() -> ctypes.CDLL:
    return ctypes.CDLL(os.environ["EAGLE_HARNESS_NATIVE_LIB"])


def _prebuilt() -> ctypes.CDLL:
    path = build.PREBUILT / f"cu{torch.version.cuda.split('.')[0]}" / build.LIBRARY  # type: ignore[union-attr]
    if not path.exists():
        raise RuntimeError(f"no prebuilt binary {path}")
    return ctypes.CDLL(str(path))


@functools.lru_cache(maxsize=None)
def library() -> Optional[ctypes.CDLL]:
    """The bound native library, or None where the Triton kernels run instead."""
    if not torch.version.cuda:  # ROCm or a CPU build of torch
        return None
    reasons = []
    for load in (*([_explicit] if "EAGLE_HARNESS_NATIVE_LIB" in os.environ else []), _prebuilt, _jit):
        try:
            lib = _bind(load())
            logger.info("eagle-harness: native CUDA kernels loaded from %s", lib._name)
            return lib
        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            reasons.append(f"{load.__name__[1:]}: {str(e).splitlines()[0]}")
    logger.warning("eagle-harness: native CUDA kernels unavailable (%s); using Triton", "; ".join(reasons))
    return None
