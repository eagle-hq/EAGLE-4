# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""nvcc build of the native CUDA kernels (``csrc/candidate.cu``): prebuilt binaries, the JIT fallback, a per-arch check.

The library has a plain C ABI over raw pointers (no torch, no Python headers), so one binary per CUDA major (12, 13)
serves every torch and Python version; the CUDA runtime is linked statically. Each binary holds SASS for every supported
architecture and the PTX of sm_90 (newer GPUs JIT it), so it needs no rebuild per GPU.

    python -m eagle_harness.kernels.build prebuild [--cuda-home /usr/local/cuda-13.0]   # into the package, for a wheel
    python -m eagle_harness.kernels.build check [--cuda-home ...]                        # one compile per architecture

``prebuild`` writes ``_native/cu<major>/<LIBRARY>``; :mod:`eagle_harness.kernels.native` loads it, or JIT-builds the
library for the visible GPU with :func:`compile_library` when there is none for this CUDA major.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PACKAGE = Path(__file__).resolve().parent
SOURCE = PACKAGE / "csrc" / "candidate.cu"
LIBRARY = "libeagle_harness_candidate.so"
PREBUILT = PACKAGE / "_native"

# SASS architectures (CUDA 12.8 has no sm_103: that GPU runs the sm_90 PTX there) and the PTX architecture.
SASS_ARCHS = (80, 90, 100, 103, 120)
PTX_ARCH = 90
# -diag-suppress 177: the helpers of the sm_90-only kernel are unreferenced in the host pass and below sm_90.
NVCC_FLAGS = ("-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-cudart", "static", "-diag-suppress", "177")


def find_nvcc(cuda_home: Optional[str] = None) -> Optional[str]:
    """The nvcc of ``cuda_home``, ``$CUDA_HOME``, ``/usr/local/cuda`` or ``PATH``; None if there is none."""
    homes = [cuda_home, os.environ.get("CUDA_HOME"), "/usr/local/cuda"]
    candidates = [os.path.join(h, "bin", "nvcc") for h in homes if h]
    return next((c for c in candidates if os.access(c, os.X_OK)), None) or shutil.which("nvcc")


def toolkit_version(nvcc: str) -> Tuple[int, int]:
    """``(major, minor)`` of the CUDA toolkit behind ``nvcc``."""
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"release (\d+)\.(\d+)", out)
    if match is None:
        raise RuntimeError(f"cannot read the CUDA version from `{nvcc} --version`")
    return int(match.group(1)), int(match.group(2))


def supported_sass(version: Tuple[int, int]) -> Tuple[int, ...]:
    """The SASS architectures this toolkit can build (sm_103 from CUDA 12.9)."""
    return tuple(a for a in SASS_ARCHS if a != 103 or version >= (12, 9))


def _gencode(sass: Iterable[int], ptx: Optional[int]) -> Sequence[str]:
    flags = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in sass]
    if ptx is not None:
        flags.append(f"-gencode=arch=compute_{ptx},code=compute_{ptx}")
    return flags


def compile_library(out: Path, sass: Iterable[int], ptx: Optional[int] = PTX_ARCH, nvcc: Optional[str] = None) -> Path:
    """Build the shared library at ``out`` for the given architectures; raises RuntimeError with nvcc's output."""
    nvcc = nvcc or find_nvcc()
    if nvcc is None:
        raise RuntimeError("nvcc not found (set CUDA_HOME)")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Compile to a private file first: a concurrent process must never load a half-written library.
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".so", delete=False) as tmp:
        partial = Path(tmp.name)
    cmd = [nvcc, *NVCC_FLAGS, *_gencode(sass, ptx), str(SOURCE), "-o", str(partial)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"nvcc failed ({' '.join(cmd)}):\n{proc.stderr[-4000:]}")
    partial.chmod(0o755)
    os.replace(partial, out)
    return out


def _toolkit(cuda_home: Optional[str]) -> Tuple[str, Tuple[int, int]]:
    nvcc = find_nvcc(cuda_home)
    if nvcc is None:
        raise RuntimeError("nvcc not found (pass --cuda-home)")
    return nvcc, toolkit_version(nvcc)


def prebuild(cuda_home: Optional[str] = None) -> Path:
    """The binary of this toolkit's CUDA major, into ``_native/cu<major>/`` (what a wheel ships)."""
    nvcc, version = _toolkit(cuda_home)
    return compile_library(PREBUILT / f"cu{version[0]}" / LIBRARY, supported_sass(version), PTX_ARCH, nvcc)


def check(cuda_home: Optional[str] = None) -> None:
    """Compile for every architecture on its own, so an error names the architecture; nothing is kept."""
    nvcc, version = _toolkit(cuda_home)
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=len(SASS_ARCHS)) as pool:
        jobs = {
            arch: pool.submit(
                compile_library, Path(tmp) / f"sm{arch}.so", [arch], PTX_ARCH if arch == PTX_ARCH else None, nvcc
            )
            for arch in supported_sass(version)
        }
        for arch, job in jobs.items():
            job.result()  # raises the nvcc error of the first failing architecture
            logger.info("CUDA %d.%d: sm_%d ok", *version, arch)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: ``prebuild`` the native library or ``check`` that nvcc builds it for every target."""
    ap = argparse.ArgumentParser(description="Build or check the native EAGLE-Harness CUDA library.")
    ap.add_argument("command", choices=("prebuild", "check"))
    ap.add_argument("--cuda-home", help="CUDA toolkit root (default: $CUDA_HOME, /usr/local/cuda, PATH)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "prebuild":
        logger.info("built %s", prebuild(args.cuda_home))
    else:
        check(args.cuda_home)
    return 0


if __name__ == "__main__":
    sys.exit(main())
