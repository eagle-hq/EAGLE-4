# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Build the packaged TokenSpeed CUDA sources without partner prebuilt binaries.

Uses the native kernel builder's nvcc discovery and atomic publication pattern.
TVM FFI supplies tensor/stream interop; torch and Python headers are not needed.
``python -m eagle_harness.tokenspeed.tree_round.ops.build --arch 103 --output ./build/round-ops``
compiles all four modules without accessing a GPU. Runtime builds are cached
by source, toolkit, architecture and TVM FFI ABI, before graph capture.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import os
import subprocess
import tempfile
from pathlib import Path

from eagle_harness.kernels.build import find_nvcc, toolkit_version

SOURCE = Path(__file__).resolve().parent / "csrc"
MODULES = ("tree", "policy", "chain", "rejection")
NVCC_FLAGS = (
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--fmad=false",
    "-Xcompiler=-fPIC",
)


def _ffi_paths() -> tuple[list[str], str]:
    try:
        from tvm_ffi import libinfo
    except ImportError as exc:
        raise RuntimeError(
            "TokenSpeed device kernels require apache-tvm-ffi (provided by TokenSpeed)"
        ) from exc
    return [
        libinfo.find_include_path(),
        libinfo.find_dlpack_include_path(),
    ], libinfo.find_libtvm_ffi()


def compile_library(name: str, out: Path, arch: int, nvcc: str | None = None) -> Path:
    """Compile and atomically publish one library; no GPU or torch import."""
    if name not in MODULES:
        raise ValueError(f"Unknown TokenSpeed kernel module: {name!r}")
    nvcc = nvcc or find_nvcc()
    if nvcc is None:
        raise RuntimeError("nvcc not found (set CUDA_HOME to a CUDA 12+ toolkit)")
    if toolkit_version(nvcc) < (12, 0):
        raise RuntimeError("TokenSpeed device graph launches require CUDA 12+")
    if arch < 80:
        raise ValueError(
            "TokenSpeed device policy requires compute capability 8.0 or newer"
        )
    includes, ffi_library = _ffi_paths()
    include_flags = [f"-I{SOURCE}", *(f"-I{p}" for p in includes)]
    gencode = [
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        f"-gencode=arch=compute_{arch},code=compute_{arch}",
    ]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=f".{name}-") as tmp:
        obj = Path(tmp) / f"{name}.o"
        commands = [
            [
                nvcc,
                *NVCC_FLAGS,
                *gencode,
                *include_flags,
                *(["-rdc=true"] if name == "chain" else []),
                "-c",
                str(SOURCE / f"eagle_harness_{name}.cu"),
                "-o",
                str(obj),
            ]
        ]
        objects = [str(obj)]
        if name == "chain":
            dlink = Path(tmp) / "device_link.o"
            commands.append(
                [
                    nvcc,
                    *gencode,
                    "-Xcompiler=-fPIC",
                    "-dlink",
                    str(obj),
                    "-lcudadevrt",
                    "-o",
                    str(dlink),
                ]
            )
            objects.append(str(dlink))
        partial = Path(tmp) / out.name
        commands.append(
            [
                nvcc,
                "-shared",
                "-Xcompiler=-fPIC",
                *objects,
                "-cudart=static",
                *(["-lcudadevrt"] if name == "chain" else []),
                ffi_library,
                f"-Xlinker=-rpath={Path(ffi_library).parent}",
                "-o",
                str(partial),
            ]
        )
        for command in commands:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if result.returncode:
                raise RuntimeError(
                    f"nvcc failed: {' '.join(command)}\n{result.stdout[-2000:]}\n{result.stderr[-6000:]}"
                )
        partial.chmod(0o755)
        os.replace(partial, out)
    return out


@functools.cache
def _load(name: str, arch: int):
    import torch
    import tvm_ffi

    from eagle_harness.kernels.native import _cache_dir

    nvcc = find_nvcc()
    if nvcc is None:
        raise RuntimeError(
            "nvcc not found (set CUDA_HOME); TokenSpeed kernel sources need a CUDA toolkit"
        )
    includes, ffi_library = _ffi_paths()
    identity = repr(
        (
            arch,
            nvcc,
            toolkit_version(nvcc),
            NVCC_FLAGS,
            importlib.metadata.version("apache-tvm-ffi"),
            includes,
            ffi_library,
        )
    ).encode()
    # Hash this build recipe as well as the sources: flag/link changes rebuild.
    digest = hashlib.sha256(identity + Path(__file__).read_bytes())
    for path in (SOURCE / f"eagle_harness_{name}.cu", SOURCE / "tvm_ffi_utils.h"):
        digest.update(path.read_bytes())
    out = _cache_dir() / "tokenspeed_round" / digest.hexdigest()[:20] / f"{name}.so"
    if not out.exists():
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Load TokenSpeed device kernels before CUDA graph capture"
            )
        compile_library(name, out, arch, nvcc)
    return tvm_ffi.load_module(str(out))


def load_module(name: str):
    """Load for the current device, with no disk access after the first call."""
    import torch

    if name not in MODULES:
        raise ValueError(f"Unknown TokenSpeed kernel module: {name!r}")
    major, minor = torch.cuda.get_device_capability()
    return _load(name, major * 10 + minor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", type=int, required=True, help="CUDA SM, e.g. 90 or 103"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-home")
    args = parser.parse_args()
    nvcc = find_nvcc(args.cuda_home)
    for name in MODULES:
        print(compile_library(name, args.output / f"{name}.so", args.arch, nvcc))


if __name__ == "__main__":
    main()
