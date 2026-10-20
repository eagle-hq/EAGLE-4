# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Compile the Triton kernels for every target architecture and report which fail: arch errors without that hardware.

    python tools/check_targets.py [--targets sm_80,sm_90,gfx942]

Runs the kernel benchmark cases (``tools/bench_kernels.py``) once per target in a fresh process with the dispatch
(:mod:`eagle_harness.kernels.arch`) answering for the target GPU and Triton compiling for its ``GPUTarget``; kernels are
compiled, never launched (inputs are real tensors on whatever NVIDIA GPU runs this, outputs are garbage). So every
launcher takes the path the target takes: no PDL before sm_90, Gluon on sm_103 only. The native CUDA library is not
loaded (the Triton fallbacks are what is checked); it has its own check: ``kernels.build check``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))  # bench_kernels

# name -> (Triton backend, Triton arch, warp size, vendor, sm)
TARGETS: Dict[str, Tuple[str, object, int, str, int]] = {
    "sm_80": ("cuda", 80, 32, "nvidia", 80),
    "sm_89": ("cuda", 89, 32, "nvidia", 89),
    "sm_90": ("cuda", 90, 32, "nvidia", 90),
    "sm_100": ("cuda", 100, 32, "nvidia", 100),
    "sm_103": ("cuda", 103, 32, "nvidia", 103),
    "sm_120": ("cuda", 120, 32, "nvidia", 120),
    "gfx90a": ("hip", "gfx90a", 64, "amd", 0),
    "gfx942": ("hip", "gfx942", 64, "amd", 0),
    "gfx950": ("hip", "gfx950", 64, "amd", 0),
}


def _compile_only(name: str) -> Tuple[List[str], int]:
    """Run every benchmark case for target ``name`` without launching: ``(failures, kernel launches compiled)``."""
    import bench_kernels
    import torch
    from triton.backends.compiler import GPUTarget
    from triton.runtime import driver
    from triton.runtime.jit import JITFunction

    from eagle_harness.kernels import arch, native

    backend, triton_arch, warp, vendor, sm = TARGETS[name]
    fake = arch.Gpu(0, name, vendor, name, sm)
    arch._probe = lambda index: fake  # type: ignore[assignment]
    native.library = lambda: None  # type: ignore[assignment]
    driver.active.get_current_target = lambda: GPUTarget(backend, triton_arch, warp)  # type: ignore[method-assign]
    run = JITFunction.run
    launches = []

    def compile_only(self, *args, **kwargs):
        launches.append(self.fn.__name__)
        return run(self, *args, **{**kwargs, "warmup": True})

    JITFunction.run = compile_only  # type: ignore[method-assign]

    failures: List[str] = []
    case = {"name": ""}

    def compile_case(fn):
        try:
            return 0.0, fn()
        except Exception as e:  # noqa: BLE001 - report every kernel that fails to compile
            failures.append(
                f"{case['name']}: {type(e).__name__}: {str(e).strip().splitlines()[-1] if str(e).strip() else ''}"
            )
            traceback.print_exc(file=sys.stderr)
            return 0.0, None

    record = bench_kernels._record

    def labelled(results, label, fn):
        case["name"] = label
        record(results, label, fn)

    bench_kernels._time = compile_case
    bench_kernels._record = labelled
    for group, run_group in bench_kernels.GROUPS.items():
        try:
            run_group({})
        except Exception as e:  # noqa: BLE001
            failures.append(f"{group} setup: {type(e).__name__}: {e}")
    torch.cuda.synchronize()
    return failures, len(launches)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compile the Triton kernels for every target architecture.")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        failures, compiled = _compile_only(args.child)
        print("\n".join(f"FAIL {args.child} {f}" for f in failures) or f"OK   {args.child}: {compiled} kernel launches")
        return 1 if failures else 0
    bad = 0
    for name in args.targets.split(","):
        proc = subprocess.run([sys.executable, __file__, "--child", name], capture_output=True, text=True)
        print(proc.stdout.strip() or f"FAIL {name} no output: {proc.stderr.strip().splitlines()[-1:]}")
        bad += proc.returncode != 0
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
