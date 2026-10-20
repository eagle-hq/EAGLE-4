# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Tune the GEMM solutions a serving stack uses and save them with the GPU's calibration data.

1. Record the GEMM shapes of the serving configuration (one short run, e.g. ``tools/greedy_parity.py run``
   or a speed smoke against the server), with PyTorch's untuned-shape log enabled in the server process::

       PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 \\
       PYTORCH_TUNABLEOP_UNTUNED_FILENAME=/tmp/untuned.csv python -m sglang.launch_server ...

   (TunableOp appends the device ordinal: the file is ``/tmp/untuned0.csv``; one file per rank at TP > 1).

2. Tune every recorded shape offline and write ``<data_path>/gemm_solutions.csv``::

       python tools/tune_gemm_solutions.py --data-path COST --untuned /tmp/untuned0.csv [/tmp/untuned1.csv ...]

   Weights of a decode round stream from HBM, so each solution is timed over inputs rotated through a buffer
   larger than the GPU's last-level cache (``--rotating-buffer-mb``, default 1024).

The server loads the file through ``eagle_harness.core.gemm_solutions`` (tuning off); see that module.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.cuda.tunable as tunable

from eagle_harness.core.gemm_solutions import GEMM_SOLUTIONS_FILE


def untuned_lines(paths: list[Path], max_rows: int, rows: set[int] | None = None) -> list[str]:
    """Select distinct GEMM lines in first-seen order, by exact row counts or the maximum row count."""
    seen: dict[str, None] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.startswith(("Gemm", "ScaledGemm")):
                continue
            m = gemm_rows(line)
            selected = m <= max_rows if rows is None else m in rows
            if selected:
                seen.setdefault(line.strip(), None)
    return list(seen)


def decode_rows(batch_sizes: list[int], top_k: int, widths: list[int]) -> set[int]:
    """The row counts of a decode round at the served batch sizes: B (one token), B x top_k (a draft step) and
    B x width (the verify and the draft extend)."""
    return {b * m for b in batch_sizes for m in (1, top_k, *widths)}


def gemm_rows(line: str) -> int:
    """The row count (BLAS ``m``) of an untuned line's signature ``<layout>_<n>_<m>_<k>_...``."""
    return int(line.split(",")[1].split("_")[2])


def tune_line(line: str, device: torch.device) -> None:
    """Run the GEMM of one untuned line once, which tunes it.

    ``F.linear`` lines (``GemmTunableOp_<dtype>_TN``, plain leading dimensions) are replayed as the linear
    they came from, which also covers the one-row GEMMs torch's offline helper skips; everything else goes
    through that helper.
    """
    op, params = line.split(",")[:2]
    fields = params.split("_")
    if op.startswith("GemmTunableOp_") and op.endswith("_TN") and fields[4] == "ld":
        n, m, k = (int(v) for v in fields[1:4])
        ldb, lda, ldc = (int(v) for v in fields[5:8])
        if (ldb, lda, ldc) == (k, k, n):
            dtype = {"BFloat16": torch.bfloat16, "Half": torch.half, "float": torch.float32}[op.split("_")[1]]
            x = torch.randn(m, k, dtype=dtype, device=device)
            w = torch.randn(n, k, dtype=dtype, device=device)
            torch.nn.functional.linear(x, w)
            return
    tunable._process_single_offline_gemm(line, device.index or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, required=True, help="calibration directory of the GPU")
    parser.add_argument("--untuned", type=Path, nargs="+", required=True, help="TunableOp untuned-shape files")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=64,
        help="tune GEMMs of at most this many rows: the decode round's (draft, verify, no-spec); "
        "prefill GEMMs take a row count per prompt length and keep the default solution",
    )
    parser.add_argument(
        "--decode-rows",
        nargs=3,
        metavar=("BATCH_SIZES", "TOP_K", "WIDTHS"),
        help="instead of --max-rows: only the row counts of the decode rounds, e.g. 1-8 4 8,16,32 for batch "
        "sizes 1..8, draft top-k 4 and verify widths 8/16/32 (prefill lengths are left out)",
    )
    parser.add_argument("--rotating-buffer-mb", type=int, default=1024)
    parser.add_argument("--max-tuning-ms", type=int, default=20, help="time budget per candidate solution")
    parser.add_argument("--max-tuning-iterations", type=int, default=30)
    args = parser.parse_args()
    rows = None
    if args.decode_rows:
        batches, top_k, widths = args.decode_rows
        try:
            first, last = (int(v) for v in batches.split("-")) if "-" in batches else (int(batches), int(batches))
            top_k = int(top_k)
            widths = [int(width) for width in widths.split(",")]
            if not 1 <= first <= last or top_k < 1 or any(width < 1 for width in widths):
                raise ValueError
        except ValueError:
            parser.error(
                "--decode-rows requires a positive batch or ascending range, positive top-k, and positive widths"
            )
        rows = decode_rows(list(range(first, last + 1)), top_k, widths)
    lines = untuned_lines(args.untuned, args.max_rows, rows)
    if not lines:
        raise SystemExit(f"no GEMM lines in {[str(p) for p in args.untuned]}")
    out = args.data_path / GEMM_SOLUTIONS_FILE
    tunable.enable(True)
    tunable.tuning_enable(True)
    tunable.set_rotating_buffer_size(args.rotating_buffer_mb)
    tunable.set_max_tuning_duration(args.max_tuning_ms)
    tunable.set_max_tuning_iterations(args.max_tuning_iterations)
    device = torch.device("cuda", torch.cuda.current_device())
    for i, line in enumerate(lines, 1):
        start = time.perf_counter()
        tune_line(line, device)
        torch.cuda.synchronize()
        print(f"[{i}/{len(lines)}] {line.split(',')[1]} {time.perf_counter() - start:.1f}s", flush=True)
    tunable.tuning_enable(False)
    os.makedirs(args.data_path, exist_ok=True)
    if not tunable.write_file(str(out)):
        raise SystemExit(f"could not write {out}")
    print(f"{len(tunable.get_results())} GEMM solutions -> {out}")


if __name__ == "__main__":
    main()
