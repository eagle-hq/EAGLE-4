# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Measure and save the ROCm candidate top-k crossover offline.

Example: python tools/calibrate_candidate_topk.py --data-path COST --vocab-size 248320 --candidates 128
The saved policy is loaded only for the exact GPU/toolchain/shape identity, on any vendor. Nothing is timed at server startup.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from eagle_harness.kernels.candidate_ops import shard_topk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--candidates", type=int, default=128)
    args = parser.parse_args()
    generator = torch.Generator(device="cuda").manual_seed(721)
    flush = torch.empty(1024**3, device="cuda", dtype=torch.uint8)
    measurements = []
    for rows in (1, 4, 8, 16, 32, 64, 128, 256):
        scores = torch.randn(rows, args.vocab_size, device="cuda", dtype=torch.bfloat16, generator=generator)
        graphs = {}
        for label, limit in (("torch", 0), ("triton", scores.numel())):

            def run(scores=scores, limit=limit):
                return shard_topk(scores, args.candidates, 0, cuda_kernels=False, max_triton_elements=limit)[0]

            ids = run()
            values = scores.gather(-1, ids).sort(dim=-1).values
            torch.testing.assert_close(
                values, scores.topk(args.candidates, sorted=True).values.flip(-1), rtol=0, atol=0
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graphs[label] = graph
        times = {label: [] for label in graphs}
        for label in ("torch", "triton", "triton", "torch") * 5:
            flush.zero_()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[label].replay()
            end.record()
            end.synchronize()
            times[label].append(start.elapsed_time(end) * 1000)
        row = {
            "rows": rows,
            "elements": scores.numel(),
            **{label + "_us": statistics.median(values) for label, values in times.items()},
            "samples_us": times,
        }
        measurements.append(row)
        print(json.dumps(row), flush=True)
    # A crossover must be monotone on the measured grid; noisy/nonmonotone data is not a routing policy.
    winners = [row["triton_us"] <= row["torch_us"] for row in measurements]
    if winners != sorted(winners, reverse=True):
        raise RuntimeError("nonmonotone top-k crossover; raw timings printed, no profile promoted")
    limit = max((r["elements"] for r, wins in zip(measurements, winners) if wins), default=0)
    profile = {
        "identity": {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "dtype": str(torch.bfloat16),
            "vocab_size": args.vocab_size,
            "num_candidates": args.candidates,
        },
        "max_triton_elements": limit,
        "measurements": measurements,
    }
    args.data_path.mkdir(parents=True, exist_ok=True)
    (args.data_path / "candidate_topk.json").write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
