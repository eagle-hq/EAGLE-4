# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""GPU smoke test of the calibration timing sweep.

Run on one GPU: ``python tools/gpu_smoke.py --out <dir>``. Prints a JSON summary:

1. the kernel report of the device (``eagle_harness.kernels.arch``);
2. a FlashInfer attention sweep at the Qwen3.8-27B TP4 shapes (draft and
   target heads), fitted with fit_cache_costs into the calibrated cost-table layout.

The kernels have their own GPU tests (``tests/test_*_gpu.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eagle_harness.core.attention_sweeps import FlashInferTimer, HeadLayout, sweep  # noqa: E402
from eagle_harness.core.calibration import CalibrationPaths, CacheSweepPlan, MeasureParams, fit_cache_costs  # noqa: E402
from eagle_harness.core.limits import ComputeIntensity, draft_query_lens, verify_buckets  # noqa: E402
from eagle_harness.kernels import arch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out: dict = {"device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()}

    out["kernels"] = arch.report().splitlines()

    paths = CalibrationPaths(os.path.join(args.out, "calib"))
    params = MeasureParams(
        context_step=64, context_stop=2048, batch_size=1, verify_num=64, node_top_k=8, loop_top_m=64, repeat_cnt=16
    )
    plan = CacheSweepPlan.from_params(params)
    shapes = {
        "slm": (HeadLayout.for_rank(40, 40, 128, 4), draft_query_lens(1, ComputeIntensity.HIGH), params.node_top_k),
        "llm": (
            HeadLayout.for_rank(24, 4, 256, 4),
            sorted(verify_buckets(1, ComputeIntensity.HIGH)),
            params.verify_num,
        ),
    }
    sweep_seconds = {}
    for model, (heads, lens, max_q) in shapes.items():
        t0 = time.time()
        sweep(
            FlashInferTimer(heads, tp_size=4),
            category=f"cache_{model}",
            batch_size=1,
            qo_lens=lens,
            plan=plan,
            save_dir=paths.raw(1, "cache", model),
        )
        sweep_seconds[model] = round(time.time() - t0, 1)
        instability = fit_cache_costs(
            f"cache_all_{model}",
            paths.raw(1, "cache", model),
            paths.fitted(1, "cache", model),
            max_query_length=max_q,
            cache_length_start=64,
            cache_length_step=64,
            cache_length_stop=2048,
            batch_size=1,
            intensity=ComputeIntensity.HIGH,
            paths=paths,
        )
        fitted = [
            json.load(open(os.path.join(paths.fitted(1, "cache", model), f"cache_interval_{i}.json")))
            for i in (0, 15, 30)
        ]
        out[f"cache_{model}"] = {
            "instability_rate": instability,
            "us_per_layer_q1_at_intervals_0_15_30": [f["array"][0] for f in fitted],
            f"us_per_layer_q{lens[-1]}_at_intervals_0_15_30": [f["array"][lens[-1] - 1] for f in fitted],
        }
    out["sweep_seconds"] = sweep_seconds
    print(json.dumps(out, indent=2))
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
