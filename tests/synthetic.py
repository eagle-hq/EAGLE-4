# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Deterministic synthetic calibration data shared by the golden generator and the tests.

Timings follow a simple model (fixed cost + per-token cost + attention that grows
with batch and context) with seeded +-1.5% noise, so the fits are stable and the
fitted tables pass the cost model's sanity checks.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List

from eagle_harness.core.calibration import CacheSweepPlan, CalibrationPaths, MeasureParams, MeasureRecord, save_jsonl
from eagle_harness.core.limits import ComputeIntensity, draft_query_lens, verify_buckets

CONTEXT_STEP = 128
CONTEXT_STOP = 2048
TARGET_LAYERS = 16
DRAFT_TREE_DEPTH = 9


def measure_params(batch_size: int, intensity: ComputeIntensity) -> MeasureParams:
    chain = intensity is ComputeIntensity.CHAIN
    verify_num = 16 if chain else (64 if batch_size <= 8 else 32)
    return MeasureParams(
        context_step=CONTEXT_STEP,
        context_stop=CONTEXT_STOP,
        batch_size=batch_size,
        verify_num=verify_num,
        node_top_k=1 if chain else 8,
        loop_top_m=verify_num,
        repeat_cnt=16,
        cache_len=CONTEXT_STEP,
    )


def _attention_us(q: int, kv: int, bs: int, scale: float) -> float:
    return scale * (2.0 + 0.0009 * kv * bs * (1.0 + 0.04 * q))


def _noise(rng: random.Random) -> float:
    return 1.0 + rng.uniform(-0.015, 0.015)


def write_raw_measurements(data_path: str, batch_size: int, intensity: ComputeIntensity, seed: int) -> MeasureParams:
    """Write raw cache and model timings for one batch size; return its params."""
    rng = random.Random(seed * 1000 + batch_size)
    params = measure_params(batch_size, intensity)
    paths = CalibrationPaths(data_path)
    plan = CacheSweepPlan.from_params(params)
    sweeps: Dict[str, List[int]] = {
        "slm": draft_query_lens(batch_size, intensity),
        "llm": sorted(verify_buckets(batch_size, intensity)),
    }
    for model, lens in sweeps.items():
        scale = 0.6 if model == "slm" else 1.0
        for q in lens:
            records = [
                MeasureRecord(
                    time_us=int(_attention_us(q, kv, batch_size, scale) * plan.num_layers * _noise(rng)),
                    measure_count=plan.num_layers,
                    query_length=q,
                    cache_length=kv,
                    task_category=f"cache_{model}",
                    batch_size=batch_size,
                )
                for kv in plan.cache_lengths()
                if kv >= plan.kv_step
            ]
            save_jsonl(os.path.join(paths.raw(batch_size, "cache", model), f"cache_qo_len_{q}.jsonl"), records)
    contexts = list(range(CONTEXT_STEP + 7, CONTEXT_STOP, 61))
    for model, lens in sweeps.items():
        for q in lens:
            records = []
            for _ in range(2):
                for kv in contexts:
                    if model == "slm":
                        count = DRAFT_TREE_DEPTH - 1
                        step = 95.0 + 4.0 * q * batch_size + _attention_us(q, kv, batch_size, 0.6)
                    else:
                        count = 1
                        step = 1400.0 * (1.0 + 0.012 * q * batch_size) + TARGET_LAYERS * _attention_us(
                            q, kv, batch_size, 1.0
                        )
                    records.append(
                        MeasureRecord(
                            time_us=int(step * count * _noise(rng)),
                            measure_count=count,
                            query_length=q,
                            cache_length=kv,
                            task_category=f"model_{model}",
                            batch_size=batch_size,
                        )
                    )
            save_jsonl(os.path.join(paths.raw(batch_size, "model", model), f"model_qo_len_{q}.jsonl"), records)
    return params
