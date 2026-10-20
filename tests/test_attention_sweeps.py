# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The calibration sweep loop and head layout, with a fake timer (the real timers need a GPU)."""

from __future__ import annotations

import json
import os

import pytest

from eagle_harness.core.attention_sweeps import HeadLayout, sweep
from eagle_harness.core.calibration import CacheSweepPlan, MeasureParams, fit_cache_costs, load_jsonl
from eagle_harness.core.limits import ComputeIntensity


class LinearTimer:
    def time_us(self, batch_size, qo_len, kv_len, num_layers):
        return int((10 + 0.01 * kv_len * qo_len) * num_layers)


def test_head_layout():
    assert HeadLayout.for_rank(24, 4, 256, 4) == HeadLayout(6, 1, 256)
    assert HeadLayout.for_rank(24, 2, 256, 4) == HeadLayout(6, 1, 256)  # replicated KV heads
    with pytest.raises(ValueError):
        HeadLayout.for_rank(24, 2, 256, 4, replicate_kv=False)
    with pytest.raises(ValueError):
        HeadLayout.for_rank(30, 4, 128, 4)


def test_sweep_writes_records_the_fit_reads(tmp_path):
    params = MeasureParams(
        context_step=64, context_stop=2048, batch_size=2, verify_num=64, node_top_k=8, loop_top_m=64, repeat_cnt=16
    )
    plan = CacheSweepPlan.from_params(params)
    assert (plan.kv_start, plan.kv_step, plan.interval_step, plan.num_layers) == (32, 64, 2, 16)
    raw, fitted = str(tmp_path / "raw"), str(tmp_path / "fit")
    files = sweep(LinearTimer(), category="cache_slm", batch_size=2, qo_lens=[1, 2, 4, 8], plan=plan, save_dir=raw)
    assert len(files) == 4
    rows = load_jsonl(files[0])
    assert min(r["cache_length"] for r in rows) == 64  # warm-up lengths dropped
    assert rows[0] == {
        "time_intervals": int((10 + 0.64) * 16),
        "measure_count": 16,
        "query_length": 1,
        "cache_length": 64,
        "task_category": "cache_slm",
        "batch_size": 2,
    }
    instability = fit_cache_costs(
        "cache_all_slm",
        raw,
        fitted,
        max_query_length=8,
        cache_length_start=64,
        cache_length_step=64,
        cache_length_stop=2048,
        batch_size=2,
        intensity=ComputeIntensity.HIGH,
    )
    assert instability == 0.0
    first = json.load(open(os.path.join(fitted, "cache_interval_0.json")))
    last = json.load(open(os.path.join(fitted, "cache_interval_30.json")))
    assert first["array"][0] < last["array"][0] and first["array"][7] > first["array"][0]
