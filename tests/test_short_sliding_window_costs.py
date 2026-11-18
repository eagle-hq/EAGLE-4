# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""fit_model_costs prices sliding-window layers whose window is under two cache-length steps."""

from __future__ import annotations

import json

import pytest

from eagle_harness.core.calibration import fit_model_costs
from eagle_harness.core.limits import ComputeIntensity

STEP = 256
CACHE_LAYERS = 4
SWA_LAYERS = 2
QUERY_MAX = 2
INTERVALS = 4


def _fit(tmp_path, sliding_window):
    query, cache, out = (tmp_path / name for name in ("query", "cache", "out"))
    for path in (query, cache, out):
        path.mkdir()
    for q in range(1, QUERY_MAX + 1):
        (query / f"query_{q}.json").write_text(json.dumps({"mean": 1.0}))
    # Attention cost grows with the cached length: interval i costs (i + 1) per layer.
    for i in range(INTERVALS):
        (cache / f"cache_interval_{i}.json").write_text(json.dumps({"array": [float(i + 1)] * QUERY_MAX}))
    fit_model_costs(
        "unit",
        str(query),
        str(cache),
        str(out),
        max_query_length_query=QUERY_MAX,
        max_query_length_cache=QUERY_MAX,
        cache_length_start=STEP,
        cache_length_step=STEP,
        cache_layers=CACHE_LAYERS,
        batch_size=1,
        intensity=ComputeIntensity.HIGH,
        sliding_window_layers=SWA_LAYERS,
        sliding_window=sliding_window,
    )
    return [json.loads((out / f"model_interval_{i}.json").read_text())["value"][0] for i in range(INTERVALS)]


@pytest.mark.parametrize("sliding_window", [128, 384])
def test_short_window_layers_stop_growing_with_context(tmp_path, sliding_window):
    costs = _fit(tmp_path, sliding_window)

    # Interval 0 holds the window, so a sliding-window layer keeps costing interval 0's 1.0;
    # only the full-attention layers follow the cached length.
    expected = [1.0 + (i + 1) * (CACHE_LAYERS - SWA_LAYERS) + 1.0 * SWA_LAYERS for i in range(INTERVALS)]
    expected[0] = 1.0 + 1.0 * CACHE_LAYERS
    assert costs == pytest.approx(expected)


def test_window_of_two_steps_is_unchanged(tmp_path):
    # 768 // 256 - 1 = 2: intervals 0..2 are within the window, as before.
    costs = _fit(tmp_path, 768)

    expected = [1.0 + (i + 1) * CACHE_LAYERS for i in range(3)] + [1.0 + 4 * 2 + 3 * 2]
    assert costs == pytest.approx(expected)
