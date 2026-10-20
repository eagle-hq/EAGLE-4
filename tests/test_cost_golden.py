# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Cost model and calibration against golden fixtures.

``tests/golden/fork_golden.json`` holds the expected outputs on the data in ``tests/synthetic.py``.
"""

from __future__ import annotations

import json
import os

import pytest

from eagle_harness.core.calibration import AttentionShape, CalibrationPaths, fit_batch, remove_outliers
from eagle_harness.core.cost_model import CostModel, CostModelParams, ragged_verify_budget
from eagle_harness.core.cost_table import load_cost_tables
from eagle_harness.core.limits import ComputeIntensity, verify_buckets
from tests import synthetic

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "fork_golden.json")
INTENSITIES = {
    ComputeIntensity.HIGH: [1, 2, 4, 8, 16, 32],
    ComputeIntensity.CHAIN: [1, 4, 16, 64],
}


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def _calibrate(tmp_path, intensity: ComputeIntensity) -> str:
    root = str(tmp_path / intensity.value)
    paths = CalibrationPaths(root)
    for bs in INTENSITIES[intensity]:
        params = synthetic.write_raw_measurements(root, bs, intensity, seed=7)
        fit_batch(paths, params, intensity, AttentionShape(1), AttentionShape(synthetic.TARGET_LAYERS))
    return root


def _tree(root: str) -> dict:
    files = {}
    for dirpath, _, names in os.walk(root):
        for name in names:
            full = os.path.join(dirpath, name)
            with open(full, encoding="utf-8") as f:
                files[os.path.relpath(full, root)] = f.read()
    return files


@pytest.fixture(scope="module", params=list(INTENSITIES), ids=lambda i: i.value)
def calibrated(request, tmp_path_factory):
    intensity = request.param
    return intensity, _calibrate(tmp_path_factory.mktemp("cal"), intensity)


def test_fitted_cost_files_are_byte_identical(golden, calibrated):
    intensity, root = calibrated
    mine = _tree(os.path.join(root, "cost"))
    expected = golden[f"cost_files_{intensity.value}"]
    assert sorted(mine) == sorted(expected)
    for name in expected:
        assert mine[name] == expected[name], name


def test_loaded_tables_match_fork_loader(golden, calibrated):
    intensity, root = calibrated
    tables = load_cost_tables(os.path.join(root, "cost"))
    expected = golden[f"loaded_{intensity.value}"]
    assert sorted(map(str, tables)) == sorted(expected)
    for bs, table in tables.items():
        data, config = expected[str(bs)]
        assert config["max_interval_id"] == table.max_interval_id
        assert config["model_ratio_array"] == list(table.model_ratio)
        assert config["cache_length_step"] == table.cache_length_step
        assert config["cache_length_start"] == table.cache_length_start
        assert config["slm_max_qo_len"] == table.draft_max_qo_len
        assert config["llm_max_qo_len_query"] == table.target_max_qo_len_query
        assert config["llm_max_qo_len_cache"] == table.target_max_qo_len_cache
        for (llm, slm), interval in zip(data, table.intervals):
            for fork_curve, curve in ((llm, interval.target), (slm, interval.draft)):
                assert fork_curve == [
                    list(curve.value),
                    list(curve.ratio),
                    list(curve.index),
                    list(curve.array_query),
                    list(curve.array_cache),
                ]


def _model(root: str, intensity: ComputeIntensity, case: dict) -> CostModel:
    return CostModel(
        tables=load_cost_tables(os.path.join(root, "cost")),
        params=CostModelParams(depth_loss_ratio=0.04),
        intensity=intensity,
        node_top_k=case["node_top_k"],
        tree_depth=case["tree_depth"],
        verify_num=case["verify_num"],
        ragged=intensity is ComputeIntensity.CHAIN,
        eval_batch_size=case["eval_batch_size"],
    )


def test_decisions_match_fork(golden, calibrated):
    intensity, root = calibrated
    cases = golden[f"decisions_{intensity.value}"]
    assert len(cases) >= 2
    for case in cases:
        _replay(_model(root, intensity, case["case"]), case["script"])


def _replay(cm: CostModel, script: list) -> None:
    row = None
    assert len(script) > 100
    for step in script:
        op, args, out = step["op"], step["args"], step["out"]
        if op == "update_interval":
            cm.update_interval(*args)
            got = cm.interval_index
        elif op == "chain":
            got = cm.chain_stop(*args)
        elif op == "classify_chain":
            tier, scale = cm.classify_chain(*args)
            got = [int(tier), scale]
        elif op == "ragged_active":
            got = cm.ragged_active(*args)
        elif op == "history":
            got = cm.history_verify_num(*args)
        elif op == "draft_cost_row":
            row = cm.draft_cost_row(*args)
            got = [
                list(row.cost_values),
                list(row.cost_index),
                row.cost_state,
                row.adjust_cost_ratio,
                row.model_cost_ratio,
                row.batch_size_ratio,
            ]
        elif op == "tree_width":
            depth, bs, past, deltas = args
            d = cm.tree_width(depth, bs, past, row, deltas)
            got = [d.width, d.verify_num, d.stop]
        elif op == "verify_count":
            got = cm.verify_count(*args)
        elif op == "verify_paged":
            scores, requested, bs, avg, lagged = args
            try:
                got = list(cm.verify_num_paged(scores, requested, bs, avg, lagged=lagged))
            except ValueError as e:
                got = ["error", type(e).__name__]
        elif op == "record":
            cm.record_verify_num(*args)
            got = None
        else:
            raise AssertionError(op)
        assert got == out, (op, args)


def test_outlier_filter_matches_fork(golden):
    for case in golden["zscore"]:
        mean, kept, ok = remove_outliers(case["data"], convergence_threshold=case["conv"])
        assert float(mean) == case["mean"]
        assert kept == case["kept"]
        assert ok == case["ok"]


def test_ragged_budget_matches_fork(golden):
    for case in golden["budget"]:
        assert ragged_verify_budget(*case["args"]) == case["out"]


def test_verify_buckets_match_fork(golden):
    for intensity in (ComputeIntensity.HIGH, ComputeIntensity.CHAIN):
        for bs, buckets in golden["verify_buckets"][intensity.value].items():
            assert verify_buckets(int(bs), intensity) == buckets
