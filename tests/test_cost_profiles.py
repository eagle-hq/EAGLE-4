# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
"""CPU regressions for the device-selected MI355X serving profile."""

import json
from pathlib import Path

import pytest

from eagle_harness.core.config import parse_harness_config
from eagle_harness.core.cost_profiles import MI355X_PROFILE, packaged_cost_profile
from eagle_harness.core.cost_table import load_cost_tables


@pytest.fixture
def model_paths(tmp_path):
    paths = []
    for name in ("target", "draft"):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_bytes((MI355X_PROFILE / f"{name}_config.json").read_bytes())
        paths.append(str(path))
    return paths


def select(paths, **overrides):
    args = dict(device_name="AMD Instinct MI355X VF", model_path=paths[0], draft_model_path=paths[1],
                tp_size=1, dtype="bfloat16", quantization=None)
    args.update(overrides)
    return packaged_cost_profile(**args)


def test_mi355x_default_has_measured_per_batch_rows(model_paths):
    path = select(model_paths)
    assert path == str(MI355X_PROFILE)
    config = parse_harness_config(None, default_data_path=path)
    tables = load_cost_tables(config.cost_dir)
    assert set(tables) == {1, 8, 32}
    for batch in (1, 8, 32):
        assert tables[batch].batch_size == batch
        assert tables[batch].intervals
    assert (Path(path) / "gemm_solutions.csv").is_file()
    assert (Path(path) / "candidate_topk.json").is_file()


@pytest.mark.parametrize("raw", [None, "{}", '{"mode":"serve"}'])
def test_default_is_resolved_without_an_env_overlay(model_paths, raw, monkeypatch):
    monkeypatch.setenv("AMD_COST_PROFILE", "/not/a/serving/config")
    path = select(model_paths)
    assert parse_harness_config(raw, default_data_path=path).data_path == path
    assert select(model_paths, device_name="AMD Instinct MI355X") == path


def test_explicit_profile_and_static_mode_win(model_paths):
    path = select(model_paths)
    assert parse_harness_config('{"data_path":"/custom"}', default_data_path=path).data_path == "/custom"
    assert parse_harness_config('{"mode":"static"}', default_data_path=path).data_path is None
    with pytest.raises(ValueError, match="needs data_path"):
        parse_harness_config('{"mode":"calibrate"}', default_data_path=path)


@pytest.mark.parametrize("override", [
    {"device_name": "AMD Instinct MI300X"}, {"device_name": "NVIDIA B300"}, {"tp_size": 2},
    {"dtype": "float16"}, {"quantization": "fp8"}, {"model_path": "missing/checkpoint"},
])
def test_unmeasured_setups_do_not_select_mi355x_rows(model_paths, override):
    assert select(model_paths, **override) is None


@pytest.mark.parametrize("index", [0, 1])
def test_changed_model_config_does_not_select_profile(model_paths, index):
    path = Path(model_paths[index]) / "config.json"
    data = json.loads(path.read_text())
    data["hidden_size"] = 1
    path.write_text(json.dumps(data))
    assert select(model_paths) is None


def test_profile_does_not_depend_on_model_directory_name(model_paths):
    assert select(model_paths, dtype="auto") == str(MI355X_PROFILE)
