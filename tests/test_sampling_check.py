# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Serving distribution gates must reject missing or incomparable evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def sampling_check():
    path = Path(__file__).parents[1] / "tools" / "sampling_check.py"
    spec = importlib.util.spec_from_file_location("sampling_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact():
    return {
        "api": "sglang",
        "max_new_tokens": 3,
        "samples": 20,
        "settings": {
            "unrestricted": {
                "params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
                "tokens": [[[1, 2, 3]] * 20 for _ in range(4)],
                "seconds": 1.0,
            }
        },
        "seeds": {},
    }


def compare(module, tmp_path, ref, candidate):
    reference = tmp_path / "reference.json"
    output = tmp_path / "candidate.json"
    reference.write_text(json.dumps(ref))
    output.write_text(json.dumps(candidate))
    return module.compare(SimpleNamespace(reference=reference, candidate=output, alpha=0.01))


def test_unrestricted_setting_exercises_temperature_one(sampling_check):
    assert sampling_check.SETTINGS["unrestricted"] == {"temperature": 1.0, "top_p": 1.0, "top_k": -1}


def test_complete_distribution_evidence_passes(sampling_check, tmp_path):
    ref = artifact()
    assert compare(sampling_check, tmp_path, ref, copy.deepcopy(ref)) == 0


@pytest.mark.parametrize("fault", ["empty", "missing", "refused", "parameters", "prompts", "samples", "tokens"])
def test_incomplete_distribution_evidence_fails(sampling_check, tmp_path, fault):
    ref = artifact()
    candidate = copy.deepcopy(ref)
    setting = candidate["settings"]["unrestricted"]
    if fault == "empty":
        ref["settings"].clear()
        candidate["settings"].clear()
    elif fault == "missing":
        candidate["settings"].clear()
    elif fault == "refused":
        setting.clear()
        setting["refused"] = "unsupported"
    elif fault == "parameters":
        setting["params"]["temperature"] = 0.0
    elif fault == "prompts":
        setting["tokens"].pop()
    elif fault == "samples":
        setting["tokens"][0].pop()
    elif fault == "tokens":
        setting["tokens"][0][0] = [1]
    assert compare(sampling_check, tmp_path, ref, candidate) == 1


def test_changed_distribution_fails(sampling_check, tmp_path):
    ref = artifact()
    candidate = copy.deepcopy(ref)
    candidate["settings"]["unrestricted"]["tokens"] = [[[4, 5, 6]] * 20 for _ in range(4)]
    assert compare(sampling_check, tmp_path, ref, candidate) == 1


def test_compare_runs_without_the_http_client(tmp_path):
    # Only `run` talks HTTP (requests); the public test env has no requests.
    path = Path(__file__).parents[1] / "tools" / "sampling_check.py"
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(artifact()))
    code = (
        "import importlib.util, sys, types; sys.modules['requests'] = None; "
        f"spec = importlib.util.spec_from_file_location('sampling_check', {str(path)!r}); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        f"sys.exit(m.compare(types.SimpleNamespace(reference={str(ref)!r}, candidate={str(ref)!r}, alpha=0.01)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
