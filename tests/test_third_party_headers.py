# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Manifest checks must reject deletion of known third-party headers."""

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

CHECKER = Path(__file__).resolve().parents[1] / "tools" / "check_headers.py"
AMD = "# Copyright (c) 2026 Advanced Micro Devices, Inc.\n# SPDX-License-Identifier: Apache-2.0\n"
MODIFIED = "# Modified by Eagle AI Corp.\n"


def run(tmp_path, text, *, notice=AMD, modified=True, path="derived.py", args=()):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / path
    if text is not None:
        source.write_text(text)
    manifest = tmp_path / "third-party.json"
    manifest.write_text(json.dumps({"files": [{"path": path, "notice": notice, "modified": modified}]}))
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--manifest", str(manifest), "--third-party-only", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    return result, source


def test_missing_amd_header_is_not_relicensed(tmp_path):
    result, source = run(tmp_path, "answer = 42\n", args=("--fix",))
    assert result.returncode == 1
    assert "derived.py: missing original notice" in result.stdout
    assert source.read_text() == "answer = 42\n"
    result, _ = run(tmp_path, AMD + MODIFIED + "\nanswer = 42\n")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "text,reason",
    [
        (AMD + "x = 1\n", "missing modification notice"),
        (AMD.replace("Apache-2.0", "MIT") + MODIFIED, "missing original notice"),
        (AMD + MODIFIED + "# SPDX-License-Identifier: MIT\n", "changed SPDX license"),
        (AMD + MODIFIED + "# Copyright 2026 Eagle AI Corp.\n", "conflicting Eagle ownership"),
        ("x = 1\n" + AMD + MODIFIED, "missing original notice"),
        (None, "missing source"),
    ],
)
def test_manifest_rejects_invalid_source(tmp_path, text, reason):
    result, _ = run(tmp_path, text)
    assert result.returncode == 1
    assert reason in result.stdout


def test_unchanged_upstream_does_not_require_modification_notice(tmp_path):
    result, _ = run(tmp_path, AMD + "x = 1\n", modified=False)
    assert result.returncode == 0, result.stdout


def test_mit_notice_is_preserved_in_full(tmp_path):
    notice = "# Copyright (c) 2026 LightSeek Foundation\n# Permission is hereby granted, free of charge\n"
    result, _ = run(tmp_path, notice + MODIFIED, notice=notice)
    assert result.returncode == 0, result.stdout
    result, _ = run(tmp_path, notice.splitlines()[0] + "\n" + MODIFIED, notice=notice)
    assert result.returncode == 1
    assert "missing original notice" in result.stdout


def test_c_preprocessor_does_not_hide_a_missing_preamble(tmp_path):
    notice = AMD.replace("#", "//")
    text = "#include <stdint.h>\n" + notice + MODIFIED.replace("#", "//")
    result, _ = run(tmp_path, text, notice=notice, path="derived.cu")
    assert result.returncode == 1
    assert "missing original notice" in result.stdout


@pytest.mark.parametrize("prefix", ["#!/usr/bin/env python3\n# coding: utf-8\n", "\ufeff", ""])
def test_valid_preambles_are_read_only(tmp_path, prefix):
    text = prefix + AMD + MODIFIED + "x = 1\n"
    result, source = run(tmp_path, text)
    assert result.returncode == 0, result.stdout
    assert source.read_text() == text


def test_symlink_is_rejected(tmp_path):
    target = tmp_path / "target.py"
    target.write_text(AMD + MODIFIED)
    (tmp_path / "derived.py").symlink_to(target)
    result, _ = run(tmp_path, None)
    assert result.returncode == 1
    assert "unsafe symlink" in result.stdout


def test_exact_first_party_predecessor_is_not_relabelled(tmp_path):
    original = b"answer = 42\n"
    source = tmp_path / "derived.py"
    source.write_bytes(original)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "derived.py",
                        "notice": AMD,
                        "modified": True,
                        "first_party_sha256": [hashlib.sha256(original).hexdigest()],
                    }
                ]
            }
        )
    )
    check = runpy.run_path(str(CHECKER))["check_third_party"]
    assert check(tmp_path, manifest) == (set(), 0)
    source.write_bytes(original + b"adapted = True\n")
    names, errors = check(tmp_path, manifest)
    assert names == {"derived.py"}
    assert errors > 0


def test_missing_manifest_is_not_success(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "absent.json"),
            "--third-party-only",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "missing provenance manifest" in result.stdout
