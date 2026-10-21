# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""tools/release_files.py against a file-backed gh fake: stage, fetch, remove."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_files", REPO / "tools/release_files.py")
release_files = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_files)

TAG = "v0.1.1"
NAMES = ("eagle_harness-0.1.1.tar.gz", "eagle_harness-0.1.1-py3-none-manylinux_2_34_x86_64.whl",
         "eagle_harness-0.1.1-py3-none-any.whl")
GH = """
import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
assert args[-2:] == ["--repo", "eagle-hq/EAGLE-4"], args
args = args[:-2]
remote = Path(os.environ["GH_TEST_REMOTE"])
with open(os.environ["GH_TEST_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
assert args[0] == "release" and args[2] == "v0.1.1", args
if args[1] == "view":
    print(json.dumps({"tagName": "v0.1.1", "isDraft": os.environ.get("GH_TEST_DRAFT") == "true",
                      "assets": [{"name": p.name} for p in remote.iterdir()]}))
elif args[1] == "upload":
    assert len(args) == 4, args
    target = remote / Path(args[3]).name
    assert not target.exists(), "no clobber"
    shutil.copyfile(args[3], target)
elif args[1] == "download":
    name = args[args.index("--pattern") + 1]
    target = Path(args[args.index("--output") + 1])
    shutil.copyfile(remote / name, target)
    if os.environ.get("GH_TEST_CORRUPT") == name:
        target.write_bytes(b"corrupted transfer")
elif args[1] == "delete-asset":
    assert args[4:] == ["--yes"], args
    (remote / args[3]).unlink()
else:
    raise AssertionError(args)
"""


@pytest.fixture
def release(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    fake = tmp_path / "bin/gh"
    fake.parent.mkdir()
    fake.write_text(f"#!{sys.executable}\n{GH}")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GH_TEST_REMOTE", str(remote))
    monkeypatch.setenv("GH_TEST_LOG", str(tmp_path / "gh.jsonl"))
    monkeypatch.setenv("GH_TEST_DRAFT", "true")
    return remote, tmp_path / "gh.jsonl"


def dist(tmp_path):
    root = tmp_path / "dist"
    (root / "pure").mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        (root / "pure" / name if name.endswith("-any.whl") else root / name).write_bytes(name.encode() * 3)
    (root / "SHA256SUMS").write_text("ignored\n")
    (root / "build.txt").write_text("ignored\n")
    return root


def uploads(log):
    return [Path(json.loads(line)[3]).name for line in log.read_text().splitlines() if json.loads(line)[1] == "upload"]


def test_stage_fetch_round_trip(tmp_path, release):
    remote, log = release
    staged = release_files.stage(TAG, dist(tmp_path))
    assert sorted(staged) == sorted(NAMES)
    assert uploads(log)[-1] == release_files.MANIFEST  # the manifest goes last
    assert release_files.stage(TAG, dist(tmp_path)) == staged  # a rerun verifies and uploads nothing
    assert len(uploads(log)) == 4
    out = tmp_path / "out"
    assert release_files.fetch(TAG, out) == staged
    assert (out / "pure" / NAMES[2]).read_bytes() == NAMES[2].encode() * 3
    assert {p.name for p in out.iterdir()} == {NAMES[0], NAMES[1], "pure", "SHA256SUMS"}


def test_stage_refuses_a_published_release_and_a_different_existing_file(tmp_path, release, monkeypatch):
    remote, _ = release
    monkeypatch.setenv("GH_TEST_DRAFT", "false")
    with pytest.raises(ValueError, match="published"):
        release_files.stage(TAG, dist(tmp_path))
    monkeypatch.setenv("GH_TEST_DRAFT", "true")
    (remote / NAMES[0]).write_bytes(b"other bytes")
    with pytest.raises(ValueError, match="refusing to replace"):
        release_files.stage(TAG, dist(tmp_path))
    assert {p.name for p in remote.iterdir()} == {NAMES[0]}


def test_fetch_fails_on_a_corrupt_file(tmp_path, release, monkeypatch):
    release_files.stage(TAG, dist(tmp_path))
    monkeypatch.setenv("GH_TEST_CORRUPT", NAMES[1])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        release_files.fetch(TAG, tmp_path / "out")


@pytest.mark.parametrize("names", [NAMES[:2], (*NAMES, "eagle_harness-0.1.1-cp312-cp312-win_amd64.whl"),
                                   ("eagle_harness-0.1.0.tar.gz", *NAMES[1:])])
def test_classify_wants_exactly_the_three_files_of_the_version(names):
    with pytest.raises(ValueError):
        release_files.classify(list(names), "0.1.1")


def test_remove_waits_for_pypi_digests(tmp_path, release):
    remote, _ = release
    staged = release_files.stage(TAG, dist(tmp_path))
    with pytest.raises(ValueError, match="PyPI does not serve"):
        release_files.remove(TAG, published={**staged, NAMES[1]: "0" * 64})
    assert len(list(remote.iterdir())) == 4
    removed = release_files.remove(TAG, published=staged)
    assert removed == [*sorted(NAMES), release_files.MANIFEST]
    assert not list(remote.iterdir())
