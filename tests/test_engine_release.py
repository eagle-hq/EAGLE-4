# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Exercise the real publisher CLI against an isolated, file-backed gh fake."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools/engine_release.py"
TAG = "v0.1.0"


@pytest.fixture
def release(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    log = tmp_path / "gh.jsonl"
    fake = tmp_path / "bin/gh"
    fake.parent.mkdir()
    fake.write_text(
        f"#!{sys.executable}\n"
        + """
import json
import os
from pathlib import Path
import shutil
import sys
args = sys.argv[1:]
with open(os.environ["GH_TEST_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
repository = os.environ.get("GH_TEST_REPOSITORY", "eagle-hq/EAGLE-4")
if args[:2] == ["repo", "view"]:
    assert args[2:] == [repository, "--json", "isPrivate"], args
    print(json.dumps({"isPrivate": os.environ.get("GH_TEST_PRIVATE") == "true"}))
    sys.exit(0)
assert args[-2:] == ["--repo", repository], args
args = args[:-2]
assert args[0] == "release" and args[2] == "v0.1.0", args
remote = Path(os.environ["GH_TEST_REMOTE"])
if args[1] == "view":
    print(json.dumps({"tagName": "v0.1.0", "isDraft": os.environ.get("GH_TEST_DRAFT") == "true",
                      "assets": [{"name": p.name} for p in remote.iterdir()]}))
elif args[1] == "upload":
    assert len(args) == 4, args
    source = Path(args[3])
    target = remote / source.name
    assert not target.exists(), "no clobber"
    shutil.copyfile(source, target)
elif args[1] == "download":
    name = args[args.index("--pattern") + 1]
    if "--dir" in args:
        target = Path(args[args.index("--dir") + 1]) / name
    else:
        target = Path(args[args.index("--output") + 1])
    shutil.copyfile(remote / name, target)
    if os.environ.get("GH_TEST_CORRUPT") == name:
        target.write_bytes(b"corrupted transfer")
else:
    raise AssertionError(args)
"""
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GH_TEST_REMOTE", str(remote))
    monkeypatch.setenv("GH_TEST_LOG", str(log))
    return remote, log


def wheels(tmp_path, channel):
    versions = (
        {"sglang": "0.5.23.dev252+eagle.4.rocm"}
        if channel == "rocm"
        else {
            "sglang": "0.5.23.dev252+eagle.4",
            "vllm": "0.32.0rc1.dev283+eagle.4",
            "tokenspeed": "0.1.1+eagle.4",
            "tokenspeed-kernel": "0.1.1+eagle.4",
            "tokenspeed-scheduler": "0.1.1+eagle.4",
        }
    )
    result = []
    for name, version in versions.items():
        path = tmp_path / f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{name}.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
        result.append(path)
    return result


def command(*args):
    return subprocess.run([sys.executable, str(TOOL), *(str(a) for a in args)], text=True, capture_output=True)


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_upload_verify_rerun_and_pages_without_wheel_payloads(tmp_path, release):
    remote, log = release
    expected = set()
    for channel in ("cu130", "rocm"):
        artifacts = wheels(tmp_path, channel)
        result = command("upload", TAG, channel, *artifacts)
        assert result.returncode == 0, result.stderr
        expected.update(p.name for p in artifacts)
        expected.add(f"engine-{channel}-SHA256SUMS")
        uploads = [args[3] for args in calls(log) if args[1] == "upload"]
        assert Path(uploads[-1]).name == f"engine-{channel}-SHA256SUMS"
        result = command("upload", TAG, channel, *artifacts)
        assert result.returncode == 0, result.stderr
        assert [args[3] for args in calls(log) if args[1] == "upload"] == uploads
    assert {p.name for p in remote.iterdir()} == expected
    site = tmp_path / "site"
    result = command("pages", TAG, site)
    assert result.returncode == 0, result.stderr
    assert not list(site.rglob("*.whl"))
    for channel in ("cu130", "rocm"):
        assert (site / "whl" / channel / "SHA256SUMS").read_bytes() == (
            remote / f"engine-{channel}-SHA256SUMS"
        ).read_bytes()
        page = (site / "whl" / channel / "sglang/index.html").read_text()
        assert "https://github.com/eagle-hq/EAGLE-4/releases/download/v0.1.0/sglang-" in page
        assert "#sha256=" in page
        assert ("eagle.4.rocm" in page) == (channel == "rocm")


def test_pages_publishes_only_released_channels(tmp_path, release):
    remote, _ = release
    artifacts = wheels(tmp_path, "cu130")
    assert command("upload", TAG, "cu130", *artifacts).returncode == 0
    # Neither a harness manifest nor an unknown platform's manifest is an index channel.
    (remote / "harness-SHA256SUMS").write_text(f"{'0' * 64}  eagle_harness-0.1.0-py3-none-any.whl\n")
    (remote / "engine-cu999-SHA256SUMS").write_text(f"{'0' * 64}  invented-0.1.0-py3-none-any.whl\n")
    site = tmp_path / "site"
    result = command("pages", TAG, site)
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in (site / "whl").iterdir()) == ["cu130"]
    released = {p.name for p in remote.iterdir()}
    for page in site.rglob("*.html"):
        for href in re.findall(r'href="([^"]+)"', page.read_text()):
            url = href.split("#", 1)[0]
            if url.startswith("https://"):
                assert unquote(url.rsplit("/", 1)[1]) in released, (page, href)
            else:
                assert (page.parent / url).exists(), (page, href)


def test_pages_requires_a_released_channel(tmp_path, release):
    remote, log = release
    (remote / "harness-SHA256SUMS").write_text(f"{'0' * 64}  eagle_harness-0.1.0-py3-none-any.whl\n")
    result = command("pages", TAG, tmp_path / "site")
    assert result.returncode != 0
    assert "no engine channel manifest" in result.stderr
    assert not (tmp_path / "site").exists()
    assert not any(args[1] == "download" for args in calls(log))


def test_upload_conflict_preflight_has_no_write_side_effects(tmp_path, release):
    remote, log = release
    artifacts = wheels(tmp_path, "cu130")
    (remote / artifacts[-1].name).write_bytes(b"other released bytes")
    result = command("upload", TAG, "cu130", *artifacts)
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr
    assert not any(args[1] == "upload" for args in calls(log))
    assert len(list(remote.iterdir())) == 1


def test_upload_verifies_download_before_finalizing_manifest(tmp_path, release, monkeypatch):
    remote, _ = release
    artifacts = wheels(tmp_path, "rocm")
    monkeypatch.setenv("GH_TEST_CORRUPT", artifacts[0].name)
    result = command("upload", TAG, "rocm", *artifacts)
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr
    assert not (remote / "engine-rocm-SHA256SUMS").exists()


def test_manifest_changes_are_refused_without_uploading(tmp_path, release):
    remote, log = release
    artifacts = wheels(tmp_path, "rocm")
    (remote / "engine-rocm-SHA256SUMS").write_text("a different finalized channel\n")
    result = command("upload", TAG, "rocm", *artifacts)
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr
    assert not any(args[1] == "upload" for args in calls(log))


@pytest.mark.parametrize("case", ["missing-companions", "wrong-channel", "harness", "duplicate", "upstream"])
def test_invalid_upload_is_rejected_before_contacting_github(tmp_path, release, case):
    _, log = release
    artifacts = wheels(tmp_path, "cu130")
    channel = "cu130"
    if case == "missing-companions":
        artifacts = artifacts[:1]
    elif case == "wrong-channel":
        channel = "rocm"
    elif case == "duplicate":
        artifacts.append(artifacts[0])
    else:
        name = "eagle-harness" if case == "harness" else "stock-engine"
        version = "0.1.0" if case == "upstream" else "0.1.0+eagle.4"
        with zipfile.ZipFile(artifacts[0], "w") as archive:
            archive.writestr("test.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
    result = command("upload", TAG, channel, *artifacts)
    assert result.returncode != 0
    assert not log.exists()


@pytest.mark.parametrize("entry", ["../escape.whl", "nested/file.whl", "-option.whl", "file[1].whl"])
def test_pages_rejects_unsafe_manifest_paths(tmp_path, release, entry):
    remote, log = release
    (remote / "engine-cu130-SHA256SUMS").write_text(f"{'0' * 64}  {entry}\n")
    result = command("pages", TAG, tmp_path / "site")
    assert result.returncode != 0
    assert "invalid checksum entry" in result.stderr
    assert not (tmp_path / "site").exists()
    assert not any(entry in args for args in calls(log))


def test_pages_checksum_mismatch_leaves_no_partial_site(tmp_path, release):
    remote, _ = release
    artifacts = wheels(tmp_path, "cu130")
    assert command("upload", TAG, "cu130", *artifacts).returncode == 0
    (remote / artifacts[0].name).write_bytes(b"wrong bytes")
    result = command("pages", TAG, tmp_path / "site")
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr
    assert not (tmp_path / "site").exists()


def test_manifest_hash_is_of_uploaded_bytes(tmp_path, release):
    remote, _ = release
    artifacts = wheels(tmp_path, "rocm")
    result = command("upload", TAG, "rocm", *artifacts)
    assert result.returncode == 0, result.stderr
    digest = hashlib.sha256((remote / artifacts[0].name).read_bytes()).hexdigest()
    assert (remote / "engine-rocm-SHA256SUMS").read_text() == f"{digest}  {artifacts[0].name}\n"


@pytest.mark.parametrize("version,ok", [("0.1.0", True), ("0.2.0", False)])
def test_private_harness_wheels_have_tag_version(tmp_path, release, monkeypatch, version, ok):
    remote, _ = release
    monkeypatch.setenv("GH_TEST_PRIVATE", "true")
    monkeypatch.setenv("GH_TEST_DRAFT", "true")
    artifacts = []
    for platform in ("any", "manylinux_2_34_x86_64"):
        wheel = tmp_path / f"eagle_harness-{version}-py3-none-{platform}.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("eagle_harness.dist-info/METADATA", f"Name: eagle-harness\nVersion: {version}\n")
        artifacts.append(wheel)
    result = command("upload", "--draft", TAG, "harness", *artifacts)
    assert (result.returncode == 0) == ok, result.stderr
    if ok:
        assert (remote / "harness-SHA256SUMS").exists()
        assert len(list(remote.glob("*.whl"))) == 2
    else:
        assert "version must equal" in result.stderr
        assert not list(remote.iterdir())


def test_harness_channel_refuses_engines(tmp_path, release):
    remote, _ = release
    result = command("upload", TAG, "harness", *wheels(tmp_path, "rocm"))
    assert result.returncode != 0
    assert "not a harness wheel" in result.stderr
    assert not list(remote.iterdir())


def test_private_draft_upload_verifies_bytes_without_publishing(tmp_path, release, monkeypatch):
    remote, log = release
    monkeypatch.setenv("GH_TEST_REPOSITORY", "example/private-stage")
    monkeypatch.setenv("GH_TEST_PRIVATE", "true")
    monkeypatch.setenv("GH_TEST_DRAFT", "true")
    artifacts = wheels(tmp_path, "rocm")
    args = ("upload", "--repository", "example/private-stage", "--draft", TAG, "rocm", *artifacts)
    result = command(*args)
    assert result.returncode == 0, result.stderr
    assert command(*args).returncode == 0
    assert (remote / artifacts[0].name).read_bytes() == artifacts[0].read_bytes()
    assert len([c for c in calls(log) if c[:2] == ["release", "upload"]]) == 2
    assert not any(c[:2] in (["release", "create"], ["release", "edit"]) for c in calls(log))


@pytest.mark.parametrize("private,draft", [(False, True), (True, False)])
def test_draft_upload_refuses_public_repo_or_published_release(tmp_path, release, monkeypatch, private, draft):
    remote, log = release
    monkeypatch.setenv("GH_TEST_PRIVATE", str(private).lower())
    monkeypatch.setenv("GH_TEST_DRAFT", str(draft).lower())
    result = command("upload", "--draft", TAG, "rocm", *wheels(tmp_path, "rocm"))
    assert result.returncode != 0
    assert "private" in result.stderr or "draft" in result.stderr
    assert not list(remote.iterdir())
    assert not any(c[:2] == ["release", "upload"] for c in calls(log))


def test_draft_requires_explicit_opt_in_and_cannot_make_pages(tmp_path, release, monkeypatch):
    monkeypatch.setenv("GH_TEST_DRAFT", "true")
    result = command("upload", TAG, "rocm", *wheels(tmp_path, "rocm"))
    assert result.returncode != 0
    assert "published" in result.stderr
    result = command("pages", TAG, tmp_path / "site")
    assert result.returncode != 0
    assert not (tmp_path / "site").exists()
