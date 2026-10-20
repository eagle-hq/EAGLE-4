# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Exercise the real publisher CLI against an isolated, file-backed gh fake."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
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
    source = Path(args[3])
    target = remote / source.name
    if args[4:] == ["--clobber"]:
        assert source.name.endswith("-SHA256SUMS"), "only a manifest is ever replaced"
    else:
        assert len(args) == 4, args
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


@pytest.fixture
def rocm_runtime(tmp_path, monkeypatch):
    source = tmp_path / "audited-source"
    (source / "tools").mkdir(parents=True)
    (source / "docker/rocm").mkdir(parents=True)
    for name in ("engine_release.py", "mk_index.py"):
        shutil.copyfile(REPO / "tools" / name, source / "tools" / name)
    monkeypatch.setattr(sys.modules[__name__], "TOOL", source / "tools/engine_release.py")
    artifacts = wheels(tmp_path, "rocm")
    torch = tmp_path / "torch-2.9.1-py3-none-any.whl"
    with zipfile.ZipFile(torch, "w") as archive:
        archive.writestr("torch.dist-info/METADATA", "Name: torch\nVersion: 2.9.1\n")
    artifacts.append(torch)
    lock = source / "docker/rocm/requirements.lock"
    lock.write_text("".join(
        f"{p.name.split('-')[0]}=={p.name.split('-')[1]} --hash=sha256:{hashlib.sha256(p.read_bytes()).hexdigest()}\n"
        for p in artifacts
    ))
    return artifacts, lock


def test_runtime_upload_verifies_complete_locked_bundle(tmp_path, release, rocm_runtime):
    remote, log = release
    artifacts, _ = rocm_runtime
    result = command("upload", TAG, "rocm-runtime", *artifacts)
    assert result.returncode == 0, result.stderr
    assert {p.name for p in remote.iterdir()} == {p.name for p in artifacts} | {"runtime-rocm-SHA256SUMS"}
    uploads = [args[3] for args in calls(log) if args[1] == "upload"]
    assert Path(uploads[-1]).name == "runtime-rocm-SHA256SUMS"
    assert command("upload", TAG, "rocm-runtime", *artifacts).returncode == 0
    assert [args[3] for args in calls(log) if args[1] == "upload"] == uploads
    for artifact in artifacts:
        assert (remote / artifact.name).read_bytes() == artifact.read_bytes()


@pytest.mark.parametrize("case", ["missing", "wrong-version", "empty-lock", "unhashed", "duplicate-lock"])
def test_runtime_upload_rejects_bad_closure_before_github(release, rocm_runtime, case):
    _, log = release
    artifacts, lock = rocm_runtime
    if case == "missing":
        artifacts = artifacts[:1]
    elif case == "wrong-version":
        lock.write_text(lock.read_text().replace("torch==2.9.1", "torch==2.9.2"))
    elif case == "empty-lock":
        lock.write_text("")
    elif case == "unhashed":
        lock.write_text("torch==2.9.1\n")
    else:
        lock.write_text(lock.read_text() * 2)
    result = command("upload", TAG, "rocm-runtime", *artifacts)
    assert result.returncode != 0
    assert "ROCm" in result.stderr
    assert not log.exists()


def test_pages_publishes_only_released_channels(tmp_path, release):
    remote, _ = release
    artifacts = wheels(tmp_path, "cu130")
    assert command("upload", TAG, "cu130", *artifacts).returncode == 0
    # Neither a harness or ROCm runtime manifest nor an unknown platform's manifest is an index channel.
    (remote / "harness-SHA256SUMS").write_text(f"{'0' * 64}  eagle_harness-0.1.0-py3-none-any.whl\n")
    (remote / "runtime-rocm-SHA256SUMS").write_text(f"{'0' * 64}  torch-2.9.1-py3-none-any.whl\n")
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


ALLOWED_THIRD_PARTY = (
    ("torch", "2.9.1+rocm7.2.0.lw.git7e1940d4", "torch-2.9.1+rocm7.2.0.lw.git7e1940d4-1eagle-cp310-cp310-linux_x86_64.whl"),
    ("sglang-kernel", "0.4.7", "sglang_kernel-0.4.7-1eagle-cp310-cp310-linux_x86_64.whl"),
)


@pytest.fixture
def third_party(tmp_path, monkeypatch):
    """Allow-listed third-party wheels plus an audited source tree whose lock pins their hashes."""
    source = tmp_path / "audited-source"
    (source / "tools").mkdir(parents=True)
    (source / "docker/rocm").mkdir(parents=True)
    for name in ("engine_release.py", "mk_index.py"):
        shutil.copyfile(REPO / "tools" / name, source / "tools" / name)
    monkeypatch.setattr(sys.modules[__name__], "TOOL", source / "tools/engine_release.py")
    extra = []
    for project, version, filename in ALLOWED_THIRD_PARTY:
        path = tmp_path / filename
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("x.dist-info/METADATA", f"Name: {project}\nVersion: {version}\n")
        extra.append(path)
    lock = source / "docker/rocm/requirements.lock"
    lock.write_text("".join(
        f"{project}=={version} --hash=sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        for (project, version, _), path in zip(ALLOWED_THIRD_PARTY, extra)
    ))
    return extra, lock


def test_extend_adds_allow_listed_third_party_wheels_and_indexes_them(tmp_path, release, third_party):
    remote, log = release
    extra, _ = third_party
    engine = wheels(tmp_path, "rocm")
    assert command("upload", TAG, "rocm", *engine).returncode == 0
    released = (remote / "engine-rocm-SHA256SUMS").read_text()
    # Without --extend a released manifest never changes.
    result = command("upload", TAG, "rocm", *engine, *extra)
    assert result.returncode != 0 and "SHA256 mismatch" in result.stderr
    assert (remote / "engine-rocm-SHA256SUMS").read_text() == released
    result = command("upload", "--extend", TAG, "rocm", *engine, *extra)
    assert result.returncode == 0, result.stderr
    manifest = (remote / "engine-rocm-SHA256SUMS").read_text()
    assert released in manifest and len(manifest.splitlines()) == 3
    uploads = [args for args in calls(log) if args[1] == "upload"]
    assert "--clobber" in uploads[-1] and uploads[-1][3].endswith("engine-rocm-SHA256SUMS")
    assert not any("--clobber" in args for args in uploads[:-1])
    assert {Path(args[3]).name for args in uploads[-3:-1]} == {p.name for p in extra}
    # A rerun verifies bytes and uploads nothing.
    assert command("upload", "--extend", TAG, "rocm", *engine, *extra).returncode == 0
    assert [args for args in calls(log) if args[1] == "upload"] == uploads
    site = tmp_path / "site"
    result = command("pages", TAG, site)
    assert result.returncode == 0, result.stderr
    for project, _, filename in ALLOWED_THIRD_PARTY:
        page = (site / "whl/rocm" / project / "index.html").read_text()
        digest = hashlib.sha256((remote / filename).read_bytes()).hexdigest()
        assert f"releases/download/{TAG}/{filename.replace('+', '%2B')}#sha256={digest}" in page
    assert not list(site.rglob("*.whl"))


@pytest.mark.parametrize("case", ["drop-released", "change-released"])
def test_extend_never_drops_or_changes_released_entries(tmp_path, release, third_party, case):
    remote, log = release
    extra, _ = third_party
    engine = wheels(tmp_path, "rocm")
    assert command("upload", TAG, "rocm", *engine, extra[0]).returncode == 0
    released = (remote / "engine-rocm-SHA256SUMS").read_bytes()
    uploads = [args for args in calls(log) if args[1] == "upload"]
    if case == "drop-released":
        result = command("upload", "--extend", TAG, "rocm", *engine, extra[1])
        assert "--extend must keep every entry" in result.stderr
    else:
        with zipfile.ZipFile(engine[0], "a") as archive:
            archive.writestr("extra.txt", "rebuilt")
        result = command("upload", "--extend", TAG, "rocm", *engine, *extra)
        assert "SHA256 mismatch" in result.stderr
    assert result.returncode != 0
    assert (remote / "engine-rocm-SHA256SUMS").read_bytes() == released
    assert [args for args in calls(log) if args[1] == "upload"] == uploads


@pytest.mark.parametrize("case", ["unlisted-upstream", "hash-not-locked", "other-channel"])
def test_third_party_wheels_must_be_allow_listed_and_locked(tmp_path, release, third_party, case):
    _, log = release
    extra, lock = third_party
    channel, artifacts = "rocm", [*wheels(tmp_path, "rocm"), *extra]
    if case == "unlisted-upstream":
        # torchvision is available from AMD's index, so it is never served here.
        path = tmp_path / "torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp310-cp310-linux_x86_64.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("x.dist-info/METADATA", "Name: torchvision\nVersion: 0.24.0+rocm7.2.0.gitb919bd0c\n")
        lock.write_text(lock.read_text() + f"torchvision==0.24.0+rocm7.2.0.gitb919bd0c --hash=sha256:"
                        f"{hashlib.sha256(path.read_bytes()).hexdigest()}\n")
        artifacts.append(path)
        expected = "not an Eagle engine wheel"
    elif case == "hash-not-locked":
        lock.write_text(lock.read_text().replace(hashlib.sha256(extra[0].read_bytes()).hexdigest(), "0" * 64))
        expected = "does not match the ROCm lock"
    else:
        channel, artifacts = "cu130", [*wheels(tmp_path, "cu130"), *extra]
        expected = "not an Eagle engine wheel"
    result = command("upload", TAG, channel, *artifacts)
    assert result.returncode != 0
    assert expected in result.stderr
    assert not log.exists()


def test_allow_list_is_pinned_by_the_shipped_rocm_lock():
    spec = importlib.util.spec_from_file_location("engine_release", TOOL)
    sys.path.insert(0, str(TOOL.parent))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(TOOL.parent))
    lock = module.read_lock(module.ROCM_LOCK)
    assert set(module.THIRD_PARTY) == {"rocm"}
    for filename in module.THIRD_PARTY["rocm"]:
        name, version = filename.split("-")[:2]
        assert lock[module.mk_index.normalize(name)][0] == version, filename
