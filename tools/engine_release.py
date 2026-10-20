#!/usr/bin/env python3
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Upload immutable engine assets or build a wheel-free Pages site from them.

Uploads require a complete channel's wheels. The checksum manifest is uploaded
last, so an interrupted upload cannot advertise an incomplete channel. Reruns
verify existing bytes instead of replacing assets. Nothing creates a release.
Pages indexes exactly the channels whose manifest the release carries.

A channel serves our engine wheels plus, explicitly allow-listed, third-party
wheels we rebuilt or repacked because no official index carries those builds.
Each must match the channel's audited hash lock. ``upload --extend`` adds wheels
to a released channel: the new manifest keeps every released entry unchanged.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import mk_index

REPOSITORY = "eagle-hq/EAGLE-4"
CHANNELS = ("cu130", "rocm")
CHANNEL_LABELS = {"cu130": "CUDA 13.0", "rocm": "ROCm"}
UPLOAD_CHANNELS = (*CHANNELS, "harness", "rocm-runtime")
ROCM_LOCK = Path(__file__).resolve().parents[1] / "docker/rocm/requirements.lock"
REQUIRED = {
    "harness": {"eagle-harness"},
    "cu130": {"sglang", "vllm", "tokenspeed", "tokenspeed-kernel", "tokenspeed-scheduler"},
    "rocm": {"sglang"},
}
WHEEL_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+\-]*\.whl")
# Never list a package users can get from its official index (PyPI, PyTorch, AMD).
THIRD_PARTY = {
    "rocm": {
        # AMD's vendor wheel with only its Triton requirement corrected; tools/repack_rocm_torch.py.
        "torch-2.9.1+rocm7.2.0.lw.git7e1940d4-1eagle-cp310-cp310-linux_x86_64.whl",
        # HIP build for gfx950 (PyPI's 0.4.7 is a CUDA build) plus its upstream THIRDPARTYNOTICES.txt.
        "sglang_kernel-0.4.7-1eagle-cp310-cp310-linux_x86_64.whl",
        # AITER as built in upstream SGLang's MI35x image; AMD publishes no wheel of it.
        "amd_aiter-0.1.21.dev48+g4ad998328.d20260918-cp310-cp310-linux_x86_64.whl",
    },
}


def validate_tag(tag: str) -> None:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise ValueError("release tag must have the form vX.Y.Z")


def gh(*args: str, repository: str = REPOSITORY) -> str:
    return subprocess.check_output(["gh", *args, "--repo", repository], text=True)


def manifest_name(channel: str) -> str:
    if channel == "rocm-runtime":
        return "runtime-rocm-SHA256SUMS"
    return "harness-SHA256SUMS" if channel == "harness" else f"engine-{channel}-SHA256SUMS"


def read_lock(lock: Path) -> dict[str, tuple[str, str]]:
    expected = {}
    for line in lock.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==(\S+) --hash=sha256:([a-f0-9]{64})", line)
        if not match:
            raise ValueError(f"invalid pinned ROCm requirement: {line!r}")
        project = mk_index.normalize(match[1])
        if project in expected:
            raise ValueError(f"duplicate locked ROCm package: {project}")
        expected[project] = (match[2], match[3])
    if not expected:
        raise ValueError("empty ROCm runtime lock")
    return expected


def validate_rocm_runtime(wheels: list[Path], lock: Path) -> None:
    expected = read_lock(lock)
    actual = {}
    for wheel in wheels:
        if not WHEEL_NAME.fullmatch(wheel.name):
            raise ValueError(f"invalid wheel filename: {wheel.name}")
        meta = mk_index.metadata(wheel)
        project = mk_index.normalize(meta["name"])
        if project in actual:
            raise ValueError(f"duplicate ROCm runtime package: {project}")
        if expected.get(project) != (meta["version"], mk_index.sha256(wheel)):
            raise ValueError(f"wheel does not match ROCm runtime lock: {wheel.name}")
        actual[project] = wheel
    missing = expected.keys() - actual.keys()
    if missing:
        raise ValueError(f"missing locked ROCm packages: {', '.join(sorted(missing))}")
    validate_wheels("rocm", [actual["sglang"]] if "sglang" in actual else [], rocm_lock=lock)


def validate_wheels(channel: str, wheels: list[Path], *, rocm_lock: Path | None = None) -> None:
    if channel == "rocm-runtime":
        validate_rocm_runtime(wheels, rocm_lock if rocm_lock is not None else ROCM_LOCK)
        return
    names = set()
    projects = set()
    lock = rocm_lock if rocm_lock is not None else ROCM_LOCK
    for wheel in wheels:
        if not WHEEL_NAME.fullmatch(wheel.name) or wheel.name in names:
            raise ValueError(f"invalid or duplicate wheel filename: {wheel.name}")
        names.add(wheel.name)
        meta = mk_index.metadata(wheel)
        project = mk_index.normalize(meta["name"])
        version = meta["version"]
        if channel == "harness":
            if project != "eagle-harness":
                raise ValueError(f"not a harness wheel: {wheel.name}")
            projects.add(project)
            continue
        if wheel.name in THIRD_PARTY.get(channel, ()):
            if read_lock(lock).get(project) != (version, mk_index.sha256(wheel)):
                raise ValueError(f"third-party wheel does not match the ROCm lock: {wheel.name}")
            continue
        if project == "eagle-harness" or "+eagle." not in version:
            raise ValueError(f"not an Eagle engine wheel: {wheel.name}")
        if (".rocm" in version) != (channel == "rocm"):
            raise ValueError(f"wheel belongs to the other platform channel: {wheel.name}")
        projects.add(project)
    missing = REQUIRED[channel] - projects
    if missing:
        raise ValueError(f"{channel}: missing engine wheels: {', '.join(sorted(missing))}")


def read_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        if not match or not WHEEL_NAME.fullmatch(match[2]):
            raise ValueError(f"invalid checksum entry in {path.name}: {line!r}")
        digest, name = match.groups()
        if name in entries:
            raise ValueError(f"duplicate checksum entry: {name}")
        entries[name] = digest
    if not entries:
        raise ValueError(f"empty checksum manifest: {path.name}")
    return entries


def verify(path: Path, digest: str) -> None:
    if mk_index.sha256(path) != digest:
        raise ValueError(f"SHA256 mismatch: {path.name}; released assets must never change")


def upload(tag: str, channel: str, wheels: list[Path], *, repository: str = REPOSITORY,
           draft: bool = False, extend: bool = False) -> None:
    validate_tag(tag)
    validate_wheels(channel, wheels)
    if channel == "harness" and any(mk_index.metadata(w)["version"] != tag.removeprefix("v") for w in wheels):
        raise ValueError("harness wheel version must equal the release tag")
    wheels = sorted(wheels, key=lambda p: p.name)
    checksums = {p.name: mk_index.sha256(p) for p in wheels}
    if draft:
        repo = json.loads(subprocess.check_output(
            ["gh", "repo", "view", repository, "--json", "isPrivate"], text=True))
        if not repo["isPrivate"]:
            raise ValueError("draft upload requires a private repository")
    release = json.loads(gh("release", "view", tag, "--json", "tagName,isDraft,assets", repository=repository))
    if release["tagName"] != tag or release["isDraft"] != draft:
        state = "draft" if draft else "published"
        raise ValueError(f"upload requires the existing, {state} release with the exact tag")
    existing = {asset["name"] for asset in release["assets"]}
    with tempfile.TemporaryDirectory(prefix="engine-upload-") as temporary:
        work = Path(temporary)
        manifest = work / manifest_name(channel)
        manifest.write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
        files = [*wheels, manifest]
        checksums[manifest.name] = mk_index.sha256(manifest)
        before = work / "before"
        before.mkdir()
        replace_manifest = False
        # Preflight every conflict before uploading anything. Only --extend may replace the
        # manifest, and only with a superset whose released entries keep their digests.
        for path in files:
            if path.name in existing:
                gh("release", "download", tag, "--pattern", path.name, "--dir", str(before), repository=repository)
                if path is manifest and extend and mk_index.sha256(before / path.name) != checksums[path.name]:
                    released = read_manifest(before / path.name)
                    kept = {name: digest for name, digest in checksums.items() if name in released}
                    if kept != released:
                        raise ValueError(f"--extend must keep every entry of {path.name} unchanged")
                    replace_manifest = True
                    continue
                verify(before / path.name, checksums[path.name])
        for path in files:
            verify(path, checksums[path.name])
            if path.name not in existing:
                gh("release", "upload", tag, str(path), repository=repository)
            elif path is manifest and replace_manifest:
                gh("release", "upload", tag, str(path), "--clobber", repository=repository)
            after = work / "after" / path.name
            after.parent.mkdir(exist_ok=True)
            gh("release", "download", tag, "--pattern", path.name, "--output", str(after), repository=repository)
            verify(after, checksums[path.name])
            print(f"verified {tag}/{path.name} sha256={checksums[path.name]}")


def pages(tag: str, output: Path) -> None:
    validate_tag(tag)
    if output.exists():
        raise ValueError(f"output already exists: {output}; use a fresh staging directory")
    release = json.loads(gh("release", "view", tag, "--json", "tagName,isDraft,assets"))
    if release["tagName"] != tag or release["isDraft"]:
        raise ValueError("Pages requires an existing, published release with the exact tag")
    # A manifest is uploaded only after its whole channel verified, so its presence marks a released channel.
    published = {asset["name"] for asset in release["assets"]}
    channels = [channel for channel in CHANNELS if manifest_name(channel) in published]
    if not channels:
        raise ValueError(f"{tag} has no engine channel manifest; upload a complete channel first")
    with tempfile.TemporaryDirectory(prefix="engine-index-") as temporary:
        work = Path(temporary)
        site = work / "site"
        for channel in channels:
            assets = work / channel
            assets.mkdir()
            manifest = manifest_name(channel)
            gh("release", "download", tag, "--pattern", manifest, "--dir", str(assets))
            entries = read_manifest(assets / manifest)
            wheels = []
            for name, digest in entries.items():
                gh("release", "download", tag, "--pattern", name, "--dir", str(assets))
                wheel = assets / name
                verify(wheel, digest)
                wheels.append(wheel)
            validate_wheels(channel, wheels)
            index = work / f"index-{channel}"
            mk_index.main(index, wheels, f"https://github.com/{REPOSITORY}/releases/download/{tag}")
            destination = site / "whl" / channel
            shutil.copytree(index / "simple", destination)
            shutil.copyfile(assets / manifest, destination / "SHA256SUMS")
        links = "".join(f'<a href="whl/{channel}/">{CHANNEL_LABELS[channel]}</a>\n' for channel in channels)
        (site / "index.html").write_text(f"<!DOCTYPE html>\n<html><body>{links}</body></html>\n")
        (site / ".nojekyll").touch()
        shutil.copytree(site, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    upload_parser = commands.add_parser("upload", help="publish wheels to an existing GitHub Release")
    upload_parser.add_argument("--repository", default=REPOSITORY)
    upload_parser.add_argument("--draft", action="store_true", help="require a private repository and a draft release")
    upload_parser.add_argument("--extend", action="store_true",
                               help="add wheels to a released channel; its released entries must not change")
    upload_parser.add_argument("tag")
    upload_parser.add_argument("channel", choices=UPLOAD_CHANNELS)
    upload_parser.add_argument("wheels", type=Path, nargs="+")
    pages_parser = commands.add_parser("pages", help="download and verify assets, then generate Pages HTML")
    pages_parser.add_argument("tag")
    pages_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "upload":
            upload(args.tag, args.channel, args.wheels, repository=args.repository, draft=args.draft,
                   extend=args.extend)
        else:
            pages(args.tag, args.output)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
