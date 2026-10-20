#!/usr/bin/env python3
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Upload immutable engine assets or build a wheel-free Pages site from them.

Uploads require a complete channel's wheels. The checksum manifest is uploaded
last, so an interrupted upload cannot advertise an incomplete channel. Reruns
verify existing bytes instead of replacing assets. Nothing creates a release.
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
UPLOAD_CHANNELS = (*CHANNELS, "harness")
REQUIRED = {
    "harness": {"eagle-harness"},
    "cu130": {"sglang", "vllm", "tokenspeed", "tokenspeed-kernel", "tokenspeed-scheduler"},
    "rocm": {"sglang"},
}
WHEEL_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+\-]*\.whl")


def validate_tag(tag: str) -> None:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise ValueError("release tag must have the form vX.Y.Z")


def gh(*args: str, repository: str = REPOSITORY) -> str:
    return subprocess.check_output(["gh", *args, "--repo", repository], text=True)


def manifest_name(channel: str) -> str:
    return "harness-SHA256SUMS" if channel == "harness" else f"engine-{channel}-SHA256SUMS"


def validate_wheels(channel: str, wheels: list[Path]) -> None:
    names = set()
    projects = set()
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
           draft: bool = False) -> None:
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
        # Preflight every conflict before uploading anything; never use --clobber.
        for path in files:
            if path.name in existing:
                gh("release", "download", tag, "--pattern", path.name, "--dir", str(before), repository=repository)
                verify(before / path.name, checksums[path.name])
        for path in files:
            verify(path, checksums[path.name])
            if path.name not in existing:
                gh("release", "upload", tag, str(path), repository=repository)
            after = work / "after" / path.name
            after.parent.mkdir(exist_ok=True)
            gh("release", "download", tag, "--pattern", path.name, "--output", str(after), repository=repository)
            verify(after, checksums[path.name])
            print(f"verified {tag}/{path.name} sha256={checksums[path.name]}")


def pages(tag: str, output: Path) -> None:
    validate_tag(tag)
    if output.exists():
        raise ValueError(f"output already exists: {output}; use a fresh staging directory")
    release = json.loads(gh("release", "view", tag, "--json", "tagName,isDraft"))
    if release["tagName"] != tag or release["isDraft"]:
        raise ValueError("Pages requires an existing, published release with the exact tag")
    with tempfile.TemporaryDirectory(prefix="engine-index-") as temporary:
        work = Path(temporary)
        site = work / "site"
        for channel in CHANNELS:
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
        (site / "index.html").write_text(
            '<!DOCTYPE html>\n<html><body><a href="whl/cu130/">CUDA 13.0</a>\n'
            '<a href="whl/rocm/">ROCm</a></body></html>\n'
        )
        (site / ".nojekyll").touch()
        shutil.copytree(site, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    upload_parser = commands.add_parser("upload", help="publish wheels to an existing GitHub Release")
    upload_parser.add_argument("--repository", default=REPOSITORY)
    upload_parser.add_argument("--draft", action="store_true", help="require a private repository and a draft release")
    upload_parser.add_argument("tag")
    upload_parser.add_argument("channel", choices=UPLOAD_CHANNELS)
    upload_parser.add_argument("wheels", type=Path, nargs="+")
    pages_parser = commands.add_parser("pages", help="download and verify assets, then generate Pages HTML")
    pages_parser.add_argument("tag")
    pages_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "upload":
            upload(args.tag, args.channel, args.wheels, repository=args.repository, draft=args.draft)
        else:
            pages(args.tag, args.output)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
