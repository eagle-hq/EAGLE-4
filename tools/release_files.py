#!/usr/bin/env python3
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Carry the qualified eagle-harness files to PyPI through the GitHub release.

    release_files.py stage <tag> <dist dir>    upload the sdist, the binary wheel and the pure wheel to the DRAFT
                                               release, then harness-SHA256SUMS; read every file back
    release_files.py fetch <tag> <out dir>     download them, verify every SHA256, lay them out as build_wheel.sh does
    release_files.py remove <tag>              once PyPI serves the same SHA256s, delete them from the release

The release files are built and GPU-qualified before the release (the release images install the same wheel), so PyPI
receives exactly those files rather than a rebuild. ``publish.yml`` runs ``fetch``. Standard library and ``gh`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.request

REPOSITORY = "eagle-hq/EAGLE-4"
MANIFEST = "harness-SHA256SUMS"
PYPI_SIMPLE = "https://pypi.org/simple/eagle-harness/"


def patterns(version: str) -> dict[str, re.Pattern]:
    v = re.escape(version)
    return {"sdist": re.compile(rf"eagle_harness-{v}\.tar\.gz"),
            "binary": re.compile(rf"eagle_harness-{v}-py3-none-manylinux_\d+_\d+_x86_64\.whl"),
            "pure": re.compile(rf"eagle_harness-{v}-py3-none-any\.whl")}


def version_of(tag: str) -> str:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise ValueError("release tag must have the form vX.Y.Z")
    return tag[1:]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(names: list[str], version: str) -> dict[str, str]:
    """Exactly one sdist, one manylinux x86_64 wheel and one pure wheel of ``version``; nothing else."""
    found: dict[str, str] = {}
    for name in names:
        kinds = [kind for kind, pattern in patterns(version).items() if pattern.fullmatch(name)]
        if len(kinds) != 1 or kinds[0] in found:
            raise ValueError(f"unexpected or duplicate release file for {version}: {name}")
        found[kinds[0]] = name
    missing = patterns(version).keys() - found.keys()
    if missing:
        raise ValueError(f"missing release files for {version}: {', '.join(sorted(missing))}")
    return found


def read_manifest(text: str, version: str) -> dict[str, str]:
    entries = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  (\S+)", line)
        if not match or match[2] in entries:
            raise ValueError(f"invalid or duplicate {MANIFEST} line: {line!r}")
        entries[match[2]] = match[1]
    classify(list(entries), version)
    return entries


def gh(*args: str, repository: str) -> str:
    return subprocess.check_output(["gh", *args, "--repo", repository], text=True)


def release_state(tag: str, repository: str) -> dict:
    release = json.loads(gh("release", "view", tag, "--json", "tagName,isDraft,assets", repository=repository))
    if release["tagName"] != tag:
        raise ValueError(f"release {tag} not found")
    return release


def download(tag: str, name: str, target: Path, repository: str) -> None:
    gh("release", "download", tag, "--pattern", name, "--output", str(target), repository=repository)


def local_files(dist: Path, version: str) -> dict[str, Path]:
    paths = [p for p in [*dist.glob("eagle_harness-*"), *(dist / "pure").glob("eagle_harness-*")] if p.is_file()]
    found = classify([p.name for p in paths], version)
    by_name = {p.name: p for p in paths}
    return {kind: by_name[name] for kind, name in found.items()}


def stage(tag: str, dist: Path, repository: str = REPOSITORY) -> dict[str, str]:
    version = version_of(tag)
    files = local_files(dist, version)
    checksums = {p.name: sha256(p) for p in sorted(files.values(), key=lambda p: p.name)}
    release = release_state(tag, repository)
    if not release["isDraft"]:
        raise ValueError(f"{tag} is published; stage the files on the draft release before publishing it")
    existing = {asset["name"] for asset in release["assets"]}
    with tempfile.TemporaryDirectory(prefix="harness-stage-") as temporary:
        work = Path(temporary)
        manifest = work / MANIFEST
        manifest.write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
        uploads = [*sorted(files.values(), key=lambda p: p.name), manifest]
        expected = {**checksums, MANIFEST: sha256(manifest)}
        # Preflight: a file already on the release must be byte-identical; nothing is ever replaced.
        for path in uploads:
            if path.name in existing:
                download(tag, path.name, work / f"existing-{path.name}", repository)
                if sha256(work / f"existing-{path.name}") != expected[path.name]:
                    raise ValueError(f"{tag} already carries a different {path.name}; refusing to replace it")
        for path in uploads:  # the manifest last: its presence means every file is there
            if path.name not in existing:
                gh("release", "upload", tag, str(path), repository=repository)
            download(tag, path.name, work / f"after-{path.name}", repository)
            if sha256(work / f"after-{path.name}") != expected[path.name]:
                raise ValueError(f"read-back of {tag}/{path.name} does not match")
            print(f"staged {tag}/{path.name} sha256={expected[path.name]}")
    return checksums


def fetch(tag: str, out: Path, repository: str = REPOSITORY) -> dict[str, str]:
    version = version_of(tag)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    (out / "pure").mkdir(parents=True, exist_ok=True)
    download(tag, MANIFEST, out / MANIFEST, repository)
    entries = read_manifest((out / MANIFEST).read_text(), version)
    kinds = classify(list(entries), version)
    for kind, name in kinds.items():
        target = out / "pure" / name if kind == "pure" else out / name
        download(tag, name, target, repository)
        if sha256(target) != entries[name]:
            raise ValueError(f"SHA256 mismatch for {name}: the release file differs from {MANIFEST}")
        print(f"verified {name} sha256={entries[name]}")
    (out / MANIFEST).replace(out / "SHA256SUMS")
    return entries


def pypi_digests(url: str = PYPI_SIMPLE) -> dict[str, str]:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.pypi.simple.v1+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        files = json.load(response)["files"]
    return {f["filename"]: f["hashes"].get("sha256", "") for f in files}


def remove(tag: str, repository: str = REPOSITORY, published: dict[str, str] | None = None) -> list[str]:
    version = version_of(tag)
    with tempfile.TemporaryDirectory(prefix="harness-remove-") as temporary:
        manifest = Path(temporary) / MANIFEST
        download(tag, MANIFEST, manifest, repository)
        entries = read_manifest(manifest.read_text(), version)
    published = pypi_digests() if published is None else published
    differing = [name for name, digest in entries.items() if published.get(name) != digest]
    if differing:
        raise ValueError(f"PyPI does not serve these files with the release SHA256 yet: {', '.join(differing)}")
    present = {asset["name"] for asset in release_state(tag, repository)["assets"]}
    removed = []
    for name in [*sorted(entries), MANIFEST]:
        if name in present:
            gh("release", "delete-asset", tag, name, "--yes", repository=repository)
            removed.append(name)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repository", default=REPOSITORY)
    commands = parser.add_subparsers(dest="command", required=True)
    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("tag")
    stage_parser.add_argument("dist", type=Path)
    fetch_parser = commands.add_parser("fetch")
    fetch_parser.add_argument("tag")
    fetch_parser.add_argument("out", type=Path)
    remove_parser = commands.add_parser("remove")
    remove_parser.add_argument("tag")
    args = parser.parse_args(argv)
    try:
        if args.command == "stage":
            stage(args.tag, args.dist, args.repository)
        elif args.command == "fetch":
            fetch(args.tag, args.out, args.repository)
        else:
            for name in remove(args.tag, args.repository):
                print(f"removed {args.tag}/{name} (PyPI serves the same SHA256)")
    except (ValueError, subprocess.CalledProcessError, OSError) as error:
        print(f"release_files: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
