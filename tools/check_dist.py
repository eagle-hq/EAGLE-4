#!/usr/bin/env python3
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Check the eagle-harness release files before an upload: metadata, entry points, contents, and the license state.

    python tools/check_dist.py <dist dir>      # the out dir of tools/build_wheel.sh

Checks every file ``build_wheel.sh`` writes (the sdist, the binary wheel, the pure wheel in ``pure/``):

- name ``eagle-harness`` and one version across the files, equal to ``eagle_harness/__init__.py``'s ``__version__``;
- the ``sglang``, ``vllm`` and ``test`` extras, the ``eagle-harness-check`` script and both engines' plugin entry points;
- the binary wheel carries the native library for CUDA 12 and 13; the sdist and the pure wheel carry none;
- the wheels hold only the package and its dist-info, the sdist only the package, tests and the build files.

A placeholder LICENSE fails validation. Exit status: 0 the files are fine, 1 a check failed, 2 usage error.
Standard library only.
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from email.parser import HeaderParser
from pathlib import Path

NAME = "eagle-harness"
EXTRAS = {"sglang", "vllm", "test"}
ENTRY_POINTS = {
    "console_scripts": "eagle-harness-check = eagle_harness.compat:main",
    "sglang.srt.plugins": "eagle_harness = eagle_harness.sglang.plugin:register",
    "vllm.general_plugins": "eagle_harness = eagle_harness.vllm.plugin:register",
}
NATIVE = ("eagle_harness/kernels/_native/cu12/", "eagle_harness/kernels/_native/cu13/")
# Top-level entries an sdist may hold; anything else (a private directory, CI files) must not ship.
SDIST_TOP = {"eagle_harness", "eagle_harness.egg-info", "tests", "LICENSE", "NOTICE", "TRADEMARKS.md", "LICENSES",
             "THIRD_PARTY_NOTICES.md", "MANIFEST.in", "PKG-INFO", "README.md", "pyproject.toml", "setup.cfg", "setup.py"}
PLACEHOLDER_LICENSE = re.compile(r"not a license|\bplaceholder\b", re.I)
SOURCE = Path(__file__).resolve().parents[1]
LICENSE_FILES = ("LICENSE", "NOTICE", "TRADEMARKS.md", "THIRD_PARTY_NOTICES.md",
                 *(p.relative_to(SOURCE).as_posix() for p in sorted((SOURCE / "LICENSES").glob("*.txt"))))


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            self.failures.append(what)


def entry_points(text: str) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    group = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            group = line.strip("[]")
            groups[group] = set()
        elif line and group:
            groups[group].add(re.sub(r"\s*=\s*", " = ", line))
    return groups


def check_metadata(report: Report, label: str, metadata: str, version: str) -> None:
    headers = HeaderParser().parsestr(metadata)
    report.check(headers["Name"] == NAME, f"{label}: Name is {headers['Name']!r}")
    report.check(headers["Version"] == version, f"{label}: Version {headers['Version']} == __version__ {version}")
    extras = set(headers.get_all("Provides-Extra") or [])
    report.check(EXTRAS <= extras, f"{label}: extras {sorted(extras)} include {sorted(EXTRAS)}")
    requires = headers.get_all("Requires-Dist") or []
    for extra in ("sglang", "vllm"):
        report.check(any(f'extra == "{extra}"' in r for r in requires), f"{label}: the {extra} extra pins its engine")
    report.check(headers["License-Expression"] == "LicenseRef-Eagle-4-Source-Available",
                 f"{label}: License-Expression LicenseRef-Eagle-4-Source-Available")
    for name in LICENSE_FILES:
        report.check(name in (headers.get_all("License-File") or []), f"{label}: License-File {name}")


def check_license(report: Report, label: str, data: bytes) -> None:
    report.check(not PLACEHOLDER_LICENSE.search(data.decode(errors="replace")),
                 f"{label}: LICENSE is not a placeholder")


def check_wheel(report: Report, path: Path, version: str, native: bool) -> None:
    with zipfile.ZipFile(path) as whl:
        names = whl.namelist()
        dist_info = next(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA"))
        check_metadata(report, path.name, whl.read(f"{dist_info}/METADATA").decode(), version)
        groups = entry_points(whl.read(f"{dist_info}/entry_points.txt").decode())
        for group, line in ENTRY_POINTS.items():
            report.check(line in groups.get(group, set()), f"{path.name}: entry point [{group}] {line}")
        libs = [n for n in names if n.endswith(".so")]
        if native:
            for prefix in NATIVE:
                report.check(any(n.startswith(prefix) for n in libs), f"{path.name}: native library under {prefix}")
        else:
            report.check(not libs, f"{path.name}: no native library ({libs or 'none'})")
        stray = [n for n in names if n.split("/")[0] not in ("eagle_harness", dist_info)]
        report.check(not stray, f"{path.name}: only eagle_harness/ and {dist_info}/ ({stray[:3] or 'nothing else'})")
        for name in LICENSE_FILES:
            member = f"{dist_info}/licenses/{name}"
            report.check(member in names, f"{path.name}: {name} in the dist-info")
            if member in names:
                data = whl.read(member)
                report.check(data == (SOURCE / name).read_bytes(), f"{path.name}: {name} matches source")
                if name == "LICENSE":
                    check_license(report, path.name, data)


def check_sdist(report: Report, path: Path, version: str) -> None:
    with tarfile.open(path) as sdist:
        names = sdist.getnames()
        root = names[0].split("/")[0]
        check_metadata(report, path.name, sdist.extractfile(f"{root}/PKG-INFO").read().decode(), version)
        rel = [n[len(root) + 1:] for n in names if n.startswith(root + "/")]
        report.check(not [n for n in rel if n.endswith(".so")], f"{path.name}: no native library")
        report.check(any(n.endswith("csrc/candidate.cu") for n in rel), f"{path.name}: the CUDA source for the JIT build")
        stray = sorted({n.split("/")[0] for n in rel if n} - SDIST_TOP)
        report.check(not stray, f"{path.name}: only the package, tests and build files ({stray or 'nothing else'})")
        for name in LICENSE_FILES:
            member = f"{root}/{name}"
            report.check(member in names, f"{path.name}: {name} in the sdist")
            if member in names:
                data = sdist.extractfile(member).read()
                report.check(data == (SOURCE / name).read_bytes(), f"{path.name}: {name} matches source")
                if name == "LICENSE":
                    check_license(report, path.name, data)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    dist = Path(argv[0])
    init = Path(__file__).resolve().parents[1] / "eagle_harness" / "__init__.py"
    version = re.search(r'^__version__ = "([^"]+)"', init.read_text(), re.M).group(1)
    sdists = sorted(dist.glob("eagle_harness-*.tar.gz"))
    wheels = sorted(dist.glob("eagle_harness-*-manylinux*.whl"))
    pure = sorted(dist.glob("pure/eagle_harness-*-py3-none-any.whl"))
    report = Report()
    report.check(len(sdists) == 1 and len(wheels) == 1 and len(pure) == 1,
                 f"one sdist, one binary wheel, one pure wheel in {dist} ({len(sdists)}, {len(wheels)}, {len(pure)})")
    for path in sdists:
        check_sdist(report, path, version)
    for path in wheels:
        check_wheel(report, path, version, native=True)
    for path in pure:
        check_wheel(report, path, version, native=False)
    if report.failures:
        print(f"{len(report.failures)} check(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
