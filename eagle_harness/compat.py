# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Which serving engines (hosts) this eagle-harness runs on, and the load-time check of the installed one.

The contract with a host is its hooks: each host adapter pins the hook names and signatures it uses, and its plugin
refuses to register when the installed host differs. The host version does not decide on its own (a source checkout
without git tags reports a placeholder such as ``0.0.0.dev0``); it goes into every message, and a host that passes
the hook check but is outside :attr:`Host.versions` gets a warning that the pair is untested.

    python -m eagle_harness.compat [--kernels]     # also: eagle-harness-check

prints the report for every installed host and exits 1 when an installed host lacks a hook (or, with ``--kernels``,
when the native CUDA library does not load).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from eagle_harness import __version__

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Host:
    name: str
    versions: SpecifierSet  # versions that can carry the hooks; no release has them yet
    tested: str  # the host commit the hook table was checked against
    problems: str  # "module:function" returning the hook mismatches of the installed host


HOSTS = (
    Host(
        "sglang",
        SpecifierSet(">=0.5.23.dev0,<0.6"),  # = the `sglang` extra in pyproject.toml
        "sglang 0.5.23.dev252+eagle.1 = main 05c1539098 + eagle-harness hooks (eagle/v1)",
        "eagle_harness.sglang.hooks:hook_problems",
    ),
    Host(
        "vllm",
        SpecifierSet(">=0.32.0rc1.dev283,!=0.32.0"),  # = the `vllm` extra in pyproject.toml
        "vLLM 0.32.0rc1.dev283+eagle.4 = main 074c83a09a + eagle-harness hooks and fixes",
        "eagle_harness.vllm.adapter:missing_hooks",
    ),
    Host(
        "tokenspeed",
        SpecifierSet(">=0.1.1,<0.2"),  # = the `tokenspeed` extra in pyproject.toml
        "TokenSpeed 0.1.1+eagle.4 with eagle-harness plugin hooks (see docs/install.md)",
        "eagle_harness.tokenspeed.hooks:hook_problems",
    ),
)


@dataclass(frozen=True)
class Report:
    host: Host
    version: str | None  # None: the host is not installed
    problems: list[str]  # hook mismatches; empty when every hook matches

    @property
    def installed(self) -> bool:
        return self.version is not None

    @property
    def ok(self) -> bool:
        return self.installed and not self.problems

    @property
    def tested_version(self) -> bool:
        """Checks whether the host version is in :attr:`Host.versions` (False for an unparsable version)."""
        try:
            return Version(self.version or "") in self.host.versions
        except InvalidVersion:
            return False

    def refusal(self) -> str:
        """What is wrong and how to fix it, for a host that fails the hook check."""
        return (
            f"eagle-harness {__version__} is disabled: {self.host.name} {self.version} lacks "
            f"{len(self.problems)} of the hooks it needs ({'; '.join(self.problems)}). "
            f"Fix: install {self.host.name} with the eagle-harness hooks ({self.host.name}{self.host.versions}; "
            f"tested: {self.host.tested}), or the eagle-harness release that your {self.host.name} pins."
        )

    def untested(self) -> str:
        return (
            f"eagle-harness {__version__}: {self.host.name} {self.version} is outside the tested versions "
            f"({self.host.name}{self.host.versions}); every hook matches, so the plugin is registered"
        )


def host(name: str) -> Host:
    return next(h for h in HOSTS if h.name == name)


def _resolve(path: str) -> Callable[[], list[str]]:
    module_name, _, attr = path.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def host_version(name: str) -> str | None:
    """The version of the importable host module (what actually runs), or None if it is not installed."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    version = getattr(module, "__version__", None)
    if version is None:  # a package without the attribute (TokenSpeed): its distribution's version
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return str(version)


def check(name: str) -> Report:
    """Checks the installed host ``name`` against this package's hook table."""
    spec = host(name)
    version = host_version(name)
    return Report(spec, version, _resolve(spec.problems)() if version is not None else [])


def check_for_registration(name: str) -> bool:
    """Checks the host before a plugin registers: logs the refusal or the untested-version warning; True to proceed."""
    report = check(name)
    if not report.ok:
        logger.error(report.refusal())
        return False
    if not report.tested_version:
        logger.warning(report.untested())
    return True


def _kernels_line() -> tuple[bool, str]:
    from eagle_harness.kernels import build, native

    lib = native.library()
    if lib is None:
        return False, "native CUDA kernels: not loaded (the Triton kernels run; see the warning above)"
    if lib._name.startswith(str(build.PREBUILT)):
        where = "prebuilt"
    elif lib._name == os.environ.get("EAGLE_HARNESS_NATIVE_LIB"):
        where = "from $EAGLE_HARNESS_NATIVE_LIB"
    else:
        where = "built by nvcc on first use"
    return True, f"native CUDA kernels: {where}: {lib._name}"


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: report every installed host; exit 1 on a hook mismatch (or unloadable kernels with --kernels)."""
    ap = argparse.ArgumentParser(description="Check eagle-harness against the installed serving engines.")
    ap.add_argument("--kernels", action="store_true", help="also load the native CUDA library (needs a GPU)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    import eagle_harness

    print(f"eagle-harness {__version__} ({eagle_harness.__file__})")
    reports = [check(h.name) for h in HOSTS]
    failed = False
    for report in reports:
        name = report.host.name
        if not report.installed:
            print(f"{name}: not installed")
        elif report.ok:
            note = "" if report.tested_version else f" (untested version, expected {name}{report.host.versions})"
            print(f"{name} {report.version}: ok, every hook matches{note}")
        else:
            failed = True
            print(f"{name} {report.version}: FAILED\n  {report.refusal()}")
    if not any(r.installed for r in reports):
        failed = True
        print(f"no supported host installed (one of: {', '.join(h.name for h in HOSTS)})")
    if args.kernels:
        ok, line = _kernels_line()
        failed |= not ok
        print(line)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
