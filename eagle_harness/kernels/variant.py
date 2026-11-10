# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Which version of a kernel its caller asked for. ktest's harness backend sets it for its arm (``harness=NAME``), so
one perf run times a kernel's NAME version against its current one in one process, and the gate checks it. A kernel
module declares its version at import (``NEW = variant.name("NEW")``) and branches on ``variant.current == NEW`` at
launch; an arm naming a version no module declares is refused, so a typo can't time the current kernels against
themselves. Nothing else sets it, so served paths always take the current versions.

Branch where the kernel launches, not inside a cached planner (its first arm's answer would serve the other), and
keep the kernel in a file ktest's harness fingerprints (backends/harness.py SOURCES)."""

import contextlib

current = None
names = set()


def name(version: str) -> str:
    names.add(version)
    return version


@contextlib.contextmanager
def use(version):
    global current
    old, current = current, version
    try:
        yield
    finally:
        current = old
