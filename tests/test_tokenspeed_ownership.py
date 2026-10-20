# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Partner runtime code is an installed engine dependency, not vendored here."""

import re
from pathlib import Path


def test_no_tokenspeed_copyright_code_in_harness():
    package = Path(__file__).resolve().parents[1] / "eagle_harness"
    copied = [
        str(path.relative_to(package))
        for path in package.rglob("*")
        if path.suffix in {".py", ".cu", ".h"} and re.search(r"Copyright[^\n]*LightSeek", path.read_text())
    ]
    assert not copied, f"TokenSpeed-owned sources belong in the engine: {copied}"
