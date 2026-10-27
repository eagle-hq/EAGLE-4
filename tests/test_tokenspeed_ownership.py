# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-specific TokenSpeed-port code is owned and shipped by the harness."""

import re
from pathlib import Path


def test_no_tokenspeed_copyright_code_in_harness():
    package = Path(__file__).resolve().parents[1] / "eagle_harness"
    copied = [
        str(path.relative_to(package))
        for path in package.rglob("*")
        if path.suffix in {".py", ".cu", ".h"} and re.search(r"Copyright[^\n]*LightSeek", path.read_text())
    ]
    assert not copied, f"Eagle-owned adapter sources must carry the Eagle license: {copied}"


def test_adapter_does_not_import_engine_owned_round_implementations():
    import ast

    package = Path(__file__).resolve().parents[1] / "eagle_harness" / "tokenspeed"
    forbidden = (
        "tokenspeed.runtime.execution.drafter.tree_round",
        "tokenspeed.runtime.execution.drafter.eagle_harness",
    )
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(forbidden), path
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith(forbidden) for alias in node.names), path


def test_generic_hook_registration_does_not_require_harness_implementation():
    from eagle_harness.tokenspeed.hooks import CAPABILITY_HOOKS, HOOKS, REQUIRED_CAPABILITIES

    assert REQUIRED_CAPABILITIES == {
        "drafter.algorithm_config", "drafter.aux_hidden_capture", "drafter.tree_layout",
        "attention.windowed_tree_lanes", "drafter.round_protocol", "drafter.tree_lanes", "drafter.tree_policy",
        "sampling.tree_sampler",
    }
    assert not any("tree_round" in hook.path for hook in HOOKS + CAPABILITY_HOOKS)


def test_owned_round_sources_keep_eagle_headers_and_ship_cuda_sources():
    from eagle_harness.tokenspeed.tree_round.ops import build

    root = Path(build.__file__).parents[1]
    for path in root.rglob("*"):
        if path.suffix not in {".py", ".cu"}:
            continue
        text = path.read_text()
        assert "Copyright 2026 Eagle AI Corp." in text.splitlines()[0], path
        assert "SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available" in text.splitlines()[1], path
    for name in build.MODULES:
        assert (build.SOURCE / f"eagle_harness_{name}.cu").is_file()
    assert (build.SOURCE / "tvm_ffi_utils.h").is_file()
