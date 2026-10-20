# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The install guides run from outside the checkout and install ROCm from public indexes only."""

from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[1]
GUIDES = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]


def bash_blocks(path: Path) -> list[str]:
    return re.findall(r"```bash\n(.*?)```", path.read_text(), re.S)


def section(path: Path, heading: str) -> str:
    return path.read_text().split(f"\n{heading}\n", 1)[1].split("\n## ", 1)[0]


def release_settings() -> dict[str, str]:
    text = (REPO / "docs/release.env").read_text()
    return dict(re.findall(r"^export (EAGLE_[A-Z0-9_]+)=(.*)$", text, re.M))


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: str(p.relative_to(REPO)))
def test_guides_never_enter_the_checkout(guide):
    # `python -m` imports from the current directory, so a server started inside the checkout
    # runs the source tree's eagle_harness/ instead of the installed wheel.
    for block in bash_blocks(guide):
        assert not re.search(r"^\s*cd\s+EAGLE-4\b|&&\s*cd\s+EAGLE-4\b", block, re.M), block
        assert ". ./docs/release.env" not in block, block


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: str(p.relative_to(REPO)))
def test_guide_variables_come_from_release_settings(guide):
    settings = release_settings()
    for block in bash_blocks(guide):
        for name in re.findall(r"\$\{?(EAGLE_[A-Z0-9_]+)", block):
            assert name in settings, f"{guide.name} uses {name}, which docs/release.env does not export"


def test_rocm_install_uses_public_indexes_and_the_shipped_lock():
    install = section(REPO / "docs/install.md", "## SGLang ROCm on AMD")
    block = re.findall(r"```bash\n(.*?)```", install, re.S)[0]
    command = block.replace("\\\n", " ")
    assert "--no-index" not in command and "BUNDLE" not in command
    lock = re.search(r"--require-hashes -r EAGLE-4/(\S+)", command)
    assert lock and (REPO / lock[1]).is_file()
    settings = release_settings()
    for name in ("EAGLE_ROCM_INDEX", "EAGLE_AMD_TRITON_INDEX", "EAGLE_AMD_ROCM_WHEELS"):
        assert f'"${name}"' in command
        assert re.fullmatch(r"(['\"])(https://|\$EAGLE_WHEEL_ROOT/)\S+/\1", settings[name]), settings[name]
    assert "release owner" not in (REPO / "docs/install.md").read_text()
