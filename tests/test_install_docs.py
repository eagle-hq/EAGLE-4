# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The install guides use the release pins, run from outside the checkout and install ROCm from public indexes only."""

from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[1]
GUIDES = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]
QUICKSTARTS = [REPO / "README.md", REPO / "docs/install.md", *sorted((REPO / "docs").glob("quickstart-*.md"))]


def bash_blocks(path: Path) -> list[str]:
    return re.findall(r"```bash\n(.*?)```", path.read_text(), re.S)


def section(path: Path, heading: str) -> str:
    return path.read_text().split(f"\n{heading}\n", 1)[1].split("\n## ", 1)[0]


def release_settings() -> dict[str, str]:
    text = (REPO / "docs/release.env").read_text()
    return dict(re.findall(r"^export (EAGLE_[A-Z0-9_]+)=(.*)$", text, re.M))


def resolved_settings() -> dict[str, str]:
    """release.env values without quotes and with earlier variables substituted, as the shell reads them."""
    values: dict[str, str] = {}
    for name, raw in release_settings().items():
        value = raw[1:-1] if raw[:1] in "'\"" else raw
        if raw.startswith('"'):
            value = re.sub(r"\$\{?(\w+)\}?", lambda m: values[m[1]], value)
        values[name] = value
    return values


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


@pytest.mark.parametrize("guide", QUICKSTARTS, ids=lambda p: str(p.relative_to(REPO)))
def test_copy_paste_commands_use_the_release_pins(guide):
    # The quick starts spell the pins out so they paste without release.env; they must match it exactly.
    s = resolved_settings()
    pins = {
        "sglang": s["EAGLE_SGLANG_VERSION"],
        "vllm": s["EAGLE_VLLM_VERSION"],
        "tokenspeed": s["EAGLE_TOKENSPEED_VERSION"],
        **dict(re.findall(r"(tokenspeed-\w+)==(\S+)", s["EAGLE_TOKENSPEED_AUX_REQUIREMENTS"])),
    }
    indexes = {s[k] for k in ("EAGLE_CUDA_INDEX", "EAGLE_TORCH_CUDA_INDEX", "EAGLE_NVIDIA_INDEX")}
    images = {s[k] for k in ("EAGLE_SGLANG_IMAGE", "EAGLE_VLLM_IMAGE", "EAGLE_TOKENSPEED_IMAGE")}
    for image in re.findall(r"ghcr\.io/eagle-hq/[\w.-]+:\d+\.\d+\.\d+", guide.read_text()):
        assert image in images, f"{guide.name}: {image} does not match release.env"
    for block in bash_blocks(guide):
        for package, version in re.findall(r'"(sglang|vllm|tokenspeed(?:-\w+)?)==([^"\s]+)"', block):
            assert version == pins[package], f"{guide.name}: {package}=={version}, release.env pins {pins[package]}"
        for url in re.findall(r"--extra-index-url (https://\S+)", block):
            assert url in indexes, f"{guide.name}: {url} is not a release index"
        for image in re.findall(r"ghcr\.io/eagle-hq/\S+", block):
            assert image in images, f"{guide.name}: {image} is not a published release image"
        if re.search(r'"(sglang|vllm|tokenspeed)==[^"]*\+eagle\.', block):
            # A +eagle.N pin resolves only with the EAGLE-4 wheel index in the same command.
            assert s["EAGLE_CUDA_INDEX"] in block, block


def test_readme_quick_start_is_one_install_one_serve_one_request():
    quick = section(REPO / "README.md", "## Quick start")
    docker, install, serve, request = re.findall(r"```bash\n(.*?)```", quick, re.S)
    assert "docker run" in docker and "--speculative-algorithm EAGLE_HARNESS" in docker
    assert install.count("pip install") == 1 and '"eagle-harness[sglang]"' in install
    assert serve.startswith("sglang serve ") and re.search(r"--speculative-draft-model-path [\w-]+/Qwen3\.8-27B-EAGLE4$", serve.strip())
    assert request.startswith("curl http://localhost:30000/")
    # The README, the install guide and the SGLang guide show the same commands.
    assert install in bash_blocks(REPO / "docs/install.md")
    assert serve in bash_blocks(REPO / "docs/quickstart-sglang.md")
    assert docker in bash_blocks(REPO / "docs/quickstart-sglang.md")


def test_copy_paste_commands_never_pin_the_harness():
    # An unpinned harness cannot name an unpublished or old version.
    for guide in QUICKSTARTS:
        for block in bash_blocks(guide):
            assert not re.search(r"eagle-harness(\[\w+\])?==\d", block), f"{guide.name}: {block}"


def test_tokenspeed_install_names_only_the_new_release_source_and_builds():
    install = (REPO / "docs/install.md").read_text()
    links = re.findall(r"https://github\.com/eagle-hq/tokenspeed[^)\s]*", install)
    assert links == ["https://github.com/eagle-hq/tokenspeed/tree/eagle/v0.1.3"]
    pins = re.findall(r'"(tokenspeed(?:-\w+)?)==([^"\s]+)"', section(REPO / "docs/install.md", "## TokenSpeed CUDA"))
    assert dict(pins) == {"tokenspeed": "0.1.1+eagle.8", "tokenspeed-kernel": "0.1.4+eagle.8",
                          "tokenspeed-scheduler": "0.1.42+eagle.8", "tokenspeed-engine": "0.1.0+eagle.8"}


def test_docs_do_not_reintroduce_withdrawn_tokenspeed_pins():
    for path in [REPO / "CHANGELOG.md", *GUIDES, *sorted((REPO / "docs/release-notes").glob("*.md"))]:
        assert not re.search(r"tokenspeed[^\n]*\+eagle\.(?:5|7)\b", path.read_text(), re.I), path


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
