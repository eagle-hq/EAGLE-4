# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The release index must work without hosting wheel bytes on Pages."""

import hashlib
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/mk_index.py"
BASE_URL = "https://github.com/eagle-hq/EAGLE-4/releases/download/v0.1.0"


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs))


def wheel(directory, name="Example_Engine", version="0.1.0+eagle.4"):
    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{name}-{version}.dist-info/METADATA",
            f"Name: {name}\nVersion: {version}\nRequires-Python: >=3.10\n",
        )
    return path


def run_index(root, *artifacts, base_url=None, script=SCRIPT):
    command = [sys.executable, str(script), str(root), *(str(p) for p in artifacts)]
    if base_url is not None:
        command += ["--base-url", base_url]
    return subprocess.run(command, capture_output=True, text=True)


@pytest.mark.parametrize("base_url", [None, BASE_URL, BASE_URL + "/"])
def test_index_links_and_metadata(tmp_path, base_url):
    artifact = wheel(tmp_path)
    root = tmp_path / "index"
    result = run_index(root, artifact, base_url=base_url)
    assert result.returncode == 0, result.stderr
    page = root / "simple/example-engine/index.html"
    (link,) = Links(page.read_text()).links
    prefix = BASE_URL if base_url else "../../files"
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert link == {
        "href": f"{prefix}/{quote(artifact.name)}#sha256={digest}",
        "data-requires-python": ">=3.10",
    }
    assert Links((root / "simple/index.html").read_text()).links == [{"href": "example-engine/"}]
    assert (root / "SHA256SUMS").read_text() == f"{digest}  files/{artifact.name}\n"


@pytest.mark.parametrize("base_url", [None, BASE_URL])
def test_released_bytes_are_immutable_and_reruns_add_versions(tmp_path, base_url):
    artifact = wheel(tmp_path)
    root = tmp_path / "index"
    assert run_index(root, artifact, base_url=base_url).returncode == 0
    first_page = (root / "simple/example-engine/index.html").read_bytes()
    assert run_index(root, artifact, base_url=base_url).returncode == 0
    assert (root / "simple/example-engine/index.html").read_bytes() == first_page
    later = wheel(tmp_path, version="0.1.0+eagle.5")
    assert run_index(root, later, base_url=base_url).returncode == 0
    page = (root / "simple/example-engine/index.html").read_bytes()
    assert len(Links(page.decode()).links) == 2
    original = artifact.read_bytes()
    artifact.write_bytes(b"different released bytes")
    result = run_index(root, artifact, base_url=base_url)
    assert result.returncode != 0
    assert "a release artifact never changes" in result.stderr
    assert (root / "files" / artifact.name).read_bytes() == original
    assert (root / "simple/example-engine/index.html").read_bytes() == page


@pytest.mark.parametrize(
    "base_url",
    [
        "/relative",
        "http://example.org",
        "https://user:pass@example.org",
        "https://example.org/?x=1",
        "https://example.org/#frag",
    ],
)
def test_reject_invalid_asset_base(tmp_path, base_url):
    result = run_index(tmp_path / "index", wheel(tmp_path), base_url=base_url)
    assert result.returncode != 0
    assert "HTTPS URL without credentials, query or fragment" in result.stderr
    assert not (tmp_path / "index").exists()
