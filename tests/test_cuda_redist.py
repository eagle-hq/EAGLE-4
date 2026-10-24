# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""tools/cuda_redist.py verifies and unpacks the redistributable archives on every supported Python (3.10+)."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

SPEC = importlib.util.spec_from_file_location("cuda_redist", Path(__file__).parents[1] / "tools/cuda_redist.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as tar:
        data = b"#!/bin/sh\n"
        info = tarfile.TarInfo("cuda_nvcc-linux-x86_64-12.9.1-archive/bin/nvcc")
        info.size, info.mode = len(data), 0o755
        tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_install_without_hashlib_file_digest(tmp_path, monkeypatch):
    nvcc = archive()
    manifest = {"cuda_nvcc": {"linux-x86_64": {"relative_path": "nvcc.tar.xz", "sha256": hashlib.sha256(nvcc).hexdigest()}}}
    files = {"redistrib_12.9.1.json": json.dumps(manifest).encode(), "nvcc.tar.xz": nvcc}
    monkeypatch.setattr(MODULE.urllib.request, "urlopen", lambda url: io.BytesIO(files[url[len(MODULE.BASE):]]))
    monkeypatch.delattr(hashlib, "file_digest", raising=False)  # as on Python 3.10
    home = tmp_path / "cuda-12.9"
    MODULE.install("12.9.1", home)
    assert (home / "bin" / "nvcc").read_bytes() == b"#!/bin/sh\n"
    assert (home / "lib64").is_symlink()
