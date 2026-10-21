# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Install the nvcc toolchain of one CUDA release from NVIDIA's redistributable archives, without root.

    python tools/cuda_redist.py <release, e.g. 12.9.1> <cuda home>

Fetches ``redistrib_<release>.json``, then every component nvcc needs for a static-runtime build (nvcc, its headers,
NVVM, cudart), checks each archive's SHA256 against the manifest and unpacks it into ``<cuda home>`` (bin/, include/,
lib/ with a lib64 link, nvvm/). Stdlib only. An existing ``<cuda home>/bin/nvcc`` is kept, so a runner's tool cache
serves later builds.
"""

import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
# cuda_crt and libnvvm are separate archives from CUDA 13; absent from a manifest means part of cuda_nvcc.
COMPONENTS = ("cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cudart", "cuda_cccl")


def install(release: str, home: Path) -> None:
    if (home / "bin" / "nvcc").exists():
        return
    with urllib.request.urlopen(f"{BASE}redistrib_{release}.json") as response:
        manifest = json.load(response)
    home.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=home.name + ".", dir=home.parent))
    try:
        for component in COMPONENTS:
            if component not in manifest:
                continue
            entry = manifest[component]["linux-x86_64"]
            with tempfile.TemporaryFile() as archive:
                with urllib.request.urlopen(BASE + entry["relative_path"]) as response:
                    shutil.copyfileobj(response, archive)
                archive.seek(0)
                digest = hashlib.file_digest(archive, "sha256").hexdigest()
                if digest != entry["sha256"]:
                    raise SystemExit(f"{component}: sha256 {digest} != manifest {entry['sha256']}")
                archive.seek(0)
                with tarfile.open(fileobj=archive, mode="r:xz") as tar:
                    for member in tar.getmembers():
                        # Drop the archive's top directory (<component>-linux-x86_64-<version>-archive/).
                        parts = member.name.split("/", 1)
                        if len(parts) < 2 or not parts[1]:
                            continue
                        member.name = parts[1]
                        tar.extract(member, staging, filter="tar")
        (staging / "lib64").symlink_to("lib")  # nvcc.profile links from lib64, as in the packaged toolkit
        try:
            staging.rename(home)
        except OSError:
            if not (home / "bin" / "nvcc").exists():  # else a concurrent job installed the same release first
                raise
            shutil.rmtree(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    install(sys.argv[1], Path(sys.argv[2]))
