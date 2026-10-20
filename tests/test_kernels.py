# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Kernel packaging and hardware dispatch (CPU): no engine imports, the native C ABI, the arch table, the nvcc check."""

from __future__ import annotations

import ast
import os
import re
import shutil
from pathlib import Path

import pytest
import torch

from eagle_harness.kernels import arch, build, native

KERNELS = Path(build.PACKAGE)


def test_kernels_import_no_engine():
    for path in KERNELS.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = {
            alias.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names
        } | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not {"sglang", "vllm"} & modules, path.name


def test_native_abi_matches_the_source():
    source = build.SOURCE.read_text(encoding="utf-8")
    assert re.findall(r"^int (eagle4_\w+)\(", source, re.M) == list(native.SIGNATURES)
    assert len(re.findall(r"__global__", source)) == 2
    for name, signature in native.SIGNATURES.items():
        params = re.search(rf"^int {name}\((.*?)\) {{", source, re.M | re.S).group(1).split(",")
        kinds = ["P" if "*" in p or "cudaStream_t" in p else "q" if "int64_t" in p else "i" for p in params]
        assert "".join(kinds) == signature, name


def test_nothing_is_built_at_import():
    native.library.cache_clear()
    import importlib

    from eagle_harness.kernels import candidate, candidate_ops

    importlib.reload(candidate)
    importlib.reload(candidate_ops)
    assert native.library.cache_info().currsize == 0


def test_cuda_12_8_has_no_sm103_and_13_has_it():
    assert 103 not in build.supported_sass((12, 8)) and 100 in build.supported_sass((12, 8))
    assert build.supported_sass((13, 0)) == build.SASS_ARCHS


@pytest.fixture
def fake_gpu(monkeypatch):
    """Make ``arch`` see one GPU of the given kind (nothing else in torch changes)."""

    def install(vendor: str, arch_name: str, sm: int = 0):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(arch, "_probe", lambda index: arch.Gpu(index, "fake", vendor, arch_name, sm))

    return install


@pytest.mark.parametrize(
    "vendor,name,sm,expected",
    [
        ("nvidia", "sm_80", 80, dict(tail=True, clusters=False, pdl=False, gluon=False, records=False, prelaunch=True)),
        ("nvidia", "sm_89", 89, dict(tail=True, clusters=False, pdl=False, gluon=False, records=False, prelaunch=True)),
        ("nvidia", "sm_90", 90, dict(tail=True, clusters=True, pdl=True, gluon=False, records=False, prelaunch=True)),
        ("nvidia", "sm_100", 100, dict(tail=True, clusters=True, pdl=True, gluon=False, records=False, prelaunch=True)),
        ("nvidia", "sm_103", 103, dict(tail=True, clusters=True, pdl=True, gluon=True, records=False, prelaunch=True)),
        ("nvidia", "sm_120", 120, dict(tail=True, clusters=True, pdl=True, gluon=False, records=False, prelaunch=True)),
        ("amd", "gfx942", 0, dict(tail=False, clusters=False, pdl=False, gluon=False, records=True, prelaunch=False)),
        ("amd", "gfx950", 0, dict(tail=False, clusters=False, pdl=False, gluon=False, records=True, prelaunch=False)),
    ],
)
def test_arch_feature_table(fake_gpu, vendor, name, sm, expected):
    fake_gpu(vendor, name, sm)
    assert arch.unsupported_reason() is None
    got = dict(
        tail=arch.native_candidate_tail(),
        clusters=arch.clusters(),
        pdl=arch.pdl(),
        gluon=arch.gluon_gdn(),
        records=arch.gdn_records_replay(),
        prelaunch=arch.side_stream_prelaunch(),
    )
    assert got == expected


@pytest.mark.parametrize("version", ["3.7.1", "3.8.0"])
def test_gluon_gdn_is_not_gated_on_the_triton_version(fake_gpu, monkeypatch, version):
    # gluon_dispatch's bitwise self-check is the guard; a version list silently turned the kernel off.
    import triton

    fake_gpu("nvidia", "sm_103", 103)
    monkeypatch.setattr(triton, "__version__", version)
    assert arch.gluon_gdn()


@pytest.mark.parametrize("vendor,name,sm", [("nvidia", "sm_75", 75), ("nvidia", "sm_70", 70), ("amd", "gfx1100", 0)])
def test_unsupported_hardware_is_switched_off_with_a_reason(fake_gpu, vendor, name, sm):
    fake_gpu(vendor, name, sm)
    reason = arch.unsupported_reason()
    assert reason and name in reason
    assert not (arch.native_candidate_tail() or arch.clusters() or arch.pdl() or arch.gluon_gdn())
    assert not arch.gdn_records_replay()
    with pytest.raises(arch.UnsupportedHardware, match=name):
        arch.require_supported()


def test_no_gpu_is_a_reason_not_an_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert "no CUDA or ROCm device" in arch.unsupported_reason()
    assert arch.gpu() is None and not arch.clusters()


def _nvcc():
    for home in (os.environ.get("CUDA_HOME"), "/usr/local/cuda-13.0", "/usr/local/cuda"):
        if home and os.path.exists(os.path.join(home, "bin", "nvcc")):
            return home
    return shutil.which("nvcc") and os.path.dirname(os.path.dirname(os.path.realpath(shutil.which("nvcc"))))


@pytest.mark.slow
def test_native_library_compiles_for_every_architecture():
    home = _nvcc()
    if home is None:
        pytest.skip("nvcc not found")
    if shutil.which("g++") is None and shutil.which("c++") is None:
        pytest.skip("no host C++ compiler for nvcc")
    try:
        build.check(home)
    except RuntimeError as err:
        if "unsupported GNU version" in str(err) or "host compiler" in str(err):
            pytest.skip(f"nvcc rejects this host compiler: {err}")
        raise


def test_gluon_gdn_admits_whole_gqa_groups_at_head_dim_128():
    """Every per-rank split of a 128-wide GDN layer is admitted (TP1 to TP8 of Qwen3.8-27B), other shapes are not."""
    pytest.importorskip("triton.experimental.gluon")
    from eagle_harness.kernels.gdn_tree_gluon import admits_shape

    assert all(admits_shape(16 // tp, 48 // tp, 128, 128) for tp in (1, 2, 4, 8))
    assert not admits_shape(4, 12, 64, 64) and not admits_shape(4, 10, 128, 128) and not admits_shape(0, 0, 128, 128)
