# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""CPU checks for CUDA compilation, including device-runtime linking."""

import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from eagle_harness.tokenspeed.tree_round.ops import build


def test_builder_import_does_not_load_torch_or_tokenspeed():
    subprocess.run(
        [sys.executable, "-c", (
            "import sys; from eagle_harness.tokenspeed.tree_round.ops import build; "
            "assert 'torch' not in sys.modules; assert 'tokenspeed' not in sys.modules"
        )],
        check=True,
    )


def test_explicit_toolkit_discovery_and_version(tmp_path):
    toolkit = tmp_path / "cuda"
    compiler = toolkit / "bin" / "nvcc"
    compiler.parent.mkdir(parents=True)
    compiler.write_text(
        "#!/bin/sh\nprintf 'Cuda compilation tools, release 13.0, V13.0.1\\n'\n"
    )
    compiler.chmod(0o755)
    assert build.find_nvcc(str(toolkit)) == str(compiler)
    assert build.toolkit_version(str(compiler)) == (13, 0)
    compiler.write_text("#!/bin/sh\nprintf 'unknown compiler\\n'\n")
    with pytest.raises(RuntimeError, match="cannot read the CUDA version"):
        build.toolkit_version(str(compiler))


@pytest.mark.parametrize("name", build.MODULES)
def test_compile_recipe_and_atomic_publication(tmp_path, monkeypatch, name):
    commands = []
    monkeypatch.setattr(build, "toolkit_version", lambda _: (13, 0))
    monkeypatch.setattr(
        build,
        "_ffi_paths",
        lambda: (["/ffi/include", "/dlpack/include"], "/ffi/lib/libtvm_ffi.so"),
    )
    out = tmp_path / f"{name}.so"
    out.write_bytes(b"previous")

    def run(command, **kwargs):
        commands.append(command)
        # The published artifact is untouched throughout compilation and linking.
        assert out.read_bytes() == b"previous"
        Path(command[command.index("-o") + 1]).write_bytes(b"built")
        return CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(build.subprocess, "run", run)
    assert build.compile_library(name, out, 103, "/cuda/bin/nvcc") == out
    assert out.read_bytes() == b"built"
    assert "--fmad=false" in commands[0]
    assert "--expt-relaxed-constexpr" in commands[0]
    assert not any("-use_fast_math" in command for command in commands)
    assert "-I/dlpack/include" in commands[0]
    assert "-gencode=arch=compute_103,code=sm_103" in commands[0]
    assert "-gencode=arch=compute_103,code=compute_103" in commands[0]
    assert "-cudart=static" in commands[-1]
    if name == "chain":
        assert len(commands) == 3
        assert "-rdc=true" in commands[0]
        assert "-dlink" in commands[1]
        assert "-lcudadevrt" in commands[1] and "-lcudadevrt" in commands[2]
        assert any(p.endswith("device_link.o") for p in commands[2])
    else:
        assert len(commands) == 2
    assert list(tmp_path.iterdir()) == [out]


def test_failed_build_preserves_existing_library(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "toolkit_version", lambda _: (13, 0))
    monkeypatch.setattr(
        build, "_ffi_paths", lambda: (["/ffi/include"], "/ffi/lib/libtvm_ffi.so")
    )
    monkeypatch.setattr(
        build.subprocess,
        "run",
        lambda *a, **kw: CompletedProcess(
            a[0], returncode=1, stdout="", stderr="deliberate compiler failure"
        ),
    )
    out = tmp_path / "policy.so"
    out.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="deliberate compiler failure"):
        build.compile_library("policy", out, 103, "nvcc")
    assert out.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [out]
