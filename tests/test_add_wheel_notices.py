# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Adding notices to a built wheel changes its notices and metadata, never its runtime payload."""

import csv
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "add_wheel_notices", Path(__file__).parents[1] / "tools/add_wheel_notices.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

NAME = "sglang_kernel-0.4.7"
INFO = f"{NAME}.dist-info"


@pytest.fixture
def wheel(tmp_path):
    source = tmp_path / f"{NAME}-cp310-cp310-linux_x86_64.whl"
    with zipfile.ZipFile(source, "w") as archive:
        binary = zipfile.ZipInfo("sgl_kernel/common_ops.so", date_time=(2020, 1, 2, 3, 4, 6))
        binary.external_attr = 0o100755 << 16
        archive.writestr(binary, b"\x00native\xff")
        archive.writestr(f"{INFO}/METADATA", "Metadata-Version: 2.4\nName: sglang-kernel\nVersion: 0.4.7\n"
                         "License-File: LICENSE\n\nlong description\n")
        archive.writestr(f"{INFO}/WHEEL", "Wheel-Version: 1.0\nTag: cp310-cp310-linux_x86_64\n")
        archive.writestr(f"{INFO}/licenses/LICENSE", "Apache License\n")
        archive.writestr(f"{INFO}/RECORD", "")
    notice = tmp_path / "THIRDPARTYNOTICES.txt"
    notice.write_text("Notice for flashinfer-ai/flashinfer\n")
    return source, notice


def test_adds_notice_build_tag_and_valid_record(tmp_path, wheel):
    source, notice = wheel
    receipt = MODULE.add_notices(source, tmp_path / "out", "1eagle", {"THIRDPARTYNOTICES.txt": notice})
    output = tmp_path / "out" / receipt["output"]
    assert output.name == f"{NAME}-1eagle-cp310-cp310-linux_x86_64.whl"
    with zipfile.ZipFile(output) as archive:
        assert archive.read(f"{INFO}/licenses/THIRDPARTYNOTICES.txt") == notice.read_bytes()
        metadata = archive.read(f"{INFO}/METADATA").decode()
        assert metadata.index("License-File: THIRDPARTYNOTICES.txt") < metadata.index("\n\nlong description")
        assert "License-File: LICENSE\n" in metadata
        assert archive.read(f"{INFO}/WHEEL").decode().endswith("Build: 1eagle\n")
        assert archive.read("sgl_kernel/common_ops.so") == b"\x00native\xff"
        info = archive.getinfo("sgl_kernel/common_ops.so")
        assert (info.external_attr >> 16, info.date_time) == (0o100755, (2020, 1, 2, 3, 4, 6))
        rows = list(csv.reader(io.StringIO(archive.read(f"{INFO}/RECORD").decode())))
        assert {row[0] for row in rows} == set(archive.namelist())
    assert receipt["unchanged_payload_members"] == 2
    assert json.loads((tmp_path / "out" / f"{output.name}.json").read_text()) == receipt


@pytest.mark.parametrize("case", ["build-tagged", "bad-build", "existing-notice", "no-notices"])
def test_rejects_unsafe_inputs(tmp_path, wheel, case):
    source, notice = wheel
    build, notices = "1eagle", {"THIRDPARTYNOTICES.txt": notice}
    if case == "build-tagged":
        source = source.rename(source.with_name(f"{NAME}-1x-cp310-cp310-linux_x86_64.whl"))
    elif case == "bad-build":
        build = "eagle"
    elif case == "existing-notice":
        notices = {"LICENSE": notice}
    else:
        notices = {}
    with pytest.raises(ValueError):
        MODULE.add_notices(source, tmp_path / "out", build, notices)
    assert not list(tmp_path.glob("out/*.whl"))


def test_refuses_to_overwrite(tmp_path, wheel):
    source, notice = wheel
    MODULE.add_notices(source, tmp_path / "out", "1eagle", {"THIRDPARTYNOTICES.txt": notice})
    with pytest.raises(FileExistsError):
        MODULE.add_notices(source, tmp_path / "out", "1eagle", {"THIRDPARTYNOTICES.txt": notice})
