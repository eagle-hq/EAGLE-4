# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Saved GEMM solutions (``<data_path>/gemm_solutions.csv``) are served with tuning off, or left alone."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from eagle_harness.core.gemm_solutions import (
    AITER_GEMM_TABLE,
    AITER_GEMM_TABLE_ENV,
    GEMM_SOLUTIONS_FILE,
    gemm_solutions_path,
    load_gemm_solutions,
    serve_bf16_gemms_through_torch,
)

TOOL = Path(__file__).resolve().parents[1] / "tools" / "tune_gemm_solutions.py"


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location("tune_gemm_solutions", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTunable:
    """Records the TunableOp calls; ``read_file`` loads ``results`` when the file validates."""

    def __init__(self, results=(), validates=True):
        self.calls = []
        self._results, self._validates = tuple(results), validates
        self.enabled = self.tuning = None

    def enable(self, val=True):
        self.calls.append(("enable", val))
        self.enabled = val

    def tuning_enable(self, val=True):
        self.calls.append(("tuning_enable", val))
        self.tuning = val

    def read_file(self, filename):
        self.calls.append(("read_file", filename))
        return self._validates

    def get_results(self):
        return self._results if self._validates else ()


@pytest.fixture
def tunable(monkeypatch):
    fake = FakeTunable(
        results=[("GemmTunableOp_BFloat16_TN", "tn_34816_32_5120_ld_5120_5120_34816", "Gemm_Hipblaslt_1", 0.06)]
    )
    monkeypatch.setattr(torch.cuda, "tunable", fake, raising=False)
    return fake


def test_no_file_changes_nothing(tmp_path, tunable):
    assert gemm_solutions_path(str(tmp_path)) is None
    assert gemm_solutions_path(None) is None
    assert load_gemm_solutions(str(tmp_path)) == 0
    assert load_gemm_solutions(None) == 0
    assert tunable.calls == []


def test_file_is_served_with_tuning_off(tmp_path, tunable):
    path = tmp_path / GEMM_SOLUTIONS_FILE
    path.write_text("Validator,PT_VERSION,2.9.1\n")
    assert gemm_solutions_path(str(tmp_path)) == str(path)
    assert load_gemm_solutions(str(tmp_path)) == 1
    assert tunable.calls == [("enable", True), ("tuning_enable", False), ("read_file", str(path))]
    assert tunable.enabled is True and tunable.tuning is False


def test_unvalidated_file_leaves_tunableop_off(tmp_path, monkeypatch, caplog):
    fake = FakeTunable(validates=False)
    monkeypatch.setattr(torch.cuda, "tunable", fake, raising=False)
    (tmp_path / GEMM_SOLUTIONS_FILE).write_text("Validator,PT_VERSION,0.0.0\n")
    with caplog.at_level("WARNING"):
        assert load_gemm_solutions(str(tmp_path)) == 0
    assert fake.enabled is False
    assert "tune_gemm_solutions.py" in caplog.text


def test_rocm_serves_bf16_gemms_through_torch(monkeypatch):
    """On ROCm aiter's bf16 dispatch is pointed at the harness's empty table (its split-K picks are not
    repeatable); a table the caller set stays; CUDA is left alone."""
    monkeypatch.delenv(AITER_GEMM_TABLE_ENV, raising=False)
    monkeypatch.setattr(torch.version, "hip", None)
    assert serve_bf16_gemms_through_torch() is False
    assert AITER_GEMM_TABLE_ENV not in os.environ
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    assert serve_bf16_gemms_through_torch() is True
    assert os.environ[AITER_GEMM_TABLE_ENV] == AITER_GEMM_TABLE
    assert Path(AITER_GEMM_TABLE).read_text().splitlines() == [
        "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw"
    ]
    monkeypatch.setenv(AITER_GEMM_TABLE_ENV, "/tuned/by/the/user.csv")
    assert serve_bf16_gemms_through_torch() is False
    assert os.environ[AITER_GEMM_TABLE_ENV] == "/tuned/by/the/user.csv"


def test_tune_tool_replays_linear_lines(monkeypatch, tool):
    linears, offline = [], []
    monkeypatch.setattr(
        torch.nn.functional, "linear", lambda x, w: linears.append((tuple(x.shape), tuple(w.shape), x.dtype))
    )
    monkeypatch.setattr(
        tool.tunable, "_process_single_offline_gemm", lambda line, dev: offline.append(line), raising=False
    )
    monkeypatch.setattr(torch, "randn", lambda *shape, dtype, device: SimpleNamespace(shape=shape, dtype=dtype))
    cpu = torch.device("cpu")
    tool.tune_line("GemmTunableOp_BFloat16_TN,tn_34816_1_5120_ld_5120_5120_34816", cpu)
    tool.tune_line("GemmTunableOp_BFloat16_NN,nn_64_32_5120_ld_64_5120_64", cpu)
    tool.tune_line("GemmTunableOp_BFloat16_TN,tn_34816_8_5120_ld_6144_5120_34816", cpu)  # sub-matrix: offline helper
    assert linears == [((1, 5120), (34816, 5120), torch.bfloat16)]
    assert [line.split(",")[1] for line in offline] == [
        "nn_64_32_5120_ld_64_5120_64",
        "tn_34816_8_5120_ld_6144_5120_34816",
    ]


def test_untuned_lines_dedupe_in_order(tmp_path, tool):
    a = tmp_path / "untuned0.csv"
    b = tmp_path / "untuned1.csv"
    a.write_text(
        "Validator,PT_VERSION,2.9.1\nGemmTunableOp_BFloat16_TN,tn_1_2_3_ld_3_3_1\nGemmTunableOp_BFloat16_TN,tn_4_5_6_ld_6_6_4\n"
    )
    b.write_text(
        "GemmTunableOp_BFloat16_TN,tn_4_5_6_ld_6_6_4\nScaledGemmTunableOp_x,y_9_300_7\nGemmTunableOp_BFloat16_TN,tn_4_65_6_ld_6_6_4\n"
    )
    assert tool.untuned_lines([a, b], 64) == [
        "GemmTunableOp_BFloat16_TN,tn_1_2_3_ld_3_3_1",
        "GemmTunableOp_BFloat16_TN,tn_4_5_6_ld_6_6_4",
    ]
    assert tool.untuned_lines([b], 300) == [
        "GemmTunableOp_BFloat16_TN,tn_4_5_6_ld_6_6_4",
        "ScaledGemmTunableOp_x,y_9_300_7",
        "GemmTunableOp_BFloat16_TN,tn_4_65_6_ld_6_6_4",
    ]


def test_decode_rows_cover_the_served_batches(tmp_path, tool):
    assert tool.decode_rows([1, 2], 4, [8, 16]) == {1, 2, 4, 8, 16, 32}
    lines = [f"GemmTunableOp_BFloat16_TN,tn_5120_{rows}_5120_ld_5120_5120_5120" for rows in (1, 3, 8, 69, 256)]
    path = tmp_path / "untuned.csv"
    path.write_text("\n".join(lines))
    selected = tool.untuned_lines([path], 64, tool.decode_rows([1, 8], 4, [8, 32]))
    assert [tool.gemm_rows(line) for line in selected] == [1, 8, 256]


@pytest.mark.parametrize("values", [("0", "4", "8"), ("8-1", "4", "8"), ("1-8", "0", "8"), ("1", "4", "0")])
def test_decode_rows_reject_invalid_dimensions(tmp_path, monkeypatch, capsys, tool, values):
    path = tmp_path / "untuned.csv"
    path.write_text("GemmTunableOp_BFloat16_TN,tn_4_1_6_ld_6_6_4\n")
    monkeypatch.setattr(
        "sys.argv",
        ["tune_gemm_solutions.py", "--data-path", str(tmp_path), "--untuned", str(path), "--decode-rows", *values],
    )
    with pytest.raises(SystemExit) as error:
        tool.main()
    assert error.value.code == 2
    assert "--decode-rows" in capsys.readouterr().err
