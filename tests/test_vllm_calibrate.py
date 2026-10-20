# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Config-derived calibration geometry and real CPU fitting."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from eagle_harness.core.calibration import AttentionShape
from eagle_harness.vllm.calibrate import attention_shape, fit, main, merge, sweep, validate_tables


def test_sweep_and_fit_cover_the_requested_context(tmp_path):
    from eagle_harness.core.cost_table import load_cost_tables

    lengths = sweep(context_stop=8192)
    assert max(lengths) > 4096
    assert max(lengths) + 64 <= 8192

    def rows(widths):
        return [{"query_length": q, "cache_length": n, "time_us": 100 + q + n / 100} for q in widths for n in lengths]

    fit(
        str(tmp_path),
        rows([1, 8, 16, 32, 64]),
        rows([1, 2, 4, 8]),
        AttentionShape(4),
        AttentionShape(1),
        1,
        context_stop=8192,
    )
    table = load_cost_tables(str(tmp_path / "cost"))[1]
    assert len(table.intervals) == 31
    validate_tables(str(tmp_path), context_stop=8192)


def test_hybrid_target_counts_only_attention_layers():
    config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
    )
    assert attention_shape(config) == AttentionShape(1)


def test_sliding_draft_reads_its_own_window():
    assert attention_shape(
        SimpleNamespace(num_hidden_layers=1, layer_types=["sliding_attention"], sliding_window=2048)
    ) == AttentionShape(1, 1, 2048)


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(num_hidden_layers=1, layer_types=["unknown"]),
        SimpleNamespace(num_hidden_layers=2, layer_types=["full_attention"]),
        SimpleNamespace(num_hidden_layers=4, full_attention_interval=4),
        SimpleNamespace(num_hidden_layers=1, layer_types=["sliding_attention"]),
        SimpleNamespace(num_hidden_layers=1, layer_types=["mamba"]),
    ],
)
def test_ambiguous_attention_layout_is_not_a_guessed_qwen_shape(config):
    with pytest.raises(ValueError):
        attention_shape(config)


def test_fitter_consumes_all_tree_widths(tmp_path):
    # Synthetic timings test the file format, not hardware performance.
    def rows(widths):
        return [
            {"batch_size": 1, "query_length": q, "cache_length": length, "time_us": 100 + 5 * q + length / 100}
            for q in widths
            for length in sweep(context_stop=4096)
        ]

    assert fit(
        str(tmp_path),
        rows([1, 8, 16, 32, 64]),
        rows([1, 2, 4, 8]),
        AttentionShape(4),
        AttentionShape(1),
        1,
        context_stop=4096,
    ) == [1]
    validate_tables(str(tmp_path))
    for model in ("model_llm", "model_slm"):
        files = list((tmp_path / "cost" / "bs_1" / model).glob("*.json"))
        assert files
        for file in files:
            assert json.loads(file.read_text())


def test_fit_uses_recorded_geometry_without_reloading_models(tmp_path):
    def rows(widths):
        return [
            {"query_length": q, "cache_length": n, "time_us": 100 + q + n / 100}
            for q in widths
            for n in sweep(context_stop=4096)
        ]

    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps(
            {
                "tp": 2,
                "context_stop": 4096,
                "target_shape": asdict(AttentionShape(4)),
                "draft_shape": asdict(AttentionShape(1)),
                "target": rows([1, 8, 16, 32, 64]),
                "draft": rows([1, 2, 4, 8]),
            }
        )
    )
    main(["--model", "not-a-real-model", "--draft", "not-a-real-draft", "fit", str(tmp_path), str(raw)])
    validate_tables(str(tmp_path))
    (tmp_path / "cost/bs_1/model_slm/model_interval_0.json").write_text("{}")
    with pytest.raises(RuntimeError, match="incomplete vLLM"):
        validate_tables(str(tmp_path))


def test_measure_records_resolved_model_limit(monkeypatch, tmp_path):
    from eagle_harness.vllm import calibrate

    geometry = SimpleNamespace(num_hidden_layers=2)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=geometry, max_model_len=8192, dtype="bfloat16"),
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=geometry)),
    )
    measured = []

    def rpc(name, args):
        measured.extend(args[1])
        return [[]]

    llm = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=config), collective_rpc=rpc)
    monkeypatch.setattr(calibrate, "engine", lambda *args, **kwargs: llm)
    key = calibrate.CostKey(gpu="fixture", gpu_count=1, model="target", draft="draft", engine="vllm fixture", dtype="bfloat16")
    monkeypatch.setattr(calibrate, "cost_key", lambda *args: key)
    output = tmp_path / "raw.json"
    calibrate.measure(str(output), 64, False, [1], 256, 1, model="target", draft="draft")
    raw = json.loads(output.read_text())
    assert raw["context_stop"] == 8192
    assert raw["key"] == key.to_dict()
    assert max(measured) > 4096 and max(measured) + 64 <= 8192


def test_merge_rejects_different_tp(tmp_path):
    paths = []
    for tp in (1, 2):
        path = tmp_path / f"{tp}.json"
        path.write_text(json.dumps({"tp": tp, "target": [], "draft": []}))
        paths.append(str(path))
    with pytest.raises(ValueError, match="differ in tp"):
        merge(paths)
