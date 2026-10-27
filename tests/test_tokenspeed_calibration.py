# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""CPU calibration policy and fake-engine measurement-to-serving tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from eagle_harness.core.calibration import (
    AttentionShape,
    CalibrationPaths,
    MeasureParams,
    MeasureRecord,
    save_model_timings,
)
from eagle_harness.core.config import Mode, build_cost_model
from eagle_harness.core.cost_table import load_cost_tables
from eagle_harness.core.limits import ComputeIntensity, draft_query_lens, verify_buckets
from eagle_harness.tokenspeed.config import parse_algorithm_config
from eagle_harness.tokenspeed.measure_run import attention_config, measure_batch
from eagle_harness.tokenspeed.measurement import MeasurePolicy


@pytest.mark.parametrize("intensity,topk", [(ComputeIntensity.HIGH, 4), (ComputeIntensity.LOW, 8), (ComputeIntensity.CHAIN, 1)])
def test_measure_policy_sweeps_core_widths_without_reading_costs(intensity, topk):
    policy = MeasurePolicy(verify_num=16, node_top_k=topk, tree_depth=5,
                           compute_intensity=intensity.value, device=torch.device("cpu"))
    drafts, verifies = set(), set()
    for _ in range(120):
        policy.next_widths(8, 8)
        policy.write_round()
        drafts.add(policy.draft_width)
        verifies.add(policy.verify_width_now)
        assert policy.widths[1].item() == policy.previous_draft_width
        assert policy.widths[2].item() == policy.draft_width
        assert all(policy.go_ready(d, 12) for d in range(1, 5))
        assert not policy.go_ready(5, 12)
    assert drafts == set(w for w in draft_query_lens(8, intensity) if w <= topk)
    assert verifies == set(w for w in verify_buckets(8, intensity) if w <= 16)


class FakeEngine:
    """Emits known stage timings at the public driver/engine boundary."""

    def __init__(self, args):
        self.args = args
        self.batches = []

    def generate(self, *, input_ids, sampling_params):
        args = self.args
        bs = len(input_ids)
        self.batches.append(bs)
        assert sampling_params["ignore_eos"]
        assert sampling_params["max_new_tokens"] > 512
        paths = CalibrationPaths(args.run_dir)
        with open(paths.config + "/measure_config.json") as f:
            assert json.load(f)["batch_size"] == bs
        records = []
        intensity = ComputeIntensity(args.compute_intensity)
        for model, widths, fixed, layers in (
            ("slm", draft_query_lens(bs, intensity), 100, 1),
            ("llm", verify_buckets(bs, intensity), 1000, 4),
        ):
            for q in widths:
                if q > (args.topk if model == "slm" else args.num_draft_tokens):
                    continue
                for length in range(64, 512):
                    us = fixed + q * bs + layers * (10 + length / 1000) + (length % 3) / 10
                    records.append(MeasureRecord(int(us * 1000), 1000, q, length, f"model_{model}", bs))
        save_model_timings(paths, MeasureParams(64, 2048, bs, 16, 4, 16, 1), records)


def fake_attention(**kwargs):
    return [MeasureRecord(int((10 + length / 1000) * 1000), 1000, width, length,
                          kwargs["category"], kwargs["batch_size"])
            for width in kwargs["widths"] for length in range(64, 2048, 4)]


def test_fake_engine_calibrates_every_served_batch_and_loads_profile(tmp_path):
    args = SimpleNamespace(run_dir=str(tmp_path), compute_intensity="high_intensity", context_step=64,
                           context_stop=2048, num_draft_tokens=16, topk=4, num_steps=5, fast=True, tp_size=4)
    engine = FakeEngine(args)
    target = SimpleNamespace(num_attention_heads=16, num_key_value_heads=4, head_dim=64, num_hidden_layers=4)
    draft = SimpleNamespace(num_attention_heads=16, num_key_value_heads=4, head_dim=64, num_hidden_layers=1)
    for bs in range(1, 33):
        assert measure_batch(args, engine, [1, 2, 3], bs, target, draft, attention_samples=fake_attention)
    assert engine.batches == list(range(1, 33))
    tables = load_cost_tables(str(tmp_path / "cost"))
    assert set(tables) == set(range(1, 33))
    for bs, table in tables.items():
        for interval in table.intervals:
            assert interval.draft.index == (0, 1, 3)
            assert interval.target.index == tuple(sorted(w - 1 for w in verify_buckets(bs, ComputeIntensity.HIGH) if w <= 16))
            assert all(v > 0 for v in (*interval.draft.value, *interval.target.value))
    config = parse_algorithm_config({"mode": "serve", "data_path": str(tmp_path)})
    assert config.mode is Mode.SERVE
    model = build_cost_model(config, node_top_k=4, tree_depth=5, verify_num=16, target_is_moe=False)
    assert model is not None


def test_short_engine_run_cannot_reuse_old_samples(tmp_path):
    args = SimpleNamespace(run_dir=str(tmp_path), compute_intensity="high_intensity", context_step=64,
                           context_stop=2048, num_draft_tokens=16, topk=4, num_steps=5, fast=True, tp_size=4)
    target = SimpleNamespace(num_attention_heads=16, num_key_value_heads=4, head_dim=64, num_hidden_layers=4)
    engine = FakeEngine(args)
    measure_batch(args, engine, [1], 1, target, target, attention_samples=fake_attention)
    with pytest.raises(FileNotFoundError):
        measure_batch(args, SimpleNamespace(generate=lambda **kw: None), [1], 1, target, target,
                      attention_samples=fake_attention)


def test_hybrid_attention_layer_ids_win_over_block_names():
    config = SimpleNamespace(num_attention_heads=16, num_key_value_heads=4, head_dim=64,
                             num_hidden_layers=8, full_attention_layer_ids=[2, 6], layer_types=["attention"] * 8)
    assert attention_config(config)[1].attention_layers == 2


def test_sliding_window_draft_is_measured_with_its_window():
    # The released Qwen3.8 EAGLE-4 head: one sliding-window layer, no full-attention layer.
    draft = SimpleNamespace(num_attention_heads=40, num_key_value_heads=40, head_dim=128, num_hidden_layers=1,
                            layer_types=["sliding_attention"], sliding_window=4096, use_sliding_window=True)
    assert attention_config(draft) == ((40, 40, 128, 4096), AttentionShape(1, 1, 4096))


def test_hybrid_target_measures_full_attention_without_a_window():
    target = SimpleNamespace(text_config=SimpleNamespace(
        num_attention_heads=24, num_key_value_heads=4, head_dim=256, num_hidden_layers=4, sliding_window=None,
        layer_types=["linear_attention"] * 3 + ["full_attention"]))
    assert attention_config(target) == ((24, 4, 256, 0), AttentionShape(1))


def test_mixed_full_and_sliding_layers_are_refused():
    config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=8, head_dim=64, num_hidden_layers=2,
                             layer_types=["full_attention", "sliding_attention"], sliding_window=1024)
    with pytest.raises(ValueError, match="mixed"):
        attention_config(config)


def test_cli_serializes_plugin_config_for_engine(tmp_path, monkeypatch):
    import sys

    from eagle_harness.tokenspeed import measure_run

    seen = {}

    class Engine:
        def __init__(self, **kwargs):
            # The engine's ServerArgs parser accepts JSON text, not a dict.
            seen.update(kwargs)
            seen["parsed_config"] = json.loads(kwargs["speculative_algorithm_config"])

        def shutdown(self):
            seen["shutdown"] = True

    monkeypatch.setitem(sys.modules, "tokenspeed.runtime.configs.model_config",
                        SimpleNamespace(get_hf_text_config=lambda config: config))
    monkeypatch.setitem(sys.modules, "tokenspeed.runtime.entrypoints.engine", SimpleNamespace(Engine=Engine))
    monkeypatch.setitem(sys.modules, "tokenspeed.runtime.utils.hf_transformers_utils",
                        SimpleNamespace(get_config=lambda *a, **kw: object()))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: object()),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: lambda *a, **kw: {"input_ids": [1, 2, 3]}),
    ))
    monkeypatch.setattr(measure_run, "measure_batch", lambda *a, **kw: True)
    measure_run.main(["--model", "target", "--draft-model", "draft", "--run-dir", str(tmp_path),
                      "--tp-size", "4", "--compute-intensity", "high_intensity", "--topk", "4",
                      "--num-steps", "5", "--num-draft-tokens", "16", "--max-bs", "1",
                      "--mamba-ssm-dtype", "float32"])
    assert seen["parsed_config"] == {"mode": "calibrate", "data_path": str(tmp_path),
                                     "intensity": "high_intensity"}
    assert seen["shutdown"]


def test_recorder_pairs_next_verify_and_writes_only_complete_rank_zero_batch(tmp_path, monkeypatch):
    from eagle_harness.tokenspeed.measurement import CostMeasurement, MEASURED_ROUNDS

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1.25

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    paths = CalibrationPaths(str(tmp_path))
    paths.write_measure_params(MeasureParams(64, 2048, 2, 16, 4, 16, 8))
    for rank in (0, 1):
        recorder = CostMeasurement(run_dir=str(tmp_path), tree_depth=5, tp_group=None,
                                   rank=rank, device=torch.device("cpu"))
        recorder.on_extend()
        recorder.record_verify(2)
        assert recorder.verify_rounds == 0
        recorder.end_draft(1, 4, 16, 100)
        assert recorder.pending_verify is None
        for i in range(MEASURED_ROUNDS):
            recorder.begin_draft()
            recorder.end_draft(2, 4, 16, 100 + i)
            recorder.record_verify(2)
        assert recorder.finished
        assert recorder.complete() is (rank == 0)
        if rank == 0:
            assert len(recorder.records) == 2 * MEASURED_ROUNDS
            assert recorder.records[-1].query_length == 16
            assert recorder.records[-1].cache_length == 100 + MEASURED_ROUNDS - 1
            assert recorder.records[0].measure_count == 4
            recorder.write_model_records()
            assert not recorder.complete()
        else:
            assert recorder.records == []
        recorder.on_extend()
        assert not recorder.finished and recorder.pending_verify is None
