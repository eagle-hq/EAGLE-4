# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""tools/stop_check.py refuses a workload file too short for the requested batches before sending anything."""

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [10] if text == "\n\n" else [ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


def load(monkeypatch):
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "requests", types.ModuleType("requests"))
    spec = importlib.util.spec_from_file_location("stop_check", Path(__file__).parents[1] / "tools/stop_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_short_data_is_a_usage_error(tmp_path, monkeypatch):
    module = load(monkeypatch)
    data = tmp_path / "question_multiturn.jsonl"
    data.write_text("".join(json.dumps({"turns": [f"question {i}"]}) + "\n" for i in range(3)))
    argv = ["--api", "sglang", "--port", "1", "--data", str(data), "--tokenizer", str(tmp_path)]
    with pytest.raises(SystemExit, match="has 3 conversations.*needs 32"):
        module.main([*argv, "--out", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()
