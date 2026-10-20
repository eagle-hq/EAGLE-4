# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Both model ports apply the same full-vocabulary draft contract."""

import importlib

import pytest

transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
@pytest.mark.parametrize("draft_vocab", ["omitted", None, 0, 1000, 320])
def test_draft_vocabulary_admission(engine, draft_vocab):
    pytest.importorskip(engine)
    path = "sglang.models.llama_eagle4" if engine == "sglang" else "vllm.models"
    check_supported = importlib.import_module(f"eagle_harness.{path}").check_supported
    config = transformers.LlamaConfig(
        architectures=["LlamaForCausalLMEagle4"],
        num_hidden_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        vocab_size=1000,
        efficient_lm_head_config={"type": "low_rank", "rank": 32, "num_candidates": 64},
    )
    if draft_vocab != "omitted":
        config.draft_vocab_size = draft_vocab
    if draft_vocab == 320:
        with pytest.raises(NotImplementedError, match="reduced vocab"):
            check_supported(config)
    else:
        check_supported(config)
