# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Native EAGLE3 metadata must not be reinterpreted as EAGLE4 metadata."""

from types import SimpleNamespace

from eagle_harness.vllm.config import configure_speculative_config


def test_native_eagle3_preserves_input_layer_ids():
    draft = SimpleNamespace(
        architectures=["Eagle3LlamaForCausalLM"],
        eagle_aux_hidden_state_layer_ids=[1, 16, 31],
    )
    spec = SimpleNamespace(
        method_config={"mode": "static", "node_top_k": 4, "tree_depth": 5, "max_verify_width": 8},
        draft_model_config=SimpleNamespace(hf_config=draft),
        num_speculative_tokens=None,
        num_speculative_tokens_per_batch_size=None,
        update_arch_=lambda: None,
    )
    configure_speculative_config(spec)
    assert draft.eagle_aux_hidden_state_layer_ids == [1, 16, 31]
    assert draft.architectures == ["Eagle3ForHarness"]
