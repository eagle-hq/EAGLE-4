# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The EAGLE-4 structures both engine ports build (``eagle4_config.check_ported``), on synthetic configs."""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")

from eagle_harness.models.eagle4_config import UNPORTED, check_ported  # noqa: E402


def _config(**extra):
    base = dict(
        architectures=["LlamaForCausalLMEagle4"],
        num_hidden_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        vocab_size=1000,
        efficient_lm_head_config={"type": "low_rank", "rank": 32, "num_candidates": 64},
    )
    base.update(extra)
    return transformers.LlamaConfig(**base)


def test_dense_one_layer_draft_with_the_low_rank_head_is_ported():
    check_ported(_config())


@pytest.mark.parametrize("draft_vocab", [None, 0, 1000])
def test_full_draft_vocab_is_ported(draft_vocab):
    check_ported(_config(draft_vocab_size=draft_vocab))


def test_reduced_draft_vocab_is_refused():
    with pytest.raises(NotImplementedError, match="reduced vocab"):
        check_ported(_config(draft_vocab_size=320))


@pytest.mark.parametrize(
    "extra",
    [
        {"attn_norm": True},
        {"enable_post_mlp_norm": True},
        {"use_rope": False},
        {"num_hidden_layers": 2},
        {"efficient_lm_head_config": None},
    ],
)
def test_unported_structures_are_refused(extra):
    with pytest.raises(NotImplementedError):
        check_ported(_config(**extra))


def test_an_engine_can_refuse_more():
    # sglang's draft worker always shares the target embedding
    config = _config(share_draft_embedding_from_target=False)
    check_ported(config)
    with pytest.raises(NotImplementedError, match="share_draft_embedding_from_target"):
        check_ported(config, {**UNPORTED, "share_draft_embedding_from_target": False})
