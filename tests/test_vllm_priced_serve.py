# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Automatic serving must not turn missing prices into maximum speculation."""

from types import SimpleNamespace

import pytest

from eagle_harness.vllm.config import configure_speculative_config


def _spec(data_path, **overrides):
    return SimpleNamespace(
        method_config={"data_path": str(data_path), "intensity": "chain_intensity", **overrides},
        draft_model_config=None,
        num_speculative_tokens_per_batch_size=None,
        num_speculative_tokens=None,
    )


def test_unpriced_automatic_chain_refuses_startup(tmp_path):
    with pytest.raises(ValueError, match="acceptance profile"):
        configure_speculative_config(_spec(tmp_path))


def test_explicit_chain_width_preserves_unpriced_opt_in(tmp_path):
    config = _spec(tmp_path, max_verify_width=8)
    configure_speculative_config(config)
    assert config.num_speculative_tokens_per_batch_size == [(1, 512, 7), (513, 1 << 16, 0)]
