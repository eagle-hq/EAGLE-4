# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Removed options fail clearly; context extrapolation is visible."""

import dataclasses

import pytest

from eagle_harness.core.config import harness_config_from_dict
from eagle_harness.core.cost_model import CostModel
from tests.test_chain_width import _table


@pytest.mark.parametrize("key,value", [("moe_top_k", 6), ("packed_candidate_gather", False)])
def test_removed_draft_options_are_unknown(key, value):
    with pytest.raises(ValueError, match="unknown keys"):
        harness_config_from_dict({"mode": "static", "draft_model": {key: value}})


def test_unused_ragged_history_is_not_model_state():
    assert "ragged_recent_verify_nums" not in {field.name for field in dataclasses.fields(CostModel)}


def test_tail_word_covers_the_verify_limit():
    from eagle_harness.core.limits import MAX_VERIFY_NUM, TAIL_MASK_BITS

    assert MAX_VERIFY_NUM <= TAIL_MASK_BITS == 64


def test_context_clamp_warns_once_per_table(caplog):
    table = _table(1, lambda w: 100.0, 10.0)
    assert table.interval_index(511) == 0
    assert not caplog.records
    assert table.interval_index(768) == 1
    assert table.interval_index(1024) == 1
    assert len(caplog.records) == 1
    assert "clamp" in caplog.text and "768" in caplog.text
