# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Tree verify width per batch size from served rounds: the high-load default of the vLLM tree."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import pytest

from eagle_harness.core.chain_width import (
    TREE_ROUNDS_FILE,
    load_tree_rounds,
    merge_tree_rounds,
    served_tree_schedule,
    width_at,
)


def _row(bs, width, round_us, accept, count=50):
    return {"batch_size": bs, "width": width, "round_us": round_us, "accept": accept, "count": count}


# The release numbers at TP4: the 16-wide tree wins at batch 8 and loses at 64 (served round times, ms).
RELEASE = [
    _row(8, 16, 12_500, 4.43),
    _row(8, 4, 9_000, 2.9),
    _row(8, 1, 6_000, 1.0),
    _row(64, 16, 47_900, 3.96),
    _row(64, 4, 20_000, 2.9),
    _row(64, 1, 12_000, 1.0),
]


def _rounds(rows):
    out = {}
    for r in rows:
        out.setdefault(r["batch_size"], {})[r["width"]] = (r["round_us"], r["accept"], r["count"])
    return out


def test_packaged_tp4_rounds_preserve_low_batch_and_narrow_from_twenty():
    profile = files("eagle_harness.core").joinpath("profiles/vllm-qwen38-tp4")
    rounds = load_tree_rounds(str(profile))
    assert rounds is not None, "shipped TP4 policy requires its measured rounds"
    assert hashlib.sha256(profile.joinpath(TREE_ROUNDS_FILE).read_bytes()).hexdigest() == (
        "589c4abcf7523bf1087390c8b32644e351010fc7b3c8d53d1a43c67d10d8043c"
    )
    schedule = served_tree_schedule(rounds, 16, 512)
    assert schedule is not None
    for batch in (1, 8, 19):
        assert width_at(schedule, batch) == 16
    for batch in (20, 64, 512):
        assert width_at(schedule, batch) == 4


def test_low_batch_keeps_the_configured_width_and_high_batch_narrows():
    schedule = served_tree_schedule(_rounds(RELEASE), 16, 512)
    assert width_at(schedule, 1) == 16
    assert width_at(schedule, 8) == 16
    # 64: 2.9 / 20 ms beats 3.96 / 47.9 ms and 1 / 12 ms.
    assert width_at(schedule, 64) == 4
    assert width_at(schedule, 512) == 4
    assert schedule[0][0] == 1 and schedule[-1][1] == 512


def test_a_few_rounds_of_a_drained_batch_do_not_move_the_low_batch_width():
    """TP1 calibration shape: the tails of two passes leave a few rounds at batch sizes 2-7 where the remaining requests
    differ between the servers. Thousands of rounds at batch 8 say 16 wins there; the switch stays above it."""
    rows = [
        _row(1, 16, 14_218, 9.91, 63), _row(1, 4, 11_611, 3.35, 163),
        _row(2, 16, 12_000, 2.11, 21), _row(2, 4, 12_125, 2.30, 21),
        _row(6, 16, 19_888, 2.96, 99), _row(6, 4, 15_727, 2.52, 77),
        _row(7, 16, 22_581, 3.17, 31), _row(7, 4, 17_605, 2.73, 70),
        _row(8, 16, 23_073, 4.51, 1654), _row(8, 4, 16_550, 2.81, 2834),
        _row(20, 16, 39_769, 3.75, 53), _row(20, 4, 19_500, 2.50, 40),
        _row(42, 16, 70_849, 4.12, 212),
        _row(43, 16, 72_997, 4.34, 247), _row(43, 4, 27_000, 2.70, 30),
    ]
    assert served_tree_schedule(_rounds(rows), 16, 512) == [(1, 19, 16), (20, 512, 4)]


def test_narrowing_never_widens_at_a_larger_batch():
    rows = RELEASE + [_row(256, 16, 150_000, 3.9), _row(256, 4, 40_000, 2.8), _row(256, 2, 25_000, 1.9)]
    rows += [_row(512, 16, 300_000, 3.9), _row(512, 4, 70_000, 2.8), _row(512, 2, 60_000, 1.9)]
    schedule = served_tree_schedule(_rounds(rows), 16, 512)
    assert (width_at(schedule, 64), width_at(schedule, 256), width_at(schedule, 512)) == (4, 2, 2)


def test_batch_sizes_below_the_first_loss_keep_the_configured_width():
    # Only batch 64 was measured: everything below it keeps 16, even batch sizes nearer to nothing else.
    rows = [r for r in RELEASE if r["batch_size"] == 64]
    schedule = served_tree_schedule(_rounds(rows), 16, 512)
    assert width_at(schedule, 63) == 16
    assert width_at(schedule, 64) == 4


def test_off_wins_where_every_draft_loses():
    rows = RELEASE[:3] + [_row(64, 16, 47_900, 3.96), _row(64, 4, 40_000, 2.9), _row(64, 1, 9_500, 1.0)]
    assert width_at(served_tree_schedule(_rounds(rows), 16, 512), 64) == 1


def test_no_change_without_a_loss_or_without_the_configured_width():
    assert served_tree_schedule(_rounds([_row(8, 16, 12_500, 4.43), _row(8, 4, 9_000, 2.9)]), 16, 512) is None
    # A batch size that did not time the configured width cannot show that it loses.
    assert served_tree_schedule(_rounds([_row(64, 4, 20_000, 2.9)]), 16, 512) is None
    assert served_tree_schedule({}, 16, 512) is None


def test_wider_than_configured_is_never_chosen():
    rows = RELEASE + [_row(64, 32, 1_000, 8.0)]
    assert width_at(served_tree_schedule(_rounds(rows), 16, 512), 64) == 4


def test_merge_keeps_other_widths_and_replaces_measured_cells(tmp_path):
    merge_tree_rounds(str(tmp_path), [_row(8, 16, 12_000, 4.4), _row(64, 16, 48_000, 3.9)])
    merge_tree_rounds(str(tmp_path), [_row(64, 16, 47_900, 3.96), _row(64, 4, 20_000, 2.9)])
    rounds = load_tree_rounds(str(tmp_path))
    assert rounds == {8: {16: (12_000.0, 4.4, 50)}, 64: {16: (47_900.0, 3.96, 50), 4: (20_000.0, 2.9, 50)}}
    data = json.loads((tmp_path / TREE_ROUNDS_FILE).read_text())
    assert sorted((r["batch_size"], r["width"]) for r in data["rounds"]) == [(8, 16), (64, 4), (64, 16)]


def test_load_is_none_without_measurements(tmp_path):
    assert load_tree_rounds(str(tmp_path)) is None


@pytest.mark.parametrize(
    "bad",
    [
        _row(0, 16, 1000, 2.0),
        _row(8, 0, 1000, 2.0),
        _row(8, 16, 0, 2.0),
        _row(8, 16, 1000, 0.5),  # a round commits at least the bonus token
    ],
)
def test_merge_rejects_invalid_rows(tmp_path, bad):
    with pytest.raises(ValueError):
        merge_tree_rounds(str(tmp_path), [bad])
