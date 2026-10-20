# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Offline crossover files cannot silently select tactics on another stack."""

import json

import pytest

from eagle_harness.core.candidate_calibration import load_candidate_crossover


def test_candidate_topk_preserves_the_calibrated_crossover(monkeypatch):
    pytest.importorskip("triton")
    from eagle_harness.kernels import candidate_ops

    seen = []
    expected = (object(), object())

    def head(*args, max_triton_elements=None, **kwargs):
        seen.append(max_triton_elements)
        return object(), object(), expected

    monkeypatch.setattr(candidate_ops, "candidate_head", head)
    result = candidate_ops.candidate_topk(
        None, None, None, num_candidates=128, vocab_start=0, top_k=4,
        max_triton_elements=32768,
    )
    assert seen == [32768]
    assert result is expected


def test_missing_or_other_device_keeps_default(tmp_path):
    assert load_candidate_crossover(None, {}) is None
    assert load_candidate_crossover(tmp_path, {}) is None
    (tmp_path / "candidate_topk.json").write_text(
        json.dumps({"identity": {"device": "other"}, "max_triton_elements": 42})
    )
    assert load_candidate_crossover(tmp_path, {"device": "this"}) is None


def test_exact_saved_identity_is_required(tmp_path):
    identity = {
        "device": "gpu",
        "torch": "version",
        "triton": "compiler",
        "dtype": "bf16",
        "vocab_size": 4096,
        "num_candidates": 128,
    }
    (tmp_path / "candidate_topk.json").write_text(json.dumps({"identity": identity, "max_triton_elements": 32768}))
    assert load_candidate_crossover(tmp_path, identity) == 32768
    for key in identity:
        assert load_candidate_crossover(tmp_path, {**identity, key: "different"}) is None


@pytest.mark.parametrize("value", [-1, 1.5, True, "32000000"])
def test_malformed_matching_profile_fails_closed(tmp_path, value):
    (tmp_path / "candidate_topk.json").write_text(json.dumps({"identity": {}, "max_triton_elements": value}))
    with pytest.raises(ValueError, match="crossover"):
        load_candidate_crossover(tmp_path, {})
