# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Native OFF is distinct from root-only speculative verification."""

from types import SimpleNamespace

import pytest

from eagle_harness.core.chain_width import chain_widths, save_accept_profile, save_round_times, served_chain_width
from eagle_harness.vllm.config import verify_width_schedule
from eagle_harness.core.limits import ComputeIntensity
from tests.test_chain_width import PROFILE


def test_native_decode_is_a_distinct_opt_in_action(tmp_path):
    save_accept_profile(str(tmp_path), 100, [90, 80, 70])
    save_round_times(
        str(tmp_path),
        [
            {"batch_size": 1, "width": 0, "round_us": 100, "count": 10},
            {"batch_size": 1, "width": 1, "round_us": 200, "count": 10},
            {"batch_size": 1, "width": 4, "round_us": 1000, "count": 10},
        ],
    )
    assert chain_widths(str(tmp_path), 4, 4) == [(1, 4, 1)]
    assert chain_widths(str(tmp_path), 4, 4, allow_no_spec=True) == [(1, 4, 0)]
    # K=0 still runs draft extend on vLLM: never silently claim it is native OFF.
    with pytest.raises(ValueError, match="native.*decode"):
        verify_width_schedule(ComputeIntensity.CHAIN, 4, str(tmp_path))


def _native_schedule_config(tmp_path, *, capable=True, schedule=None, mode="serve"):
    from eagle_harness.vllm.config import configure_speculative_config

    save_accept_profile(str(tmp_path), 100, [90, 80, 70])
    save_round_times(
        str(tmp_path),
        [
            {"batch_size": batch, "width": width, "round_us": us, "count": 10}
            for batch, times in [(1, {0: 1000, 1: 100, 4: 1000}), (8, {0: 100, 1: 200, 4: 1000})]
            for width, us in times.items()
        ],
    )
    config = SimpleNamespace(
        method_config={"mode": mode, "intensity": "chain_intensity", "max_verify_width": 4, "data_path": str(tmp_path)},
        draft_model_config=None,
        num_speculative_tokens=None,
        num_speculative_tokens_per_batch_size=schedule,
    )
    if capable:
        config.native_decode_batch_sizes = None
    configure_speculative_config(config)
    return config


def test_canonical_schedule_keeps_native_off_distinct_from_root_only(tmp_path):
    config = _native_schedule_config(tmp_path)
    expected = chain_widths(str(tmp_path), 4, 512, allow_no_spec=True)
    assert expected is not None
    assert {width for _, _, width in expected} == {0, 1}
    assert config.native_decode_batch_sizes == [(first, last) for first, last, width in expected if width == 0]
    assert all(k == 0 for _, _, k in config.num_speculative_tokens_per_batch_size)
    assert config.num_speculative_tokens == 1
    # Reconstructing an already configured object must keep its OFF metadata.
    from eagle_harness.vllm.config import configure_speculative_config

    configure_speculative_config(config)
    assert config.native_decode_batch_sizes == [(first, last) for first, last, width in expected if width == 0]


def test_native_off_requires_an_explicit_host_capability(tmp_path):
    with pytest.raises(ValueError, match="native.*decode"):
        _native_schedule_config(tmp_path, capable=False)


@pytest.mark.parametrize("mode,schedule", [("static", None), ("serve", [(1, 65536, 0)])])
def test_static_and_explicit_k_zero_do_not_opt_into_native_off(tmp_path, mode, schedule):
    config = _native_schedule_config(tmp_path, schedule=schedule, mode=mode)
    assert config.native_decode_batch_sizes is None
    assert config.num_speculative_tokens_per_batch_size == schedule


def test_native_decode_wins_ties_without_changing_existing_widths():
    assert served_chain_width({0: 100, 1: 100, 2: 190}, PROFILE, 2, allow_no_spec=True) == 0
    assert served_chain_width({0: 100, 1: 100, 2: 190}, PROFILE, 2) == 1
