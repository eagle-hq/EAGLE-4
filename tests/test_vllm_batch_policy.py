# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The per-batch policy behind vLLM's optional ``suspend_drafting`` / ``verify_budget`` speculator hooks, on both
engine shapes: the 0.1.0 release engine (no hooks: native ranges go through ``native_decode_batch_sizes``) and review
PR 3 (hooks: the speculator answers them)."""

import logging
import sys
import types
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from eagle_harness.vllm import batch_policy
from eagle_harness.vllm.batch_policy import BatchPolicy, host_speculator_hooks
from eagle_harness.vllm.config import configure_speculative_config, native_decode_ranges
from tests.test_chain_no_spec import _native_schedule_config

HOOKS = frozenset(batch_policy.SPECULATOR_HOOKS)
SCHEDULE = [(1, 19, 15), (20, 512, 3), (513, 1 << 16, 0)]


def test_suspend_follows_the_native_ranges_only():
    policy = BatchPolicy([(8, 16), (40, 64)], None, 64)
    assert [policy.suspend(n) for n in (1, 7, 8, 16, 17, 40, 64)] == [False, False, True, True, False, True, True]
    assert policy.suspended == 4
    assert not BatchPolicy((), None, 64).suspend(64)


def test_verify_budget_keeps_the_schedule_unless_the_batch_size_verifies_fewer():
    policy = BatchPolicy((), SCHEDULE, 64)
    drafted_small = np.array([15, 15, 15], dtype=np.int32)
    # the batch drafted at its own batch size: vLLM keeps it as scheduled (same object)
    assert policy.verified_drafts(3, drafted_small) is drafted_small
    within = np.array([3, 2], dtype=np.int32)
    assert policy.verified_drafts(20, within) is within
    # drafted at batch size < 20 (width 16), verified in a batch of 20 (width 4)
    trimmed = policy.verified_drafts(20, np.array([15, 3, 0], dtype=np.int32))
    assert trimmed.tolist() == [3, 3, 0]
    # batch sizes past max_num_reqs take the last entry
    assert policy.verified_drafts(99, np.array([15], dtype=np.int32)).tolist() == [3]
    assert (policy.batches, policy.trimmed) == (4, 2)


def test_without_a_schedule_every_scheduled_draft_is_verified():
    drafts = np.array([7, 7], dtype=np.int32)
    assert BatchPolicy([(1, 4)], None, 64).verified_drafts(2, drafts) is drafts
    empty = np.array([], dtype=np.int32)
    assert BatchPolicy((), SCHEDULE, 64).verified_drafts(0, empty) is empty


def test_policy_logs_the_first_event_then_once_a_minute(caplog):
    now = [100.0]
    policy = BatchPolicy([(64, 64)], SCHEDULE, 64, clock=lambda: now[0])
    with caplog.at_level(logging.INFO, logger=batch_policy.__name__):
        policy.verified_drafts(20, np.array([15], dtype=np.int32))
        policy.suspend(64)
        now[0] += batch_policy.LOG_INTERVAL_S
        policy.suspend(64)
    lines = [record.getMessage() for record in caplog.records]
    assert lines == [
        "eagle_harness: verify_budget trimmed 1 of 1 draft batches to their batch size's width; "
        "suspend_drafting suspended 0 batches",
        "eagle_harness: verify_budget trimmed 1 of 1 draft batches to their batch size's width; "
        "suspend_drafting suspended 2 batches",
    ]


def _fake_vllm(monkeypatch, base):
    module = types.ModuleType("vllm.v1.worker.gpu.spec_decode.speculator")
    module.BaseSpeculator = base
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_host_hooks_are_found_by_name_and_parameters(monkeypatch):
    class Release:
        pass

    class ReviewPr3:
        def suspend_drafting(self, num_reqs, req_ids):
            return False

        def verify_budget(self, req_ids, num_draft_tokens):
            return num_draft_tokens

    class Renamed:
        def suspend_drafting(self, batch_size, req_ids):
            return False

    _fake_vllm(monkeypatch, Release)
    assert host_speculator_hooks() == frozenset()
    _fake_vllm(monkeypatch, ReviewPr3)
    assert host_speculator_hooks() == HOOKS
    _fake_vllm(monkeypatch, Renamed)
    assert host_speculator_hooks() == frozenset()


def test_installed_host_hooks_are_recognized_when_present():
    speculator = pytest.importorskip("vllm.v1.worker.gpu.spec_decode.speculator")
    present = {name for name in batch_policy.SPECULATOR_HOOKS if hasattr(speculator.BaseSpeculator, name)}
    # a hook vLLM has under another signature would be silently ignored: pin the contract
    assert host_speculator_hooks() == present


@pytest.mark.parametrize("hooks", [frozenset(), HOOKS], ids=["release-engine", "hook-engine"])
def test_native_ranges_reach_the_host_its_own_way(tmp_path, hooks):
    with mock.patch.object(batch_policy, "host_speculator_hooks", return_value=hooks):
        config = _native_schedule_config(tmp_path, capable=not hooks)
    ranges = native_decode_ranges(config)
    assert ranges and ranges[0][0] >= 1
    if hooks:
        assert not hasattr(config, "native_decode_batch_sizes")  # the speculator answers suspend_drafting
    else:
        assert config.native_decode_batch_sizes == ranges


def test_neither_hook_nor_field_refuses_native_ranges(tmp_path):
    with mock.patch.object(batch_policy, "host_speculator_hooks", return_value=frozenset()):
        with pytest.raises(ValueError, match="suspend_drafting"):
            _native_schedule_config(tmp_path, capable=False)


@pytest.mark.parametrize("mode,schedule", [("static", None), ("serve", [(1, 65536, 0)])])
def test_native_ranges_are_empty_unless_configure_installed_the_schedule(tmp_path, mode, schedule):
    with mock.patch.object(batch_policy, "host_speculator_hooks", return_value=HOOKS):
        config = _native_schedule_config(tmp_path, capable=False, schedule=schedule, mode=mode)
    assert native_decode_ranges(config) == []


def test_native_ranges_survive_a_serialized_schedule(tmp_path):
    with mock.patch.object(batch_policy, "host_speculator_hooks", return_value=HOOKS):
        config = _native_schedule_config(tmp_path, capable=False)
    expected = native_decode_ranges(config)
    as_lists = SimpleNamespace(
        method_config=dict(config.method_config),
        num_speculative_tokens_per_batch_size=[list(entry) for entry in config.num_speculative_tokens_per_batch_size],
    )
    assert native_decode_ranges(as_lists) == expected
    configure_speculative_config(config)  # reconstructing keeps the schedule and the ranges
    assert native_decode_ranges(config) == expected


def test_speculator_hooks_delegate_to_its_batch_policy():
    pytest.importorskip("vllm")
    from eagle_harness.vllm.speculator import HarnessSpeculator

    speculator = HarnessSpeculator.__new__(HarnessSpeculator)
    speculator.batch_policy = BatchPolicy([(4, 8)], SCHEDULE, 64)
    assert speculator.suspend_drafting(4, ["r"] * 4) and not speculator.suspend_drafting(3, ["r"] * 3)
    trimmed = speculator.verify_budget([f"r{i}" for i in range(20)], np.full(20, 15, dtype=np.int32))
    assert trimmed.tolist() == [3] * 20
