# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Platform-specific draft prelaunch ordering."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("side_stream", [True, False])
def test_prelaunch_follows_the_side_stream_predicate(monkeypatch, side_stream):
    from contextlib import nullcontext

    pytest.importorskip("sglang")
    from eagle_harness.kernels import arch
    from eagle_harness.sglang.drafter import DynamicDrafter

    class Stream:
        def wait_stream(self, other):
            pass

        def record_event(self):
            return self

    caller, side = Stream(), Stream()
    monkeypatch.setattr(arch, "side_stream_prelaunch", lambda device=None: side_stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: caller)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: side)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    drafter = object.__new__(DynamicDrafter)
    drafter._prelaunch_stream = drafter._pending = drafter._device = None
    drafted, extended = [], []
    drafter._chain_width = None
    drafter.live = None
    drafter.policy = lambda bs: "fixed"
    drafter._queue_until_decision = lambda bs, policy, context: drafted.append(bs) or 2
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(kv=SimpleNamespace(req_pool_idx=3))],
        req_pool_indices=SimpleNamespace(record_stream=lambda stream: None),
    )
    fence = drafter.prelaunch(batch, extend=lambda: extended.append(True), inputs=())
    assert extended == [True]
    assert fence is (side if side_stream else caller)
    assert drafted == ([1] if side_stream else [])
    if not side_stream:
        assert drafter._pending is None
        assert drafter._prelaunch_stream is None
    else:
        assert drafter._pending.key == (3,)
        assert drafter._pending.next_depth == 2
