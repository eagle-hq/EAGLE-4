# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""When :class:`~eagle_harness.sglang.verify_tail.VerifyTail` captures a graph: after a key's eager runs, and only
over static inputs (the capture itself is faked; ``tests/test_verify_tail_gpu.py`` replays real graphs)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang")

from eagle_harness.sglang import verify_tail  # noqa: E402


class _Graph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


@pytest.fixture
def captures(monkeypatch):
    made = []

    def capture(body):
        made.append(_Graph())
        return made[-1], body()

    monkeypatch.setattr(verify_tail, "_capture", capture)
    return made


def test_a_key_runs_eagerly_until_its_runs_would_have_paid_for_a_capture(captures):
    graphs, calls = verify_tail._Graphs(), []

    def body():
        calls.append(None)
        return len(calls)

    n = verify_tail.CAPTURE_AFTER
    outputs = [graphs.run(("bs", 512), body) for _ in range(n + 3)]
    assert outputs == list(range(1, n + 1)) + [n + 1] * 3  # eager results, then the captured graph's outputs
    assert len(captures) == 1 and captures[0].replays == 3
    assert graphs.run(("bs", 511), body) == n + 2  # another batch size starts over, eagerly
    assert graphs.run(("bs", 512), body, static=False) == n + 3  # inputs a graph would not read: eager
    assert len(captures) == 1


def test_a_selection_is_captured_only_over_the_ranking_graphs_outputs(monkeypatch, captures):
    monkeypatch.setattr(verify_tail, "CAPTURE_AFTER", 1)
    tree = SimpleNamespace(
        rank=lambda bs, depth, count: (torch.arange(count).expand(bs, count), torch.ones(bs, count + 1)),
        select=lambda bs, steps, ranked, width: ranked[:, : width - 1].clone(),
    )
    tail = verify_tail.VerifyTail(tree, lambda bs, selection, steps, width: selection)
    eager = tail.rank(2, 3, 5)
    for _ in range(3):  # an eager ranking's tensors are not the ones a later round refills
        tail.verify_input(2, 3, 4, eager)
    assert not captures
    ranking = tail.rank(2, 3, 5)  # second run: captured
    assert len(captures) == 1 and ranking is tail.rank(2, 3, 5)
    tail.verify_input(2, 3, 4, ranking)
    tail.verify_input(2, 3, 4, ranking)
    assert len(captures) == 2
    tail.verify_input(2, 3, 4, eager)  # the selection graph reads the ranking graph's buffers: not these
    assert captures[1].replays == 1
