# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""A capture between served rounds keeps the allocator's caches and the GPU queue (skipped without CUDA).

Run in a GPU job: ``pytest -q tests/test_graph_capture_gpu.py``.
"""

from unittest import mock

import pytest
import torch

pytest.importorskip("sglang")

from eagle_harness.sglang import graph_pool  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def test_capture_replays_the_body_without_flushing_or_syncing():
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)

    def body():
        torch.mul(x, 2.0, out=out)
        return out + 1.0

    body()  # eager first round, as the callers do
    graph_pool.capture(body)  # the pool's first capture may flush once (it anchors the pool)
    reserved = torch.cuda.memory_reserved()
    with (
        mock.patch.object(torch.cuda, "empty_cache", side_effect=AssertionError("empty_cache")),
        mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("synchronize")),
    ):
        graph, result = graph_pool.capture(body)
    assert torch.cuda.memory_reserved() >= reserved  # nothing the rounds cached was released

    x.fill_(3.0)  # written on the current stream: the replay must see it
    graph.replay()
    torch.cuda.current_stream().synchronize()
    assert torch.equal(result, torch.full_like(x, 7.0))
    assert torch.equal(out, torch.full_like(x, 6.0))


def test_capture_into_a_pool_whose_graphs_were_all_freed():
    x = torch.ones(8, device="cuda")
    for _ in range(3):  # each round frees every graph it captured, as a finished test or request does
        graph, out = graph_pool.capture(lambda: x * 2.0)
        graph.replay()
        assert torch.equal(out, torch.full_like(x, 2.0))
        del graph, out
