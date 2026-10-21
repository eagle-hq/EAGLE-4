# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The CUDA graph memory pool the harness's graphs share with sglang's, and captures between served rounds."""

from __future__ import annotations

from typing import Callable, Dict, Tuple, TypeVar

import torch

from sglang.srt.model_executor.runner_utils.pool import get_global_graph_memory_pool, set_global_graph_memory_pool


def graph_memory_pool():
    """sglang's global graph memory pool, created on first use."""
    pool = get_global_graph_memory_pool()
    if pool is None:
        pool = torch.cuda.graph_pool_handle()
        set_global_graph_memory_pool(pool)
    return pool


T = TypeVar("T")
_capture_streams: Dict[int, torch.cuda.Stream] = {}
_anchors: Dict[object, torch.cuda.CUDAGraph] = {}  # pool handle -> a graph that keeps the pool's use count above 0


def capture(body: Callable[[], T]) -> Tuple[torch.cuda.CUDAGraph, T]:
    """Capture ``body()`` into a CUDA graph in the shared pool, for a graph first needed while serving.

    ``torch.cuda.graph`` synchronizes the device and empties the allocator's device and pinned-host caches before it
    captures. Between served rounds that drains the GPU queue and frees every cached block, which the next rounds
    allocate again: 20-30 ms per first-use capture on a TP4 batch-1 round. A capture only records work, so this
    captures on a side stream ordered after the current one, with neither.

    The cache flush also deletes a pool whose graphs were all freed; capturing into such a pool again without it
    fails an allocator assert. So the first capture into a pool goes through ``torch.cuda.graph`` once and keeps a
    small anchor graph in it, and the pool stays alive for every later capture.
    """
    pool = graph_memory_pool()
    if pool not in _anchors:
        anchor = torch.cuda.CUDAGraph()
        with torch.cuda.graph(anchor, pool=pool, capture_error_mode="thread_local"):
            torch.zeros(1, device="cuda")
        _anchors[pool] = anchor
    device = torch.cuda.current_device()
    stream = _capture_streams.get(device)
    if stream is None:
        stream = _capture_streams[device] = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        graph.capture_begin(pool=pool, capture_error_mode="thread_local")
        try:
            outputs = body()
        finally:
            graph.capture_end()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, outputs
