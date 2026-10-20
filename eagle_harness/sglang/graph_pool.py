# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The CUDA graph memory pool the harness's graphs share with sglang's."""

from __future__ import annotations

import torch

from sglang.srt.model_executor.runner_utils.pool import get_global_graph_memory_pool, set_global_graph_memory_pool


def graph_memory_pool():
    """sglang's global graph memory pool, created on first use."""
    pool = get_global_graph_memory_pool()
    if pool is None:
        pool = torch.cuda.graph_pool_handle()
        set_global_graph_memory_pool(pool)
    return pool
