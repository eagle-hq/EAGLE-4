# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-Harness device-chained rounds (EAGLE-4): CUDA graphs instantiated
for device launch, and the kernel inside each draft depth's graph that
tail-launches the round's next graph; see ``csrc/eagle_harness_chain.cu``.

Graph and exec handles are the raw ``cudaGraph_t`` / ``cudaGraphExec_t``
pointers as integers (``torch.cuda.CUDAGraph(keep_graph=True)``'s
``raw_cuda_graph()``), streams the raw ``cudaStream_t`` (``cuda_stream``)."""

import torch

from eagle_harness.tokenspeed.tree_round.ops.build import load_module

__all__ = [
    "eagle_harness_chain_destroy",
    "eagle_harness_chain_instantiate",
    "eagle_harness_chain_load",
    "eagle_harness_chain_next",
]


def eagle_harness_chain_load() -> None:
    """Load the kernel library (no launch), before a graph capture uses it."""
    load_module("chain")


def eagle_harness_chain_instantiate(graph: int, stream: int) -> int:
    """Instantiate the captured ``graph`` for device launch and upload it on
    ``stream``; returns the exec handle (``eagle_harness_chain_destroy``
    frees it). The graph may hold kernel, memcpy (device or pinned memory)
    and memset nodes, fork/join edges included; no event, host, memory
    allocation or conditional nodes, and no dynamic parallelism beyond this
    module's own launch (CUDA's device graph requirements)."""
    return load_module("chain").eagle_harness_chain_instantiate(graph, stream)


def eagle_harness_chain_destroy(exec_handle: int) -> None:
    """Free an exec handle no queued or running chain still launches."""
    load_module("chain").eagle_harness_chain_destroy(exec_handle)


def eagle_harness_chain_next(
    execs: torch.Tensor,
    go: torch.Tensor,
    sampled: torch.Tensor,
    depth: int,
    last_depth: int,
) -> None:
    """Tail-launch depth+1 when go says so, otherwise the greedy finalize.

    execs keeps the partner [last_depth+3] ABI; the final column is reserved.
    sampled must contain false. A missing handle or sampled round traps.
    """
    load_module("chain").eagle_harness_chain_next(execs, go, sampled, depth, last_depth)
