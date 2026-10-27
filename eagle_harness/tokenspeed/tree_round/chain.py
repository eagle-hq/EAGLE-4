# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-Harness device-chained rounds: a round's captured graphs launch one
another on the GPU, so the host launches depth 1 and never waits on the
round's decisions.

Each depth's graph holds a kernel, right after the depth's decision, that
tail-launches the round's next graph of its padded batch: depth ``d + 1``
while ``go[d + 1]`` is set, else the greedy or sampled finalize graph. A tail
launch starts once the launching graph has completed, and the stream that
launched depth 1 runs its later work after the whole round. Depth 1 is an
ordinary graph the host launches; the later depths and the finalize are
instantiated for device launch, which allows kernel, memcpy and memset nodes
and fork/join edges, but no event nodes. A device launch skips torch's
``CUDAGraph.replay``; these round graphs must capture no torch random
op. A missing graph handle traps on the device. ``instantiate`` requires
a contiguous depth prefix and one greedy, optionally one sampled, finalizer.
"""

from __future__ import annotations

import torch

from eagle_harness.tokenspeed.tree_round.ops.chain import (
    eagle_harness_chain_destroy,
    eagle_harness_chain_instantiate,
    eagle_harness_chain_load,
    eagle_harness_chain_next,
)

__all__ = ["DeviceChain"]


class DeviceChain:
    """The device-launchable round graphs of each padded batch.

    Args:
        max_bs: The largest padded batch.
        last_depth: The deepest depth a round drafts (``tree_depth - 1``).
        device: The CUDA device the rounds run on.
    """

    def __init__(self, max_bs: int, last_depth: int, device: torch.device) -> None:
        if not 1 <= max_bs <= 512 or not 1 <= last_depth <= 31:
            raise ValueError(
                "DeviceChain needs max_bs in 1..512 and last_depth in 1..31"
            )
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DeviceChain requires a CUDA device")
        self.last_depth = last_depth
        # Per padded batch: depth d's exec at column d (from 2), then the
        # greedy and sampled finalizers at fixed columns; uncaptured slots stay zero.
        self._execs = torch.zeros(
            (max_bs + 1, last_depth + 3), dtype=torch.int64, device=device
        )
        # Per padded batch: the exec handles it owns.
        self._handles: dict[int, list[int]] = {}
        # Loaded here: the first launch may be inside a graph capture.
        with torch.cuda.device(self.device):
            eagle_harness_chain_load()

    def end_depth(
        self, rows: int, depth: int, go: torch.Tensor, sampled: torch.Tensor
    ) -> None:
        """Enqueue, inside depth ``depth``'s graph at padded batch ``rows``,
        the launch of the round's next graph, which starts once that graph
        has ended. ``sampled`` selects the captured finalizer."""
        if not 1 <= rows < self._execs.shape[0] or not 1 <= depth <= self.last_depth:
            raise ValueError("DeviceChain rows/depth outside the captured shape")
        eagle_harness_chain_next(self._execs[rows], go, sampled, depth, self.last_depth)

    def instantiate(
        self,
        rows: int,
        depths: list[torch.cuda.CUDAGraph],
        finalize: list[torch.cuda.CUDAGraph],
        stream: torch.cuda.Stream,
    ) -> None:
        """Instantiate the graphs of padded batch ``rows`` (``keep_graph``
        captures) for device launch: ``depths`` is a prefix starting at depth 2;
        ``finalize`` is greedy, then optionally sampled. Publish on the upload stream; callers
        must order that stream before replay. Call release before recapturing."""
        if not 1 <= rows < self._execs.shape[0]:
            raise ValueError("DeviceChain rows outside the captured shape")
        if rows in self._handles:
            raise ValueError(
                f"DeviceChain batch {rows} already instantiated; release before replacing"
            )
        if len(depths) > self.last_depth - 1 or len(finalize) not in (1, 2):
            raise ValueError(
                "DeviceChain requires a depth prefix from 2 and one or two finalizers"
            )
        if stream.device != self._execs.device:
            raise ValueError("DeviceChain upload stream must be on the graph device")
        handles = []
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            try:
                for graph in (*depths, *finalize):
                    handles.append(
                        eagle_harness_chain_instantiate(
                            graph.raw_cuda_graph(), stream.cuda_stream
                        )
                    )
                row = [0] * (self.last_depth + 3)
                row[2 : 2 + len(depths)] = handles[: len(depths)]
                finalize_col = self.last_depth + 1
                row[finalize_col : finalize_col + len(finalize)] = handles[
                    len(depths) :
                ]
                self._execs[rows].copy_(torch.tensor(row, dtype=torch.int64))
            except Exception:
                stream.synchronize()
                for handle in handles:
                    eagle_harness_chain_destroy(handle)
                raise
        self._handles[rows] = handles

    def release(self) -> None:
        """Free every exec handle, once no round runs."""
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            for handles in self._handles.values():
                for handle in handles:
                    eagle_harness_chain_destroy(handle)
            self._handles.clear()
            self._execs.zero_()
