# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-Harness device-chained rounds: from an ordinary first graph, graphs
captured by torch and instantiated for device launch run the depths the go
flags select and then the greedy finalize, and the
launching stream orders its later work after the whole chain."""

import sys

import pytest
import torch

pytest.importorskip("tvm_ffi", reason="TokenSpeed round kernels require apache-tvm-ffi")

from eagle_harness.tokenspeed.tree_round.chain import DeviceChain

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU is required"
)

_LAST_DEPTH = 4
_FINALIZE = 100


@pytest.mark.parametrize("rows", [4, 33, 512])
@pytest.mark.parametrize("captured_depth", [2, _LAST_DEPTH])
@pytest.mark.parametrize("sampled_finalize", [False, True])
def test_chain_follows_the_go_flags_to_the_selected_finalize(
    rows, captured_depth, sampled_finalize
) -> None:
    device = torch.device("cuda")
    log = torch.zeros(64, dtype=torch.int64, device=device)
    count = torch.zeros(1, dtype=torch.int64, device=device)
    go = torch.zeros(_LAST_DEPTH + 1, dtype=torch.int32, device=device)
    sampled = torch.zeros(1, dtype=torch.bool, device=device)
    chain = DeviceChain(rows, _LAST_DEPTH, device)
    side = torch.cuda.Stream()

    def note(tag: int) -> None:
        log.index_fill_(0, count, tag)
        count.add_(1)

    stream = torch.cuda.Stream()
    graphs = []
    with torch.cuda.stream(stream):
        for depth in range(1, captured_depth + 1):
            # Depth 1 is an ordinary graph; the rest launch on the device.
            graph = torch.cuda.CUDAGraph(keep_graph=depth > 1)
            with torch.cuda.graph(graph, stream=stream):
                # The launch is enqueued on a side branch, as after a depth's decision.
                side.wait_stream(stream)
                with torch.cuda.stream(side):
                    chain.end_depth(rows, depth, go, sampled)
                torch.cuda._sleep(100_000)
                note(depth)
                stream.wait_stream(side)
            graphs.append(graph)
        for finalize in (
            (_FINALIZE, _FINALIZE + 1) if sampled_finalize else (_FINALIZE,)
        ):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                note(finalize)
            graphs.append(graph)
    chain.instantiate(rows, graphs[1:captured_depth], graphs[captured_depth:], stream)
    # Uploaded on the capture stream; the rounds launch from the default one.
    stream.synchronize()

    def run(go_flags: list[int], draw: bool) -> list[int]:
        go.copy_(torch.tensor([0, *go_flags], dtype=torch.int32))
        sampled.fill_(draw)
        count.zero_()
        graphs[0].replay()
        # Ordered after the whole chain.
        note(-1)
        torch.cuda.synchronize()
        return log[: int(count)].tolist()

    # go[d + 1] is read once depth d's graph ends; the last depth always finalizes.
    assert run([int(d <= captured_depth) for d in range(1, 5)], False) == [
        *range(1, captured_depth + 1),
        _FINALIZE,
        -1,
    ]
    assert run([1, 0, 1, 1], False) == [1, _FINALIZE, -1]
    assert run([1, 1, 0, 1], False) == [1, 2, _FINALIZE, -1]
    if sampled_finalize:
        assert run([1, 0, 0, 0], True) == [1, _FINALIZE + 1, -1]
        assert run([1, 1, 0, 0], True) == [1, 2, _FINALIZE + 1, -1]
    torch.cuda.synchronize()
    chain.release()
    chain.release()  # Releasing twice owns no handles the second time.


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
