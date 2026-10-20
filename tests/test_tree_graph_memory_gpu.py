# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Serial tree-growth shapes must not retain one full workspace per graph."""

import contextlib
from types import MethodType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph memory needs a GPU")


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("counts", [(7,), (3, 7, 15)])
def test_growth_graphs_share_workspace_and_replay_in_any_shape_order(monkeypatch, conditional, counts):
    pytest.importorskip("vllm")
    import vllm.distributed.parallel_state as ps

    from eagle_harness.vllm.tree_drafter import TreeDrafter
    from tests.test_depth_stop import _Model

    @contextlib.contextmanager
    def graph_capture(device):
        yield SimpleNamespace(stream=torch.cuda.Stream(device=device))

    monkeypatch.setattr(ps, "graph_capture", graph_capture)
    device = torch.device("cuda")
    attn = SimpleNamespace(
        impl=SimpleNamespace(scale=64**-0.5), sliding_window=None, num_heads=2, num_kv_heads=2, head_size=64
    )
    drafter = TreeDrafter(
        _Model(2, 64, 300, "cuda"),
        attn,
        top_k=4,
        depth=6,
        max_reqs=512,
        device=device,
        min_gain=0.0 if conditional else None,
    )
    drafter.graphs = True
    elements = 16 * 1024 * 1024  # a 64 MiB transient logits-like workspace

    def growth(self, num_reqs, count, committed, block_table, stop=None):
        self._go.fill_(True)
        with self._capture_if(stop):
            workspace = committed[0].float().expand(elements).clone()
            result = workspace[:: elements // 16].clone(), torch.full((num_reqs, count), num_reqs, device=device)
        return result

    drafter._grow_and_select = MethodType(growth, drafter)
    committed = torch.zeros(512, dtype=torch.int64, device=device)
    table = torch.zeros((512, 1), dtype=torch.int32, device=device)
    sizes = [1, 32, 33, 64, 128, 256, 512]
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved()
    for count in counts:
        for bs in sizes:
            for value in (1, 2):  # eager compile, then capture/replay
                committed.fill_(value)
                out, parents = drafter.grow_and_select(bs, count, committed[:bs], table[:bs])
                assert out.tolist() == [float(value)] * 16
                assert torch.all(parents == bs)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    retained = torch.cuda.memory_reserved() - before
    assert retained < 2 * elements * 4, f"graphs retained {retained} bytes of workspace"
    assert len(drafter._graphs) == len(sizes) * len(counts)
    replay_shapes = [(bs, count) for count in reversed(counts) for bs in (512, 1, 128, 33, 64, 32, 256, 1)]
    for value, (bs, count) in enumerate(replay_shapes, 3):
        committed.fill_(value)
        out, parents = drafter.grow_and_select(bs, count, committed[:bs], table[:bs])
        assert out.tolist() == [float(value)] * 16
        assert torch.all(parents == bs)
