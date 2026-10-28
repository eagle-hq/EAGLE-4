# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The draft candidates' all-gather (:mod:`eagle_harness.vllm.candidate_gather`): fallback and one-rank identity."""

from types import SimpleNamespace

import pytest
import torch


class _Group:
    world_size = 2

    def __init__(self):
        self.calls = []

    def all_gather(self, x, dim):
        self.calls.append((tuple(x.shape), dim))
        return torch.cat([x, x], dim)


def test_candidate_gather_falls_back_to_the_group_all_gather():
    from eagle_harness.vllm.candidate_gather import CandidateGather

    group = _Group()
    gather = CandidateGather(group, max_rows=4)
    gather._tried = True  # no symmetric memory here
    x = torch.arange(12, dtype=torch.int64).view(3, 4)
    out = gather(x)
    assert torch.equal(out, torch.cat([x, x])) and group.calls == [((3, 4), 0)]
    # a symmetric buffer that is too small or of another dtype is not used either
    gather.out = torch.zeros(2 * 4 * 4, dtype=torch.int64)
    assert torch.equal(gather(torch.zeros(5, 4, dtype=torch.int64)), torch.zeros(10, 4, dtype=torch.int64))
    assert torch.equal(gather(torch.zeros(2, 4, dtype=torch.int32)), torch.zeros(4, 4, dtype=torch.int32))
    assert len(group.calls) == 3


def test_candidate_gather_hands_the_op_a_rank_major_output_of_the_input_rank(monkeypatch):
    """The multicast op checks ``out.dim() == x.dim()`` and ``out.shape[0] == world * rows`` (a flat output failed at TP4)."""
    from eagle_harness.vllm import candidate_gather as module

    def fake_op(x, group_name, out):
        assert out.dim() == x.dim() and out.shape == (2 * x.shape[0], x.shape[1]) and out.is_contiguous()
        assert group_name == "tp"
        out.copy_(torch.cat([x, x + 100]))

    monkeypatch.setattr(module, "_multimem_all_gather_out", fake_op)
    group = _Group()
    group.device_group = SimpleNamespace(group_name="tp")
    gather = module.CandidateGather(group, max_rows=4)
    gather._tried = True
    gather.out = torch.zeros(2 * 4 * 3, dtype=torch.int64)
    x = torch.arange(6, dtype=torch.int64).view(2, 3)
    out = gather(x)
    assert torch.equal(out, torch.cat([x, x + 100])) and group.calls == []
    assert out.data_ptr() == gather.out.data_ptr()  # the result lives in the symmetric buffer


def _two_rank_gather(rank: int, port: int, results) -> None:
    import torch.distributed as dist

    from eagle_harness.vllm.candidate_gather import CandidateGather

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        group = SimpleNamespace(
            world_size=2,
            device_group=dist.group.WORLD,
            all_gather=lambda x, dim: _nccl_all_gather(x),
        )
        gather = CandidateGather(group, max_rows=64)
        for rows in (16, 4, 16):  # the buffer is sized once; smaller gathers use its prefix
            x = torch.arange(rows * 24, dtype=torch.int64, device="cuda").view(rows, 24) + 1000 * rank
            want = torch.cat([x - 1000 * rank, x - 1000 * rank + 1000])
            got = gather(x)
            assert got.shape == want.shape and torch.equal(got, want), rows
        results.put((rank, "multicast" if gather.out is not None else "fallback"))
    finally:
        dist.destroy_process_group()


def _nccl_all_gather(x: torch.Tensor) -> torch.Tensor:
    import torch.distributed as dist

    out = x.new_empty(2 * x.shape[0], *x.shape[1:])
    dist.all_gather_into_tensor(out, x)
    return out


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_candidate_gather_two_ranks_is_the_rank_major_concat():
    """Two ranks: the multicast path (or the NCCL fallback) returns both ranks' rows, rank-major, for any row count."""
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    mp.start_processes(_two_rank_gather, args=(29541, results), nprocs=2, start_method="spawn")
    print(sorted(results.get() for _ in range(2)))


def _two_stream_gather(rank: int, world: int, port: int, results) -> None:
    """Gathers issued from two caller streams in turn while one rank lags at a time (vLLM's startup: warm-up, then a
    new stream per graph capture). Each caller copies its result on its own stream before the next gather, as the
    draft does; every copy must hold that step's rows."""
    import time

    import torch.distributed as dist

    from eagle_harness.vllm.candidate_gather import CandidateGather

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        group = SimpleNamespace(world_size=world, device_group=dist.group.WORLD, all_gather=None)
        gather = CandidateGather(group, max_rows=64)
        streams = [torch.cuda.Stream() for _ in range(2)]
        copies, bad = [], 0
        for step in range(400):
            if rank == step % world and step % 7 == 0:
                time.sleep(0.02)
            with torch.cuda.stream(streams[step % 2]):
                x = torch.full((16, 24), 1000 * step + rank, dtype=torch.int64, device="cuda")
                copies.append((step, gather(x).clone()))
            if step % 50 == 49:
                torch.cuda.synchronize()
                for s, got in copies:
                    want = torch.arange(world, device="cuda").repeat_interleave(16)[:, None] + 1000 * s
                    bad += not torch.equal(got, want.expand(-1, 24))
                copies.clear()
        results.put((rank, "multicast" if gather.out is not None else "fallback", bad))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_candidate_gather_from_two_streams_keeps_every_result():
    """Two streams, lagging ranks: no gather returns or keeps another step's rows, and no rank hangs."""
    import torch.multiprocessing as mp

    world = min(torch.cuda.device_count(), 8)
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    mp.start_processes(_two_stream_gather, args=(world, 29547, results), nprocs=world, start_method="spawn")
    got = sorted(results.get() for _ in range(world))
    print(got)
    assert all(bad == 0 for _, _, bad in got), got


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="needs a GPU")
def test_candidate_gather_single_rank_is_the_input():
    """One rank: the multicast path (when the host has it) returns the input itself."""
    pytest.importorskip("vllm")
    import torch.distributed as dist

    from eagle_harness.vllm.candidate_gather import CandidateGather

    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29533", rank=0, world_size=1)
    group = SimpleNamespace(world_size=1, device_group=dist.group.WORLD, all_gather=lambda x, dim: x.clone())
    gather = CandidateGather(group, max_rows=64)
    x = torch.randint(0, 1 << 40, (16, 32), dtype=torch.int64, device="cuda")
    assert torch.equal(gather(x), x)
    print("multicast" if gather.out is not None else "fallback")
