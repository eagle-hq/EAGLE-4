# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Graph-handle ownership on CPU, with graph uploads replaced by a test driver."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from eagle_harness.tokenspeed.tree_round import chain as module


def chain_driver(monkeypatch):
    chain = object.__new__(module.DeviceChain)
    chain.device = torch.device("cpu")
    chain.last_depth = 3
    chain._execs = torch.zeros((5, 6), dtype=torch.int64)
    chain._handles = {}
    destroyed = []
    monkeypatch.setattr(torch.cuda, "device", lambda *_: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda *_: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(module, "eagle_harness_chain_destroy", destroyed.append)
    stream = SimpleNamespace(
        device=torch.device("cpu"), cuda_stream=123, synchronize=lambda: None
    )
    depths = [
        SimpleNamespace(raw_cuda_graph=lambda: 10),
        SimpleNamespace(raw_cuda_graph=lambda: 20),
    ]
    finalize = [SimpleNamespace(raw_cuda_graph=lambda: 30)]
    return chain, stream, depths, finalize, destroyed


def test_chain_publishes_and_releases_owned_handles(monkeypatch):
    chain, stream, depths, finalize, destroyed = chain_driver(monkeypatch)
    monkeypatch.setattr(
        module, "eagle_harness_chain_instantiate", lambda graph, stream: graph + 1
    )
    chain.instantiate(4, depths, finalize, stream)
    assert chain._execs[4].tolist() == [0, 0, 11, 21, 31, 0]
    with pytest.raises(ValueError, match="already instantiated"):
        chain.instantiate(4, depths, finalize, stream)
    chain.release()
    chain.release()
    assert destroyed == [11, 21, 31]
    assert not chain._execs.any()


def test_partial_upload_failure_destroys_handles_without_publishing(monkeypatch):
    chain, stream, depths, finalize, destroyed = chain_driver(monkeypatch)

    def upload(graph, stream):
        if graph == 20:
            raise RuntimeError("upload failed")
        return graph + 1

    monkeypatch.setattr(module, "eagle_harness_chain_instantiate", upload)
    with pytest.raises(RuntimeError, match="upload failed"):
        chain.instantiate(4, depths, finalize, stream)
    assert destroyed == [11]
    assert not chain._handles
    assert not chain._execs.any()


def test_chain_refuses_invalid_capture_counts(monkeypatch):
    chain, stream, depths, finalize, destroyed = chain_driver(monkeypatch)
    for ds, fs in ((depths * 2, finalize), (depths, finalize * 3), (depths, [])):
        with pytest.raises(ValueError, match="depth prefix"):
            chain.instantiate(4, ds, fs, stream)
    assert not destroyed
