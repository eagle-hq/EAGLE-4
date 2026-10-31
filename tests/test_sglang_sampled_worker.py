# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The T=1 sampled tree is wired only on engines whose verify takes ``sampling_step``.

The draft worker is built the way sglang builds it: sglang's own ``EagleDraftWorker.__init__`` and
``init_attention_backends`` run, with only the distributed contexts, the draft model build and the GPU-side
harness objects replaced, so a name the harness reads from the wrong scope fails here instead of in a GPU job.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang.srt.speculative.eagle_worker_v2")

from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.speculative import eagle_worker_v2  # noqa: E402

from eagle_harness.sglang import worker as worker_mod  # noqa: E402


def _verify_without_hook(batch, *, target_worker, req_to_token_pool, token_to_kv_pool_allocator, plan_stream,
                         plan_stream_ctx, topk, num_draft_tokens, device, metadata_ready_pre_pad, finalize_tree_path,
                         grammar_barrier, accept_step=None):
    return SimpleNamespace(called_with={"accept_step": accept_step})


def _verify_with_hook(batch, *, target_worker, req_to_token_pool, token_to_kv_pool_allocator, plan_stream,
                      plan_stream_ctx, topk, num_draft_tokens, device, metadata_ready_pre_pad, finalize_tree_path,
                      grammar_barrier, accept_step=None, sampling_step=None):
    return SimpleNamespace(called_with={"accept_step": accept_step, "sampling_step": sampling_step})


@pytest.mark.parametrize("engine_verify, seed", [(_verify_without_hook, None), (_verify_with_hook, 721)])
def test_draft_worker_built_like_sglang_seeds_the_sampled_tree_only_with_the_hook(monkeypatch, engine_verify, seed):
    for name in dir(eagle_worker_v2):
        if name.endswith(("_context", "_scope")) and callable(getattr(eagle_worker_v2, name)):
            monkeypatch.setattr(eagle_worker_v2, name, lambda *args, **kwargs: contextlib.nullcontext())
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=8),
        model=SimpleNamespace(fc=SimpleNamespace(in_features=16)),
        get_attention_sliding_window_size=lambda: None,
    )
    runner = SimpleNamespace(model=model, dtype=torch.float32, model_config=SimpleNamespace(hf_config=SimpleNamespace()))
    monkeypatch.setattr(
        eagle_worker_v2, "TpModelWorker", lambda **kwargs: SimpleNamespace(model_runner=runner, init_attention_backends=lambda: None)
    )
    monkeypatch.setattr(eagle_worker_v2, "get_plan_stream", lambda device: (None, None))
    monkeypatch.setattr(eagle_worker_v2, "get_pp_stage_load_group", lambda: None)
    built = {}
    monkeypatch.setattr(worker_mod, "run_eagle_verify", engine_verify)
    monkeypatch.setattr(worker_mod, "DraftBackendFactory", lambda *args: SimpleNamespace(create_draft_extend_backend=lambda: None))
    for name in ("DraftTree", "DraftRowAttention", "DraftExtend"):
        monkeypatch.setattr(worker_mod, name, lambda *args, **kwargs: SimpleNamespace())
    steps = object()
    monkeypatch.setattr(worker_mod, "DraftSteps", lambda *args, **kwargs: built.update(kwargs) or steps)
    target = SimpleNamespace(random_seed=721, model_runner=SimpleNamespace(model_config=SimpleNamespace(context_len=64)))
    policy = SimpleNamespace(intensity=None, max_history_verify_num=16, launch_depths=())
    with get_context().override_server_args(
        speculative_eagle_topk=4, speculative_num_steps=6, speculative_num_draft_tokens=16, device="cpu",
        max_running_requests=8,
    ):
        draft = worker_mod.HarnessDraftWorker(None, 0, 0, target, policy=policy)
        draft.init_attention_backends()
    assert built == {"sampled_seed": seed}
    assert draft.steps is steps  # every sglang-built draft worker has its DraftSteps before the first round


@pytest.mark.parametrize("vocab_size", [32, 128256, 248320])
def test_startup_softmax_uses_the_served_vocabulary(monkeypatch, vocab_size):
    """Shape-based backend dispatch must see the same vocabulary during warm-up and serving."""
    worker = SimpleNamespace(
        device="cuda", topk=1,
        draft_runner=SimpleNamespace(model=SimpleNamespace(logits_processor=SimpleNamespace(vocab_size=vocab_size))),
    )
    zeros = torch.zeros
    allocations, calls = [], []

    def allocate(*shape, dtype, device):
        allocations.append((shape, dtype, device))
        return zeros(*shape, dtype=dtype)

    def softmax(logits):
        calls.append((tuple(logits.shape), logits.dtype))
        return logits.softmax(dim=-1)

    monkeypatch.setattr(worker_mod.torch, "zeros", allocate)
    monkeypatch.setattr(worker_mod.spec_utils, "sampling_softmax", softmax)
    worker_mod.HarnessDraftWorker._warm_up_kernels(worker)
    assert allocations == [((1, vocab_size), torch.float32, "cuda")]
    assert calls == [((1, vocab_size), torch.float32)]


def test_tree_warm_up_moves_the_kv_of_the_target_pool(monkeypatch):
    """A tree's verify moves the accepted tokens' KV in the target's pool: its move kernels compile before serving."""
    pool = object()
    moved, accept_tail = [], []
    monkeypatch.setattr(worker_mod, "warm_up_accept_tail", lambda *args: accept_tail.append(args))
    monkeypatch.setattr(worker_mod, "warm_up_kv_moves", lambda kv_cache, device: moved.append((kv_cache, device)))
    monkeypatch.setattr(worker_mod.spec_utils, "sampling_softmax", None)
    allocator = SimpleNamespace(get_kvcache=lambda: pool)
    worker = SimpleNamespace(
        device="cuda", topk=4, extend=SimpleNamespace(hidden_rows=lambda rows: rows),
        target_worker=SimpleNamespace(model_runner=SimpleNamespace(dtype=torch.bfloat16,
                                                                   token_to_kv_pool_allocator=allocator)),
    )
    worker_mod.HarnessDraftWorker._warm_up_kernels(worker)
    assert moved == [(pool, "cuda")] and len(accept_tail) == 1
    worker.topk = 1  # a chain's accepted tokens are already in place: nothing moves
    moved.clear()
    worker_mod.HarnessDraftWorker._warm_up_kernels(worker)
    assert moved == []


def test_kv_move_warm_up_covers_every_program_a_round_can_compile():
    """sglang's tiled KV move compiles one program per next power of two of the moved slots (up to its chunk) and
    per slot count specialization (1, a multiple of 16, any other): the warm-up runs each, on the padding slot."""
    from eagle_harness.sglang.verify_step import KV_MOVE_MAX_SLOTS, warm_up_kv_moves

    moves = []
    warm_up_kv_moves(SimpleNamespace(move_kv_cache=lambda tgt, src: moves.append((tgt.clone(), src.clone()))), "cpu")
    assert all(tgt.dtype == torch.int64 and not tgt.any() and not src.any() for tgt, src in moves)

    def program(count):
        return 1 << (count - 1).bit_length(), "one" if count == 1 else "x16" if count % 16 == 0 else "other"

    warmed = {program(len(tgt)) for tgt, _ in moves}
    assert warmed == {program(count) for count in range(1, KV_MOVE_MAX_SLOTS + 1)}
    assert KV_MOVE_MAX_SLOTS >= 256  # sglang's largest chunk (smaller KV strides)


@pytest.mark.parametrize("engine_verify, active", [(_verify_without_hook, False), (_verify_with_hook, False), (_verify_with_hook, True)])
def test_verify_passes_the_tree_sampler_only_for_an_active_sampled_tree(monkeypatch, engine_verify, active):
    monkeypatch.setattr(worker_mod, "run_eagle_verify", engine_verify)
    monkeypatch.setattr(worker_mod, "fused_accept_supported", lambda *args: False)
    sampled = None if engine_verify is _verify_without_hook else SimpleNamespace(active=active, sample=lambda *args: args)
    worker = SimpleNamespace(
        _draft_worker=SimpleNamespace(steps=SimpleNamespace(sampled=sampled)),
        topk=4,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
        **dict.fromkeys(("target_worker", "req_to_token_pool", "plan_stream", "plan_stream_ctx", "device")),
    )
    batch = SimpleNamespace(spec_info="verify-input", enable_overlap=False)
    result, staged = worker_mod.EagleHarnessWorker._verify(worker, batch, 8, None)
    assert not staged
    step = result.called_with.get("sampling_step")
    if active:
        assert step("target", "coins", "final") == ("verify-input", "target", "coins", "final")
    else:
        assert step is None


def _sampled_tree(rows=6):
    from eagle_harness.sglang.sampled_tree import SampledTree

    shape = SimpleNamespace(max_depth=3, top_k=2, layer_start=lambda depth: 1 + (depth - 1) * 2)
    tree = SimpleNamespace(shape=shape, tokens=torch.zeros((rows, 9), dtype=torch.int64), max_bs=rows)
    return SampledTree(tree, 8, 721, rows=rows)


def test_prefill_records_each_request_sampling_settings_at_its_slot(monkeypatch):
    """A prefill stages its requests' sampler settings, so their later proposals follow the target's sampler."""
    sampled = _sampled_tree()
    calls = []
    parent = worker_mod.EagleHarnessWorker.__mro__[1]
    monkeypatch.setattr(parent, "forward_batch_generation", lambda self, *args, **kwargs: calls.append(args) or "prefilled")
    worker = worker_mod.EagleHarnessWorker.__new__(worker_mod.EagleHarnessWorker)
    worker._chain_rounds = SimpleNamespace(interrupt=lambda: None)
    worker._drafter = SimpleNamespace(cancel_prelaunch=lambda: None)
    worker._draft_worker = SimpleNamespace(steps=SimpleNamespace(sampled=sampled))
    sampled.root_valid.fill_(True)
    info = SimpleNamespace(temperatures=torch.tensor([[0.5], [1.0]]), top_ks=torch.tensor([20, 1 << 30], dtype=torch.int32),
                           top_ps=torch.tensor([0.95, 1.0]), min_ps=torch.tensor([0.0, 0.1]))
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True), is_extend_in_batch=True,
                            req_pool_indices=torch.tensor([3, 0]), sampling_info=info)
    assert worker.forward_batch_generation(batch) == "prefilled" and calls
    identity = torch.tensor([1.0, float(1 << 30), 1.0, 0.0])
    torch.testing.assert_close(sampled.filters[3], torch.tensor([2.0, 20.0, 0.95, 0.0]))
    torch.testing.assert_close(sampled.filters[0], torch.tensor([1.0, float(1 << 30), 1.0, 0.1]))
    for slot in (1, 2, 4, 5):
        torch.testing.assert_close(sampled.filters[slot], identity)
    assert sampled.root_valid.tolist() == [False, True, True, False, True, True]
