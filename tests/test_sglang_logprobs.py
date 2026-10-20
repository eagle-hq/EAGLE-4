# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Logprobs of an EAGLE_HARNESS tree round on sglang.

The worker's verify (``EagleHarnessWorker._verify``) runs sglang's ``run_eagle_verify`` on the harness's tree, and
the scheduler turns its result into each request's logprobs. Accepted token k of a request must carry the logprobs
of the verify row at depth k of its ACCEPTED PATH. A tree's accepted path is not a prefix of its rows: here request 0
accepts rows 0, 2, 5, 7 and request 1 rows 0, 1, 4, so row k would give request 0 the logprobs of nodes 1, 2, 3.
The target forward, the KV moves and the state commit are stubbed; the tree walk is a stand-in for sglang's CUDA
kernel (``verify_tree_greedy``) with the same contract.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("sglang.srt.speculative.eagle_worker_common")

from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # noqa: E402
from sglang.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.runtime_context import get_context, get_parallel  # noqa: E402
from sglang.srt.speculative import eagle_utils, eagle_worker_common  # noqa: E402

from eagle_harness.sglang.worker import EagleHarnessWorker  # noqa: E402

W, VOCAB, TOP = 8, 64, 5
PARENTS = [-1, 0, 0, 1, 1, 2, 3, 5]  # one tree shape for both requests
PATHS = [[0, 2, 5, 7], [0, 1, 4]]  # the second child at depth 1: neither path is a prefix of the rows
TOKEN_IDS = [None, [3, 40]]  # request 1 also asks for the logprobs of two given tokens


def _tables(bs):
    """sglang's retrieve tables of ``bs`` trees (first child, next sibling; global row ids)."""
    index = torch.arange(bs * W).view(bs, W)
    first = torch.full((bs, W), -1, dtype=torch.int64)
    sibling = torch.full((bs, W), -1, dtype=torch.int64)
    for node in range(W - 1, 0, -1):
        parent = PARENTS[node]
        sibling[:, node] = first[:, parent]
        first[:, parent] = node
    return index, first, sibling


def _round(paths):
    """Draft tokens and target logits whose greedy walk accepts ``paths``."""
    generator = torch.Generator().manual_seed(0)
    bs = len(paths)
    draft = torch.stack([10 + b * W + torch.arange(W) for b in range(bs)])  # unique per request, never a bonus
    logits = torch.randn(bs * W, VOCAB, generator=generator)
    for b, path in enumerate(paths):
        for here, child in zip(path, path[1:]):
            logits[b * W + here, draft[b, child]] += 8.0
        logits[b * W + path[-1], 1 + b] += 8.0  # the bonus token: no child drafted it
    return draft, logits


def _walk(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
    topk=-1,
):
    """``verify_tree_greedy``: follow the child that drafted the target's token, down to the bonus token."""
    for b in range(candidates.shape[0]):
        node, depth = 0, 0
        accept_index[b, 0] = retrieve_index[b, 0]
        while True:
            child = int(retrieve_next_token[b, node])
            while child != -1 and int(candidates[b, child]) != int(target_predict[b, node]):
                child = int(retrieve_next_sibling[b, child])
            predicts[retrieve_index[b, node]] = target_predict[b, node]
            if child == -1:
                break
            depth += 1
            accept_index[b, depth] = retrieve_index[b, child]
            node = child
        accept_token_num[b] = depth
    return predicts, accept_index, accept_token_num


def _fill_bonus(accept_tokens, accept_lens, bonus_tokens, accept_stride, batch_size):
    for b in range(batch_size):
        bonus_tokens[b] = accept_tokens.reshape(-1)[b * accept_stride + int(accept_lens[b]) - 1]


def _verify(paths, token_ids, logits, draft):
    """``EagleHarnessWorker._verify`` of a logprob batch on CPU; returns the batch and the result."""
    bs = len(paths)
    index, first, sibling = _tables(bs)
    verify_input = SimpleNamespace(
        draft_token=draft.reshape(-1),
        draft_token_num=W,
        max_tree_depth=max(map(len, paths)),  # spec_steps + 1
        tree_topk=4,
        retrieve_index=index,
        retrieve_next_token=first,
        retrieve_next_sibling=sibling,
    )
    sampling = SimpleNamespace(
        is_all_greedy=True,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        has_watermark_candidates=False,
    )
    batch = SimpleNamespace(
        spec_info=verify_input,
        seq_lens=torch.tensor([10, 20][:bs]),
        req_pool_indices=torch.arange(bs),
        input_ids=None,
        out_cache_loc=torch.arange(bs * W),
        forward_mode=ForwardMode.DECODE,
        sampling_info=sampling,
        has_grammar=False,
        return_logprob=True,
        top_logprobs_nums=[TOP] * bs,
        token_ids_logprobs=token_ids,
        hisparse_coordinator=None,
        mamba_track_indices=None,
        device="cpu",
    )
    forward = SimpleNamespace(
        logits_output=LogitsProcessorOutput(next_token_logits=logits.clone(), hidden_states=None),
        routed_experts_output=None,
        indexer_topk_output=None,
    )
    runner = SimpleNamespace(watermark_state=None, attn_backend=SimpleNamespace(verify_mask=None))
    worker = SimpleNamespace(
        topk=4,
        device="cpu",
        plan_stream=None,
        plan_stream_ctx=contextlib.nullcontext(),
        req_to_token_pool=SimpleNamespace(req_to_token=None),
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: SimpleNamespace()),
        target_worker=SimpleNamespace(model_runner=runner, forward_batch_generation=lambda **kw: forward),
        _draft_worker=SimpleNamespace(
            steps=SimpleNamespace(seq_lens=batch.seq_lens, req_pool_indices=batch.req_pool_indices), extend=None
        ),
        _tail_graphs=None,
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(get_context().override_server_args(speculative_num_draft_tokens=W))
        # ROCm's greedy verifier checks the TP group even for this single-rank CPU round.
        stack.enter_context(get_parallel().override(tp_group=SimpleNamespace(world_size=1)))
        for module, name, value in (
            (eagle_worker_common, "eagle_prepare_for_verify", lambda *a, **k: (SimpleNamespace(), False)),
            (eagle_worker_common, "commit_mamba_states_after_verify", lambda *a, **k: None),
            (eagle_worker_common, "move_accept_tokens_to_target_kvcache", lambda *a, **k: None),
            (eagle_worker_common, "fill_bonus_tokens_func", _fill_bonus),
            (eagle_utils, "verify_tree_greedy_func", _walk),
        ):
            stack.enter_context(mock.patch.object(module, name, value))
        result, staged = EagleHarnessWorker._verify(worker, batch, W, None)
    assert not staged  # a logprob batch runs sglang's accept code, not the fused accept step
    return batch, result


def _request_logprobs(token_ids, batch, result):
    """The scheduler's per-request logprobs of the round (its decode-output path)."""
    stride = result.speculative_num_draft_tokens
    ids, lens = result.next_token_ids.tolist(), result.accept_lens.tolist()
    tokens = [ids[i * stride : i * stride + n] for i, n in enumerate(lens)]
    scheduler = SimpleNamespace(_resolve_spec_v2_tokens=lambda result, batch: tokens)
    batch.spec_algorithm = SimpleNamespace(is_none=lambda: False)
    output = result.logits_output
    tokens, token_logprobs = SchedulerBatchResultProcessor._normalize_decode_outputs(
        scheduler, batch=batch, result=result, logits_output=output, next_token_ids=result.next_token_ids
    )
    reqs = []
    for i, ask in enumerate(token_ids):
        lists = ("token_logprobs", "top_logprobs", "token_ids_logprobs")
        state = {f"output_{name}_{part}": [] for name in lists for part in ("val", "idx")}
        req = SimpleNamespace(logprob=SimpleNamespace(top_logprobs_num=TOP, token_ids_logprob=ask, **state))
        SchedulerBatchResultProcessor._apply_decode_logprobs(
            scheduler,
            req=req,
            i=i,
            batch=batch,
            next_token_id=tokens[i],
            next_token_logprobs=token_logprobs,
            logits_output=output,
        )
        reqs.append(req.logprob)
    return reqs


@pytest.mark.parametrize("hip", [False, True], ids=["non-hip", "hip"])
@pytest.mark.parametrize("requests", [[0, 1], [0], [1]], ids=["bs2", "bs1-deep", "bs1-sibling"])
def test_logprobs_come_from_the_accepted_path_rows_in_path_order(requests, hip, monkeypatch):
    """Batch size 1 also checks the round leaves the fused accept step, which has no logprobs."""
    monkeypatch.setattr(eagle_utils, "_is_hip", hip)
    paths, token_ids = [PATHS[r] for r in requests], [TOKEN_IDS[r] for r in requests]
    draft, logits = _round(paths)
    batch, result = _verify(paths, token_ids, logits, draft)
    reqs = _request_logprobs(token_ids, batch, result)
    log_probs = torch.log_softmax(logits, dim=-1)
    for b, (path, ask, lp) in enumerate(zip(paths, token_ids, reqs)):
        rows = [b * W + node for node in path]
        tokens = log_probs[rows].argmax(-1)
        assert lp.output_token_logprobs_idx == tokens.tolist()
        torch.testing.assert_close(torch.tensor(lp.output_token_logprobs_val), log_probs[rows, tokens])
        top = log_probs[rows].topk(TOP, dim=-1)
        assert [list(x) for x in lp.output_top_logprobs_idx] == top.indices.tolist()
        assert [row[0] for row in lp.output_top_logprobs_idx] == tokens.tolist()  # top-1 is the emitted token
        torch.testing.assert_close(torch.tensor(lp.output_top_logprobs_val), top.values)
        if ask is None:
            assert lp.output_token_ids_logprobs_val == []
        else:
            torch.testing.assert_close(torch.tensor(lp.output_token_ids_logprobs_val), log_probs[rows][:, ask])
