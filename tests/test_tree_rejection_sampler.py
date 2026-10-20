# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The tree rejection sampler's output contract: vLLM reads the logprobs and the sampling mask of sampled token k of
a request from logits row k of that request, so those rows must be the accepted path's rows, not the first rows of
the tree (CPU through the Triton interpreter, or a GPU)."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

pytest.importorskip("vllm")

from eagle_harness.vllm.tree_accept import TreeRejectionSampler  # noqa: E402
from eagle_harness.vllm.verify_layout import TreeStore, VerifyStage  # noqa: E402
from tests.triton_probe import needs_triton  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, VOCAB, SLOTS, TOP_K, NUM_LOGPROBS = 8, 64, 4, 3, 3
# Two requests (state slots 3 and 1). The first verifies 8 rows and accepts rows 0, 2, 5, 7; the second verifies
# 5 rows and accepts rows 0, 1, 4. Neither path is a prefix of the rows.
TREES = [([-1, 0, 0, 1, 1, 2, 3, 5], [0, 2, 5, 7], 3), ([-1, 0, 0, 1, 1], [0, 1, 4], 1)]


class _Sampler:
    """The parts of vLLM's ``Sampler`` the rejection sampler reads; ``top_k`` > 0 keeps only the top-k logits."""

    def __init__(self, logprobs_mode: str, top_k: int):
        self.compute_nans = False
        self.use_fp64_gumbel = False
        self.logprobs_mode = logprobs_mode
        self.return_sampling_mask = top_k > 0
        self.top_k = top_k
        prefill_len = torch.zeros(SLOTS, dtype=torch.int32, device=DEV)
        self.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=prefill_len))
        self.sampling_states = SimpleNamespace(
            max_num_logprobs=lambda idx_mapping_np: NUM_LOGPROBS,
            top_k=SimpleNamespace(np=np.full(SLOTS, top_k)),
            temperature=SimpleNamespace(gpu=torch.zeros(SLOTS, device=DEV)),  # greedy
            seeds=SimpleNamespace(gpu=torch.zeros(SLOTS, dtype=torch.int64, device=DEV)),
        )

    def apply_sampling_params(self, logits, *args):
        if not self.top_k:
            return logits  # vLLM returns the raw logits themselves when no request needs processing
        kth = logits.topk(self.top_k, dim=-1).values[:, -1:]
        return logits.float().masked_fill(logits < kth, float("-inf"))


def _batch():
    """Seeded target logits, and draft tokens chosen so that the greedy walk accepts each request's path."""
    gen = torch.Generator().manual_seed(7)
    sizes = [len(parents) for parents, _, _ in TREES]
    logits = torch.randn(sum(sizes), VOCAB, generator=gen)
    drafted = torch.randint(0, VOCAB, (sum(sizes),), generator=gen)
    start = 0
    for (parents, path, _), size in zip(TREES, sizes):
        argmax = logits[start : start + size].argmax(-1)
        for row in range(1, size):  # off-path rows never match their parent's argmax
            if drafted[start + row] == argmax[parents[row]]:
                drafted[start + row] = (argmax[parents[row]] + 1) % VOCAB
        for parent, child in zip(path, path[1:]):
            drafted[start + child] = argmax[parent]
        start += size
    return logits, drafted, sizes


def _run(logprobs_mode: str, top_k: int, *, structured: bool = False):
    logits, drafted, sizes = _batch()
    num_reqs, num_logits = len(TREES), sum(sizes)
    store, stage = TreeStore(SLOTS, W, DEV), VerifyStage(SLOTS, num_logits, W, DEV)
    stage.active = False  # positions are already chain positions
    for parents, _, slot in TREES:
        stage.slot_parents[slot, : len(parents)] = torch.tensor(parents, dtype=torch.int32)
    slots = [slot for _, _, slot in TREES]
    cu = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    batch = SimpleNamespace(
        input_ids=drafted.to(DEV),
        positions=torch.cat([torch.arange(size) + 40 for size in sizes]).to(DEV),
        num_tokens=num_logits,
        logits_indices=torch.arange(num_logits, device=DEV),
        expanded_idx_mapping=torch.tensor(np.repeat(slots, sizes), dtype=torch.int32, device=DEV),
        expanded_local_pos=torch.cat([torch.arange(size, dtype=torch.int32) for size in sizes]).to(DEV),
        idx_mapping=torch.tensor(slots, dtype=torch.int32, device=DEV),
        idx_mapping_np=np.array(slots),
        cu_num_logits=torch.from_numpy(cu).to(DEV),
        cu_num_logits_np=cu,
        num_reqs=num_reqs,
        has_structured_output_reqs=structured,
        seq_lens=torch.tensor([40 + size for size in sizes], dtype=torch.int32, device=DEV),
        seq_lens_cpu_upper_bound=torch.tensor([40 + size for size in sizes], dtype=torch.int32),
    )
    spec = SimpleNamespace(num_speculative_tokens=W - 1, enable_adaptive_verification=False, rejection_sample_method="")
    sampler = _Sampler(logprobs_mode, top_k)
    out = TreeRejectionSampler(sampler, spec, DEV, store, stage)(logits.clone().to(DEV), batch)
    reference = sampler.apply_sampling_params(logits) if logprobs_mode.startswith("processed") else logits
    return out, reference, cu


@needs_triton
@pytest.mark.parametrize(("logprobs_mode", "top_k"), [("raw_logprobs", 0), ("processed_logprobs", TOP_K)])
def test_logprobs_and_sampling_mask_come_from_the_accepted_path_rows(logprobs_mode, top_k):
    out, reference, cu = _run(logprobs_mode, top_k)
    lp = out.logprobs_tensors
    for b, (_, path, _) in enumerate(TREES):
        assert int(out.num_sampled[b]) == len(path)
        tokens = out.sampled_token_ids[b, : len(path)].cpu()
        assert tokens.tolist() == [int(reference[cu[b] + row].argmax()) for row in path]
        rows = slice(cu[b], cu[b] + len(path))
        top = reference[cu[b] + torch.tensor(path)].topk(NUM_LOGPROBS, dim=-1)
        log_probs = reference[cu[b] + torch.tensor(path)].log_softmax(-1)
        assert lp.logprob_token_ids[rows, 0].cpu().tolist() == tokens.tolist()
        assert lp.logprob_token_ids[rows, 1:].cpu().tolist() == top.indices.tolist()  # top-1 is the emitted token
        torch.testing.assert_close(lp.logprobs[rows, 0].cpu(), log_probs.gather(-1, tokens[:, None])[:, 0])
        torch.testing.assert_close(lp.logprobs[rows, 1:].cpu(), log_probs.gather(-1, top.indices))
        if top_k:
            mask = out.sampling_mask_tensors
            for k, row in enumerate(path):
                kept = torch.isfinite(reference[cu[b] + row]).nonzero()[:, 0]
                assert mask.token_ids[b * W + k, :top_k].cpu().tolist() == kept.tolist()


@needs_triton
@pytest.mark.parametrize(("logprobs_mode", "top_k"), [("raw_logprobs", 0), ("processed_logprobs", TOP_K)])
def test_structured_output_never_accepts_a_row_with_a_chain_grammar_mask(logprobs_mode, top_k):
    # vLLM supplies masks for a chain, not each tree node's ancestry. Only the root mask is valid;
    # all other logits stand for arbitrary, potentially permissive masks after an invalid draft.
    out, reference, cu = _run(logprobs_mode, top_k, structured=True)
    assert out.num_sampled.cpu().tolist() == [1, 1]
    for b in range(len(TREES)):
        token = int(reference[cu[b]].argmax())
        assert int(out.sampled_token_ids[b, 0]) == token
        assert int(out.logprobs_tensors.logprob_token_ids[cu[b], 0]) == token
        torch.testing.assert_close(
            out.logprobs_tensors.logprobs[cu[b], 0].cpu(), reference[cu[b]].log_softmax(-1)[token]
        )
