# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The tree rejection sampler is lossless: decoded through draft trees over several rounds, seeded requests emit
token sequences whose distribution is the target's (a chi-square goodness of fit against the exact sequence
probabilities), and a request's seed fixes its text (CPU through the Triton interpreter, or a GPU).

The target is a toy model: the logits after a context are a seeded function of the context. Each round, every
request drafts a fixed-shape tree of its context's draft top tokens and is verified in one batch with the others.
"""

import os
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

pytest.importorskip("vllm")

from vllm.v1.worker.gpu.sample.bad_words import BadWordsState  # noqa: E402
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample  # noqa: E402
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState  # noqa: E402
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState  # noqa: E402
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS  # noqa: E402

from eagle_harness.vllm.tree_accept import TreeRejectionSampler  # noqa: E402
from eagle_harness.vllm.verify_layout import TreeStore, VerifyStage  # noqa: E402
from tests.triton_probe import needs_triton  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB, TOP_K, TEMPERATURE = 4, 3, 0.8  # top-k 3 of 4 tokens: one token is never sampled
PARENTS = [-1, 0, 0, 1, 3]  # rows 1 and 2 are siblings, 1 -> 3 -> 4 a chain below them
CHILD_RANK = [None, 0, 1, 0, 0]  # each row's token: the draft's rank-th token after its parent's context
PROMPT_LEN, LENGTH, ROUNDS = 40, 3, 3  # every round emits at least one token
REQUESTS = 4096 if torch.cuda.is_available() else 512


def _logits(context: tuple, salt: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(zlib.crc32(repr((salt, context)).encode()))
    return 1.5 * torch.randn(VOCAB, generator=gen)


def target_logits(context: tuple) -> torch.Tensor:
    return _logits(context, 0)


def processed(logits: torch.Tensor) -> torch.Tensor:
    """Temperature, then top-k: what vLLM's ``apply_sampling_params`` hands the sampler."""
    logits = logits.float() / TEMPERATURE
    kth = logits.topk(TOP_K, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def draft_tree(context: tuple) -> list:
    """The tree's tokens, row 0 (the context's last token) first; the draft ranks tokens by its own toy logits."""
    contexts, tokens = [context], [context[-1]]
    for parent, rank in zip(PARENTS[1:], CHILD_RANK[1:]):
        draft = target_logits(contexts[parent]) + _logits(contexts[parent], 1)  # a draft that is often right
        token = int(draft.argsort(descending=True)[rank])
        contexts.append(contexts[parent] + (token,))
        tokens.append(token)
    return tokens


class _Sampler:
    """The parts of vLLM's ``Sampler`` the rejection sampler reads: temperature and top-k, one seed per request."""

    def __init__(self, seeds: torch.Tensor, temperature: float):
        slots = seeds.shape[0]
        self.compute_nans = False
        self.use_fp64_gumbel = False
        self.logprobs_mode = "raw_logprobs"
        self.return_sampling_mask = False
        prefill_len = torch.zeros(slots, dtype=torch.int32, device=DEV)
        self.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=prefill_len))
        self.sampling_states = SimpleNamespace(
            max_num_logprobs=lambda idx_mapping_np: NO_LOGPROBS,
            temperature=SimpleNamespace(gpu=torch.full((slots,), temperature, device=DEV)),
            seeds=SimpleNamespace(gpu=seeds.to(DEV)),
        )

    def apply_sampling_params(self, logits, *args):
        return processed(logits)


def decode(seeds: list, temperature: float = TEMPERATURE, prompts: list | None = None) -> list:
    """``ROUNDS`` tree rounds of every request (one per seed, prompt ``(seed % 2,)`` unless ``prompts`` gives each
    request's prompt token); the emitted tokens of each."""
    n = len(seeds)
    store, stage = TreeStore(n, len(PARENTS), DEV), VerifyStage(n, n * len(PARENTS), len(PARENTS), DEV)
    store.parents[:] = torch.tensor(PARENTS, dtype=torch.int32)
    sampler = _Sampler(torch.tensor(seeds, dtype=torch.int64), temperature)
    spec = SimpleNamespace(
        num_speculative_tokens=len(PARENTS) - 1, enable_adaptive_verification=False, rejection_sample_method=""
    )
    rejection = TreeRejectionSampler(sampler, spec, DEV, store, stage)
    width, slots = len(PARENTS), np.arange(n)
    # the prompt's last token starts each context
    emitted = [[seed % 2] for seed in seeds] if prompts is None else [[prompt] for prompt in prompts]
    for _ in range(ROUNDS):
        trees = [draft_tree(tuple(tokens)) for tokens in emitted]
        contexts = []
        for tokens, tree in zip(emitted, trees):
            rows = [tuple(tokens)]
            for parent, token in zip(PARENTS[1:], tree[1:]):
                rows.append(rows[parent] + (token,))
            contexts += rows
        logits = torch.stack([target_logits(context) for context in contexts]).to(DEV)
        lengths = torch.tensor([PROMPT_LEN + len(tokens) - 1 for tokens in emitted])  # the root's position
        positions = (lengths[:, None] + torch.arange(width)).flatten().to(DEV)
        qsl = torch.arange(0, (n + 1) * width, width, dtype=torch.int32, device=DEV)
        idx_mapping = torch.from_numpy(slots).to(torch.int32).to(DEV)
        stage.stage(store, idx_mapping, qsl, qsl, positions, n, n * width, tree=True)
        batch = SimpleNamespace(
            input_ids=torch.tensor(trees, dtype=torch.int64).flatten().to(DEV),
            positions=positions,
            num_tokens=n * width,
            logits_indices=torch.arange(n * width, device=DEV),
            expanded_idx_mapping=idx_mapping.repeat_interleave(width),
            expanded_local_pos=torch.arange(width, dtype=torch.int32, device=DEV).repeat(n),
            idx_mapping=idx_mapping,
            idx_mapping_np=slots,
            cu_num_logits=qsl,
            cu_num_logits_np=qsl.cpu().numpy(),
            num_reqs=n,
            seq_lens=(lengths + width).to(torch.int32).to(DEV),
            seq_lens_cpu_upper_bound=(lengths + width).to(torch.int32),
        )
        out = rejection(logits, batch)
        sampled, count = out.sampled_token_ids.cpu().tolist(), out.num_sampled.cpu().tolist()
        for tokens, row, k in zip(emitted, sampled, count):
            tokens += row[:k]
    return [tokens[1:] for tokens in emitted]


def decode_without_speculation(seeds: list, steps: int) -> list:
    """What vLLM's sampler emits one token at a time: Gumbel-max of the processed logits, keyed by seed and position."""
    n = len(seeds)
    slots = torch.arange(n, dtype=torch.int32, device=DEV)
    temperature = torch.full((n,), TEMPERATURE, device=DEV)
    emitted = [[seed % 2] for seed in seeds]
    for step in range(steps):
        logits = torch.stack([processed(target_logits(tuple(tokens))) for tokens in emitted]).to(DEV)
        pos = torch.full((n,), PROMPT_LEN + step, dtype=torch.int64, device=DEV)  # the last token's position
        seed = torch.tensor(seeds, dtype=torch.int64, device=DEV)
        tokens = gumbel_sample(logits, slots, temperature, seed, pos, apply_temperature=False, is_drafting=False)
        for text, token in zip(emitted, tokens.tolist()):
            text.append(token)
    return [tokens[1:] for tokens in emitted]


def exact_probability(prompt: int, sequence: tuple) -> float:
    context, prob = (prompt,), 1.0
    for token in sequence:
        prob *= float(torch.softmax(processed(target_logits(context)), -1)[token])
        context += (token,)
    return prob


def chi_square_p_value(observed: dict, expected: dict, count: int) -> float:
    """Goodness of fit; cells expected fewer than 5 times are pooled into one, and an impossible cell fails outright."""
    if any(expected.get(cell, 0.0) == 0.0 for cell in observed):
        return 0.0
    big = [cell for cell, p in expected.items() if count * p >= 5]
    pairs = [(observed.get(cell, 0), count * expected[cell]) for cell in big]
    rest = count - sum(count * expected[cell] for cell in big)
    if rest > 1e-9:
        pairs.append((count - sum(o for o, _ in pairs), rest))
    stat = sum((o - e) ** 2 / e for o, e in pairs)
    dof = torch.tensor((len(pairs) - 1) / 2.0, dtype=torch.float64)
    return float(torch.special.gammaincc(dof, torch.tensor(stat / 2.0, dtype=torch.float64)))


@needs_triton
def test_emitted_sequences_follow_the_target_distribution():
    seeds = list(range(1000, 1000 + REQUESTS))
    texts = decode(seeds)
    assert all(len(text) >= LENGTH for text in texts)
    cells = [tuple(map(int, np.unravel_index(i, (VOCAB,) * LENGTH))) for i in range(VOCAB**LENGTH)]
    for prompt in (0, 1):
        prefixes = [tuple(text[:LENGTH]) for seed, text in zip(seeds, texts) if seed % 2 == prompt]
        observed: dict = {}
        for prefix in prefixes:
            observed[prefix] = observed.get(prefix, 0) + 1
        expected = {cell: exact_probability(prompt, cell) for cell in cells}
        assert abs(sum(expected.values()) - 1.0) < 1e-5
        assert chi_square_p_value(observed, expected, len(prefixes)) > 1e-4, (prompt, observed)
    accepted = sum(len(text) for text in texts) / (REQUESTS * ROUNDS)
    assert accepted > 1.2, accepted  # the drafts are accepted, not only the bonus token


@needs_triton
def test_a_seed_fixes_the_text_and_greedy_is_the_argmax_path():
    seeds = [7, 8, 7, 9, 8, 10, 11, 12]
    texts = decode(seeds)
    assert texts[0] == texts[2] and texts[1] == texts[4]
    # Each emitted token takes the noise of its own position, as without speculation: the same text for a seed.
    reference = decode_without_speculation(seeds, max(len(text) for text in texts))
    assert [ref[: len(text)] for text, ref in zip(texts, reference)] == texts
    greedy = decode([3, 4], temperature=0.0)
    for prompt, text in zip((1, 0), greedy):
        context = (prompt,)
        for token in text:
            assert token == int(processed(target_logits(context)).argmax())
            context += (token,)


@needs_triton
def test_different_seeds_sample_different_text():
    """The QA matrix's seed probe (resume/qa-matrix.md item 1): at temperature 1 the tree verify returned the same text
    for seeds 1 and 2 on every prompt, because it walked the argmax. Each of 4 prompts with seeds 1 and 2 must give
    two different texts but at most once by chance."""
    prompts = [0, 1, 2, 3] * 2
    texts = decode([1] * 4 + [2] * 4, temperature=1.0, prompts=prompts)
    same = sum(texts[i] == texts[i + 4] for i in range(4))
    assert same <= 1, texts


def _processing_sampler(slots: int, custom: bool = False) -> SimpleNamespace:
    """A sampler whose per-request processing flags a test sets: vLLM's three native logits processors (built
    without their tensors), a thinking budget, and optionally a custom processor."""
    penalties, bad_words, bias = (object.__new__(cls) for cls in (PenaltiesState, BadWordsState, LogitBiasState))
    penalties.use_penalty = np.zeros(slots, dtype=bool)
    bad_words.num_bad_words = SimpleNamespace(np=np.zeros(slots, dtype=np.int32))
    processors = [bias, penalties, bad_words] + ([SimpleNamespace()] if custom else [])
    return SimpleNamespace(
        needs_logits_processing=np.zeros(slots, dtype=bool),
        logits_processors=processors,
        penalties=penalties,
        bad_words=bad_words,
        thinking_budget_state=SimpleNamespace(enabled=False),  # no reasoning config: no per-request flags
    )


def test_prefix_reading_processing_is_verified_as_chains():
    """Penalties, bad words, a thinking budget and custom processors read each row's prefix, which in a tree is not
    the row's ancestors: such a batch must be verified as chains. Temperature, top-k/top-p and logit bias must not
    cost the tree."""
    rejection = object.__new__(TreeRejectionSampler)
    batch, other = np.array([0, 1]), np.array([2, 3])
    rejection.sampler = sampler = _processing_sampler(4)
    assert not rejection.reads_draft_prefix(batch)
    sampler.needs_logits_processing[:] = True  # temperature, top-k/top-p, min-p or logit bias
    assert not rejection.reads_draft_prefix(batch)
    for flag in (
        lambda s: s.penalties.use_penalty.__setitem__(2, True),
        lambda s: s.bad_words.num_bad_words.np.__setitem__(3, 1),
        lambda s: setattr(
            s, "thinking_budget_state", SimpleNamespace(enabled=True, use_thinking_budget=np.eye(4)[2] > 0)
        ),
    ):
        rejection.sampler = sampler = _processing_sampler(4)
        sampler.needs_logits_processing[:] = True
        flag(sampler)
        assert rejection.reads_draft_prefix(other) and not rejection.reads_draft_prefix(batch)
    rejection.sampler = sampler = _processing_sampler(4, custom=True)
    assert not rejection.reads_draft_prefix(batch)  # no request asks for processing
    sampler.needs_logits_processing[1] = True
    assert rejection.reads_draft_prefix(batch)
