# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Serving distribution gates must reject missing or incomparable evidence, pass a correct sampler whose logits
move by bf16 rounding between runs, and fail a biased one."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def sampling_check():
    path = Path(__file__).parents[1] / "tools" / "sampling_check.py"
    spec = importlib.util.spec_from_file_location("sampling_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact():
    return {
        "api": "sglang",
        "max_new_tokens": 3,
        "samples": 20,
        "settings": {
            "unrestricted": {
                "params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0},
                "tokens": [[[1, 2, 3]] * 20 for _ in range(4)],
                "seconds": 1.0,
            }
        },
        "seeds": {},
    }


def compare(module, tmp_path, ref, candidate, scores=None, deterministic=False):
    reference = tmp_path / "reference.json"
    output = tmp_path / "candidate.json"
    reference.write_text(json.dumps(ref))
    output.write_text(json.dumps(candidate))
    score_path = None
    if scores is not None:
        score_path = tmp_path / "scores.json"
        score_path.write_text(json.dumps(scores))
    args = dict(reference=reference, candidate=output, alpha=0.01, scores=score_path, deterministic=deterministic)
    return module.compare(SimpleNamespace(**args))


def test_unrestricted_setting_exercises_temperature_one(sampling_check):
    assert sampling_check.SETTINGS["unrestricted"] == {
        "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0
    }


def test_every_setting_names_each_filter(sampling_check):
    # A server fills an unnamed filter from the model's generation_config (Qwen3.8: top-k 20, top-p 0.95).
    for params in sampling_check.SETTINGS.values():
        assert params.keys() == sampling_check.NEUTRAL.keys()
    assert not sampling_check.truncates(sampling_check.SETTINGS["temperature"])
    assert not sampling_check.truncates(sampling_check.SETTINGS["penalty"])


def test_complete_distribution_evidence_passes(sampling_check, tmp_path):
    ref = artifact()
    assert compare(sampling_check, tmp_path, ref, copy.deepcopy(ref)) == 0


@pytest.mark.parametrize(
    "fault", ["empty", "missing", "refused", "parameters", "prompts", "samples", "tokens", "unnamed"]
)
def test_incomplete_distribution_evidence_fails(sampling_check, tmp_path, fault):
    ref = artifact()
    candidate = copy.deepcopy(ref)
    setting = candidate["settings"]["unrestricted"]
    if fault == "empty":
        ref["settings"].clear()
        candidate["settings"].clear()
    elif fault == "missing":
        candidate["settings"].clear()
    elif fault == "refused":
        setting.clear()
        setting["refused"] = "unsupported"
    elif fault == "parameters":
        setting["params"]["temperature"] = 0.0
    elif fault == "prompts":
        setting["tokens"].pop()
    elif fault == "samples":
        setting["tokens"][0].pop()
    elif fault == "tokens":
        setting["tokens"][0][0] = [1]
    elif fault == "unnamed":
        for doc in (ref, candidate):
            del doc["settings"]["unrestricted"]["params"]["top_p"]
    assert compare(sampling_check, tmp_path, ref, candidate) == 1


def test_changed_distribution_fails(sampling_check, tmp_path):
    ref = artifact()
    candidate = copy.deepcopy(ref)
    candidate["settings"]["unrestricted"]["tokens"] = [[[4, 5, 6]] * 20 for _ in range(4)]
    assert compare(sampling_check, tmp_path, ref, candidate) == 1


@pytest.mark.parametrize("deterministic", [False, True])
def test_seeds_are_judged_only_on_a_deterministic_server(sampling_check, tmp_path, deterministic):
    ref = artifact()
    ref["seeds"] = {str(s): [s, 1, 2] for s in range(32)}
    ref["seeds_reproduced"] = 32
    candidate = copy.deepcopy(ref)
    candidate["seeds_reproduced"] = 31
    assert compare(sampling_check, tmp_path, ref, candidate, deterministic=deterministic) == int(deterministic)


def test_compare_runs_without_the_http_client(tmp_path):
    # Only `run` talks HTTP (requests); the public test env has no requests.
    path = Path(__file__).parents[1] / "tools" / "sampling_check.py"
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(artifact()))
    code = (
        "import importlib.util, sys, types; sys.modules['requests'] = None; "
        f"spec = importlib.util.spec_from_file_location('sampling_check', {str(path)!r}); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        f"sys.exit(m.compare(types.SimpleNamespace(reference={str(ref)!r}, candidate={str(ref)!r}, alpha=0.01, "
        "scores=None, deterministic=False)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# A synthetic target: 40 tokens whose fp32 logits fall 0.45 (3.6 bf16 ulps at [16, 32)) per rank from 24, except that
# ranks 8 and 9 sit 0.04 apart, a near tie right at a top-8 cut. A server start moves every logit by up to a quarter
# bf16 ulp and the rank-8 one a further ulp up (odd starts) or down (even starts), so odd starts keep rank 8 and even
# starts rank 9, as a real Qwen3.8 top-p cut did between two no-spec vLLM starts (a token drawn 0 of 1428 times by one
# start and 115 of 1397 times by the other).
VOCAB, SCORED, SAMPLES = 40, 24, 2000
PROMPTS = [[101], [102], [103], [104]]
TOP_K = {"temperature": 1.0, "top_k": 8, "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0}
TEMPERATURE = {"temperature": 0.7, "top_k": -1, "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0}


def fp32_logits(context):
    """{token: logit} of a context."""
    ranks = [24.0 - 0.45 * r for r in range(VOCAB)]
    ranks[8] = ranks[7] - 0.04
    order = list(range(VOCAB))
    random.Random(f"order {context}").shuffle(order)
    return {token: logit for token, logit in zip(order, ranks)}


def served_logits(context, start):
    """The logits a server start computes: fp32 plus rounding (see above), within the fp32 arbiter's 2 ulps a pair."""
    noise = random.Random(f"start {start} {context}")
    logits = fp32_logits(context)
    eighth = sorted(logits, key=logits.get, reverse=True)[7]
    served = {}
    for token, value in logits.items():
        ulp = 2.0 ** (math.frexp(abs(value))[1] - 8)
        served[token] = value + noise.uniform(-0.25, 0.25) * ulp + (ulp if start % 2 else -ulp) * (token == eighth)
    return served


def processed(logits, params):
    """The sampled distribution: temperature, then the top-k cut (no other filter is used here)."""
    order = sorted(logits, key=logits.get, reverse=True)
    if params["top_k"] > 0:
        order = order[: params["top_k"]]
    weights = [math.exp((logits[t] - logits[order[0]]) / params["temperature"]) for t in order]
    return {t: w / sum(weights) for t, w in zip(order, weights)}


def draw(probs, rng):
    u, total = rng.random(), 0.0
    for token, p in probs.items():
        total += p
        if u < total:
            return token
    return token


def biased_chain_verify(p, q, rng):
    """One draft token y ~ q, accepted with min(1, p_y / q_y); on rejection the classic mistake of subtracting only
    y's q mass from p (tests/test_sampled_tree.py's mutation) instead of drawing from norm(max(p - q, 0))."""
    y = draw(q, rng)
    if rng.random() * q[y] < p.get(y, 0.0):
        return y
    residual = dict(p)
    residual[y] = max(p.get(y, 0.0) - q[y], 0.0)
    total = sum(residual.values())
    return draw({t: v / total for t, v in residual.items()}, rng)


def sampled_run(params, start, sampler=None, seed=0):
    """A run artifact: SAMPLES completions of 3 tokens per prompt, each token drawn by ``sampler(logits, rng)``
    from the start's logits (by default the processed distribution)."""
    rng = random.Random(f"draws {start} {seed}")
    sampler = sampler or (lambda logits, rng: draw(processed(logits, params), rng))
    tokens = []
    for prompt in PROMPTS:
        rows = []
        for _ in range(SAMPLES):
            context = list(prompt)
            for _ in range(3):
                context.append(sampler(served_logits(tuple(context), start), rng))
            rows.append(context[len(prompt) :])
        tokens.append(rows)
    return {
        "api": "vllm",
        "max_new_tokens": 3,
        "samples": SAMPLES,
        "prompt_ids": PROMPTS,
        "settings": {"setting": {"params": dict(params), "tokens": tokens, "seconds": 1.0}},
        "seeds": {},
    }


def fp32_scores(module, reference):
    """What ``score`` writes for the reference: the top SCORED fp32 logits of every judged context."""
    contexts = {}
    for _, i, prefix in module.judged_contexts(reference):
        ids = reference["prompt_ids"][i] + list(prefix)
        logits = fp32_logits(tuple(ids))
        top = sorted(logits, key=logits.get, reverse=True)[:SCORED]
        contexts[module.context_key(ids)] = {"vocab": VOCAB, "top": [[logits[t], t] for t in top]}
    return {"tolerance_ulps": 2.0, "contexts": contexts}


def test_cut_bounds_leave_near_ties_out(sampling_check):
    top = [[24.0 - 0.45 * r, r] for r in range(SCORED)]
    top[8][0] = top[7][0] - 0.04
    kept, possible = sampling_check.cut_bounds(top, VOCAB, TOP_K, 2.0)
    assert kept == set(range(7)) and possible == set(range(9))
    # A clear cut (3.6 ulps) is no near tie.
    kept, possible = sampling_check.cut_bounds(top, VOCAB, dict(TOP_K, top_k=5), 2.0)
    assert kept == possible == set(range(5))
    # top-p 0.6: shares 0.57 / 0.21 / 0.21, so whichever of the near-tied second and third ranks first is kept.
    flat = [[21.0, 0], [20.0, 1], [19.98, 2], [17.0, 3]] + [[10.0 - r, 4 + r] for r in range(SCORED - 4)]
    kept, possible = sampling_check.cut_bounds(flat, VOCAB, dict(TOP_K, top_k=-1, top_p=0.6), 2.0)
    assert kept == {0} and possible == {0, 1, 2}
    # min-p 0.05 at temperature 1: a token log(20) = 3.0 below the top, near the floor, is a near tie.
    edge = [[20.0, 0], [20.0 - math.log(20.0) + 0.01, 1], [15.0, 2]] + [[10.0 - r, 3 + r] for r in range(SCORED - 3)]
    kept, possible = sampling_check.cut_bounds(edge, VOCAB, dict(TOP_K, top_k=-1, min_p=0.05), 2.0)
    assert kept == {0} and possible == {0, 1}
    with pytest.raises(ValueError, match="past the 8 scored"):
        sampling_check.cut_bounds(top[:8], VOCAB, TOP_K, 2.0)
    with pytest.raises(ValueError, match="not modelled"):
        sampling_check.cut_bounds(top, VOCAB, dict(TOP_K, repetition_penalty=1.5), 2.0)


def test_score_keeps_as_many_tokens_as_bound_the_cut(sampling_check):
    # min-p 0.05 on a flat distribution keeps every token within log(20) = 3.0 of the top: 300 here.
    pairs = [[20.0 - 0.01 * r, r] for r in range(1000)]
    min_p = dict(TOP_K, top_k=-1, min_p=0.05)
    top = sampling_check.bounded_top(pairs, 5000, [TOP_K, min_p], 2.0)
    assert len(top) == 1000  # 64 and 256 do not bound it
    kept, possible = sampling_check.cut_bounds(top, 5000, min_p, 2.0)
    # Near-ties: within one ulp (0.125) either side of the top and of the floor, i.e. 25 ranks around rank 300.
    assert kept == set(range(275)) and possible == set(range(325))
    assert len(sampling_check.bounded_top(pairs, 5000, [TOP_K], 2.0)) == 64


def test_self_control_passes_through_bf16_cut_flips(sampling_check, tmp_path):
    ref, cand = sampled_run(TOP_K, start=1), sampled_run(TOP_K, start=2)
    # The flips are real: per draw position, the two starts' supports differ, which failed the old per-token test.
    first = [[row[0] for row in doc["settings"]["setting"]["tokens"][0]] for doc in (ref, cand)]
    assert set(first[0]) != set(first[1])
    assert sampling_check.homogeneity(*first)[0] < 1e-6
    assert compare(sampling_check, tmp_path, ref, cand, fp32_scores(sampling_check, ref)) == 0


def test_truncated_setting_needs_fp32_scores(sampling_check, tmp_path):
    ref = sampled_run(TOP_K, start=1)
    assert compare(sampling_check, tmp_path, ref, copy.deepcopy(ref)) == 1


def test_sampler_that_ignores_the_cut_fails(sampling_check, tmp_path):
    ref = sampled_run(TOP_K, start=1)
    untruncated = dict(TOP_K, top_k=-1)
    cand = sampled_run(TOP_K, start=2, sampler=lambda logits, rng: draw(processed(logits, untruncated), rng))
    assert compare(sampling_check, tmp_path, ref, cand, fp32_scores(sampling_check, ref)) == 1


@pytest.mark.parametrize("biased", [False, True])
def test_biased_verify_residual_fails(sampling_check, tmp_path, biased):
    """An under-confident draft (the target at temperature 2): a correct chain verify keeps the target's
    distribution, the mutated residual moves 6% of it (TV 0.063) off the top token."""

    def verify(logits, rng):
        p = processed(logits, TOP_K)
        q = processed(logits, dict(TOP_K, temperature=2.0))
        if biased:
            return biased_chain_verify(p, q, rng)
        y = draw(q, rng)
        if rng.random() * q[y] < p[y]:
            return y
        residual = {t: max(p[t] - q.get(t, 0.0), 0.0) for t in p}
        total = sum(residual.values())
        return draw({t: v / total for t, v in residual.items()}, rng)

    ref = sampled_run(TOP_K, start=1)
    cand = sampled_run(TOP_K, start=2, sampler=verify)
    assert compare(sampling_check, tmp_path, ref, cand, fp32_scores(sampling_check, ref)) == int(biased)


def test_ignored_temperature_fails(sampling_check, tmp_path):
    ref = sampled_run(TEMPERATURE, start=1)
    assert compare(sampling_check, tmp_path, ref, sampled_run(TEMPERATURE, start=2)) == 0
    hot = dict(TEMPERATURE, temperature=1.0)
    cand = sampled_run(TEMPERATURE, start=2, sampler=lambda logits, rng: draw(processed(logits, hot), rng))
    assert compare(sampling_check, tmp_path, ref, cand) == 1
