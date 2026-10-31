#!/usr/bin/env python3
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Lossless sampling under speculation: a speculative server must sample what the no-spec server samples.

``run`` sends ``--samples`` short completions of each of a few open-ended prompts to a running server under each
sampling setting (temperature, top-k with top-p, min-p, a repetition penalty), ``--concurrency`` requests at a
time so they share decode rounds, and saves the output token ids. The first token comes from the prefill sampler;
every later one from a speculative verify round. Every setting names temperature, top-k, top-p, min-p and the
repetition penalty: a server fills unnamed ones from the model's ``generation_config`` (Qwen3.8 sets top-k 20 and
top-p 0.95), which would truncate a setting meant to be untruncated.

``compare`` tests, per setting and prompt, that the speculative server's tokens come from the same distribution as
the reference's, with two-sample chi-square tests of homogeneity (cells seen fewer than 10 times in both runs
together are pooled) at a Bonferroni-corrected level ``--alpha / tests``, and prints the total variation distance
of each test. A server that ignores a sampling parameter (for example, verifies greedily at temperature > 0) fails
it by a wide margin.

An untruncated setting is tested on the joint distribution of the first three tokens, on the marginal of every
position, and on tokens 2 and 3 given the reference's most common prefix ("verify draws given the modal prefix",
which isolates the verify rounds).

A truncated setting (top-k, top-p or min-p) is tested only at fixed contexts: token 1 given the prompt, and tokens
2 and 3 given the reference's most common prefix. A truncation cut is discontinuous in the logits: a token whose
logit sits within bf16 rounding of the cut is kept by one server start or batch shape and dropped by another, and
its whole probability moves with it, so two correct no-spec servers differ there. ``score`` re-forwards each of
these contexts with the unmodified target in float32 (``tools/fp32_arbiter.py``'s reference model, on a GPU) and
``compare --scores`` splits the vocabulary with the fp32 arbiter's tolerance (each logit may move by one bf16 ulp
at its magnitude, so a pair's gap by two): tokens every such logit keeps, near-ties at the cut, and tokens no such
logit keeps. Truncation renormalizes the kept tokens proportionally, so draws among the always-kept tokens follow
the same distribution whichever near-ties a run kept; the test runs on those draws. A draw of a token no tolerated
logit keeps fails the setting outright (a sampler that ignores the cut).

With ``--seeds S`` (``run``) every one of S seeds is sent twice, in batches of a different composition, at
temperature 0.7 for ``--seed-tokens`` tokens; ``compare`` (and ``run`` itself) report how many seeds reproduced
their text, and ``compare`` how many seeded texts equal the reference server's for the same seed. A seed reproduces
across batches only on a batch-invariant server (sglang with ``--enable-deterministic-inference``), so ``compare``
requires every seed to reproduce only with ``--deterministic``.

With ``--speed-data`` (a SPEED-Bench jsonl) ``run`` also decodes the first user turn of ``--speed-prompts``
conversations one at a time (batch size 1, chat template, thinking off) at temperature 0 and at temperature 1, and
``compare`` prints both decode rates against the reference's and how many greedy outputs equal the reference's.

    python tools/sampling_check.py run --api vllm --port P --tokenizer <model dir> --out nospec.json --seeds 32
    python tools/sampling_check.py run --api vllm --port P --tokenizer <model dir> --out spec.json --seeds 32
    python tools/sampling_check.py score --model <model dir> nospec.json --out scores.json
    python tools/sampling_check.py compare nospec.json spec.json --scores scores.json

``--api`` names the server's HTTP API: ``sglang`` and ``tokenspeed`` (``/generate``, TokenSpeed with the served
model's name) or ``vllm`` (``/v1/completions``).

Exit code 1 (``compare``) when a test rejects at the corrected level, a draw falls outside every tolerated cut, or
(``--deterministic``) a seed did not reproduce.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import concurrent.futures
import importlib.util
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

PROMPTS = (
    "Here is a list of ten random animals:",
    "Once upon a time, in a",
    "My favorite word is",
    "The weather today is",
)
# Every setting names each filter: a server fills unnamed ones from the model's generation_config.
NEUTRAL = {"temperature": 1.0, "top_k": -1, "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0}
SETTINGS: Dict[str, dict] = {
    "unrestricted": dict(NEUTRAL),
    "temperature": {**NEUTRAL, "temperature": 0.7},
    "top_k_top_p": {**NEUTRAL, "top_k": 8, "top_p": 0.7},
    "min_p": {**NEUTRAL, "min_p": 0.05},
    "penalty": {**NEUTRAL, "temperature": 0.8, "repetition_penalty": 1.5},
}
SEED_SETTING = SETTINGS["temperature"]
JOINT = 3  # tokens in the joint test
POOL = 10  # cells seen fewer times than this in both runs together are pooled


def generate(api: str, base: str, model: Optional[str], prompt: List[int], params: dict, max_new: int, seed=None):
    """One completion's output token ids (and sglang's ``spec_*`` meta info)."""
    import requests  # only the HTTP client needs it; compare runs in test envs without it
    if api in ("sglang", "tokenspeed"):
        sp = {"max_new_tokens": max_new, "ignore_eos": True, **params}
        if seed is not None:
            sp["sampling_seed"] = seed
        body = {"input_ids": prompt, "sampling_params": sp}
        if api == "tokenspeed":  # TokenSpeed's /generate routes by model name and answers with a one-element list
            body["model"] = model
        r = requests.post(f"{base}/generate", json=body, timeout=3600)
        r.raise_for_status()
        out = r.json()
        out = out[0] if isinstance(out, list) else out
        meta = out.get("meta_info") or {}
        ids = out["output_ids"] if "output_ids" in out else meta["output_ids"]
        return ids, {k: v for k, v in meta.items() if k.startswith("spec_")}
    body = {"model": model, "prompt": prompt, "max_tokens": max_new, "ignore_eos": True, "return_token_ids": True}
    body.update(params)
    if seed is not None:
        body["seed"] = seed
    r = requests.post(f"{base}/v1/completions", json=body, timeout=3600)
    r.raise_for_status()
    return r.json()["choices"][0]["token_ids"], {}


def vllm_spec_counters(base: str) -> Dict[str, float]:
    """vLLM's speculative decoding counters (drafts, accepted tokens), from /metrics."""
    import requests
    counters: Dict[str, float] = collections.defaultdict(float)
    for line in requests.get(f"{base}/metrics", timeout=60).text.splitlines():
        if line.startswith("vllm:spec_decode_num_") and "_total" in line.split("{")[0]:
            counters[line.split("{")[0]] += float(line.rsplit(" ", 1)[1])
    return dict(counters)


def run(args) -> int:
    import requests
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    base = f"http://127.0.0.1:{args.port}"
    model = requests.get(f"{base}/v1/models", timeout=60).json()["data"][0]["id"] if args.api != "sglang" else None
    prompts = [tokenizer.encode(p, add_special_tokens=False) for p in PROMPTS]
    result: dict = {
        "api": args.api,
        "max_new_tokens": args.max_new_tokens,
        "samples": args.samples,
        "concurrency": args.concurrency,
        "prompt_ids": prompts,
        "settings": {},
        "seeds": {},
    }
    pool = concurrent.futures.ThreadPoolExecutor(args.concurrency)
    for name in args.settings.split(","):
        params = SETTINGS[name]
        before = vllm_spec_counters(base) if args.api == "vllm" else {}
        jobs = [(i, prompt) for _ in range(args.samples) for i, prompt in enumerate(prompts)]
        t0 = time.time()
        try:
            outs = list(
                pool.map(lambda job: generate(args.api, base, model, job[1], params, args.max_new_tokens), jobs)
            )
        except requests.HTTPError as error:  # the server refuses the setting: nothing to compare
            result["settings"][name] = {"params": params, "refused": f"{error}: {error.response.text[:300]}"}
            print(f"{name}: refused: {result['settings'][name]['refused']}", flush=True)
            continue
        seconds = time.time() - t0
        tokens: List[List[List[int]]] = [[] for _ in prompts]
        accept = []
        for (i, _), (ids, meta) in zip(jobs, outs):
            tokens[i].append(ids[: args.max_new_tokens])
            if meta.get("spec_verify_ct"):
                accept.append(len(ids) / meta["spec_verify_ct"])
        entry = {"params": params, "tokens": tokens, "seconds": round(seconds, 2)}
        if accept:
            entry["mean_tokens_per_verify"] = round(sum(accept) / len(accept), 3)
        if args.api == "vllm":
            after = vllm_spec_counters(base)
            entry["spec_counters"] = {k: after[k] - before.get(k, 0.0) for k in after}
        result["settings"][name] = entry
        save(args.out, result)
        print(
            f"{name}: {len(jobs)} requests in {seconds:.1f} s {entry.get('mean_tokens_per_verify', '')} "
            f"{entry.get('spec_counters', '')}",
            flush=True,
        )
    if args.speed_data:
        chats = []
        with open(args.speed_data, encoding="utf-8") as f:
            for line in f:
                turn = [{"role": "user", "content": json.loads(line)["turns"][0]}]
                text = tokenizer.apply_chat_template(
                    turn, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                chats.append(tokenizer.encode(text, add_special_tokens=False))
                if len(chats) == args.speed_prompts:
                    break
        result["speed"] = {}
        for temperature in (0.0, 1.0):
            t0, outs = time.time(), []
            for chat in chats:
                outs.append(generate(args.api, base, model, chat, {"temperature": temperature}, args.speed_tokens)[0])
            seconds = time.time() - t0
            rate = sum(map(len, outs)) / seconds
            result["speed"][str(temperature)] = {"tokens": outs, "seconds": round(seconds, 2), "tok_s": round(rate, 1)}
            save(args.out, result)
            print(f"speed at temperature {temperature}: {rate:.1f} tok/s (batch size 1)", flush=True)
    if args.seeds:
        seeds = list(range(9000, 9000 + args.seeds))
        prompt = prompts[1]
        passes = []
        for order in (seeds, seeds[::-1]):  # the same seeds in batches of another composition
            outs = pool.map(lambda s: generate(args.api, base, model, prompt, SEED_SETTING, args.seed_tokens, s), order)
            passes.append({s: ids for s, (ids, _) in zip(order, outs)})
        result["seeds"] = {str(s): passes[0][s] for s in seeds}
        same = sum(passes[0][s] == passes[1][s] for s in seeds)
        result["seeds_reproduced"] = same
        print(f"seeds: {same}/{len(seeds)} reproduced their text", flush=True)
    save(args.out, result)
    return 0


def save(path: str, result: dict) -> None:
    """Write what ``run`` has so far, so a failure in a later section keeps the earlier ones."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f)


def chi2_sf(stat: float, dof: int) -> float:
    """P(X >= stat) for X chi-square with ``dof`` degrees of freedom (regularized upper incomplete gamma)."""
    a, x = dof / 2.0, stat / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:  # series for the lower function
        term = total = 1.0 / a
        for n in range(1, 1000):
            term *= x / (a + n)
            total += term
            if term < total * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b, c, d = x + 1 - a, 1e300, 1 / (x + 1 - a)  # continued fraction for the upper function
    h = d
    for n in range(1, 1000):
        an = -n * (n - a)
        b += 2
        d = an * d + b
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def homogeneity(ref: Sequence, test: Sequence) -> Tuple[float, float, int]:
    """Two-sample chi-square test of homogeneity: (p value, total variation distance, cells after pooling)."""
    a, b = collections.Counter(ref), collections.Counter(test)
    na, nb = len(ref), len(test)
    if not na or not nb:
        return 1.0, 0.0, 0
    tv = 0.5 * sum(abs(a[c] / na - b[c] / nb) for c in set(a) | set(b))
    cells: List[Tuple[int, int]] = []
    rest = [0, 0]
    for c in set(a) | set(b):
        if a[c] + b[c] >= POOL:
            cells.append((a[c], b[c]))
        else:
            rest[0] += a[c]
            rest[1] += b[c]
    if sum(rest) > 0:
        cells.append((rest[0], rest[1]))
    if len(cells) < 2:
        return 1.0, tv, len(cells)
    stat = 0.0
    for x, y in cells:
        total = x + y
        for observed, n in ((x, na), (y, nb)):
            expected = total * n / (na + nb)
            stat += (observed - expected) ** 2 / expected
    return chi2_sf(stat, len(cells) - 1), tv, len(cells)


TRUNCATION = ("top_k", "top_p", "min_p")


def truncates(params: dict) -> bool:
    """Whether the setting cuts the vocabulary (top-k, top-p or min-p)."""
    return params["top_k"] not in (-1, 0) or params["top_p"] < 1.0 or params["min_p"] > 0.0


def _sibling(name: str):
    """Another tool of this directory, imported by path (tools/ is not a package)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # its dataclasses resolve their annotations through sys.modules
        spec.loader.exec_module(module)
    return sys.modules[name]


def cut_bounds(
    top: Sequence[Sequence[float]], vocab: int, params: dict, tolerance_ulps: float, *, joint: bool = False
) -> Tuple[Set, Set]:
    """(tokens every tolerated logit keeps, tokens some tolerated logit keeps) under the setting's cut.

    ``top``: fp32 ``[logit, token]`` pairs, best first; the ``vocab - len(top)`` other tokens are at most the last
    logit. Each logit may move by ``tolerance_ulps / 2`` bf16 ulps at its magnitude, so the gap of a pair moves by up
    to ``tolerance_ulps`` (the fp32 arbiter's rule). SGLang applies top-k and top-p jointly to the full
    distribution. Otherwise top-p uses the renormalized top-k distribution. Min-p alone is also supported."""
    ulp = _sibling("fp32_arbiter").bf16_ulp
    temperature, k, top_p, min_p = params["temperature"], params["top_k"], params["top_p"], params["min_p"]
    k = None if k in (-1, 0) else k
    if min_p > 0.0 and (k is not None or top_p < 1.0):
        raise ValueError("min-p together with top-k or top-p is not modelled")
    if params.get("repetition_penalty", 1.0) != 1.0:
        raise ValueError("a repetition penalty together with a cut is not modelled")
    tokens = [int(t) for _, t in top]
    lo = [v - tolerance_ulps / 2 * ulp(abs(v)) for v, _ in top]
    hi = [v + tolerance_ulps / 2 * ulp(abs(v)) for v, _ in top]
    # The unscored tokens as one entry of multiplicity ``rest``, anywhere at or below the last scored logit.
    rest = vocab - len(top)
    lo.append(-math.inf)
    hi.append(hi[-1])
    count = [1] * len(tokens) + [rest]
    n = len(count)
    shift = max(hi)

    def mass(x: float) -> float:
        return math.exp((x - shift) / temperature)

    def counter(bounds: List[float]):
        """Tokens whose bound is above x, by bisection (the scored set can run to thousands of tokens)."""
        pairs = sorted(zip(bounds, count))
        keys = [b for b, _ in pairs]
        tail = list(itertools.accumulate([c for _, c in pairs][::-1]))[::-1] + [0]
        return lambda x: tail[bisect.bisect_right(keys, x)]

    above_hi, above_lo = counter(hi), counter(lo)

    # In the top k for every tolerated logit (fewer than k others can reach it) / for some (fewer than k others are
    # surely above it). Joint top-p uses all tokens, independently of the top-k cut.
    in_k = [k is None or above_hi(lo[i]) - count[i] < k for i in range(n)]
    maybe_k = [count[i] > 0 and (k is None or above_lo(hi[i]) < k) for i in range(n)]
    in_p = [joint or keep for keep in in_k]
    pool = [j for j in range(n) if joint or maybe_k[j]]
    by_hi = sorted(range(n), key=lambda j: hi[j], reverse=True)[:2]  # the two largest bounds of each side, for min-p
    by_lo = sorted(range(n), key=lambda j: lo[j], reverse=True)[:2]
    kept, possible = [], []
    for i in range(n):
        sure, maybe = in_k[i], maybe_k[i]
        if top_p < 1.0 and maybe:
            # Kept when the renormalized mass of the tokens above it is below top_p: bound that share both ways.
            others = [j for j in pool if j != i]
            above_max = sum(count[j] * mass(hi[j]) for j in others if hi[j] > lo[i])
            below_min = mass(lo[i]) + sum(count[j] * mass(lo[j]) for j in others if hi[j] < lo[i] and in_p[j])
            sure = sure and above_max / (above_max + below_min) < top_p
            above_min = sum(count[j] * mass(lo[j]) for j in others if lo[j] > hi[i] and in_p[j])
            rest_max = count[i] * mass(hi[i]) + sum(count[j] * mass(hi[j]) for j in others if lo[j] <= hi[i])
            maybe = above_min / (above_min + rest_max) < top_p
        if min_p > 0.0:
            # Kept when its probability is at least min_p times the largest.
            floor = temperature * math.log(min_p)
            others_hi = hi[by_hi[1] if by_hi[0] == i else by_hi[0]]
            others_lo = lo[by_lo[1] if by_lo[0] == i else by_lo[0]]
            sure = sure and lo[i] - max(lo[i], others_hi) >= floor
            maybe = maybe and hi[i] - max(hi[i], others_lo) >= floor
        kept.append(sure and i < len(tokens))
        possible.append(maybe)
    if possible[-1]:
        raise ValueError(f"the cut may keep tokens past the {len(tokens)} scored ones")
    return ({t for t, keep in zip(tokens, kept) if keep}, {t for t, keep in zip(tokens, possible) if keep})


def modal_prefix(tokens: Sequence[Sequence[int]], depth: int) -> Tuple[int, ...]:
    """The most common first ``depth`` tokens (the first seen among equally common ones)."""
    return collections.Counter(tuple(t[:depth]) for t in tokens).most_common(1)[0][0]


def judged_contexts(artifact: dict):
    """(setting, prompt, prefix) of every context a truncated setting of this reference is judged at: the prompt,
    then the reference's most common prefix of each length below ``JOINT``."""
    for name, entry in artifact["settings"].items():
        if "refused" in entry or not truncates(entry["params"]):
            continue
        for i, tokens in enumerate(entry["tokens"]):
            for depth in range(JOINT):
                yield name, i, modal_prefix(tokens, depth)


def context_key(ids: Sequence[int]) -> str:
    return ",".join(map(str, ids))


def bounded_top(pairs: Sequence[Sequence[float]], vocab: int, settings: Sequence[dict], tolerance_ulps: float) -> list:
    """The shortest head of ``pairs`` that bounds every setting for both joint and sequential filtering."""
    size = min(64, len(pairs))
    while True:
        try:
            for params in settings:
                for joint in (False, True):
                    cut_bounds(pairs[:size], vocab, params, tolerance_ulps, joint=joint)
            return list(pairs[:size])
        except ValueError:
            if size == len(pairs):
                raise
            size = min(4 * size, len(pairs))


def score(args) -> int:
    """fp32 logits of every judged context of the references: as many of the best as bound each setting's cut,
    and the vocabulary size."""
    import torch

    fp32 = _sibling("fp32_arbiter")
    judged: Dict[str, Tuple[list, list]] = {}  # context key: (ids, params of the settings judged there)
    for path in args.references:
        with open(path, encoding="utf-8") as f:
            artifact = json.load(f)
        for name, i, prefix in judged_contexts(artifact):
            ids = list(artifact["prompt_ids"][i]) + list(prefix)
            judged.setdefault(context_key(ids), (ids, []))[1].append(artifact["settings"][name]["params"])
    model, seconds = fp32.load_reference_model(args.model)
    print(f"fp32 reference loaded in {seconds:.0f} s; {len(judged)} contexts", flush=True)
    contexts: Dict[str, dict] = {}
    for key, (ids, settings) in judged.items():
        batch = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            out = model(input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1].double()
        values, indices = torch.sort(logits, descending=True)
        pairs = [[v, int(t)] for v, t in zip(values.tolist(), indices.tolist())]
        top = bounded_top(pairs, logits.numel(), settings, fp32.TOLERANCE_ULPS)
        contexts[key] = {"vocab": logits.numel(), "top": top}
        print(f"context {key}: {len(top)} tokens scored", flush=True)
    save(args.out, {"model": args.model, "tolerance_ulps": fp32.TOLERANCE_ULPS, "contexts": contexts})
    return 0

def validate_pair(ref: dict, cand: dict) -> None:
    """Reject partial evidence before a statistical test can silently omit it."""
    if ref.get("api") != cand.get("api"):
        raise ValueError("reference and candidate must use the same engine API")
    if not ref["settings"] or ref["settings"].keys() != cand["settings"].keys():
        raise ValueError("reference and candidate must contain the same nonempty settings")
    length = ref["max_new_tokens"]
    if length < JOINT or cand["max_new_tokens"] != length:
        raise ValueError("matched token lengths must cover the joint test")
    if ref.get("prompt_ids") != cand.get("prompt_ids"):
        raise ValueError("reference and candidate prompt token ids differ")
    for name in ref["settings"]:
        r, c = ref["settings"][name], cand["settings"][name]
        if "refused" in r or "refused" in c:
            raise ValueError(f"{name}: refused setting is not distribution evidence")
        if r["params"] != c["params"]:
            raise ValueError(f"{name}: sampling parameters differ")
        unnamed = [key for key in ("temperature", *TRUNCATION) if key not in r["params"]]
        if unnamed:
            raise ValueError(f"{name}: {', '.join(unnamed)} left to the server's defaults (the model's may truncate)")
        if len(r["tokens"]) != len(PROMPTS) or len(c["tokens"]) != len(PROMPTS):
            raise ValueError(f"{name}: missing prompt results")
        for r_tokens, c_tokens in zip(r["tokens"], c["tokens"]):
            if not r_tokens or len(r_tokens) != len(c_tokens):
                raise ValueError(f"{name}: empty or unequal sample counts")
            for artifact, tokens in ((ref, r_tokens), (cand, c_tokens)):
                if len(tokens) != artifact.get("samples", len(tokens)):
                    raise ValueError(f"{name}: sample count differs from the requested count")
                if any(len(ids) != length for ids in tokens):
                    raise ValueError(f"{name}: incomplete output token sequence")


def mixed_tests(name: str, prompt: int, r_tokens: list, c_tokens: list) -> list:
    """An untruncated setting's tests: the joint first tokens, every position's marginal, and tokens 2 and 3 given
    the reference's most common prefix (the verify round's own draws, free of any difference in the prefill's)."""
    tests = []
    r_joint = [tuple(t[:JOINT]) for t in r_tokens]
    c_joint = [tuple(t[:JOINT]) for t in c_tokens]
    tests.append((name, prompt, f"joint 1-{JOINT}", *homogeneity(r_joint, c_joint)))
    for pos in range(min(len(r_tokens[0]), len(c_tokens[0]))):
        marginals = [t[pos] for t in r_tokens], [t[pos] for t in c_tokens]
        tests.append((name, prompt, f"token {pos + 1}", *homogeneity(*marginals)))
    for pos in (1, 2):
        prefix = modal_prefix(r_tokens, pos)
        r_next = [t[pos] for t in r_tokens if tuple(t[:pos]) == prefix]
        c_next = [t[pos] for t in c_tokens if tuple(t[:pos]) == prefix]
        if r_next and c_next:
            tests.append((name, prompt, f"token {pos + 1} | modal prefix", *homogeneity(r_next, c_next)))
    return tests


def cut_tests(
    name: str,
    params: dict,
    prompt: int,
    prompt_ids: list,
    r_tokens: list,
    c_tokens: list,
    scores: dict,
    tally: List[int],
    outside: list,
    *,
    joint: bool = False,
) -> list:
    """A truncated setting's tests at its judged contexts, on the draws among the tokens every tolerated logit keeps.
    Counts near-tie and outside draws into ``tally`` and lists the outside tokens in ``outside``."""
    tests = []
    for depth in range(JOINT):
        prefix = modal_prefix(r_tokens, depth)
        context = scores["contexts"].get(context_key(list(prompt_ids) + list(prefix)))
        if context is None:
            raise ValueError(f"{name}: prompt {prompt} prefix {list(prefix)} was not scored")
        kept, possible = cut_bounds(
            context["top"], context["vocab"], params, scores["tolerance_ulps"], joint=joint
        )
        draws = [[t[depth] for t in tokens if tuple(t[:depth]) == prefix] for tokens in (r_tokens, c_tokens)]
        what = "token 1" if depth == 0 else f"token {depth + 1} | modal prefix"
        tests.append((name, prompt, what, *homogeneity(*[[t for t in d if t in kept] for d in draws])))
        for slot, d in enumerate(draws):
            tally[slot] += sum(t in possible and t not in kept for t in d)
            tally[2 + slot] += sum(t not in possible for t in d)
        stray = [sorted({t for t in d if t not in possible}) for d in draws]
        if stray[0] or stray[1]:
            outside.append((name, prompt, what, *stray))
    return tests


def compare(args) -> int:
    with open(args.reference, encoding="utf-8") as f:
        ref = json.load(f)
    with open(args.candidate, encoding="utf-8") as f:
        cand = json.load(f)
    try:
        validate_pair(ref, cand)
    except (KeyError, TypeError, ValueError) as error:
        print(f"INVALID distribution evidence: {error}")
        return 1
    scores = None
    if args.scores:
        with open(args.scores, encoding="utf-8") as f:
            scores = json.load(f)
    tests = []  # (setting, prompt, what, p, tv, cells)
    cuts: Dict[str, List[int]] = {}  # setting: draws [ref near-tie, candidate near-tie, ref outside, candidate outside]
    outside: list = []  # (setting, prompt, what, reference tokens, candidate tokens) outside every tolerated cut
    try:
        for name, entry in cand["settings"].items():
            params = entry["params"]
            if truncates(params) and scores is None:
                raise ValueError(f"{name} truncates: judge it with --scores (see score)")
            for i, (r_tokens, c_tokens) in enumerate(zip(ref["settings"][name]["tokens"], entry["tokens"])):
                if truncates(params):
                    assert scores is not None
                    tally = cuts.setdefault(name, [0, 0, 0, 0])
                    prompt_ids = ref["prompt_ids"][i]
                    tests += cut_tests(
                        name, params, i, prompt_ids, r_tokens, c_tokens, scores, tally, outside,
                        joint=ref.get("api") == "sglang",
                    )
                else:
                    tests += mixed_tests(name, i, r_tokens, c_tokens)
    except (KeyError, ValueError) as error:  # unscored contexts, or a cut the scores cannot bound
        print(f"INVALID distribution evidence: {error}")
        return 1
    level = args.alpha / max(1, len(tests))
    failed = False
    for name in cand["settings"]:
        rows = [t for t in tests if t[0] == name]
        if not rows:
            continue
        worst = min(rows, key=lambda t: t[3])
        joint = [t[4] for t in rows if t[2].startswith("joint")]
        bad = [t for t in rows if t[3] < level]
        stray = [t for t in outside if t[0] == name]
        conditional = [t for t in rows if "modal prefix" in t[2]]
        cond_bad = [t for t in conditional if t[3] < level]
        failed |= bool(bad) or bool(stray)
        extra = cand["settings"][name].get("mean_tokens_per_verify") or cand["settings"][name].get("spec_counters", "")
        print(
            f"{name:12s} {'FAIL' if bad or stray else 'PASS'}: {len(rows)} tests, {len(bad)} below {level:.1e}; min p "
            f"{worst[3]:.2e} ({worst[2]}, prompt {worst[1]}, TV {worst[4]:.3f})"
            f"{f'; max joint TV {max(joint):.3f}' if joint else ''}; "
            f"{cand['settings'][name]['seconds']} s (ref {ref['settings'][name]['seconds']} s) {extra}"
        )
        if name in cuts:
            tally = cuts[name]
            print(
                f"    fp32 cut ({scores['tolerance_ulps']:g} bf16 ulps): near-tie draws left out {tally[0]} ref / "
                f"{tally[1]} candidate; draws outside every tolerated cut {tally[2]} ref / {tally[3]} candidate"
            )
        cond_worst = min(conditional, key=lambda t: t[3]) if conditional else None
        if cond_worst:
            print(
                f"    verify draws given the modal prefix: {'FAIL' if cond_bad else 'PASS'}, {len(cond_bad)} of "
                f"{len(conditional)} below; min p {cond_worst[3]:.2e} ({cond_worst[2]}, prompt {cond_worst[1]})"
            )
        for t in sorted(bad, key=lambda t: t[3])[:5]:
            print(f"    {t[2]}, prompt {t[1]}: p {t[3]:.2e}, TV {t[4]:.3f}, {t[5]} cells")
        for t in stray[:5]:
            print(f"    {t[2]}, prompt {t[1]}: tokens outside every tolerated cut: ref {t[3]}, candidate {t[4]}")
    for temperature, entry in cand.get("speed", {}).items():
        ref_entry = ref.get("speed", {}).get(temperature, {})
        line = f"speed at temperature {temperature}: {entry['tok_s']} tok/s (reference {ref_entry.get('tok_s')})"
        if float(temperature) == 0.0 and ref_entry:
            same = sum(a == b for a, b in zip(entry["tokens"], ref_entry["tokens"]))
            line += f"; greedy outputs equal to the reference's: {same}/{len(entry['tokens'])}"
        print(line)
    if cand.get("seeds"):
        seeds = cand["seeds"]
        reproduced = cand.get("seeds_reproduced", 0)
        same_as_ref = sum(ref.get("seeds", {}).get(s) == ids for s, ids in seeds.items())
        prefix = [
            next((j for j, (x, y) in enumerate(zip(ids, ref["seeds"][s])) if x != y), len(ids))
            for s, ids in seeds.items()
            if s in ref.get("seeds", {})
        ]
        # A seed reproduces its text in another batch only on a batch-invariant server (vLLM's default is not: its
        # no-spec server reproduced 24 to 32 of 32 by chance), so only --deterministic makes it a gate.
        print(
            f"seeds: {reproduced}/{len(seeds)} reproduced their text; {same_as_ref}/{len(seeds)} equal the "
            f"reference's text for the same seed (mean common prefix {sum(prefix) / max(1, len(prefix)):.1f} tokens)"
            f"{'' if args.deterministic else '; not judged (no --deterministic)'}"
        )
        failed |= args.deterministic and reproduced != len(seeds)
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--api", choices=("sglang", "vllm", "tokenspeed"), required=True)
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--settings", default=",".join(SETTINGS))
    r.add_argument("--samples", type=int, default=2000, help="completions per prompt and setting")
    r.add_argument("--max-new-tokens", type=int, default=6)
    r.add_argument("--concurrency", type=int, default=32)
    r.add_argument("--seeds", type=int, default=0)
    r.add_argument("--seed-tokens", type=int, default=32)
    r.add_argument("--speed-data", help="SPEED-Bench jsonl: also time batch-size-1 decoding")
    r.add_argument("--speed-prompts", type=int, default=8)
    r.add_argument("--speed-tokens", type=int, default=512)
    s = sub.add_parser("score", help="fp32 logits of the truncated settings' judged contexts (a GPU and the model)")
    s.add_argument("--model", required=True)
    s.add_argument("references", nargs="+", help="reference run outputs (their modal prefixes are judged)")
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument("--alpha", type=float, default=0.01, help="family-wise level, Bonferroni-corrected")
    c.add_argument("--scores", help="score's output for the reference: required by truncated settings")
    c.add_argument("--deterministic", action="store_true", help="the servers are batch-invariant: seeds must reproduce")
    args = parser.parse_args(argv)
    return {"run": run, "score": score, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
