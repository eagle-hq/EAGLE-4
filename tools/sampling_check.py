#!/usr/bin/env python3
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Lossless sampling under speculation: a speculative server must sample what the no-spec server samples.

``run`` sends ``--samples`` short completions of each of a few open-ended prompts to a running server under each
sampling setting (temperature, top-k with top-p, min-p, a repetition penalty), ``--concurrency`` requests at a
time so they share decode rounds, and saves the output token ids. The first token comes from the prefill sampler;
every later one from a speculative verify round.

``compare`` tests, per setting and prompt, that the speculative server's tokens come from the same distribution as
the reference's: a two-sample chi-square test of homogeneity on the joint distribution of the first three tokens and
on the marginal of every position (cells seen fewer than 10 times in both runs together are pooled), at a
Bonferroni-corrected level ``--alpha / tests``. It also prints the total variation distance of each test. A server
that ignores a sampling parameter (for example, verifies greedily at temperature > 0) fails it by a wide margin.
The first token comes from the prefill, so a speculative server whose prefill runs other attention kernels can
differ there where a top-k or top-p cut falls between near-tied tokens; tokens 2 and 3 are therefore also tested
given the reference's most common prefix ("verify draws given the modal prefix"), which isolates the verify rounds.
Run the no-spec server twice to see the engine's own run-to-run level.

With ``--seeds S`` (``run``) every one of S seeds is sent twice, in batches of a different composition, at
temperature 0.7 for ``--seed-tokens`` tokens; ``compare`` (and ``run`` itself) report how many seeds reproduced
their text, and ``compare`` how many seeded texts equal the reference server's for the same seed. sglang honors
seeds only with ``--enable-deterministic-inference``.

With ``--speed-data`` (a SPEED-Bench jsonl) ``run`` also decodes the first user turn of ``--speed-prompts``
conversations one at a time (batch size 1, chat template, thinking off) at temperature 0 and at temperature 1, and
``compare`` prints both decode rates against the reference's and how many greedy outputs equal the reference's.

    python tools/sampling_check.py run --api vllm --port P --tokenizer <model dir> --out nospec.json --seeds 32
    python tools/sampling_check.py run --api vllm --port P --tokenizer <model dir> --out spec.json --seeds 32
    python tools/sampling_check.py compare nospec.json spec.json

Exit code 1 (``compare``) when a test rejects at the corrected level, or a seed did not reproduce.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import math
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

PROMPTS = (
    "Here is a list of ten random animals:",
    "Once upon a time, in a",
    "My favorite word is",
    "The weather today is",
)
SETTINGS: Dict[str, dict] = {
    "unrestricted": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
    "temperature": {"temperature": 0.7},
    "top_k_top_p": {"temperature": 1.0, "top_k": 8, "top_p": 0.7},
    "min_p": {"temperature": 1.0, "min_p": 0.05},
    "penalty": {"temperature": 0.8, "repetition_penalty": 1.5},
}
SEED_SETTING = {"temperature": 0.7}
JOINT = 3  # tokens in the joint test
POOL = 10  # cells seen fewer times than this in both runs together are pooled


def generate(api: str, base: str, model: Optional[str], prompt: List[int], params: dict, max_new: int, seed=None):
    """One completion's output token ids (and sglang's ``spec_*`` meta info)."""
    import requests  # only the HTTP client needs it; compare runs in test envs without it
    if api == "sglang":
        sp = {"max_new_tokens": max_new, "ignore_eos": True, **params}
        if seed is not None:
            sp["sampling_seed"] = seed
        r = requests.post(f"{base}/generate", json={"input_ids": prompt, "sampling_params": sp}, timeout=3600)
        r.raise_for_status()
        out = r.json()
        return out["output_ids"], {k: v for k, v in out["meta_info"].items() if k.startswith("spec_")}
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
    model = requests.get(f"{base}/v1/models", timeout=60).json()["data"][0]["id"] if args.api == "vllm" else None
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


def validate_pair(ref: dict, cand: dict) -> None:
    """Reject partial evidence before a statistical test can silently omit it."""
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
    tests = []  # (setting, prompt, what, p, tv, cells)
    for name, entry in cand["settings"].items():
        for i, (r_tokens, c_tokens) in enumerate(zip(ref["settings"][name]["tokens"], entry["tokens"])):
            k = min(len(r_tokens[0]), len(c_tokens[0]))
            r_joint = [tuple(t[:JOINT]) for t in r_tokens]
            c_joint = [tuple(t[:JOINT]) for t in c_tokens]
            tests.append((name, i, f"joint 1-{JOINT}", *homogeneity(r_joint, c_joint)))
            for pos in range(k):
                tests.append(
                    (name, i, f"token {pos + 1}", *homogeneity([t[pos] for t in r_tokens], [t[pos] for t in c_tokens]))
                )
            # Tokens 2 and 3 given the reference's most common prefix: the verify round's own draws, free of any
            # difference in the prefill's sampling (other attention kernels move near-tied top-k / top-p cuts).
            for pos in (1, 2):
                prefix = collections.Counter(tuple(t[:pos]) for t in r_tokens).most_common(1)[0][0]
                r_next = [t[pos] for t in r_tokens if tuple(t[:pos]) == prefix]
                c_next = [t[pos] for t in c_tokens if tuple(t[:pos]) == prefix]
                if r_next and c_next:
                    tests.append((name, i, f"token {pos + 1} | modal prefix", *homogeneity(r_next, c_next)))
    level = args.alpha / max(1, len(tests))
    failed = False
    for name in cand["settings"]:
        rows = [t for t in tests if t[0] == name]
        if not rows:
            continue
        worst = min(rows, key=lambda t: t[3])
        joint_tv = max(t[4] for t in rows if t[2].startswith("joint"))
        bad = [t for t in rows if t[3] < level]
        conditional = [t for t in rows if "modal prefix" in t[2]]
        cond_bad = [t for t in conditional if t[3] < level]
        failed |= bool(bad)
        extra = cand["settings"][name].get("mean_tokens_per_verify") or cand["settings"][name].get("spec_counters", "")
        print(
            f"{name:12s} {'FAIL' if bad else 'PASS'}: {len(rows)} tests, {len(bad)} below {level:.1e}; min p "
            f"{worst[3]:.2e} ({worst[2]}, prompt {worst[1]}, TV {worst[4]:.3f}); max joint TV {joint_tv:.3f}; "
            f"{cand['settings'][name]['seconds']} s (ref {ref['settings'][name]['seconds']} s) {extra}"
        )
        cond_worst = min(conditional, key=lambda t: t[3]) if conditional else None
        if cond_worst:
            print(
                f"    verify draws given the modal prefix: {'FAIL' if cond_bad else 'PASS'}, {len(cond_bad)} of "
                f"{len(conditional)} below; min p {cond_worst[3]:.2e} ({cond_worst[2]}, prompt {cond_worst[1]})"
            )
        for t in sorted(bad, key=lambda t: t[3])[:5]:
            print(f"    {t[2]}, prompt {t[1]}: p {t[3]:.2e}, TV {t[4]:.3f}, {t[5]} cells")
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
        print(
            f"seeds: {reproduced}/{len(seeds)} reproduced their text; {same_as_ref}/{len(seeds)} equal the "
            f"reference's text for the same seed (mean common prefix {sum(prefix) / max(1, len(prefix)):.1f} tokens)"
        )
        failed |= reproduced != len(seeds)
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--api", choices=("sglang", "vllm"), required=True)
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
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument("--alpha", type=float, default=0.01, help="family-wise level, Bonferroni-corrected")
    args = parser.parse_args(argv)
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
