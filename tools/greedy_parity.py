#!/usr/bin/env python3
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Greedy parity between a speculative server and a no-spec server, token for token.

``run`` sends the first user turn of a stratified subset of the SPEED-Bench
conversations to a running ``sglang.launch_server`` through ``/generate``
(chat template applied client side, temperature 0) and saves the output token
ids plus the server's ``spec_*`` meta info. With ``--top-logprobs`` (no-spec
server only) it also saves the top-5 logprobs of every generated position, so
``compare`` can say whether a divergence is a numerical near-tie: the
candidate's token is in the reference's top 5 there and at most ``--tie-gap``
below the reference's best (the reference logits are bf16, one ulp is 0.125 for
logits in [16, 32)).

``compare`` reports, per prompt, identical / diverged at token i with the
reference's logprob gap to the candidate's token, plus the mean acceptance length.

    python tools/greedy_parity.py run --port P --data question_multiturn.jsonl \
        --tokenizer <model dir> --out nospec.json --top-logprobs
    python tools/greedy_parity.py compare nospec.json static.json
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from typing import Dict, List

import requests

logger = logging.getLogger("greedy_parity")
TOP_LOGPROBS = 5


def load_prompts(path: str, count: int) -> List[dict]:
    """``count`` conversations spread over the categories (round robin, file order)."""
    by_category: Dict[str, List[dict]] = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_category[row["category"]].append(row)
    picked: List[dict] = []
    for i in range(max(len(v) for v in by_category.values())):
        for category in sorted(by_category):
            if i < len(by_category[category]) and len(picked) < count:
                picked.append(by_category[category][i])
    return picked


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    base = f"http://127.0.0.1:{args.port}"
    results = []
    t0 = time.time()
    for row in load_prompts(args.data, args.prompts):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": row["turns"][0]},
        ]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=True, reasoning_effort="xhigh"
        )
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        payload = {
            "input_ids": list(input_ids),
            "sampling_params": {"temperature": 0, "top_k": 1, "max_new_tokens": args.max_new_tokens},
        }
        if args.top_logprobs:
            payload.update(return_logprob=True, top_logprobs_num=TOP_LOGPROBS)
        start = time.time()
        reply = requests.post(f"{base}/generate", json=payload, timeout=3600)
        if not reply.ok:
            raise RuntimeError(f"/generate {reply.status_code}: {reply.text[:2000]}")
        data = reply.json()
        meta = data["meta_info"]
        entry = {
            "question_id": row["question_id"],
            "category": row["category"],
            "prompt_tokens": len(input_ids),
            "output_ids": data["output_ids"],
            "latency": time.time() - start,
            "spec_verify_ct": meta.get("spec_verify_ct"),
            "spec_accept_length": meta.get("spec_accept_length"),
            "finish_reason": meta.get("finish_reason"),
        }
        if args.top_logprobs:
            entry["top"] = [[[lp, tok] for lp, tok, *_ in pos] for pos in meta["output_top_logprobs"]]
        results.append(entry)
        logger.info(
            "%s %s: %d tokens, accept %s",
            row["category"],
            row["question_id"][:8],
            len(entry["output_ids"]),
            entry["spec_accept_length"],
        )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"port": args.port, "wall": time.time() - t0, "results": results}, f)


def compare(args: argparse.Namespace) -> int:
    with open(args.reference, encoding="utf-8") as f:
        ref = {r["question_id"]: r for r in json.load(f)["results"]}
    with open(args.candidate, encoding="utf-8") as f:
        cand = json.load(f)["results"]
    identical, near_ties, real = 0, 0, 0
    tokens = verify = 0
    for r in cand:
        base = ref[r["question_id"]]
        a, b = base["output_ids"], r["output_ids"]
        n = min(len(a), len(b))
        first = next((i for i in range(n) if a[i] != b[i]), None)
        if r.get("spec_verify_ct"):
            tokens += len(b)
            verify += r["spec_verify_ct"]
        if first is None and len(a) == len(b):
            identical += 1
            print(f"{r['question_id'][:8]} {r['category']:<14} identical ({len(a)} tokens)")
            continue
        if first is None:
            print(f"{r['question_id'][:8]} {r['category']:<14} prefix-equal, lengths {len(a)} vs {len(b)}")
            real += 1
            continue
        gap = None
        top = base.get("top", [])
        if first < len(top):
            ranked = {tok: lp for lp, tok in top[first]}
            if b[first] in ranked:
                gap = top[first][0][0] - ranked[b[first]]
        tie = gap is not None and gap <= args.tie_gap
        near_ties += tie
        real += not tie
        print(
            f"{r['question_id'][:8]} {r['category']:<14} diverges at {first}/{n}: ref {a[first]} vs {b[first]}"
            f", ref logprob gap to it {'not in ref top-5' if gap is None else round(gap, 4)}"
            f"{' (near-tie)' if tie else ''}"
        )
    print(f"identical {identical}/{len(cand)}, near-tie divergences {near_ties}, other {real}")
    if verify:
        print(f"acceptance length (tokens / verify rounds) {tokens / verify:.3f} over {verify} rounds")
    return 0 if real == 0 else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--prompts", type=int, default=20)
    r.add_argument("--max-new-tokens", type=int, default=512)
    r.add_argument("--top-logprobs", action="store_true")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument("--tie-gap", type=float, default=0.13, help="max ref logprob gap counted as a near-tie (1 bf16 ulp)")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
