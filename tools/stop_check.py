#!/usr/bin/env python3
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Per-request stop conditions under batched speculation: one batch mixes every kind of stop.

Sends ``--concurrency`` greedy requests at once to a running server, so they share decode rounds, cycling
four kinds of request:

- ``ignore_eos``: must run to ``--max-new-tokens`` (an EOS or another request's stop must not end it);
- ``stop_token``: ``stop_token_ids=[<"\\n\\n">]``; the stop token may only be the last output token;
- ``stop_string``: ``stop=[" and"]``; the text may not go on past the stop string (kept in the output);
- ``eos``: neither; an EOS may only be the last output token.

Every kind must also end for its own reason only: a stop-token finish must end on the request's own stop
token or EOS, a request without stop tokens must not end on one. A speculative round accepts several
tokens per request, so an engine that checked one request's stop set for the whole batch, or checked only
the last accepted token, fails here. Greedy, thinking off, first user turns of ``--data``.

The checker is checked by the engine's no-spec server (0 violations). With ``--reference`` (the no-spec
server's output file) it also counts the rows whose tokens equal the reference's.

    python tools/stop_check.py --api sglang --port P --data question_multiturn.jsonl \\
        --tokenizer <model dir> --out stop_nospec.json
    python tools/stop_check.py --api vllm --port P ... --reference stop_nospec.json --out stop_spec.json

``--api tokenspeed`` is sglang's ``/generate`` with the model name TokenSpeed's gateway routes by.

Exit code 1 when any request violates its stop condition or fails.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import requests

KINDS = ("ignore_eos", "stop_token", "stop_string", "eos")
STOP_STRING = " and"
STOP_TOKEN_TEXT = "\n\n"


def first_turns(path: str, count: int) -> List[str]:
    """The first user turn of the first ``count`` conversations."""
    turns = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            turns.append(json.loads(line)["turns"][0])
            if len(turns) == count:
                break
    return turns


def eos_ids(tokenizer, model_dir: str) -> Set[int]:
    """The tokenizer's EOS id and the generation config's EOS ids (what the engines stop on by default)."""
    ids = {tokenizer.eos_token_id}
    try:
        with open(f"{model_dir}/generation_config.json", encoding="utf-8") as f:
            configured = json.load(f).get("eos_token_id")
    except FileNotFoundError:
        configured = None
    if configured is not None:
        ids.update(configured if isinstance(configured, list) else [configured])
    ids.discard(None)
    return ids


def violation(
    kind: str, tokens: Sequence[int], text: str, max_new: int, stop_token: int, eos: Set[int], last_chars: int
) -> Optional[str]:
    """Why a finished request broke its stop condition, or None. ``last_chars``: the length of the last token's
    text (a stop string can end inside it, and the engines keep the whole token)."""
    if not tokens:
        return "no tokens"
    inner, last = tokens[:-1], tokens[-1]
    if kind == "ignore_eos":
        return None if len(tokens) == max_new else f"{len(tokens)} of {max_new} tokens"
    ends_by_length = len(tokens) == max_new
    allowed_last = set(eos) | ({stop_token} if kind == "stop_token" else set())
    if kind == "stop_token" and stop_token in inner:
        return f"stop token at {inner.index(stop_token)} of {len(tokens)}"
    if kind == "stop_string":
        at = text.find(STOP_STRING)
        past = len(text) - at - len(STOP_STRING)
        if at >= 0 and past >= last_chars:
            return f"text goes on {past} chars past the stop string"
        if at >= 0:
            return None  # ended on the stop string
    hit = [i for i, t in enumerate(inner) if t in eos]
    if hit:
        return f"EOS at {hit[0]} of {len(tokens)}"
    if not ends_by_length and last not in allowed_last:
        return f"ended early on token {last}, not one of its stops"
    return None


def generate(api: str, base: str, model: Optional[str], prompt: List[int], kind: str, args, stop_token: int):
    """One greedy request; returns (output token ids, text, finish reason)."""
    stop_ids = [stop_token] if kind == "stop_token" else []
    stop = [STOP_STRING] if kind == "stop_string" else []
    if api in ("sglang", "tokenspeed"):
        params = {
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": kind == "ignore_eos",
            "no_stop_trim": True,
            "skip_special_tokens": False,
            "stop_token_ids": stop_ids,
            "stop": stop,
        }
        body = {"input_ids": prompt, "sampling_params": params}
        if model is not None:  # TokenSpeed's gateway routes /generate by model name
            body["model"] = model
        r = requests.post(f"{base}/generate", json=body, timeout=3600)
        r.raise_for_status()
        out = r.json()
        if isinstance(out, list):  # TokenSpeed's gateway wraps the one result in a list
            (out,) = out
        return out["output_ids"], out["text"], out["meta_info"].get("finish_reason")
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": args.max_new_tokens,
        "temperature": 0,
        "ignore_eos": kind == "ignore_eos",
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
        "return_token_ids": True,
        "stop_token_ids": stop_ids,
        "stop": stop,
    }
    r = requests.post(f"{base}/v1/completions", json=body, timeout=3600)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    return choice["token_ids"], choice["text"], {"type": choice["finish_reason"], "matched": choice.get("stop_reason")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Per-request stop conditions under batched speculation.")
    parser.add_argument("--api", choices=("sglang", "vllm", "tokenspeed"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--data", required=True, help="SPEED-Bench question_multiturn.jsonl (first turns are used)")
    parser.add_argument("--tokenizer", required=True, help="the target model directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, default=16, help="requests in flight together (one batch)")
    parser.add_argument("--waves", type=int, default=2, help="batches sent one after the other")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--reference", help="a no-spec run's --out: count rows with the same tokens")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stop_token = tokenizer.encode(STOP_TOKEN_TEXT, add_special_tokens=False)
    if len(stop_token) != 1:
        raise SystemExit(f"{STOP_TOKEN_TEXT!r} is not one token in this tokenizer: {stop_token}")
    stop_token = stop_token[0]
    eos = eos_ids(tokenizer, args.tokenizer)
    base = f"http://127.0.0.1:{args.port}"
    model = requests.get(f"{base}/v1/models", timeout=60).json()["data"][0]["id"] if args.api != "sglang" else None
    turns = first_turns(args.data, args.concurrency * args.waves)
    prompts = [
        tokenizer.encode(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": turn}], add_generation_prompt=True, tokenize=False, enable_thinking=False
            ),
            add_special_tokens=False,
        )
        for turn in turns
    ]

    rows: List[dict] = []
    t0 = time.time()
    for wave in range(args.waves):
        jobs: List[Tuple[int, str]] = [
            (wave * args.concurrency + i, KINDS[(i + wave) % len(KINDS)]) for i in range(args.concurrency)
        ]
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            futures = {
                pool.submit(generate, args.api, base, model, prompts[i], kind, args, stop_token): (i, kind)
                for i, kind in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                i, kind = futures[future]
                row: Dict = {"index": i, "wave": wave, "kind": kind}
                try:
                    tokens, text, finish = future.result()
                    row.update(
                        tokens=tokens,
                        text=text,
                        finish=finish,
                        violation=violation(
                            kind,
                            tokens,
                            text,
                            args.max_new_tokens,
                            stop_token,
                            eos,
                            len(tokenizer.decode(tokens[-1:])),
                        ),
                    )
                except Exception as e:  # noqa: BLE001 -- a failed request is a result
                    row["error"] = f"{type(e).__name__}: {e}"
                rows.append(row)
    rows.sort(key=lambda r: r["index"])

    bad = [r for r in rows if r.get("violation") or r.get("error")]
    summary: Dict[str, object] = {
        kind: {
            "requests": sum(r["kind"] == kind for r in rows),
            "violations": sum(r["kind"] == kind and bool(r.get("violation")) for r in rows),
            "errors": sum(r["kind"] == kind and "error" in r for r in rows),
        }
        for kind in KINDS
    }
    if args.reference:
        with open(args.reference, encoding="utf-8") as f:
            reference = {r["index"]: r.get("tokens") for r in json.load(f)["rows"]}
        summary["same_tokens_as_reference"] = sum(reference.get(r["index"]) == r.get("tokens") for r in rows)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {"api": args.api, "args": vars(args), "wall_s": time.time() - t0, "summary": summary, "rows": rows}, f
        )
    for kind in KINDS:
        print(f"{kind:<12} {summary[kind]}")
    if args.reference:
        print(f"same tokens as the reference: {summary['same_tokens_as_reference']} of {len(rows)}")
    for r in bad:
        print(f"  BAD #{r['index']} {r['kind']}: {r.get('violation') or r.get('error')} finish={r.get('finish')}")
    print(f"violations+errors {len(bad)} of {len(rows)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
