# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Chain calibration on sglang from served rounds: forced widths while real requests run, then the widths per batch size.

usage: python -m eagle_harness.sglang.calibrate measure --port P --data question_multiturn.jsonl --tokenizer MODEL
           [--concurrency 1,8,64,256] [--max-new-tokens 512]
       python -m eagle_harness.sglang.calibrate report <data_path> [--max-width 8] [--max-batch-size 256]

``measure`` drives a running server started in the calibration mode::

    python -m sglang.launch_server ... --speculative-algorithm EAGLE_HARNESS --speculative-eagle-topk 1 \\
        --speculative-num-steps 7 --speculative-num-draft-tokens 8 --max-running-requests 256 \\
        --speculative-algorithm-config '{"mode": "calibrate", "intensity": "chain_intensity", "data_path": "<dir>"}'

The server forces every chain width up to ``--speculative-num-draft-tokens`` in turn, then native decode (width 0),
and records the
period of each decode round per batch size and width, plus the accept lengths of the widest rounds
(:mod:`eagle_harness.sglang.chain_rounds`). ``measure`` sends, per concurrency level, one batch of that many greedy
requests (SPEED-Bench first turns, ``ignore_eos`` so the batch stays full) and then one single-token request, whose
prefill makes the server write what it recorded: the raw rounds, ``<data_path>/rounds/chain.json`` and
``accept/chain.json``. A serve-mode chain server with
that ``data_path`` reads them. ``report`` prints the round times per batch size and width, the acceptance profile and
the widths per batch size that server will use (:func:`eagle_harness.core.chain_width.chain_widths`).

After native decode begins, requests stay OFF until completion because their draft KV is stale. A prefill resets
the calibration sequence for fresh requests. The runtime requires DP1 and disabled Mamba prefix-cache tracking.

These are served-path timings on purpose: the period of a served round holds the scheduler, the draft extend and the
launches, which isolated forward timings miss.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Sequence

from eagle_harness.core.chain_width import chain_widths, load_accept_profile
from eagle_harness.sglang.chain_rounds import MIN_ROUNDS, RAW_FILE, round_cells


def first_turns(path: str) -> list[str]:
    """The first user turns of a SPEED-Bench file, round robin over its categories."""
    by_category: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_category[row["category"]].append(row["turns"][0])
    turns: list[str] = []
    for i in range(max(len(v) for v in by_category.values())):
        turns += [by_category[c][i] for c in sorted(by_category) if i < len(by_category[c])]
    return turns


def measure(port: int, data: str, tokenizer_path: str, levels: Sequence[int], max_new_tokens: int) -> None:
    """One batch of ``c`` concurrent greedy requests per level, each followed by a flushing prefill."""
    import requests
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    turns = first_turns(data)
    base = f"http://127.0.0.1:{port}"

    def prompt(i: int) -> list[int]:
        messages = [{"role": "user", "content": turns[i % len(turns)]}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return tokenizer.encode(text, add_special_tokens=False)

    def generate(input_ids: list[int], new_tokens: int) -> int:
        params = {"temperature": 0, "max_new_tokens": new_tokens, "ignore_eos": True}
        reply = requests.post(
            f"{base}/generate", json={"input_ids": input_ids, "sampling_params": params}, timeout=7200
        )
        reply.raise_for_status()
        return len(reply.json()["output_ids"])

    for c in levels:
        inputs = [prompt(i) for i in range(c)]
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(c) as pool:
            tokens = sum(pool.map(lambda ids: generate(ids, max_new_tokens), inputs))
        wall = time.time() - start
        generate(prompt(0), 1)  # its prefill writes the rounds recorded so far
        print(f"C{c}: {tokens} tokens in {wall:.1f} s ({tokens / wall:.0f} tok/s, widths mixed)", flush=True)


def report(data_path: str, max_width: int, max_batch_size: int) -> None:
    """Print the round times, the acceptance profile and the widths per batch size a serve-mode server will use."""
    with open(os.path.join(data_path, RAW_FILE), encoding="utf-8") as f:
        raw = json.load(f)
    cells = round_cells(raw["rounds"], raw["widths"])
    profile = load_accept_profile(data_path)
    if not cells or profile is None:
        raise SystemExit(f"no batch size timed every width {raw['widths']} at least {MIN_ROUNDS} times, or no profile")
    print(f"acceptance at width {raw['max_width']}: " + " ".join(f"{r:.3f}" for r in profile.rates))
    print("batch size: width median round ms (rounds) tokens per request per s")
    for bs, row in cells.items():
        parts = [
            f"w{w} {us / 1000:.2f} ({n}) {profile.expected_tokens(max(1, w)) / us * 1e6:.0f}"
            for w, (us, n) in row.items()
        ]
        print(f"  bs {bs}: " + ", ".join(parts))
    print(f"chain widths per batch size: {chain_widths(data_path, max_width, max_batch_size, allow_no_spec=True)}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Chain calibration on sglang from served rounds.")
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--port", type=int, required=True)
    m.add_argument("--data", required=True, help="SPEED-Bench question_multiturn.jsonl")
    m.add_argument("--tokenizer", required=True)
    m.add_argument("--concurrency", default="1,8,64,256", help="comma-separated concurrency levels")
    m.add_argument("--max-new-tokens", type=int, default=512)
    r = sub.add_parser("report")
    r.add_argument("data_path")
    r.add_argument("--max-width", type=int, default=8, help="the server's --speculative-num-draft-tokens")
    r.add_argument("--max-batch-size", type=int, default=256, help="the server's --max-running-requests")
    args = parser.parse_args(argv)
    if args.command == "measure":
        levels = [int(x) for x in args.concurrency.split(",")]
        measure(args.port, args.data, args.tokenizer, levels, args.max_new_tokens)
    else:
        report(args.data_path, args.max_width, args.max_batch_size)


if __name__ == "__main__":
    main(sys.argv[1:])
