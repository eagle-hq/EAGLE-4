# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Profile bs=1 decode rounds of a running sglang server and summarize the trace.

``run``: one warm-up request, then ``/start_profile`` for ``--steps`` scheduler
steps (CPU + GPU activities by default, rank 0's trace is what ``summarize`` reads) while a
single greedy request decodes, from its start or ``--profile-after`` seconds into it.

``summarize``: from a torch profiler chrome trace, the GPU time per scheduler
step, the share of the step the GPU is busy, and the top kernels by total time
(names truncated, grouped by prefix), so the gap to a reference is attributed
to kernels vs host overhead.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import glob
import gzip
import json
import logging
import os
import time

import requests

logger = logging.getLogger("profile_decode")

PROMPT = (
    "Explain, step by step and with a worked numerical example, how a speculative decoding "
    "system with a tree-shaped draft decides which drafted tokens to accept."
)


def run(args: argparse.Namespace) -> None:
    base = f"http://127.0.0.1:{args.port}"
    payload = {"text": PROMPT, "sampling_params": {"temperature": 0, "max_new_tokens": args.max_new_tokens}}
    requests.post(f"{base}/generate", json=payload, timeout=600).raise_for_status()
    os.makedirs(args.out_dir, exist_ok=True)
    profile = {"output_dir": args.out_dir, "num_steps": args.steps, "activities": args.activities.split(",")}
    if args.profile_after > 0:
        # Profile the request's steps after ``profile_after`` seconds of decoding (a longer context).
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            pending = pool.submit(requests.post, f"{base}/generate", json=payload, timeout=600)
            time.sleep(args.profile_after)
            requests.post(f"{base}/start_profile", json=profile, timeout=60).raise_for_status()
            reply = pending.result()
    else:
        requests.post(f"{base}/start_profile", json=profile, timeout=60).raise_for_status()
        reply = requests.post(f"{base}/generate", json=payload, timeout=600)
    reply.raise_for_status()
    meta = reply.json()["meta_info"]
    logger.info("profiled request: %d tokens, %s verify rounds", meta["completion_tokens"], meta.get("spec_verify_ct"))
    deadline = time.time() + 300
    while time.time() < deadline and not glob.glob(os.path.join(args.out_dir, "*TP-0*.trace.json.gz")):
        time.sleep(5)


def _events(path: str) -> list:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)["traceEvents"]


def _kernel_group(name: str) -> str:
    return name.split("<")[0].split("(")[0][:90]


STAGES = ("draft", "verify", "draft_extend", "draft_prelaunch")


def _stage_spans(events: list, stages) -> dict:
    """CPU spans of the speculative stages (``spec_stage_span``), by name, sorted by start."""
    spans: dict = {name: [] for name in stages}
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("name") in spans
            and e.get("cat") in ("user_annotation", "cpu_op", "python_function")
        ):
            spans[e["name"]].append((e["ts"], e["ts"] + e["dur"]))
    return {name: sorted(v) for name, v in spans.items()}


def _covering(spans: list, t: float) -> bool:
    lo, hi = 0, len(spans)
    while lo < hi:  # last span starting at or before t
        mid = (lo + hi) // 2
        if spans[mid][0] <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo > 0 and spans[lo - 1][1] >= t


def summarize(args: argparse.Namespace) -> None:
    events = _events(args.trace)
    kernels = sorted(
        (e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e),
        key=lambda e: e["ts"],
    )
    if not kernels:
        raise SystemExit("no GPU kernels in the trace")
    stages = tuple(args.stages.split(","))
    spans = _stage_spans(events, stages)
    rounds = spans["draft"] or spans["verify"]
    busy = sum(k["dur"] for k in kernels)
    span = kernels[-1]["ts"] + kernels[-1]["dur"] - kernels[0]["ts"]
    n = max(len(rounds), 1)
    print(f"rounds {len(rounds)}  GPU busy {busy / 1e3:.2f} ms over {span / 1e3:.2f} ms ({100 * busy / span:.1f}%)")
    if len(rounds) > 1:
        walls = [b[0] - a[0] for a, b in zip(rounds, rounds[1:])]
        walls.sort()
        print(f"round wall mean {sum(walls) / len(walls) / 1e3:.3f} ms, median {walls[len(walls) // 2] / 1e3:.3f} ms")
    print(f"GPU time per round {busy / n / 1e3:.3f} ms, kernels per round {len(kernels) / n:.0f}")
    for name in stages:
        if spans[name]:
            host = sum(b - a for a, b in spans[name])
            gpu = sum(k["dur"] for k in kernels if _covering(spans[name], k["ts"]))
            print(
                f"  {name:13s} host {host / n / 1e3:7.3f} ms/round, GPU kernels running inside {gpu / n / 1e3:7.3f} ms/round"
            )
    # GPU idle gaps, attributed to the stage whose host span covers the gap's start.
    idle: dict = collections.defaultdict(float)
    end = kernels[0]["ts"] + kernels[0]["dur"]
    for k in kernels[1:]:
        gap = k["ts"] - end
        if gap > 0:
            owner = next((name for name in stages if _covering(spans[name], end)), "other (scheduler)")
            idle[owner] += gap
        end = max(end, k["ts"] + k["dur"])
    print("GPU idle per round, by the stage the host is in:")
    for name, t in sorted(idle.items(), key=lambda kv: -kv[1]):
        print(f"  {name:18s} {t / n / 1e3:7.3f} ms")
    totals: dict = collections.defaultdict(lambda: [0.0, 0])
    for k in kernels:
        t = totals[_kernel_group(k["name"])]
        t[0] += k["dur"]
        t[1] += 1
    print(f"{'us/round':>9} {'calls/rnd':>10} {'share':>6}  kernel")
    for name, (dur, calls) in sorted(totals.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"{dur / n:9.1f} {calls / n:10.1f} {100 * dur / busy:5.1f}%  {name}")
    if args.host_ops:
        ops: dict = collections.defaultdict(lambda: [0.0, 0])
        for e in events:
            if e.get("ph") == "X" and e.get("cat") == "cpu_op" and any(_covering(spans[s], e["ts"]) for s in stages):
                o = ops[e["name"]]
                o[0] += e["dur"]
                o[1] += 1
        print(f"{'host us/rnd':>11} {'calls/rnd':>10}  cpu op inside the stages (inclusive time)")
        for name, (dur, calls) in sorted(ops.items(), key=lambda kv: -kv[1][0])[: args.host_ops]:
            print(f"{dur / n:11.1f} {calls / n:10.1f}  {name[:90]}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--steps", type=int, default=40)
    r.add_argument("--max-new-tokens", type=int, default=1024)
    r.add_argument(
        "--activities", default="CPU,GPU", help="profiler activities; GPU alone keeps the host timing unperturbed"
    )
    r.add_argument(
        "--profile-after", type=float, default=0.0, help="seconds into the profiled request to start the profile"
    )
    r.set_defaults(fn=run)
    s = sub.add_parser("summarize")
    s.add_argument("trace")
    s.add_argument("--top", type=int, default=30)
    s.add_argument("--host-ops", type=int, default=25, help="top host ops inside the stages (0: off)")
    s.add_argument(
        "--stages", default=",".join(STAGES), help="stage span names, comma separated (the first counts rounds)"
    )
    s.set_defaults(fn=summarize)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
