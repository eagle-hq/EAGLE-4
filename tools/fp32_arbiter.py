#!/usr/bin/env python3
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""fp32 arbiter for greedy-parity divergences above the near-tie gap.

``greedy_parity.py compare`` calls a first divergence a near tie when the bf16 reference's logprob gap to the
candidate's token is at most 0.13 (one bf16 ulp for logits in [16, 32)). Every divergence above that is decided
here: the target is re-forwarded in float32 on the exact common prefix (same weights, the unmodified HF model,
eager attention, no fused kernels), ``d = |logit_fp32(reference token) - logit_fp32(candidate token)|`` is taken,
and with ``ulp = 2 ** (floor(log2(max(|a|, |b|))) - 7)`` (bf16 spacing at that magnitude) the divergence
**passes** when ``d <= 2 ulp`` (``fp32_pass``) or when the candidate's own token is the fp32 argmax at that prefix
(``fp32_argmax``: the bf16 reference is the one that is off), and **fails** otherwise. Exact bf16 ties fall under
the same rule. Each divergence is reported as identical / near_tie / fp32_pass / fp32_argmax / fail with the fp32
logits and token ids.

    python tools/fp32_arbiter.py from-parity --model <model dir> --data question_multiturn.jsonl \\
        --reference nospec.json --candidate live.json [static.json ...] --out arbiter.json
    python tools/fp32_arbiter.py prefix --model <model dir> --prefix-ids ids.json --tokens 4087 310
    python tools/fp32_arbiter.py rejudge arbiter.json --out arbiter.rejudged.json

``rejudge`` re-decides a saved report under the current rule from the fp32 logits it recorded, without a model,
and writes a new file next to the original evidence.

The scorer (an fp32 forward at the last position of a prefix) is the only part that needs a GPU and the model;
the rule and the parity classification are plain functions, tested on CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

TIE_GAP = 0.13
TOLERANCE_ULPS = 2.0
STATUSES = ("identical", "near_tie", "fp32_pass", "fp32_argmax", "fail", "length")

# fp32 logits of the requested tokens at the position after ``prefix_ids``, plus the top-k (value, token) pairs.
Scorer = Callable[[Sequence[int], Sequence[int]], "Scores"]


@dataclass
class Scores:
    logits: Dict[int, float]
    top: List[List[float]]  # [[logit, token], ...] best first
    seconds: float = 0.0


@dataclass
class Verdict:
    reference_token: int
    candidate_token: int
    reference_logit: float
    candidate_logit: float
    d: float
    ulp: float
    tolerance_ulps: float
    fp32_argmax: int
    status: str  # fp32_pass, fp32_argmax or fail
    passes: bool


def bf16_ulp(magnitude: float) -> float:
    """bf16 spacing at ``magnitude``: 2 ** (floor(log2(magnitude)) - 7); 0.125 on [16, 32)."""
    if magnitude <= 0.0:
        return math.ldexp(1.0, -133)  # smallest bf16 subnormal: a tie at 0 is still a tie
    mantissa, exponent = math.frexp(magnitude)  # magnitude = mantissa * 2 ** exponent, mantissa in [0.5, 1)
    return math.ldexp(1.0, exponent - 8)


def arbitrate(
    reference_token: int, candidate_token: int, scores: Scores, tolerance_ulps: float = TOLERANCE_ULPS
) -> Verdict:
    """Within ``tolerance_ulps`` of each other passes; beyond it, the candidate passes only as the fp32 argmax."""
    a, b = scores.logits[reference_token], scores.logits[candidate_token]
    d = abs(a - b)
    ulp = bf16_ulp(max(abs(a), abs(b)))
    best_logit, best_token = scores.top[0]
    if d <= tolerance_ulps * ulp:
        status = "fp32_pass"
    elif b >= best_logit:  # a tie at the top still makes the candidate an fp32 argmax
        status = "fp32_argmax"
    else:
        status = "fail"
    return Verdict(
        reference_token, candidate_token, a, b, d, ulp, tolerance_ulps, int(best_token), status, status != "fail"
    )


def first_divergence(reference: dict, candidate: dict) -> Optional[int]:
    a, b = reference["output_ids"], candidate["output_ids"]
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)


def reference_gap(reference: dict, index: int, token: int) -> Optional[float]:
    """The bf16 reference's logprob gap from its best to ``token`` at ``index`` (None when not in its saved top-k)."""
    top = reference.get("top", [])
    if index >= len(top):
        return None
    ranked = {tok: lp for lp, tok in top[index]}
    return top[index][0][0] - ranked[token] if token in ranked else None


def classify(
    reference: dict,
    candidate: dict,
    prompt_ids: Sequence[int],
    score: Scorer,
    tie_gap: float = TIE_GAP,
    tolerance_ulps: float = TOLERANCE_ULPS,
) -> dict:
    """One prompt's parity verdict (one of ``STATUSES``), with the fp32 evidence when scored."""
    a, b = reference["output_ids"], candidate["output_ids"]
    index = first_divergence(reference, candidate)
    entry = {"question_id": candidate["question_id"], "index": index, "status": None}
    if index is None:
        entry["status"] = "identical" if len(a) == len(b) else "length"
        entry["lengths"] = [len(a), len(b)]
        return entry
    ref_token, cand_token = a[index], b[index]
    gap = reference_gap(reference, index, cand_token)
    entry.update(reference_token=ref_token, candidate_token=cand_token, reference_gap=gap)
    if gap is not None and gap <= tie_gap:
        entry["status"] = "near_tie"
        return entry
    prefix = list(prompt_ids) + list(a[:index])
    scores = score(prefix, [ref_token, cand_token])
    verdict = arbitrate(ref_token, cand_token, scores, tolerance_ulps)
    entry.update(
        status=verdict.status,
        prefix_length=len(prefix),
        fp32=asdict(verdict),
        fp32_top=scores.top,
        fp32_argmax=verdict.fp32_argmax,
        seconds=scores.seconds,
    )
    return entry


def summarize(rows: List[dict]) -> dict:
    """Counts per status; ``passes`` iff no row is ``fail`` or ``length``."""
    counts = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    return {"counts": counts, "passes": counts["fail"] == 0 and counts["length"] == 0, "rows": rows}


def classify_parity(
    reference_doc: dict,
    candidate_doc: dict,
    prompts: Mapping[str, Sequence[int]],
    score: Scorer,
    tie_gap: float = TIE_GAP,
    tolerance_ulps: float = TOLERANCE_ULPS,
) -> dict:
    """Every prompt of a ``greedy_parity.py run`` pair; ``passes`` iff no prompt is ``fail`` or ``length``."""
    reference = {r["question_id"]: r for r in reference_doc["results"]}
    rows = [
        classify(reference[c["question_id"]], c, prompts[c["question_id"]], score, tie_gap, tolerance_ulps)
        for c in candidate_doc["results"]
    ]
    return {"tie_gap": tie_gap, "tolerance_ulps": tolerance_ulps, **summarize(rows)}


def rejudge_row(row: dict) -> dict:
    """``row`` decided again by ``arbitrate`` from the fp32 logits it saved (no re-scoring); unscored rows as they are."""
    if "fp32" not in row:
        return dict(row)
    saved, ref_token, cand_token = row["fp32"], row["reference_token"], row["candidate_token"]
    scores = Scores({ref_token: saved["reference_logit"], cand_token: saved["candidate_logit"]}, row["fp32_top"])
    verdict = arbitrate(ref_token, cand_token, scores, saved["tolerance_ulps"])
    return {**row, "status": verdict.status, "fp32": asdict(verdict), "fp32_argmax": verdict.fp32_argmax}


def rejudge(report: dict) -> dict:
    """A saved report (``from-parity``'s ``candidates`` or a list of ``verdicts``) under the current rule, as a copy."""
    if "candidates" not in report and "verdicts" not in report:
        raise ValueError("not an fp32 arbiter report: no 'candidates' or 'verdicts'")

    def again(result: dict) -> dict:
        return {**result, **summarize([rejudge_row(r) for r in result["rows"]])}

    out = dict(report)
    if "candidates" in report:
        out["candidates"] = {name: again(result) for name, result in report["candidates"].items()}
    if "verdicts" in report:
        out["verdicts"] = [again(result) for result in report["verdicts"]]
    return out


def format_row(row: dict) -> str:
    head = f"{row['question_id'][:8]} {row['status']:<11}"
    if row["index"] is None:
        return f"{head} lengths {row.get('lengths')}"
    line = f"{head} at {row['index']}: ref {row['reference_token']} vs {row['candidate_token']}, ref bf16 gap {row['reference_gap']}"
    if "fp32" in row:
        v = row["fp32"]
        line += (
            f"; fp32 {v['reference_logit']:.4f} vs {v['candidate_logit']:.4f}, d {v['d']:.4f}, ulp {v['ulp']}"
            f" ({v['d'] / v['ulp']:.2f} ulp, limit {v['tolerance_ulps']}), fp32 argmax {row['fp32_argmax']}"
        )
    return line


def load_reference_model(model_path: str, device: str = "cuda"):
    """The unmodified HF model in float32 with eager attention: the reference, not a serving path."""
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    cls = getattr(transformers, config.architectures[0], transformers.AutoModelForCausalLM)
    started = time.monotonic()
    # No device_map: it needs `accelerate`, which the serving venvs do not carry; the fp32 27B fits one B300.
    model = cls.from_pretrained(model_path, dtype=torch.float32, attn_implementation="eager").to(device).eval()
    dtypes = {str(p.dtype) for p in model.parameters()}
    assert dtypes == {"torch.float32"}, dtypes
    module = sys.modules[type(model).__module__]
    fast = getattr(module, "is_fast_path_available", False)
    assert not fast, "the reference must run the torch fallback kernels, not the fused fast path"
    return model, time.monotonic() - started


def hf_scorer(model, device: str = "cuda", topk: int = 10) -> Scorer:
    import torch

    def score(prefix_ids: Sequence[int], tokens: Sequence[int]) -> Scores:
        ids = torch.tensor([list(prefix_ids)], dtype=torch.long, device=device)
        started = time.monotonic()
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1].double()
        values, indices = torch.topk(logits, topk)
        return Scores(
            {int(t): logits[t].item() for t in tokens},
            [[v, int(i)] for v, i in zip(values.tolist(), indices.tolist())],
            time.monotonic() - started,
        )

    return score


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("from-parity", help="arbitrate every first divergence of greedy_parity run outputs")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="question_multiturn.jsonl the parity runs used")
    p.add_argument("--prompts", type=int, default=20, help="prompt count the parity runs used")
    p.add_argument("--reference", required=True, help="no-spec parity json with --top-logprobs")
    p.add_argument("--candidate", nargs="+", required=True)
    p.add_argument("--tie-gap", type=float, default=TIE_GAP)
    p.add_argument("--tolerance-ulps", type=float, default=TOLERANCE_ULPS)
    p.add_argument("--out", required=True)
    q = sub.add_parser("prefix", help="score two tokens after one explicit prefix")
    q.add_argument("--model", required=True)
    q.add_argument(
        "--prefix-ids", required=True, help="json file with the full token id list (prompt + accepted tokens)"
    )
    q.add_argument("--tokens", type=int, nargs=2, required=True, metavar=("REFERENCE", "CANDIDATE"))
    q.add_argument("--tolerance-ulps", type=float, default=TOLERANCE_ULPS)
    q.add_argument("--out")
    r = sub.add_parser("rejudge", help="re-decide a saved report from its fp32 logits (no model)")
    r.add_argument("report")
    r.add_argument("--out", required=True, help="a new file: the saved report stays the raw evidence")
    args = ap.parse_args(argv)

    if args.command == "rejudge":
        if Path(args.out).resolve() == Path(args.report).resolve():
            ap.error("--out must not overwrite the saved report")
        report = rejudge(json.loads(Path(args.report).read_text()))
        report["rejudged_from"] = args.report
        results = list(report.get("candidates", {}).items())
        results += [(v.get("candidate", str(i)), v) for i, v in enumerate(report.get("verdicts", []))]
        for name, result in results:
            print(f"== {name}")
            for row in result["rows"]:
                if row["status"] not in ("identical", "near_tie"):
                    print(format_row(row))
            print(f"{'PASS' if result['passes'] else 'FAIL'} {json.dumps(result['counts'])}")
        Path(args.out).write_text(json.dumps(report, indent=1))
        return 0 if all(result["passes"] for _, result in results) else 1

    model, load_seconds = load_reference_model(args.model)
    score = hf_scorer(model)
    if args.command == "prefix":
        prefix = json.loads(Path(args.prefix_ids).read_text())
        scores = score(prefix, args.tokens)
        verdict = arbitrate(args.tokens[0], args.tokens[1], scores, args.tolerance_ulps)
        report = {"model": args.model, "prefix_length": len(prefix), "fp32": asdict(verdict), "fp32_top": scores.top}
        print(json.dumps(report["fp32"]))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1))
        return 0 if verdict.passes else 1

    from transformers import AutoTokenizer

    # Imported here: greedy_parity is the HTTP client (it needs requests); the rule and rejudge do not.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from greedy_parity import encode_prompt, load_prompts

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = {row["question_id"]: encode_prompt(tokenizer, row) for row in load_prompts(args.data, args.prompts)}
    reference_doc = json.loads(Path(args.reference).read_text())
    report = {"model": args.model, "load_seconds": load_seconds, "reference": args.reference, "candidates": {}}
    passes = True
    for path in args.candidate:
        result = classify_parity(
            reference_doc, json.loads(Path(path).read_text()), prompts, score, args.tie_gap, args.tolerance_ulps
        )
        report["candidates"][path] = result
        print(f"== {path}")
        for row in result["rows"]:
            print(format_row(row))
        print(f"{'PASS' if result['passes'] else 'FAIL'} {json.dumps(result['counts'])}", flush=True)
        passes &= result["passes"]
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())
