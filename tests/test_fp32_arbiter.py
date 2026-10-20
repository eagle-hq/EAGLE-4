# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""fp32 arbiter: the bf16-ulp rule and the parity classification, without a model (the scorer is injected)."""

import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "fp32_arbiter.py"
_spec = importlib.util.spec_from_file_location("fp32_arbiter", _TOOL)
assert _spec is not None and _spec.loader is not None
arbiter = importlib.util.module_from_spec(_spec)
sys.modules["fp32_arbiter"] = arbiter  # dataclasses with postponed annotations resolve through sys.modules
_spec.loader.exec_module(arbiter)


@pytest.mark.parametrize(
    "magnitude, ulp",
    [
        (16.0, 0.125),
        (27.75, 0.125),
        (31.999, 0.125),
        (32.0, 0.25),
        (1.0, 2**-7),
        (0.3, 2**-9),
        (100.0, 0.5),
        (0.0, 2**-133),
    ],
)
def test_bf16_ulp_is_the_spacing_at_that_magnitude(magnitude, ulp):
    assert arbiter.bf16_ulp(magnitude) == ulp
    if magnitude > 0:
        assert arbiter.bf16_ulp(magnitude) == 2 ** (math.floor(math.log2(magnitude)) - 7)


def scores(**logits):
    table = {int(k): v for k, v in logits.items()}
    top = sorted(([v, t] for t, v in table.items()), reverse=True)
    return arbiter.Scores(table, top)


def test_arbitrate_passes_within_two_ulps_and_fails_beyond():
    # The RAG 87791f97 token-2 divergence: fp32 4087 = 27.7906, 310 = 27.5972: d 0.193 <= 2 * 0.125.
    v = arbiter.arbitrate(4087, 310, scores(**{"4087": 27.790557861328125, "310": 27.597150802612305}))
    assert v.passes and v.ulp == 0.125 and v.d == pytest.approx(0.1934, abs=1e-4)
    assert (
        v.reference_logit > v.candidate_logit
    )  # fp32 prefers the native token: the pass is the rule, not an argmax match
    assert arbiter.arbitrate(1, 2, scores(**{"1": 27.75, "2": 27.5})).passes  # exactly 2 ulp
    assert not arbiter.arbitrate(1, 2, scores(**{"1": 27.75, "2": 27.4})).passes  # 2.8 ulp
    assert arbiter.arbitrate(1, 2, scores(**{"1": 27.75, "2": 27.75})).passes  # an exact tie
    # The larger magnitude sets the ulp (0.25): 12 > 0.5 (the candidate is not the fp32 argmax here).
    wide = arbiter.arbitrate(2, 1, scores(**{"1": 20.0, "2": 32.0}))
    assert not wide.passes and wide.ulp == 0.25
    assert not arbiter.arbitrate(1, 2, scores(**{"1": 27.75, "2": 27.5}), tolerance_ulps=1.0).passes


def parity_docs():
    prompts = {"a": [10, 11], "b": [20, 21], "c": [30], "d": [40], "e": [50]}
    ref = {
        "results": [
            {"question_id": "a", "output_ids": [1, 2, 3], "top": [[[-0.1, 1], [-3.0, 9]]] * 3},
            {
                "question_id": "b",
                "output_ids": [1, 2, 3],
                "top": [[[-0.5, 1], [-0.6, 7]], [[-0.5, 2], [-0.6, 7]], [[-0.5, 3], [-0.6, 7]]],
            },
            {
                "question_id": "c",
                "output_ids": [1, 2, 3],
                "top": [[[-0.5, 1], [-0.75, 7]]] * 3,
            },  # gap 0.25 > 0.13: arbitrated
            {"question_id": "d", "output_ids": [1, 2, 3], "top": [[[-0.5, 1], [-0.75, 7]]] * 3},
            {"question_id": "e", "output_ids": [1, 2, 3], "top": [[[-0.5, 1], [-0.75, 7]]] * 3},
        ]
    }
    cand = {
        "results": [
            {"question_id": "a", "output_ids": [1, 2, 3]},  # identical
            {"question_id": "b", "output_ids": [1, 7, 3]},  # bf16 gap 0.1 <= 0.13: near tie, never scored
            {"question_id": "c", "output_ids": [1, 2, 7]},  # fp32 d = 0.2 at |logit| ~27: 1.6 ulp, passes
            {"question_id": "d", "output_ids": [7, 2, 3]},  # fp32 d = 0.5: 4 ulp, fails
            {"question_id": "e", "output_ids": [1, 2]},  # prefix-equal, shorter: length
        ]
    }
    return prompts, ref, cand


def test_classify_parity_scores_only_the_divergences_above_the_tie_gap():
    prompts, ref, cand = parity_docs()
    calls = []

    def score(prefix, tokens):
        calls.append((list(prefix), list(tokens)))
        return {("c", 3): scores(**{"3": 27.7, "7": 27.5}), ("d", 1): scores(**{"1": 27.75, "7": 27.25})}[
            ("c", 3) if prefix == [30, 1, 2] else ("d", 1)
        ]

    result = arbiter.classify_parity(ref, cand, prompts, score)
    by_id = {r["question_id"]: r for r in result["rows"]}
    assert [r["status"] for r in result["rows"]] == ["identical", "near_tie", "fp32_pass", "fail", "length"]
    assert result["counts"] == {
        "identical": 1,
        "near_tie": 1,
        "fp32_pass": 1,
        "fp32_argmax": 0,
        "fail": 1,
        "length": 1,
    }
    assert not result["passes"]
    # The scored prefix is the prompt plus the reference tokens before the divergence, and only c and d are scored.
    assert calls == [([30, 1, 2], [3, 7]), ([40], [1, 7])]
    assert by_id["c"]["prefix_length"] == 3 and by_id["c"]["fp32"]["d"] == pytest.approx(0.2)
    assert by_id["d"]["fp32"]["d"] == 0.5 and by_id["d"]["fp32_argmax"] == 1
    assert "fp32" not in by_id["b"] and by_id["b"]["reference_gap"] == pytest.approx(0.1)
    assert by_id["e"]["lengths"] == [3, 2]
    for row in result["rows"]:
        assert row["question_id"] in arbiter.format_row(row)


def test_classify_parity_passes_when_every_scored_divergence_is_within_tolerance():
    prompts, ref, cand = parity_docs()
    cand["results"] = cand["results"][:3]
    result = arbiter.classify_parity(ref, cand, prompts, lambda prefix, tokens: scores(**{"3": 27.7, "7": 27.5}))
    assert result["passes"] and result["counts"]["fail"] == 0


def test_candidate_token_outside_the_reference_top_k_is_still_arbitrated():
    ref = {"results": [{"question_id": "x", "output_ids": [1, 2], "top": [[[-0.5, 1], [-0.6, 9]]] * 2}]}
    cand = {"results": [{"question_id": "x", "output_ids": [1, 5]}]}
    result = arbiter.classify_parity(ref, cand, {"x": [0]}, lambda prefix, tokens: scores(**{"2": 10.0, "5": 9.97}))
    row = result["rows"][0]
    assert row["reference_gap"] is None and row["status"] == "fp32_pass" and row["fp32"]["ulp"] == 2**-4


# An fp32 re-forward where the candidate's token 3437 (21.0152) beats the reference's 5586 (20.6798) by d = 0.3354:
# more than 2 bf16 ulp (0.25) at that size, so only the candidate-argmax rule passes it.
FLIP = {"5586": 20.679800033569336, "3437": 21.015199661254883, "11": 18.5}


def test_candidate_that_is_the_fp32_argmax_passes_above_the_ulp_limit():
    v = arbiter.arbitrate(5586, 3437, scores(**FLIP))
    assert v.d > v.tolerance_ulps * v.ulp == 0.25  # the pair-gap rule alone fails it
    assert (v.status, v.passes, v.fp32_argmax) == ("fp32_argmax", True, 3437)


def test_reference_that_is_the_fp32_argmax_still_fails_with_the_ids_swapped():
    v = arbiter.arbitrate(3437, 5586, scores(**FLIP))
    assert v.d > v.tolerance_ulps * v.ulp
    assert (v.status, v.passes, v.fp32_argmax) == ("fail", False, 3437)


def test_pair_gap_rule_decides_first_and_ties_at_the_top_count_as_argmax():
    # Within 2 ulp the divergence is fp32_pass whichever token wins, so the two categories stay distinct.
    assert arbiter.arbitrate(1, 2, scores(**{"1": 27.5, "2": 27.75})).status == "fp32_pass"
    # Candidate 2 ties the top logit that topk reports under another token: it is still an fp32 argmax.
    tied = arbiter.Scores({1: 27.0, 2: 28.0}, [[28.0, 9], [28.0, 2], [27.0, 1]])
    assert arbiter.arbitrate(1, 2, tied).status == "fp32_argmax"
    # A token above the candidate makes it fail even when the candidate beats the reference.
    above = arbiter.Scores({1: 27.0, 2: 27.5}, [[28.0, 9], [27.5, 2], [27.0, 1]])
    assert arbiter.arbitrate(1, 2, above).status == "fail"


def flip_docs(reference_token, candidate_token):
    """One prompt whose first divergence is at index 1, with a bf16 gap (0.375) above the near-tie gate."""
    top = [[[-0.1, reference_token], [-0.475, candidate_token]]] * 2
    ref = {"results": [{"question_id": "q", "output_ids": [7, reference_token], "top": top}]}
    cand = {"results": [{"question_id": "q", "output_ids": [7, candidate_token]}]}
    return ref, cand


def test_classify_parity_uses_each_rows_own_candidate_token():
    # Two runs of the same prompt diverge in opposite directions; the fp32 forward of the prefix is the same.
    score = lambda prefix, tokens: scores(**FLIP)  # noqa: E731
    ours = arbiter.classify_parity(*flip_docs(5586, 3437), {"q": [0]}, score)
    swapped = arbiter.classify_parity(*flip_docs(3437, 5586), {"q": [0]}, score)
    assert ours["rows"][0]["status"] == "fp32_argmax" and ours["passes"]
    assert ours["counts"]["fp32_argmax"] == 1 and ours["counts"]["fail"] == 0
    assert swapped["rows"][0]["status"] == "fail" and not swapped["passes"]
    assert ours["rows"][0]["fp32_argmax"] == swapped["rows"][0]["fp32_argmax"] == 3437
    assert "fp32_argmax" in arbiter.format_row(ours["rows"][0])


def saved_old_rule_report():
    """A report written before the candidate-argmax rule: both flips saved as fail, with their fp32 evidence."""
    rows = []
    for ref_token, cand_token in ((5586, 3437), (3437, 5586)):
        s = scores(**FLIP)
        a, b = s.logits[ref_token], s.logits[cand_token]
        old = {
            "reference_token": ref_token,
            "candidate_token": cand_token,
            "reference_logit": a,
            "candidate_logit": b,
            "d": abs(a - b),
            "ulp": 0.125,
            "tolerance_ulps": 2.0,
            "passes": False,
        }
        rows.append(
            {
                "question_id": f"q{ref_token}",
                "index": 1,
                "status": "fail",
                "reference_token": ref_token,
                "candidate_token": cand_token,
                "reference_gap": 0.375,
                "fp32": old,
                "fp32_top": s.top,
                "fp32_argmax": s.top[0][1],
            }
        )
    rows.append(
        {
            "question_id": "n",
            "index": 3,
            "status": "near_tie",
            "reference_token": 1,
            "candidate_token": 2,
            "reference_gap": 0.125,
        }
    )
    counts = {"identical": 0, "near_tie": 1, "fp32_pass": 0, "fail": 2, "length": 0}
    return {"tie_gap": 0.13, "tolerance_ulps": 2.0, "counts": counts, "passes": False, "rows": rows}


@pytest.mark.parametrize("layout", ["verdicts", "candidates"])
def test_rejudge_applies_the_current_rule_to_a_saved_report_without_changing_it(layout):
    result = saved_old_rule_report()
    report = {"verdicts": [result]} if layout == "verdicts" else {"candidates": {"live.json": result}}
    original = copy.deepcopy(report)
    again = arbiter.rejudge(report)
    assert report == original  # the saved evidence is never rewritten
    redone = again["verdicts"][0] if layout == "verdicts" else again["candidates"]["live.json"]
    assert [r["status"] for r in redone["rows"]] == ["fp32_argmax", "fail", "near_tie"]
    assert redone["counts"] == {"identical": 0, "near_tie": 1, "fp32_pass": 0, "fp32_argmax": 1, "fail": 1, "length": 0}
    assert not redone["passes"] and redone["tie_gap"] == 0.13
    assert redone["rows"][0]["fp32"]["d"] == result["rows"][0]["fp32"]["d"]  # same logits, new decision


def test_rejudge_cli_writes_a_new_file_and_refuses_to_overwrite(tmp_path, capsys):
    result = saved_old_rule_report()
    result["rows"] = result["rows"][:1]
    saved = tmp_path / "verdicts.json"
    saved.write_text(json.dumps({"verdicts": [result]}))
    before = saved.read_bytes()
    out = tmp_path / "verdicts.rejudged.json"
    assert arbiter.main(["rejudge", str(saved), "--out", str(out)]) == 0
    assert saved.read_bytes() == before
    written = json.loads(out.read_text())
    assert written["rejudged_from"] == str(saved) and written["verdicts"][0]["counts"]["fp32_argmax"] == 1
    assert "PASS" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        arbiter.main(["rejudge", str(saved), "--out", str(saved)])
    assert saved.read_bytes() == before


def test_rule_and_rejudge_import_without_the_parity_http_client():
    # Only from-parity needs greedy_parity, whose HTTP client imports requests; the public test env has no requests.
    code = (
        "import importlib.util, sys; sys.modules['requests'] = None; "
        f"spec = importlib.util.spec_from_file_location('fp32_arbiter', {str(_TOOL)!r}); "
        "m = importlib.util.module_from_spec(spec); sys.modules['fp32_arbiter'] = m; spec.loader.exec_module(m); "
        "print(m.bf16_ulp(20.0))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0.125"
