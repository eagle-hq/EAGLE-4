# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Chain verify width per batch size, from the cost tables and a measured acceptance profile.

A chain round of width ``w`` drafts ``w - 1`` tokens and verifies them with the root. Under greedy acceptance a
request gains ``1 + a_1 + ... + a_{w-1}`` tokens per round, where ``a_d`` is the share of rounds whose first ``d``
drafted tokens are all accepted: a property of the draft, the target and the workload, not of the batch size. The
round takes ``V(B, w) + (w - 1) * D(B)`` at batch size ``B``: the target's verify forward of ``w`` tokens per request
and ``w - 1`` draft steps of one token per request, both from the cost tables. The width with the most tokens per
unit of time serves batch size ``B``; width 1 verifies only the root and may still extend draft KV.

Small batches are bound by weight reads, so ``V`` barely grows with ``w`` and the widest chain wins; large batches are
bound by compute, ``V`` grows with the tokens it verifies, and the optimum narrows down to plain decoding. The width
is fixed per batch size, so a round reads nothing back from the GPU to choose it.

The acceptance profile is ``<data_path>/accept/chain.json``::

    {"drafts": <rounds counted>, "accepted_per_position": [<rounds whose drafted token d + 1 was accepted>, ...]}

the counters vLLM keeps as ``spec_decode_num_drafts`` and ``spec_decode_num_accepted_tokens_per_pos``, measured at the
widest chain on the workload to be served. Positions past the profile have no measured acceptance, so wider chains
are not considered.

``V + (w - 1) D`` from the cost tables prices a round by its forwards alone. A served round also pays the scheduler,
the draft extend and the launches, and at large batch sizes those dominate: timings of isolated forwards priced the
vLLM chain round at batch size 64 about 3x too cheap. Served round times replace the price when they were measured:
``<data_path>/rounds/chain.json``::

    {"rounds": [{"batch_size": 64, "width": 4, "round_us": 21500.0, "count": 180}, ...]}

the median period between consecutive decode rounds of a serving engine at that batch size with every round of that
width. Width 1 is root-only verification. Width 0 means native decode in the same engine, with no draft
forward or draft KV update, and is considered only when the adapter supports that action. It never means
starting a separate engine. The width of batch size ``B`` is then
``argmax_w (1 + a_1 + ... + a_{w-1}) / R(B, w)``.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from eagle_harness.core.cost_table import BatchCostTable, CostCurve, load_cost_tables
from eagle_harness.core.limits import nearest_cost_batch_size
from eagle_harness.core.live_costs import CostKey, check_key

PROFILE_FILE = os.path.join("accept", "chain.json")
ROUNDS_FILE = os.path.join("rounds", "chain.json")
TREE_ROUNDS_FILE = os.path.join("rounds", "tree.json")


@dataclass(frozen=True)
class AcceptProfile:
    """Acceptance of a chain by draft position: ``rates[d]`` is the share of rounds that accept drafted token ``d + 1``
    (and so every token before it)."""

    rates: Tuple[float, ...]

    @classmethod
    def from_counts(cls, drafts: int, accepted_per_position: Sequence[int]) -> "AcceptProfile":
        """A profile from round counts (vLLM's spec-decode counters)."""
        if drafts <= 0:
            raise ValueError(f"an acceptance profile needs at least one round, got drafts={drafts}.")
        counts = list(accepted_per_position)
        if any(b > a for a, b in zip(counts, counts[1:])) or any(c < 0 or c > drafts for c in counts):
            raise ValueError(f"accepted_per_position must be non-increasing counts in [0, drafts]: {counts}")
        return cls(tuple(c / drafts for c in counts))

    def expected_tokens(self, width: int) -> float:
        """Tokens a request gains per round at verify width ``width`` (``width - 1`` drafts plus the bonus token)."""
        if not 1 <= width <= len(self.rates) + 1:
            raise ValueError(f"the profile covers widths 1..{len(self.rates) + 1}, got {width}.")
        return 1.0 + sum(self.rates[: width - 1])


def load_accept_profile(data_path: str) -> Optional[AcceptProfile]:
    """The chain acceptance profile under ``data_path``, None if it was not measured."""
    path = os.path.join(data_path, PROFILE_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    return AcceptProfile.from_counts(data["drafts"], data["accepted_per_position"])


def save_accept_profile(data_path: str, drafts: int, accepted_per_position: Sequence[int]) -> str:
    """Write the counts as ``<data_path>/accept/chain.json`` (validated first); returns the path."""
    AcceptProfile.from_counts(drafts, accepted_per_position)
    path = os.path.join(data_path, PROFILE_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"drafts": drafts, "accepted_per_position": list(accepted_per_position)}, f)
    return path


def median_cost(curves: Sequence[CostCurve], query_length: int) -> Optional[float]:
    """Median cost over the context intervals at ``query_length`` tokens per request; None if any interval lacks it."""
    if not curves or any(query_length - 1 not in curve.index for curve in curves):
        return None
    return statistics.median(curve.value[curve.index.index(query_length - 1)] for curve in curves)


def chain_width(table: BatchCostTable, profile: AcceptProfile, max_width: int) -> Optional[int]:
    """The verify width with the most tokens per unit of time at the table's batch size (1: root-only verification).

    Candidates are the widths up to ``max_width`` the table timed and the profile covers. None when the table did not
    time root-only verification or a draft step.
    """
    targets = [interval.target for interval in table.intervals]
    plain = median_cost(targets, 1)
    step = median_cost([interval.draft for interval in table.intervals], 1)
    if plain is None or step is None or plain <= 0 or step <= 0:
        return None
    best, best_rate = 1, 1.0 / plain
    for width in range(2, min(max_width, len(profile.rates) + 1) + 1):
        verify = median_cost(targets, width)
        if verify is None:
            continue
        rate = profile.expected_tokens(width) / (verify + (width - 1) * step)
        if rate > best_rate:
            best, best_rate = width, rate
    return best


def chain_width_schedule(
    tables: Mapping[int, BatchCostTable], profile: AcceptProfile, max_width: int, max_batch_size: int
) -> Optional[List[Tuple[int, int, int]]]:
    """``(first batch size, last batch size, width)`` ranges for batch sizes ``1..max_batch_size``.

    A batch size between calibrated ones takes the width of the nearest calibrated batch size (ties go to the larger),
    as the cost model reads its tables. None if any calibrated batch size cannot price a chain.
    """
    if not tables:
        return None
    widths: Dict[int, int] = {}
    for bs, table in tables.items():
        width = chain_width(table, profile, max_width)
        if width is None:
            return None
        widths[bs] = width
    return _ranges(widths, max_batch_size)


def served_chain_width(
    round_us: Mapping[int, float], profile: AcceptProfile, max_width: int, *, allow_no_spec: bool = False
) -> Optional[int]:
    """The verify width with the most tokens per unit of time from served round times at one batch size.

    ``round_us[w]`` is the measured period (``1``: root-only verify, ``0``: native decode). Candidates are the
    widths up to ``max_width`` that were timed and that the profile covers; None when root-only verification was not timed.
    """
    plain = round_us.get(1)
    if plain is None or plain <= 0:
        return None
    best, best_rate = 1, 1.0 / plain
    no_spec = round_us.get(0) if allow_no_spec else None
    if no_spec is not None and 0 < no_spec <= plain:
        best, best_rate = 0, 1.0 / no_spec
    for width, us in sorted(round_us.items()):
        if not 2 <= width <= min(max_width, len(profile.rates) + 1) or us <= 0:
            continue
        rate = profile.expected_tokens(width) / us
        if rate > best_rate:
            best, best_rate = width, rate
    return best


def served_width_schedule(
    rounds: Mapping[int, Mapping[int, float]],
    profile: AcceptProfile,
    max_width: int,
    max_batch_size: int,
    *,
    allow_no_spec: bool = False,
) -> Optional[List[Tuple[int, int, int]]]:
    """:func:`chain_width_schedule` from served round times (``rounds[batch size][width]``, microseconds)."""
    if not rounds:
        return None
    widths: Dict[int, int] = {}
    for bs, round_us in rounds.items():
        width = served_chain_width(round_us, profile, max_width, allow_no_spec=allow_no_spec)
        if width is None:
            return None
        widths[bs] = width
    return _ranges(widths, max_batch_size)


def load_round_times(data_path: str) -> Optional[Dict[int, Dict[int, float]]]:
    """The served round times under ``data_path`` as ``{batch size: {width: microseconds}}``, None if not measured."""
    try:
        with open(os.path.join(data_path, ROUNDS_FILE), encoding="utf-8") as f:
            rows = json.load(f)["rounds"]
    except FileNotFoundError:
        return None
    rounds: Dict[int, Dict[int, float]] = {}
    for row in rows:
        rounds.setdefault(int(row["batch_size"]), {})[int(row["width"])] = float(row["round_us"])
    return rounds


def save_round_times(data_path: str, rows: Sequence[Mapping[str, float]]) -> str:
    """Write ``rows`` (``batch_size``, ``width``, ``round_us`` and the ``count`` of rounds behind it) as
    ``<data_path>/rounds/chain.json``; returns the path."""
    for row in rows:
        if row["batch_size"] < 1 or row["width"] < 0 or row["round_us"] <= 0:
            raise ValueError(f"a served round time needs batch_size >= 1, width >= 0 and round_us > 0: {dict(row)}")
    path = os.path.join(data_path, ROUNDS_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"rounds": [dict(row) for row in rows]}, f, indent=1)
    return path


def chain_widths(
    data_path: str,
    max_width: int,
    max_batch_size: int,
    *,
    allow_no_spec: bool = False,
    key: Optional[CostKey] = None,
) -> Optional[List[Tuple[int, int, int]]]:
    """The chain widths per batch size range under ``data_path``: from the served round times when measured, else
    from the cost tables; both need the acceptance profile. None when either part is missing. With the server's
    ``key``, a directory measured on another setup is refused (:func:`~eagle_harness.core.live_costs.check_key`)."""
    profile = load_accept_profile(data_path)
    if profile is None:
        return None
    if key is not None:
        check_key(data_path, key)
    rounds = load_round_times(data_path)
    if rounds is not None:
        return served_width_schedule(rounds, profile, max_width, max_batch_size, allow_no_spec=allow_no_spec)
    tables = load_cost_tables(os.path.join(data_path, "cost"))
    return chain_width_schedule(tables, profile, max_width, max_batch_size) if tables else None


def width_at(schedule: Sequence[Tuple[int, int, int]], batch_size: int) -> int:
    """The width a schedule of ``(first, last, width)`` ranges gives ``batch_size``."""
    for first, last, width in schedule:
        if first <= batch_size <= last:
            return width
    raise ValueError(f"batch size {batch_size} is outside the chain width schedule {list(schedule)}")


def merge_ranges(rows: Iterable[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """``(first, last, value)`` rows in batch-size order, consecutive rows with the same value merged into one range."""
    ranges: List[Tuple[int, int, int]] = []
    for first, last, value in rows:
        if ranges and ranges[-1][2] == value:
            ranges[-1] = (ranges[-1][0], last, value)
        else:
            ranges.append((first, last, value))
    return ranges


def _ranges(widths: Mapping[int, int], max_batch_size: int) -> List[Tuple[int, int, int]]:
    """Per-batch-size widths of the calibrated batch sizes as ranges over ``1..max_batch_size``."""
    return merge_ranges((bs, bs, widths[nearest_cost_batch_size(bs, widths)]) for bs in range(1, max_batch_size + 1))


# ---------------------------------------------------------------- the tree's verify width under load
# A tree speculator on an engine whose scheduler fixes the verify width per batch size (vLLM) verifies the best
# ``width`` nodes of its draft tree. Wide trees win where batches are bound by weight reads and lose where the verify
# forward is bound by compute. ``<data_path>/rounds/tree.json`` holds what the serving engine measured::
#
#     {"rounds": [{"batch_size": 64, "width": 16, "round_us": 47900.0, "accept": 3.96, "count": 200}, ...]}
#
# per batch size and verify width: the median period between consecutive decode rounds and the mean tokens a request
# committed per round (``accept``: 1 + accepted drafts). Width 1 verifies only the root (no drafts). Unlike the chain's
# acceptance profile, the tree's acceptance is measured per width, since the best ``width`` nodes of a tree are not a
# prefix of a chain.


def _check_tree_row(row: Mapping[str, float]) -> None:
    if row["batch_size"] < 1 or row["width"] < 1 or row["round_us"] <= 0 or row["accept"] < 1 or row["count"] < 1:
        raise ValueError(
            "a served tree round needs batch_size >= 1, width >= 1, round_us > 0, accept >= 1 and count >= 1: "
            f"{dict(row)}"
        )


def load_tree_rounds(
    data_path: str, key: Optional[CostKey] = None
) -> Optional[Dict[int, Dict[int, Tuple[float, float]]]]:
    """``{batch size: {width: (round us, accept)}}`` from ``<data_path>/rounds/tree.json``, None if not measured.
    With the server's ``key``, rounds measured on another setup are refused (as by :func:`chain_widths`)."""
    try:
        with open(os.path.join(data_path, TREE_ROUNDS_FILE), encoding="utf-8") as f:
            rows = json.load(f)["rounds"]
    except FileNotFoundError:
        return None
    if key is not None:
        check_key(data_path, key)
    rounds: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for row in rows:
        _check_tree_row(row)
        cell = (float(row["round_us"]), float(row["accept"]))
        rounds.setdefault(int(row["batch_size"]), {})[int(row["width"])] = cell
    return rounds


def merge_tree_rounds(data_path: str, rows: Sequence[Mapping[str, float]]) -> str:
    """Write ``rows`` into ``<data_path>/rounds/tree.json``, replacing the (batch size, width) cells they measured and
    keeping the others (each calibration server times its own widths); returns the path."""
    for row in rows:
        _check_tree_row(row)
    path = os.path.join(data_path, TREE_ROUNDS_FILE)
    cells: Dict[Tuple[int, int], dict] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for row in json.load(f)["rounds"]:
                cells[(int(row["batch_size"]), int(row["width"]))] = dict(row)
    except FileNotFoundError:
        pass
    for row in rows:
        cells[(int(row["batch_size"]), int(row["width"]))] = dict(row)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"rounds": [cells[key] for key in sorted(cells)]}, f, indent=1)
    os.replace(tmp, path)
    return path


def served_tree_schedule(
    rounds: Mapping[int, Mapping[int, Tuple[float, float]]], width: int, max_batch_size: int
) -> Optional[List[Tuple[int, int, int]]]:
    """``(first, last, verify width)`` ranges over ``1..max_batch_size`` for a tree configured ``width`` wide.

    At each measured batch size that timed ``width``, the measured width up to ``width`` with the most tokens per unit
    of time (``accept / round_us``; ties keep the wider). Every batch size below the first measured one where another
    width wins keeps ``width``, so the low-batch behavior never changes; from there each batch size takes the choice of
    the nearest measured one (ties go to the larger). None when no measured batch size beats ``width``: the tree then
    keeps one width, as without measurements.
    """
    choices: Dict[int, int] = {}
    for bs, cells in rounds.items():
        if width not in cells:
            continue
        best, best_rate = width, cells[width][1] / cells[width][0]
        for w, (us, accept) in sorted(cells.items(), reverse=True):
            if w <= width and accept / us > best_rate:
                best, best_rate = w, accept / us
        choices[bs] = best
    losing = [bs for bs, best in choices.items() if best != width]
    if not losing:
        return None
    switch = min(losing)
    ranges = _ranges({bs: w for bs, w in choices.items() if bs >= switch}, max_batch_size)
    kept = [(1, switch - 1, width)] if switch > 1 else []
    clipped = [(max(first, switch), last, w) for first, last, w in ranges if last >= switch]
    return merge_ranges(kept + clipped)
