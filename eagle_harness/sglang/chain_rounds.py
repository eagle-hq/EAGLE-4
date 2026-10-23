# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Fixed-width chain rounds: the served width per batch size, and the calibration that times it.

A chain (``--speculative-eagle-topk 1``, ``chain_intensity``) with served round times and an acceptance profile
under ``data_path`` verifies a fixed width per batch size (:func:`eagle_harness.core.chain_width.chain_widths`),
width 0 (native decode without draft extend) where speculation loses. ``"mode": "calibrate"`` measures that choice
while real requests are served: it forces every chain width the target has a verify graph for, in blocks of
:data:`BLOCK_ROUNDS` rounds, and records

- the period between consecutive decode rounds at the same batch size (wall clock, so the scheduler, the draft
  extend and the launches are in it), per batch size and width;
- the accept lengths of the widest rounds, per request (a histogram kept on the GPU, read when the file is written).

A round is recorded only between two decode rounds of the same batch size and width, so the first and last round
of a block (whose periods hold another width's prelaunched draft) and the rounds next to a prefill are not.

Rank 0 writes what it has every :data:`FLUSH_ROUNDS` rounds and at the next prefill: the raw rounds
(``<data_path>/raw/sglang_rounds.json``), the median period of every batch size that timed each width at least
:data:`MIN_ROUNDS` times (``rounds/chain.json``) and the acceptance profile (``accept/chain.json``), the two files a
serve-mode chain server with the same ``data_path`` reads.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import torch

from eagle_harness.core.chain_width import AcceptProfile, save_accept_profile, save_round_times, width_at
from eagle_harness.core.config import HarnessConfig, Mode
from eagle_harness.core.limits import ComputeIntensity, cost_batch_sizes, nearest_cost_batch_size

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)

BLOCK_ROUNDS = 8
FLUSH_ROUNDS = 256
MIN_ROUNDS = 8
# Whole-interval periods of one width before two widths are compared. Under sustained load the periods that hold
# a prefill are several times the others (coefficient of variation ~0.5), so the mean of 32 is within ~9%: enough
# to separate native decode from speculation where the choice matters, short enough to settle in a first pass.
EXPLORE_ROUNDS = 32
# A period in which more than 1/BURST_SHARE of the batch joined holds a burst of arrivals (the load changed), not the
# refills of sustained load (tokens per round / output length: under 1/8 for outputs over 64 tokens at width <= 8).
# Its prefills are the same work under every width and, landing in one width's few periods, would decide the choice.
BURST_SHARE = 8
# Speculation serves a batch key only when it beats native decode by this many standard errors of the difference of
# the two measured rates: a wrong native choice costs only the speculative gain, a wrong speculative one runs slower
# than plain decoding.
NATIVE_MARGIN_SE = 2.0
# Speculation that beats native decode's estimate (the batch's one-token verify in the live costs) by this factor,
# beyond NATIVE_MARGIN_SE standard errors, is not measured against native decode. Requests decoded natively cannot
# resume drafting and a batch holding one decodes natively, so measuring native decode keeps the batch native until it
# empties: under sustained load, for good. The estimate is a model of native decode, not a measurement (served native
# rounds ran 1.1x it at batch key 64 on B300), hence the wide factor.
NATIVE_SKIP_RATIO = 1.5
# A period longer than STALL_FACTOR times its cell's median holds one-time work, not the served load: a kernel compile
# or graph capture on a cold cache (1-3 s on B300 TP4, against 5-15 ms rounds). The refill prefills of sustained load
# stay within a few medians. One such period in a 32-period cell moves its rate several-fold (native decode at batch
# key 32 read 0.28 tokens/ms against 5.6 served), so it is counted, not priced.
STALL_FACTOR = 10
RAW_FILE = os.path.join("raw", "sglang_rounds.json")


def round_cells(
    rounds: Sequence[Sequence[float]], widths: Sequence[int], min_rounds: int = MIN_ROUNDS
) -> Dict[int, Dict[int, Tuple[float, int]]]:
    """``{batch size: {width: (median period us, rounds)}}`` of the batch sizes that timed every one of ``widths`` at
    least ``min_rounds`` times."""
    periods: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for bs, width, us, _context in rounds:
        periods[(int(bs), int(width))].append(us)
    cells: Dict[int, Dict[int, Tuple[float, int]]] = defaultdict(dict)
    for (bs, width), values in periods.items():
        if len(values) >= min_rounds:
            cells[bs][width] = (statistics.median(values), len(values))
    return {bs: dict(sorted(row.items())) for bs, row in sorted(cells.items()) if set(row) == set(widths)}


def accept_counts(accept_lens: Sequence[int]) -> Tuple[int, List[int]]:
    """Request-rounds and those accepting drafted token ``d`` (``d = 1 .. max``) from a histogram of accept lengths
    (tokens committed per request and round: 1 + accepted drafts)."""
    per_position = [sum(accept_lens[d + 1 :]) for d in range(1, len(accept_lens) - 1)]
    return sum(accept_lens), per_position


class ChainCalibration:
    """The forced width of each round (``width``) and the recorder of the served rounds (``round_start``,
    ``accepted``)."""

    def __init__(
        self, data_path: str, widths: Sequence[int], device: torch.device, *, writer: bool, native_decode: bool = False
    ):
        """``widths``: the chain widths to time (each has a target verify graph); ``writer``: this rank writes."""
        self.widths = sorted(widths)
        self.max_width = self.widths[-1]
        self._native_decode = native_decode
        self.last_width = self.widths[0]
        if native_decode:
            self.widths.append(0)
        self._data_path = data_path
        self._writer = writer
        self._planned = 0  # rounds whose width was handed out
        self._started = 0  # decode rounds started
        # The last two decode rounds since the last other step: (bs, width, start ns, mean context).
        self._recent: List[Tuple[int, int, int, int]] = []
        self.rounds: List[Tuple[int, int, float, int]] = []  # (bs, width, period us, mean context)
        self._flushed = 0  # rounds in the file
        self._accept = torch.zeros(self.max_width + 1, dtype=torch.int64, device=device)

    def width(self, bs: int) -> int:
        """The width of the next round (one call per round, in round order)."""
        if self._native_decode:
            # Collect enough settled intervals in one pass, then stay OFF: its
            # draft KV is no longer current and must not be consumed again.
            index = min(self._planned // (BLOCK_ROUNDS + MIN_ROUNDS), len(self.widths) - 1)
        else:
            index = (self._planned // BLOCK_ROUNDS) % len(self.widths)
        self.last_width = self.widths[index]
        self._planned += 1
        return self.last_width

    def round_start(self, bs: int, width: int, context: int) -> None:
        """A decode round of ``bs`` requests at ``width`` starts now; the previous round's period ends."""
        now = time.perf_counter_ns()
        recent = self._recent
        if len(recent) == 2 and all(r[:2] == (bs, width) for r in recent):
            self.rounds.append((bs, width, (now - recent[1][2]) / 1000.0, recent[1][3]))
        self._recent = recent[-1:] + [(bs, width, now, context)]
        self._started += 1
        if self._started % FLUSH_ROUNDS == 0:
            self.flush()
            self._recent = []  # the write is in this round's period

    def interrupt(self) -> None:
        """A non-decode step runs: the period in progress is not a decode round's. Writes the rounds recorded since
        the last write (a prefill after a batch, e.g. the calibration driver's next level, flushes it)."""
        self._recent = []
        if self._flushed != len(self.rounds):
            self.flush()
        if self._native_decode:
            self._planned = 0
            self.last_width = self.widths[0]

    def accepted(self, width: int, accept_lens: torch.Tensor) -> None:
        """Count the round's accept lengths (tokens committed per request, 1..width) when it is the widest."""
        if width == self.max_width:
            self._accept += torch.bincount(accept_lens.to(torch.int64), minlength=self.max_width + 1)

    def flush(self) -> None:
        """Write the raw rounds, the round times and the acceptance profile (rank 0); reads the accept histogram from
        the GPU."""
        self._flushed = len(self.rounds)
        if not self._writer:
            return
        accept_lens = self._accept.tolist()
        raw = {"max_width": self.max_width, "widths": self.widths, "accept_lens": accept_lens}
        raw["rounds"] = [list(r) for r in self.rounds]
        path = os.path.join(self._data_path, RAW_FILE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(f"{path}.tmp", "w", encoding="utf-8") as f:
            json.dump(raw, f)
        os.replace(f"{path}.tmp", path)
        cells = round_cells(self.rounds, self.widths)
        if cells:
            rows = [
                {"batch_size": bs, "width": w, "round_us": us, "count": n}
                for bs, row in cells.items()
                for w, (us, n) in row.items()
            ]
            save_round_times(self._data_path, rows)
        drafts, per_position = accept_counts(accept_lens)
        if drafts:
            save_accept_profile(self._data_path, drafts, per_position)


def served_chain(config: HarnessConfig, topk: int) -> bool:
    """Whether a server chooses its chain widths on its served rounds (:class:`ServedChainPolicy`): a ``serve``-mode
    chain without ``data_path``."""
    return (
        config.mode is Mode.SERVE and not config.data_path and config.intensity is ComputeIntensity.CHAIN and topk == 1
    )


def served_tree(config: HarnessConfig, topk: int) -> bool:
    """Whether a tree chooses between speculation and native decode on its served rounds (:class:`ServedTreePolicy`):
    every ``serve``-mode tree (top-k > 1), the default configuration; the engine must also have the native route
    (:func:`~eagle_harness.sglang.plain_decode.native_decode_available`)."""
    return config.mode is Mode.SERVE and topk > 1


def _ratio_rate(samples: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Tokens per microsecond over periods ``(us, tokens)`` (the ratio of the sums) and its variance (ratio
    estimator; infinite below two periods)."""
    n = len(samples)
    us = sum(u for u, _ in samples)
    if n == 0 or us <= 0:
        return 0.0, math.inf
    rate = sum(t for _, t in samples) / us
    if n < 2:
        return rate, math.inf
    mean_us = us / n
    residual = sum((t - rate * u) ** 2 for u, t in samples)
    return rate, residual / (n * (n - 1) * mean_us**2)


def _settled(samples: Sequence[tuple]) -> List[tuple]:
    """The periods ``(us, ...)`` of a cell without its stalls (:data:`STALL_FACTOR`)."""
    if len(samples) < 2:
        return list(samples)
    limit = STALL_FACTOR * statistics.median(sample[0] for sample in samples)
    return [sample for sample in samples if sample[0] <= limit]


class ServedChainPolicy:
    """The chain width per batch size of a ``serve``-mode chain without ``data_path``, chosen on its served rounds.

    Price: tokens per wall-clock time over whole served intervals. A round's period runs from its start to the next
    decode round's start, so the prefill steps between them are in it, and so are the costs a chain pays around them
    (a prelaunched draft dropped at a prefill or a batch change, the next round drafting from scratch). Common prefill
    work costs every action the same per committed token, so it does not move the choice; dropping those periods
    (as :class:`ChainCalibration` does) prices chains on the rounds where they are cheapest. A period is cut where
    the batch turned over completely (the server may have idled in between), and one that holds a burst of arrivals
    (:data:`BURST_SHARE`), a stall
    (:data:`STALL_FACTOR`) or the first round after a switch of width (it also finishes the previous width's work)
    counts toward the comparison without being priced.

    Search, per batch key (the nearest calibration batch size, :data:`~eagle_harness.core.limits.COST_BATCH_SIZES`):
    width 1 against the next wider captured width, then wider while the wider one wins (tokens per time is unimodal in
    the width: acceptance grows concavely, the verify cost convexly), never wider than a smaller batch key chose.
    Each comparison interleaves the two widths in
    blocks of :data:`BLOCK_ROUNDS` until each has :data:`EXPLORE_ROUNDS` periods, so both see the same load. With
    ``native_decode``, the search starts at width 2 (width 1 commits the same token as native decode with more work)
    and native decode (width 0) is measured last: its requests cannot resume drafting, so it runs right after a fresh
    block of the best width (both again after a complete turnover, so they see the same load), and the batch stays
    native until those requests finish (and the ones that join it meanwhile). So with ``native_us`` (native decode's
    microseconds per round at a batch size, from the live costs), speculation that beats that estimate by
    :data:`NATIVE_SKIP_RATIO` is served without measuring native decode. Speculation loses more as the
    batch grows (the verify becomes compute-bound), so where native decode wins every larger batch key takes it
    unmeasured, and where speculation wins no smaller batch key measures native decode.

    Decisions are made on TP rank 0 and broadcast (``sync``) at the round that completes a comparison, the same
    round on every rank. Rank 0 keeps them in ``cache_path`` (keyed like the live costs), so a restart on the same
    setup serves them from its first round. Without that file, a first start serves the ``packaged`` choices of its
    setup (:func:`~eagle_harness.core.cost_profiles.packaged_served_choices`) and searches only the other batch keys.
    shortcut: a decision is not revisited when the workload's acceptance changes; re-explore (delete the file) after
    changing the served workload class.
    """

    def __init__(
        self,
        widths: Sequence[int],
        device: torch.device,
        *,
        max_batch_size: int,
        native_decode: bool,
        writer: bool,
        sync: Callable[[object], object],
        cache_path: str | None = None,
        key: dict | None = None,
        packaged: Dict[int, int] | None = None,
        native_us: Callable[[int], float | None] | None = None,
    ):
        self.widths = sorted(widths)
        self._native = native_decode
        self._native_us = native_us
        self._native_checked: set = set()  # batch keys whose speculation was checked against native decode's estimate
        self._writer = writer
        self._sync = sync
        self._path = cache_path
        self._identity = {"key": key, "widths": self.widths, "native_decode": native_decode}
        self._grid = cost_batch_sizes(max_batch_size) or [max_batch_size]
        self._key = [0] + [nearest_cost_batch_size(bs, self._grid) for bs in range(1, max_batch_size + 1)]
        self.decided: Dict[int, int] = {}  # batch key -> served width (0: native decode)
        self._pair: Dict[int, Tuple[int, int]] = {}  # batch key -> (best width so far, challenger)
        self._started: Dict[int, int] = defaultdict(int)  # rounds of the comparison in progress, per batch key
        self._cells: Dict[Tuple[int, int], List[float]] = {}  # (batch key, width) -> [period us, requests, periods]
        # (batch key, width) -> the priced periods: (us, requests, tokens committed: a device scalar, trees only)
        self._samples: Dict[Tuple[int, int], List[Tuple[float, int, object]]] = defaultdict(list)
        # key, width, start ns, bs, request ids, whether the round is priced (not a live-cost seed round), whether it
        # switched action (its period also holds the previous action's tail: a dropped prelaunched draft, the switch)
        self._previous: Tuple[int, int, int, int, frozenset, bool, bool] | None = None
        self._last_width: int | None = None  # the width of the last round seen
        self._accept = torch.zeros(self.widths[-1] + 1, self.widths[-1] + 1, dtype=torch.int64, device=device)
        if not self._load() and packaged:
            self._start_from(packaged, "the packaged choices of this setup")

    # ---- the served width
    def width(self, bs: int) -> int:
        """The width of a round of ``bs`` requests (0: native decode)."""
        key = self._key[min(bs, len(self._key) - 1)]
        if key in self.decided:
            return self.decided[key]
        # With native decode, width 1 (the same tokens with more work) never wins: the search starts above it.
        first = self.widths[1] if self._native and len(self.widths) > 2 else self.widths[0]
        best, challenger = self._pair.setdefault(key, (first, self._wider(first)))
        if challenger == 0:  # the best width's periods, then native decode's right after them
            return 0 if self._cells.setdefault((key, best), [0.0, 0, 0])[2] >= EXPLORE_ROUNDS else best
        return challenger if (self._started[key] // BLOCK_ROUNDS) % 2 else best

    def _wider(self, width: int) -> int:
        """The next wider width; past the widest, native decode (0)."""
        return next((w for w in self.widths if w > width), 0)

    def exploring(self, bs: int) -> bool:
        return self._key[min(bs, len(self._key) - 1)] not in self.decided

    # ---- recording
    def round_start(self, bs: int, width: int, requests: Sequence, settled: bool = True) -> None:
        """A decode round of ``bs`` requests at ``width`` starts: the previous round's whole interval ends.
        ``settled``: False for a round the drafter forces to seed its live costs, whose period is not counted."""
        now = time.perf_counter_ns()
        key = self._key[min(bs, len(self._key) - 1)]
        previous, self._previous = self._previous, None
        switched, self._last_width = self._last_width not in (None, width), width
        if not self._pair and key in self.decided:
            return
        ids = frozenset(req.rid for req in requests)
        self._previous = (key, width, now, bs, ids, settled, switched)
        if previous is not None and previous[4].isdisjoint(ids):
            # The batch turned over (the server may have idled): native decode and the block of the best width it is
            # compared with are measured again together, under the load that follows.
            for k, pair in self._pair.items():
                if pair[1] == 0:
                    self._clear(k, pair[0])
                    self._clear(k, 0)
        elif previous is not None and previous[5] and previous[1] in self._pair.get(previous[0], ()):
            cell = self._cells.setdefault(previous[:2], [0.0, 0, 0])
            cell[2] += 1
            # A burst of arrivals, or the first round after a switch of action, is counted, not priced.
            if len(ids - previous[4]) * BURST_SHARE <= previous[3] and not previous[6]:
                us = (now - previous[2]) / 1000.0
                cell[0] += us
                cell[1] += previous[3]
                self._samples[previous[:2]].append((us, previous[3], self._period_tokens(previous[1])))
        if key in self.decided:
            return
        self._started[key] += 1
        pair = self._pair.get(key)
        if pair is None:
            return
        periods = [self._cells.setdefault((key, w), [0.0, 0, 0])[2] for w in pair]
        if all(n >= EXPLORE_ROUNDS for n in periods):
            self._compare(key, pair)
        elif (
            self._native_us is not None
            and pair[1] == 0
            and periods[0] >= EXPLORE_ROUNDS
            and not periods[1]
            and key not in self._native_checked
        ):
            self._check_native(key, pair[0])

    def accepted(self, width: int, accept_lens: torch.Tensor) -> None:
        """Count a round's accept lengths (tokens committed per request) under its width (rank 0 decides)."""
        if width >= 1 and self._writer:  # index_add_, not bincount: bincount sizes its output from a device read
            lens = accept_lens.to(torch.int64)
            self._accept[width].index_add_(0, lens, torch.ones_like(lens))

    def _period_tokens(self, width: int) -> object:
        """The tokens a period's round committed, when the policy measures them (trees); None: priced from the
        acceptance profile."""
        return None

    def _clear(self, key: int, width: int) -> None:
        self._cells[(key, width)] = [0.0, 0, 0]
        self._samples.pop((key, width), None)
        if width == 0:
            self._native_checked.discard(key)

    # ---- decisions
    def profile(self) -> AcceptProfile | None:
        """Acceptance by drafted position from every served width: a round of width ``w`` drafted positions
        ``1 .. w - 1``, and accepted position ``d`` when its accept length exceeds ``d``."""
        hist = self._accept.tolist()
        rates = []
        for d in range(1, len(hist) - 1):
            drafted = sum(sum(row) for row in hist[d + 1 :])
            if not drafted:
                break
            rates.append(sum(sum(row[d + 1 :]) for row in hist[d + 1 :]) / drafted)
        return AcceptProfile(tuple(rates)) if rates else None

    def _rates(self, key: int, pair: Tuple[int, int]) -> Dict[int, Tuple[float, float]]:
        """Committed tokens per microsecond of the pair's widths and the variance of each (native decode and width
        1 commit one token per request; wider chains the profile's expected tokens)."""
        profile = self.profile()
        covered = len(profile.rates) + 1 if profile else 1
        rates = {}
        for width in pair:
            if width <= covered and self._cells[(key, width)][0] > 0:
                tokens = profile.expected_tokens(width) if width > 1 else 1.0
                rates[width] = _ratio_rate([(us, tokens * n) for us, n, _ in _settled(self._samples[(key, width)])])
        return rates

    @staticmethod
    def _better(rates: Dict[int, Tuple[float, float]], best: int, challenger: int) -> int:
        """The winner of a comparison. Against native decode, speculation must win by :data:`NATIVE_MARGIN_SE`
        standard errors."""
        (best_rate, best_var), (rate, var) = rates.get(best, (0.0, math.inf)), rates.get(challenger, (0.0, math.inf))
        if challenger != 0:
            return challenger if rate > best_rate else best
        margin = NATIVE_MARGIN_SE * math.sqrt(best_var + var)
        return best if best_rate - rate > margin else 0

    def _compare(self, key: int, pair: Tuple[int, int]) -> None:
        """Both widths of the pair have their periods: rank 0 decides the next step, every rank applies it."""
        step = None
        if self._writer:
            rates = self._rates(key, pair)
            best, challenger = pair
            best = self._better(rates, best, challenger)
            # A larger batch never verifies wider than a smaller one chose, and native decode where speculation won
            # at a larger batch would only lose (and stick).
            widest = min((w for k, w in self.decided.items() if k < key and w >= 1), default=self.widths[-1])
            spec_won_above = any(w >= 1 and k >= key for k, w in self.decided.items())
            if best != 0 and challenger != 0 and best == challenger and best < widest:
                step = ("compare", best, self._wider(best))
            elif best != 0 and challenger != 0 and self._native and not spec_won_above:
                step = ("compare", best, 0)
            else:
                step = ("decide", best, None)
            logger.info(
                "Served chain rates at batch key %d (tokens per ms): %s -> %s",
                key,
                {w: (round(1000 * r, 3), round(1000 * math.sqrt(v), 3)) for w, (r, v) in sorted(rates.items())},
                step,
            )
        kind, width, challenger = self._sync(step)
        self._started[key] = 0
        for w in pair:
            self._clear(key, w)
        if kind == "compare":
            self._pair[key] = (width, challenger)
            return
        self._decide(key, width)

    def _decide(self, key: int, width: int) -> None:
        self._pair.pop(key, None)
        self.decided[key] = width
        self._infer()
        self._save()

    def _native_estimate(self, key: int, width: int) -> float | None:
        """Native decode's tokens per microsecond over the batches of ``width``'s periods at ``key``, priced by
        ``native_us``; None where it has no price."""
        if self._native_us is None:
            return None
        us = requests = 0.0
        for _, n, _ in _settled(self._samples[(key, width)]):
            price = self._native_us(n)
            if price is None or price <= 0:
                return None
            us += price
            requests += n
        return requests / us if us else None

    def _check_native(self, key: int, best: int) -> None:
        """``best`` has its periods before native decode's: rank 0 decides whether it beats native decode's estimate
        so clearly (:data:`NATIVE_SKIP_RATIO`) that native decode is not measured, and every rank applies it."""
        self._native_checked.add(key)
        step = None
        if self._writer:
            rate, var = self._rates(key, (best,)).get(best, (0.0, math.inf))
            native = self._native_estimate(key, best)
            won = native is not None and rate - NATIVE_MARGIN_SE * math.sqrt(var) > NATIVE_SKIP_RATIO * native
            step = ("decide", best, None) if won else ("compare", best, 0)
            logger.info(
                "Served chain rate at batch key %d (tokens per ms): %s, native decode estimate %s -> %s",
                key,
                (round(1000 * rate, 3), round(1000 * math.sqrt(var), 3)),
                None if native is None else round(1000 * native, 3),
                step,
            )
        kind, width, _ = self._sync(step)
        if kind == "decide":
            self._started[key] = 0
            self._clear(key, width)
            self._clear(key, 0)
            self._decide(key, width)

    def _infer(self) -> None:
        """Native decode at a batch key serves every larger one unmeasured."""
        for key in self._grid:
            if key not in self.decided and any(w == 0 for k, w in self.decided.items() if k < key):
                self.decided[key] = 0
                self._pair.pop(key, None)

    # ---- persistence
    def _start_from(self, decided: Dict[int, int], source: str) -> bool:
        """Serve ``decided`` (batch key -> width) if every width is an action of this server."""
        decided = {k: w for k, w in decided.items() if k in self._grid}
        if any(w not in (0, *self.widths) or (w == 0 and not self._native) for w in decided.values()):
            logger.info("Served chain widths %s from %s are not actions of this server; exploring.", decided, source)
            return False
        self.decided = dict(decided)
        self._infer()
        logger.info("Served chain widths per batch key from %s: %s", source, self.decided)
        return True

    def _load(self) -> bool:
        if self._path is None:
            return False
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            if {k: data.get(k) for k in self._identity} != self._identity:
                logger.info("Served chain widths in %s are for another setup; exploring again.", self._path)
                return False
            decided = {int(k): int(w) for k, w in data["chain"].items()}
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
            logger.info("Served chain widths %s unavailable or invalid (%s); exploring.", self._path, e)
            return False
        return self._start_from(decided, self._path)

    def _save(self) -> None:
        if not (self._writer and self._path):
            return
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            with open(f"{self._path}.tmp", "w", encoding="utf-8") as f:
                json.dump({**self._identity, "chain": {str(k): w for k, w in sorted(self.decided.items())}}, f)
            os.replace(f"{self._path}.tmp", self._path)
        except OSError as e:
            logger.warning("Could not write the served chain widths %s: %s", self._path, e)


class ServedTreePolicy(ServedChainPolicy):
    """Speculation or native decode per batch size for a ``serve``-mode tree, chosen on its served rounds.

    The cost model sizes every speculative round (its verify width included); this policy decides only whether a batch
    key speculates at all. Speculation is the one action ``spec_width`` (the tree's maximum verify width, whatever
    width each round verifies), compared with native decode like the best chain width of :class:`ServedChainPolicy`:
    the same whole served intervals, the same order (native decode last, sticky), the same persistence. Its price is
    the tokens its rounds committed (the sum of their accept lengths, kept on the GPU until the comparison), since a
    tree's acceptance has no per-position profile. A round that seeds the live costs (``settled=False``) forces a
    width the cost model would not choose and is not counted.

    Speculation gains less as the batch grows, so native decode at a batch key serves every larger key and
    speculation at a batch key every smaller one, both unmeasured. ``packaged`` choices map a batch key to 1
    (speculate) or 0 (native decode).
    """

    def __init__(
        self,
        spec_width: int,
        device: torch.device,
        *,
        max_batch_size: int,
        writer: bool,
        sync: Callable[[object], object],
        cache_path: str | None = None,
        key: dict | None = None,
        packaged: Dict[int, int] | None = None,
        native_us: Callable[[int], float | None] | None = None,
    ):
        self._tokens: torch.Tensor | None = None  # the committed tokens of the last speculative round
        super().__init__(
            [spec_width],
            device,
            max_batch_size=max_batch_size,
            native_decode=True,
            writer=writer,
            sync=sync,
            cache_path=cache_path,
            key=key,
            packaged=None if packaged is None else {k: spec_width if c else 0 for k, c in packaged.items()},
            native_us=native_us,
        )

    def round_start(self, bs: int, width: int, requests: Sequence, settled: bool = True) -> None:
        super().round_start(bs, self.widths[0] if width else 0, requests, settled)

    def accepted(self, width: int, accept_lens: torch.Tensor) -> None:
        if width >= 1 and self._writer:
            self._tokens = accept_lens.sum()

    def _period_tokens(self, width: int) -> object:
        return self._tokens if width and self._writer else None

    def _rates(self, key: int, pair: Tuple[int, int]) -> Dict[int, Tuple[float, float]]:
        rates = {}
        for width in pair:
            samples = _settled(self._samples[(key, width)])
            if not samples:
                continue
            if width:
                tokens = torch.stack([t for _, _, t in samples]).tolist()  # the comparison's one device read
                rates[width] = _ratio_rate([(us, t) for (us, _, _), t in zip(samples, tokens)])
            else:
                rates[width] = _ratio_rate([(us, n) for us, n, _ in samples])
        return rates

    def _infer(self) -> None:
        super()._infer()
        spec = self.widths[0]
        for key in self._grid:
            if key not in self.decided and any(w == spec for k, w in self.decided.items() if k > key):
                self.decided[key] = spec
                self._pair.pop(key, None)


class ChainRounds:
    """The served/calibrated chain width and optional recording of completed rounds.

    Drafting asks for ``draft_width`` once when it queues a round. Native decode
    reads ``plain_width`` without advancing that plan, including a prelaunched
    choice of OFF. Context is read only for calibration, without a device sync.
    A ``policy`` (:class:`ServedChainPolicy`) chooses the width of each batch on the served rounds instead.
    """

    def __init__(
        self,
        schedule: Sequence[tuple[int, int, int]] | None,
        calibration: ChainCalibration | None,
        average_context: Callable[[ScheduleBatch], int],
        policy: ServedChainPolicy | ServedTreePolicy | None = None,
    ):
        self._schedule = schedule
        self._calibration = calibration
        self._average_context = average_context
        self._policy = policy

    def draft_width(self, bs: int) -> int:
        if self._policy is not None:
            width = self._policy.width(bs)
        else:
            width = self._calibration.width(bs) if self._calibration is not None else width_at(self._schedule, bs)
        return max(1, width)

    def plain_width(self, bs: int) -> int:
        if self._policy is not None:
            return self._policy.width(bs)  # this batch's own choice: a prelaunch planned the previous batch
        return self._calibration.last_width if self._calibration is not None else width_at(self._schedule, bs)

    def interrupt(self) -> None:
        if self._calibration is not None:
            self._calibration.interrupt()

    def round_start(self, bs: int, width: int, batch: ScheduleBatch, settled: bool = True) -> None:
        """A decode round starts; ``settled``: False for a round forced to seed the live costs."""
        if self._calibration is not None:
            self._calibration.round_start(bs, width, self._average_context(batch))
        if self._policy is not None:
            self._policy.round_start(bs, width, batch.reqs, settled)

    def accepted(self, width: int, accept_lens: torch.Tensor) -> None:
        if self._calibration is not None:
            self._calibration.accepted(width, accept_lens)
        if self._policy is not None and self._policy.exploring(len(accept_lens)):
            self._policy.accepted(width, accept_lens)
