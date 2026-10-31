# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Live costs: the cost model's tables measured from the served rounds, keyed to the serving setup.

Offline calibration prices the draft step and the target verify on dummy batches, ahead of time, for one GPU, TP
size, model pair, engine build and dtype. :class:`LiveCosts` prices them on the engine's own served rounds
instead: a seed sweep forces each captured verify width for a few rounds per batch bucket (the exact kernels
served, at the real context), then every served round refines its cell (an EMA per batch bucket, verify width
and context interval). A cell's value is the fastest of its first samples, and a later sample enters the EMA at
most at :data:`OUTLIER_RATIO` times the cell, so one stalled round (a kernel compiled on first use, a host
hiccup) moves a cell by a few percent at most and a lasting change still converges. The cells become the same
:class:`~eagle_harness.core.cost_table.BatchCostTable` objects the :class:`~eagle_harness.core.cost_model.CostModel`
reads, so the decision code has one input.

A :class:`CostKey` names the setup the costs belong to. Offline tables carry it in ``key.json`` under their
``data_path``; live costs carry it in their cache file. Costs whose key differs from the server's are refused
(offline) or measured again (live); an unkeyed offline directory is accepted with a warning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from eagle_harness.core.cost_table import BatchCostTable, CostCurve, IntervalCost

logger = logging.getLogger(__name__)

KEY_FILE = "key.json"


@dataclass(frozen=True)
class CostKey:
    """What a set of costs was measured on: the costs are valid for exactly this setup."""

    gpu: str  # device name, e.g. "NVIDIA B300"
    gpu_count: int  # TP size: the per-rank shapes depend on it
    model: str  # target config digest (see config_digest)
    draft: str  # draft config digest
    engine: str  # engine name and version, e.g. "sglang 0.5.23"
    dtype: str  # target dtype, e.g. "torch.bfloat16"

    def to_dict(self) -> dict:
        """The key as JSON data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CostKey":
        """A key from JSON data."""
        return cls(**{f: data[f] for f in cls.__dataclass_fields__})

    @property
    def digest(self) -> str:
        """A short file-name-safe hash of the key."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def config_digest(model_path: str) -> str:
    """Identify a model: the digest of its ``config.json`` and its path's last component (a Hub id or a local dir)."""
    name = os.path.basename(os.path.normpath(model_path))
    config = os.path.join(model_path, "config.json")
    h = hashlib.sha256(name.encode())
    if os.path.isfile(config):
        with open(config, "rb") as f:
            h.update(f.read())
    else:
        h.update(model_path.encode())
    return f"{name}:{h.hexdigest()[:12]}"


def write_key(data_path: str, key: CostKey) -> None:
    """Record the setup a calibration directory was measured on."""
    os.makedirs(data_path, exist_ok=True)
    with open(os.path.join(data_path, KEY_FILE), "w", encoding="utf-8") as f:
        json.dump(key.to_dict(), f, indent=2)


def read_key(data_path: str) -> Optional[CostKey]:
    """The key a calibration directory carries, None when it has none (tables measured before keys existed)."""
    try:
        with open(os.path.join(data_path, KEY_FILE), encoding="utf-8") as f:
            return CostKey.from_dict(json.load(f))
    except FileNotFoundError:
        return None


def check_key(data_path: str, key: CostKey) -> None:
    """Refuse offline tables measured on another setup; warn on unkeyed ones."""
    stored = read_key(data_path)
    if stored is None:
        logger.warning(
            "Cost tables under %s carry no %s: they are assumed to be for %s. Calibrate again to key them.",
            data_path,
            KEY_FILE,
            key,
        )
        return
    if stored != key:
        differing = {f: (getattr(stored, f), getattr(key, f)) for f in key.__dataclass_fields__ if getattr(stored, f) != getattr(key, f)}
        raise ValueError(
            f"Cost tables under {data_path} were measured on another setup (stored vs this server): {differing}. "
            "Calibrate for this setup, or drop data_path to measure costs on the served rounds."
        )


# ------------------------------------------------------------------ live cells
# Cache files written under another rule for the cells are measured again.
CACHE_VERSION = 2
# A refining sample counts at most at this multiple of its cell.
OUTLIER_RATIO = 2.0


@dataclass
class Cell:
    """One timed shape: microseconds (min of the seed samples, then an EMA) and the sample count."""

    us: float = 0.0
    samples: int = 0


DRAFT_WIDTH = 0  # cell width of the draft step (one tree level, top-k rows per request)


@dataclass
class LiveCosts:
    """Round timings per (batch bucket, verify width, context interval), and the tables built from them.

    ``widths(bucket)`` lists the verify widths with a graph at the bucket (what the seed sweep must time).
    A bucket is *ready* once every width and the draft step have ``seed_samples``; until then
    :meth:`seed_width` names the width the next round must force.
    """

    widths: Callable[[int], Sequence[int]]
    key: Optional[CostKey] = None
    draft_shape: Tuple[int, int] = (1, 1)  # top-k and depth: the timed step's rows and tree-attention work
    context_step: int = 1024
    context_intervals: int = 16
    seed_samples: int = 3
    alpha: float = 0.05
    cells: Dict[Tuple[int, int, int], Cell] = field(default_factory=dict)  # (bucket, width, interval)
    # Sample counts per (bucket, width) and the ready buckets, indexed from ``cells`` (rebuilt when it is replaced):
    # the drafter asks every round, and a scan of every cell costs far more than the answer.
    _counts: Dict[Tuple[int, int], int] = field(default_factory=dict, init=False, repr=False, compare=False)
    _ready: set = field(default_factory=set, init=False, repr=False, compare=False)
    _indexed: Optional[dict] = field(default=None, init=False, repr=False, compare=False)

    def interval(self, context: int) -> int:
        """Context interval of an average committed length (the tables' ``interval_index``)."""
        return min(max((context - self.context_step) // self.context_step, 0), self.context_intervals - 1)

    def _index(self) -> None:
        """Bring the counts and ready buckets up to date with ``cells`` (a new dict after a load or a TP sync)."""
        if self._indexed is self.cells:
            return
        self._indexed = self.cells
        self._counts = {}
        for (b, w, _), c in self.cells.items():
            self._counts[b, w] = self._counts.get((b, w), 0) + c.samples
        self._ready = {b for b, _ in self._counts if self._is_ready(b)}

    def _samples(self, bucket: int, width: int) -> int:
        self._index()
        return self._counts.get((bucket, width), 0)

    def seed_width(self, bucket: int) -> Optional[int]:
        """The verify width the next round at ``bucket`` must force, None once the bucket is seeded.

        Width 1 first (the trivial verify prices the target's floor), then the others widest first. The draft
        step is timed in every seed round, so it is seeded with them.
        """
        for width in sorted(self.widths(bucket), key=lambda w: (w != 1, -w)):
            if self._samples(bucket, width) < self.seed_samples:
                return width
        return None

    def _is_ready(self, bucket: int) -> bool:
        return self.seed_width(bucket) is None and self._samples(bucket, DRAFT_WIDTH) >= self.seed_samples

    def ready(self, bucket: int) -> bool:
        """Whether the bucket's cells price every captured width and the draft step."""
        self._index()
        return bucket in self._ready

    def _bracket(self, bucket: int) -> Optional[Tuple[int, int]]:
        """The nearest ready buckets below and above an unready ``bucket``, None unless both exist."""
        if self.ready(bucket):
            return None
        below = max((b for b in self._ready if b < bucket), default=None)
        above = min((b for b in self._ready if b > bucket), default=None)
        return None if below is None or above is None else (below, above)

    def priced(self, bucket: int) -> bool:
        """Whether the bucket has a table without a seed sweep: it is ready, or it lies between two ready
        buckets, which price it by linear interpolation in the batch size (:meth:`tables`).

        Verify and draft costs grow smoothly with the batch size, so a bucket the load visits only briefly (the
        sizes a batch drains through) is priced from its neighbours instead of forcing narrow verify widths on its
        served rounds; its own served rounds refine the widths they run.
        """
        return self.ready(bucket) or self._bracket(bucket) is not None

    def record(self, bucket: int, width: int, context: int, us: float) -> None:
        """Add one served timing (``width`` 0: one draft step)."""
        if not math.isfinite(us) or us <= 0:
            return
        self._index()
        cell = self.cells.setdefault((bucket, width, self.interval(context)), Cell())
        if cell.samples < self.seed_samples:
            cell.us = us if cell.samples == 0 else min(cell.us, us)
        else:
            cell.us += self.alpha * (min(us, OUTLIER_RATIO * cell.us) - cell.us)
        cell.samples += 1
        self._counts[bucket, width] = self._counts.get((bucket, width), 0) + 1
        if bucket not in self._ready and self._is_ready(bucket):
            self._ready.add(bucket)

    # ---------------------------------------------------------------- tables
    def _value(self, bucket: int, width: int, interval: int) -> Optional[float]:
        """Use the nearest fully seeded interval, or the pooled seed minimum if none is fully seeded yet."""
        for d in range(self.context_intervals):
            for i in (interval - d, interval + d):
                cell = self.cells.get((bucket, width, i))
                if cell is not None and cell.samples >= self.seed_samples:
                    return cell.us
        # Seeding counts samples across intervals; crossing a boundary must still discard a first-use stall.
        return min((c.us for (b, w, _), c in self.cells.items() if b == bucket and w == width and c.samples), default=None)

    def verify_us(self, bucket: int, width: int) -> Optional[float]:
        """The priced verify of ``width`` tokens per request at ``bucket`` (shortest context first), None untimed."""
        value = self._value(bucket, width, 0)
        if value is None and self._bracket(bucket) is not None:
            value = self._priced(bucket, width, 0)
        return value

    def _width_value(self, bucket: int, width: int, interval: int) -> float:
        """A ready bucket's cost at ``width``, linear between its measured widths (they include 1 and the widest)."""
        if width == DRAFT_WIDTH or width in self.widths(bucket):
            return self._value(bucket, width, interval) or 0.0
        widths = sorted(self.widths(bucket))
        lo = max((w for w in widths if w < width), default=widths[0])
        hi = min((w for w in widths if w > width), default=widths[-1])
        lo_us, hi_us = self._value(bucket, lo, interval) or 0.0, self._value(bucket, hi, interval) or 0.0
        return lo_us if hi == lo else lo_us + (width - lo) / (hi - lo) * (hi_us - lo_us)

    def _priced(self, bucket: int, width: int, interval: int) -> float:
        """An unready bucket's cost: its own fully sampled cell at this interval, else linear in the batch size
        between the ready buckets around it (:meth:`priced`)."""
        own = self.cells.get((bucket, width, interval))
        if own is not None and own.samples >= self.seed_samples:
            return own.us
        bracket = self._bracket(bucket)
        if bracket is None:
            raise ValueError(f"batch bucket {bucket} has no ready buckets on both sides")
        below, above = bracket
        lo_us, hi_us = self._width_value(below, width, interval), self._width_value(above, width, interval)
        return lo_us + (bucket - below) / (above - below) * (hi_us - lo_us)

    def tables(self, also: Sequence[int] = ()) -> Dict[int, BatchCostTable]:
        """One table per priced bucket that has cells or is in ``also`` (the cost model's input), from the current
        cells: a ready bucket's own, an unready one's interpolated from its neighbours where it lacks them."""
        tables: Dict[int, BatchCostTable] = {}
        for bucket in sorted({b for b, _, _ in self.cells} | set(also)):
            if self.ready(bucket):
                value = self._value
            elif self._bracket(bucket) is not None:
                value = self._priced
            else:
                continue
            widths = sorted(self.widths(bucket))
            intervals: List[IntervalCost] = []
            ratios: List[float] = []
            for i in range(self.context_intervals):
                target = [value(bucket, w, i) or 0.0 for w in widths]
                draft = value(bucket, DRAFT_WIDTH, i) or 0.0
                intervals.append(IntervalCost(_curve(target, [w - 1 for w in widths]), _curve([draft], [0])))
                ratios.append(round(target[0] / draft, 4))
            tables[bucket] = BatchCostTable(
                batch_size=bucket,
                cache_length_start=self.context_step,
                cache_length_step=self.context_step,
                target_max_qo_len_query=widths[-1],
                target_max_qo_len_cache=widths[-1],
                draft_max_qo_len=1,
                intervals=tuple(intervals),
                model_ratio=tuple(ratios),
            )
        return tables

    # ----------------------------------------------------------------- cache
    def to_dict(self) -> dict:
        """The cells and key as JSON data."""
        return {
            "version": CACHE_VERSION,
            "key": None if self.key is None else self.key.to_dict(),
            "draft_shape": list(self.draft_shape),
            "context_step": self.context_step,
            "context_intervals": self.context_intervals,
            "cells": [[b, w, i, c.us, c.samples] for (b, w, i), c in sorted(self.cells.items())],
        }

    def save(self, path: str) -> None:
        """Write the cells to ``path`` (atomically)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        """Take the cells of a cache file whose key and grid match; False (nothing taken) otherwise."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("expected an object")
            key = None if data.get("key") is None else CostKey.from_dict(data["key"])
            if (
                data.get("version") != CACHE_VERSION
                or key != self.key or data.get("draft_shape") != list(self.draft_shape)
                or (data["context_step"], data["context_intervals"]) != (self.context_step, self.context_intervals)
            ):
                logger.info("Live cost cache %s is for another setup; measuring again.", path)
                return False
            cells = {}
            for b, w, i, us, n in data["cells"]:
                if (
                    any(type(v) is not int for v in (b, w, i, n))
                    or b <= 0 or n <= 0 or not 0 <= i < self.context_intervals
                    or w not in (DRAFT_WIDTH, *self.widths(b))
                    or isinstance(us, bool) or not math.isfinite(us) or us <= 0
                    or (b, w, i) in cells
                ):
                    raise ValueError("invalid or duplicate timing cell")
                cells[b, w, i] = Cell(us, n)
        except (OSError, ValueError, TypeError, KeyError) as e:
            logger.info("Live cost cache %s is unavailable or invalid (%s); measuring again.", path, e)
            return False
        self.cells = cells
        return True


def _curve(values: Sequence[float], index: Sequence[int]) -> CostCurve:
    base = values[0]
    return CostCurve(
        value=tuple(values),
        ratio=tuple(round(v / base, 4) for v in values),
        index=tuple(index),
        array_query=(),
        array_cache=(),
    )
