# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Calibration: measurement records, the cost fits, and the on-disk layout.

The files written here match the golden fixtures for the same raw measurements
(``tests/test_cost_golden.py``), preserving the calibrated cost-table layout.

Layout under a calibration ``data_path``::

    config/measure_config.json            MeasureParams of the batch being measured
    config/task_normal.json               [{"task_normal": bool}]: false = unstable, retry
    test/bs_<N>/{model,cache}_{slm,llm}/  raw timings (jsonl, one record per line)
    cost/bs_<N>/{cache,query,model}_{slm,llm}/  fitted costs (json)

``slm`` is the draft model, ``llm`` the target.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from eagle_harness.core.limits import (
    ComputeIntensity,
    draft_query_lens,
    verify_buckets,
)

logger = logging.getLogger(__name__)

DRAFT = "slm"
TARGET = "llm"


# --------------------------------------------------------------------- records
@dataclass
class MeasureRecord:
    """One timing: ``time_us`` for ``measure_count`` repeats of one shape."""

    time_us: int
    measure_count: int = 1
    query_length: int = 1
    cache_length: int = 64
    task_category: str = "cache_llm"
    batch_size: int = 1

    def to_dict(self) -> dict:
        """The measurement record in the on-disk JSON format."""
        return {
            "time_intervals": self.time_us,
            "measure_count": self.measure_count,
            "query_length": self.query_length,
            "cache_length": self.cache_length,
            "task_category": self.task_category,
            "batch_size": self.batch_size,
        }


def save_jsonl(path: str, records: Sequence[MeasureRecord]) -> None:
    """Write ``records`` to ``path``, one JSON object per line."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[dict]:
    """Read the JSON objects of a JSON-lines file."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line in %s", path)
    return rows


def _save_json_lines(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------- parameters
@dataclass
class MeasureParams:
    """What one batch size's calibration measures (``config/measure_config.json``)."""

    context_step: int
    context_stop: int
    batch_size: int
    verify_num: int
    node_top_k: int
    loop_top_m: int
    repeat_cnt: int
    enable_fast_measurement: bool = False
    cache_len: int = 1
    query_llm: int = 1
    query_slm: int = 1


@dataclass(frozen=True)
class CalibrationPaths:
    """Layout of a calibration directory: ``test/`` raw measurements, ``cost/`` fitted tables, ``config/``."""

    data_path: str

    @property
    def cost(self) -> str:
        """Directory of the fitted cost tables."""
        return os.path.join(self.data_path, "cost")

    @property
    def test(self) -> str:
        """Directory of the raw measurements."""
        return os.path.join(self.data_path, "test")

    @property
    def config(self) -> str:
        """Directory of the calibration settings."""
        return os.path.join(self.data_path, "config")

    def raw(self, batch_size: int, kind: str, model: str) -> str:
        """Raw measurement file of one batch size, kind and model."""
        return os.path.join(self.test, f"bs_{batch_size}", f"{kind}_{model}")

    def fitted(self, batch_size: int, kind: str, model: str) -> str:
        """Fitted cost file of one batch size, kind and model."""
        return os.path.join(self.cost, f"bs_{batch_size}", f"{kind}_{model}")

    def write_measure_params(self, params: MeasureParams) -> None:
        """Store the measurement settings."""
        _save_json_lines(os.path.join(self.config, "measure_config.json"), [asdict(params)])

    def set_task_normal(self, normal: bool) -> None:
        """Store the ``task_normal`` flag of the calibration run."""
        _save_json_lines(os.path.join(self.config, "task_normal.json"), [{"task_normal": normal}])

    def task_normal(self) -> bool:
        """The stored ``task_normal`` flag."""
        rows = load_jsonl(os.path.join(self.config, "task_normal.json"))
        return bool(rows[0]["task_normal"])


@dataclass(frozen=True)
class CacheSweepPlan:
    """Grid of one attention-cost sweep with validated cache-length bounds."""

    kv_start: int
    kv_step: int  # timings below this cache length are warm-up and dropped
    kv_stop: int
    interval_step: int
    num_layers: int

    @classmethod
    def from_params(cls, params: MeasureParams) -> "CacheSweepPlan":
        """The sweep plan of the measurement settings (context step and stop are validated)."""
        step, stop = params.context_step, params.context_stop
        if step not in (64, 128, 256, 512, 1024):
            raise ValueError(f"context_step must be 64, 128, 256, 512 or 1024, got {step}.")
        if not 2048 <= stop <= 128 * 1024:
            raise ValueError(f"context_stop must be in [2048, 131072], got {stop}.")
        if stop % step != 0 or not 16 <= stop // step <= 128:
            raise ValueError(f"context_stop must be 16..128 multiples of context_step, got {stop} / {step}.")
        num_layers = params.repeat_cnt
        interval_step = step // 32
        if params.enable_fast_measurement:
            interval_step = step // 16
            if stop // step >= 32 and stop >= 32 * 1024:
                num_layers = max(num_layers // 4, 1)
            elif (stop // step >= 32 and stop >= 8 * 1024) or stop >= 16 * 1024:
                num_layers = max(num_layers // 2, 1)
        return cls(step // 2, step, stop, interval_step, num_layers)

    def cache_lengths(self) -> range:
        """The cache lengths the sweep measures."""
        return range(self.kv_start, self.kv_stop + 1, self.interval_step)


def query_lens(category: str, batch_size: int, intensity: ComputeIntensity, max_query_length: int) -> List[int]:
    """Query lengths a fit reads: draft widths for the draft, verify buckets for the target."""
    if category.endswith(DRAFT):
        return [q for q in draft_query_lens(batch_size, intensity) if q <= max_query_length]
    if category.endswith(TARGET):
        return sorted(q for q in verify_buckets(batch_size, intensity) if q <= max_query_length)
    return list(range(1, max_query_length + 1))


# ------------------------------------------------------------------- fitting
def remove_outliers(
    data: Sequence[float],
    initial_threshold: float = 6.00,
    threshold_decrement: float = 0.20,
    min_threshold: float = 1.00,
    convergence_threshold: float = 0.20,
    max_iterations: int = 50,
    sample_rate: float = 0.75,
) -> Tuple[float, List[float], bool]:
    """Iterative z-score filter. Returns (mean, kept values, stable)."""
    if len(data) == 0:
        raise ValueError("Input data is empty.")
    kept = list(data)
    threshold = initial_threshold
    range_rate = 1.00
    mean = np.mean(kept)
    for _ in range(max_iterations):
        values = np.asarray(kept, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.abs((values - values.mean()) / values.std())
        kept = [x for i, x in enumerate(kept) if z[i] < threshold]
        if kept:
            mean = np.mean(kept)
        if len(kept) < sample_rate * len(data):
            break
        range_rate = (np.max(kept) - np.min(kept)) / mean
        if mean != 0 and range_rate <= convergence_threshold:
            break
        threshold = max(threshold - threshold_decrement, min_threshold)
    stable = (range_rate <= convergence_threshold) and (len(kept) >= sample_rate * len(data))
    return mean, kept, stable


def _interval_ids(path: str, prefix: str) -> int:
    max_id = -1
    for name in os.listdir(path):
        if name.startswith(prefix) and name.endswith(".json"):
            try:
                max_id = max(max_id, int(name.split("_")[-1].split(".")[0]))
            except ValueError as e:
                raise RuntimeError(f"Unexpected file name format: {name}") from e
    if max_id == -1:
        raise FileNotFoundError(f"No valid '{prefix}*.json' files found in {path}")
    return max_id


def fit_cache_costs(
    category: str,
    source: str,
    target: str,
    *,
    max_query_length: int,
    cache_length_start: int,
    cache_length_step: int,
    cache_length_stop: int,
    batch_size: int,
    intensity: ComputeIntensity,
    fast: bool = False,
    paths: Optional[CalibrationPaths] = None,
) -> float:
    """Per-interval attention cost per layer, per query length. Returns the
    instability rate; above the limit, ``paths`` gets ``task_normal=false``."""
    interval_data: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for qo_len in query_lens(category, batch_size, intensity, max_query_length):
        file_path = os.path.join(source, f"cache_qo_len_{qo_len}.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found: {file_path}")
        for record in load_jsonl(file_path):
            kv_len = record["cache_length"]
            if not cache_length_start <= kv_len < cache_length_stop:
                continue
            interval_id = (kv_len - cache_length_start) // cache_length_step
            interval_data[interval_id][qo_len].append(record["time_intervals"] / record["measure_count"])

    means: Dict[int, list] = defaultdict(lambda: [0.0] * max_query_length)
    total = unstable = 0
    for interval_id, by_q in interval_data.items():
        for qo_len, times in by_q.items():
            total += 1
            mean, _, stable = remove_outliers(times, convergence_threshold=0.25 if fast else 0.35)
            unstable += not stable
            means[interval_id][qo_len - 1] = mean
    instability = unstable / total if total > 0 else 0.0
    if instability >= (0.20 if fast else 0.25):
        logger.warning("High cache-cost instability: %.2f%%", instability * 100)
        if paths is not None:
            paths.set_task_normal(False)

    for interval_id in range((cache_length_stop - cache_length_start) // cache_length_step):
        sequence = means.get(interval_id, [0.0] * max_query_length)
        ratio = [round(v / sequence[0], 4) for v in sequence] if sequence and sequence[0] > 0 else []
        _write_json(
            os.path.join(target, f"cache_interval_{interval_id}.json"),
            {
                "value": [round(v, 4) for v in sequence],
                "index": list(range(len(sequence))),
                "ratio": [round(v, 4) for v in ratio],
                "array": [round(v, 4) for v in sequence],
                "batch_size": batch_size,
                "max_query_length": max_query_length,
                "cache_length_start": cache_length_start,
                "cache_length_step": cache_length_step,
                "cache_length_stop": cache_length_stop,
                "instability_rate": round(instability, 2),
            },
        )
    return instability


def _load_cache_arrays(path: str) -> Dict[int, list]:
    arrays = {}
    for i in range(_interval_ids(path, "cache_interval_") + 1):
        file_path = os.path.join(path, f"cache_interval_{i}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expected cache interval file not found: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            arrays[i] = json.load(f).get("array", [])
    return arrays


def fit_query_costs(
    category: str,
    model_source: str,
    cache_source: str,
    target: str,
    *,
    max_query_length: int,
    cache_length_start: int,
    cache_length_step: int,
    cache_layers: int,
    batch_size: int,
    intensity: ComputeIntensity,
    average: bool = False,
    sliding_window_layers: int = 0,
    sliding_window: int = 0,
    paths: Optional[CalibrationPaths] = None,
) -> float:
    """Non-attention cost per query length: measured step minus fitted attention."""
    cache_costs = _load_cache_arrays(cache_source)
    lens = query_lens(category, batch_size, intensity, max_query_length)
    query_costs: Dict[int, List[float]] = defaultdict(list)
    for qo_len in lens:
        file_path = os.path.join(model_source, f"model_qo_len_{qo_len}.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found: {file_path}")
        for record in load_jsonl(file_path):
            kv_len = record["cache_length"]
            if kv_len < cache_length_start:
                continue
            interval_id = (kv_len - cache_length_start) // cache_length_step
            if interval_id not in cache_costs or len(cache_costs[interval_id]) < qo_len:
                continue
            cost_cache = cache_costs[interval_id][qo_len - 1]
            cost = record["time_intervals"] / record["measure_count"] - cost_cache * cache_layers
            if sliding_window_layers > 0 and kv_len > sliding_window:
                swa_id = max((sliding_window - cache_length_start) // cache_length_step, 0)
                cost += sliding_window_layers * (cost_cache - cache_costs[swa_id][qo_len - 1])
            query_costs[qo_len].append(cost)

    total = unstable = 0
    for qo_len in lens:
        costs = query_costs.get(qo_len, [])
        if not costs:
            raise RuntimeError(f"No query cost data available for query length {qo_len}.")
        total += 1
        mean, _, stable = remove_outliers(costs, convergence_threshold=0.50 if average else 0.30)
        unstable += not stable
        _write_json(
            os.path.join(target, f"query_{qo_len}.json"),
            {"mean": round(mean, 4), "array": [round(v, 4) for v in costs], "batch_size": batch_size},
        )
    instability = unstable / total if total > 0 else 0.0
    if instability >= 0.30:
        logger.warning("High query-cost instability: %.2f%%", instability * 100)
        if paths is not None:
            paths.set_task_normal(False)
    for qo_len in lens:
        file_path = os.path.join(target, f"query_{qo_len}.json")
        with open(file_path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["instability_rate"] = round(instability, 4)
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=4)
    return instability


def fit_model_costs(
    category: str,
    query_source: str,
    cache_source: str,
    target: str,
    *,
    max_query_length_query: int,
    max_query_length_cache: int,
    cache_length_start: int,
    cache_length_step: int,
    cache_layers: int,
    batch_size: int,
    intensity: ComputeIntensity,
    sliding_window_layers: int = 0,
    sliding_window: int = 0,
) -> None:
    """Model cost per interval and query length = query cost + attention cost.

    Writes the ``model_interval_<i>.json`` files :mod:`cost_table` reads.
    """
    lens = query_lens(category, batch_size, intensity, max_query_length_query)
    query_costs = [0.0] * len(lens)
    for i, qo_len in enumerate(lens):
        file_path = os.path.join(query_source, f"query_{qo_len}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expected query cost file not found: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            query_costs[i] = json.load(f).get("mean", 0.0)

    max_id = _interval_ids(cache_source, "cache_interval_")
    swa_index = max(sliding_window // cache_length_step - 1, 0)
    sequences: List[list] = []
    for interval_id in range(max_id + 1):
        with open(os.path.join(cache_source, f"cache_interval_{interval_id}.json"), encoding="utf-8") as f:
            cache_data = json.load(f)
        sequence = cache_data.get("array", [])
        sequences.append(sequence)
        if not sequence or len(sequence) != max_query_length_cache:
            raise RuntimeError(f"Cache cost data for interval {interval_id} is invalid.")
        costs = [0] * len(lens)
        for i, n in enumerate(lens):
            if swa_index == 0 or interval_id <= swa_index:
                costs[i] = query_costs[i] + sequence[n - 1] * cache_layers
            else:
                full = sequence[n - 1] * (cache_layers - sliding_window_layers)
                windowed = sequences[swa_index][n - 1] * sliding_window_layers
                costs[i] = query_costs[i] + full + windowed
        ratio = [round(v / costs[0], 4) for v in costs] if costs and costs[0] > 0 else []
        _write_json(
            os.path.join(target, f"model_interval_{interval_id}.json"),
            {
                "notes": "Estimated model_cost = query_cost + cache_cost",
                "value": [round(v, 4) for v in costs],
                "index": [q - 1 for q in lens],
                "ratio": ratio,
                "array_model": [round(v, 4) for v in costs],
                "array_query": [round(v, 4) for v in query_costs],
                "array_cache": [round(v * cache_layers, 4) for v in sequence],
                "batch_size": batch_size,
                "max_query_length_query": max_query_length_query,
                "max_query_length_cache": max_query_length_cache,
                "cache_length_start": cache_length_start,
                "cache_length_step": cache_length_step,
                "cache_layers_num": cache_layers,
                "instability_rate": cache_data.get("instability_rate", -1.0),
            },
        )


# ---------------------------------------------------------------- one batch
@dataclass(frozen=True)
class AttentionShape:
    """What the fits need of a model: attention layers and sliding-window layers."""

    attention_layers: int
    sliding_window_layers: int = 0
    sliding_window: int = 0


def save_model_timings(paths: CalibrationPaths, params: MeasureParams, records: Sequence[MeasureRecord]) -> None:
    """Group step timings by model and query length into ``model_qo_len_<q>.jsonl``."""
    groups: Dict[Tuple[str, int], List[MeasureRecord]] = defaultdict(list)
    for record in records:
        model = record.task_category.split("_")[-1]
        groups[(model, record.query_length)].append(record)
    for (model, qo_len), group in sorted(groups.items()):
        save_jsonl(os.path.join(paths.raw(params.batch_size, "model", model), f"model_qo_len_{qo_len}.jsonl"), group)


def fit_batch(
    paths: CalibrationPaths,
    params: MeasureParams,
    intensity: ComputeIntensity,
    draft: AttentionShape,
    target: AttentionShape,
    *,
    tp_size: int = 1,
) -> None:
    """Fit one batch size from raw timings, processing the draft before the target."""
    bs = params.batch_size
    start, step, stop = params.context_step, params.context_step, params.context_stop
    for model, max_q in ((DRAFT, params.node_top_k), (TARGET, params.verify_num)):
        fit_cache_costs(
            f"cache_all_{model}",
            paths.raw(bs, "cache", model),
            paths.fitted(bs, "cache", model),
            max_query_length=max_q,
            cache_length_start=start,
            cache_length_step=step,
            cache_length_stop=stop,
            batch_size=bs,
            intensity=intensity,
            fast=params.enable_fast_measurement,
            paths=paths,
        )
    for model, shape, max_q, average in (
        (DRAFT, draft, params.node_top_k, (not params.enable_fast_measurement) or tp_size > 1),
        (TARGET, target, params.loop_top_m, False),
    ):
        fit_query_costs(
            f"query_{model}",
            paths.raw(bs, "model", model),
            paths.fitted(bs, "cache", model),
            paths.fitted(bs, "query", model),
            max_query_length=max_q,
            cache_length_start=start,
            cache_length_step=step,
            cache_layers=shape.attention_layers,
            batch_size=bs,
            intensity=intensity,
            average=average,
            sliding_window_layers=shape.sliding_window_layers,
            sliding_window=shape.sliding_window,
            paths=paths,
        )
    for model, shape, max_q, max_cache in (
        (DRAFT, draft, params.node_top_k, params.node_top_k),
        (TARGET, target, params.loop_top_m, params.verify_num),
    ):
        fit_model_costs(
            f"model_{model}",
            paths.fitted(bs, "query", model),
            paths.fitted(bs, "cache", model),
            paths.fitted(bs, "model", model),
            max_query_length_query=max_q,
            max_query_length_cache=max_cache,
            cache_length_start=start,
            cache_length_step=step,
            cache_layers=shape.attention_layers,
            batch_size=bs,
            intensity=intensity,
            sliding_window_layers=shape.sliding_window_layers,
            sliding_window=shape.sliding_window,
        )


# ------------------------------------------------- engines that time whole steps only
@dataclass(frozen=True)
class StepTiming:
    """One timed forward of a step: ``query_length`` tokens per request at ``cache_length``, ``time_us`` microseconds."""

    query_length: int
    cache_length: int
    time_us: float


def attention_growth_records(
    timings: Sequence[StepTiming], attention_layers: int, category: str, batch_size: int = 1
) -> List[MeasureRecord]:
    """Attention cost per layer from step timings alone: how much the step grows over the shortest cache length, per layer.

    An engine without an attention-only timing (vLLM) still has what the fits need: the model cost is the query cost plus
    ``layers * cache cost``, and the cache cost is exactly what a step gains from a longer cache.
    """
    records = []
    for q in sorted({t.query_length for t in timings}):
        by_length = {t.cache_length: t.time_us for t in timings if t.query_length == q}
        base = min(by_length.values())
        records += [
            MeasureRecord(
                time_us=max(int(1000 * (time - base) / attention_layers), 1),
                measure_count=1000,
                query_length=q,
                cache_length=length,
                task_category=category,
                batch_size=batch_size,
            )
            for length, time in by_length.items()
        ]
    return records


def fit_step_timings(
    paths: CalibrationPaths,
    params: MeasureParams,
    intensity: ComputeIntensity,
    draft: Tuple[AttentionShape, Sequence[StepTiming]],
    target: Tuple[AttentionShape, Sequence[StepTiming]],
    *,
    tp_size: int = 1,
) -> None:
    """Store the raw timings of one batch size and run all fits, from whole-step timings of the draft and the target."""
    paths.write_measure_params(params)
    paths.set_task_normal(True)
    bs = params.batch_size
    for model, (shape, timings) in ((DRAFT, draft), (TARGET, target)):
        save_model_timings(
            paths,
            params,
            [
                MeasureRecord(
                    time_us=int(1000 * t.time_us),
                    measure_count=1000,
                    query_length=t.query_length,
                    cache_length=t.cache_length,
                    task_category=f"model_{model}",
                    batch_size=bs,
                )
                for t in timings
            ],
        )
        by_query: Dict[int, List[MeasureRecord]] = defaultdict(list)
        for record in attention_growth_records(timings, shape.attention_layers, f"cache_{model}", bs):
            by_query[record.query_length].append(record)
        for q, group in by_query.items():
            save_jsonl(os.path.join(paths.raw(bs, "cache", model), f"cache_qo_len_{q}.jsonl"), group)
    fit_batch(paths, params, intensity, draft[0], target[0], tp_size=tp_size)
