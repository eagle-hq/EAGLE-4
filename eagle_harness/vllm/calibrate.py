# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""vLLM whole-step calibration. Use ``eagle-harness calibrate`` for the full sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import torch

from eagle_harness.core.calibration import AttentionShape, CalibrationPaths, MeasureParams, StepTiming, fit_step_timings
from eagle_harness.core.limits import MAX_CHAIN_VERIFY_NUM, ComputeIntensity
from eagle_harness.core.live_costs import CostKey, config_digest, write_key

CONTEXT_STEP = 256
REPEATS = 30
DRAFT_ROWS = (1, 2, 4, 8)


def sweep(step: int = CONTEXT_STEP, *, context_stop: int) -> list[int]:
    """Cache lengths timed: two inside every interval of the fits' grid (intervals of ``step`` from ``step`` to the stop)."""
    if step < 4 or context_stop < 2 * step or context_stop % step:
        raise ValueError("max_model_len must be at least two context steps and a multiple of context_step")
    return [
        step * (1 + i) + offset for i in range((context_stop - step) // step) for offset in (step // 4, 3 * step // 4)
    ]


def time_graph(run: Any) -> float:
    """Median microseconds of ``run()`` replayed as a CUDA graph (captured once after two eager calls)."""
    from vllm.distributed.parallel_state import graph_capture

    run()
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with graph_capture(device=torch.device("cuda")) as capture, torch.cuda.graph(graph, stream=capture.stream):
        run()
    samples = []
    for _ in range(REPEATS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(1000 * start.elapsed_time(end))
    return statistics.median(samples)


def _elapsed_us(run: Any) -> float:
    """Microseconds of one ``run()`` on the current stream (host launch included)."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    run()
    end.record()
    end.synchronize()
    return 1000 * start.elapsed_time(end)


class CalibrationProbe:
    """Worker extension: ``collective_rpc("calibrate_target" | "calibrate_draft", ...)`` run in every worker (rank 0's rows are used)."""

    def calibrate_target(
        self, widths: list[int], lengths: list[int], batch_sizes: list[int], repeats: int = REPEATS
    ) -> list[dict]:
        """Time the target forward of ``q`` tokens per request at every batch size and cache length (``q`` must be 1 or the
        engine's width)."""
        speculator = self.model_runner.speculator
        propose, speculator.propose = speculator.propose, lambda *args, **kwargs: None  # the draft is timed separately
        try:
            return [
                {
                    "batch_size": bs,
                    "query_length": q,
                    "cache_length": length,
                    "time_us": self._time_dummy(lambda bs=bs, q=q, length=length: self._step(bs, q, length), repeats),
                }
                for bs in batch_sizes
                for q in widths
                for length in lengths
            ]
        finally:
            speculator.propose = propose

    def calibrate_chain_draft(
        self, width: int, lengths: list[int], batch_sizes: list[int], repeats: int = REPEATS
    ) -> list[dict]:
        """Time one step of vLLM's chain draft per batch size and cache length: a round (the verify forward and the
        ``width - 1`` draft steps of the engine's chain) minus the verify forward alone, timed interleaved, over the steps."""
        speculator = self.model_runner.speculator
        propose = speculator.propose

        def skip(*args, **kwargs):
            return None

        rows = []
        for bs in batch_sizes:
            for length in lengths:
                spans: dict[bool, list[float]] = {True: [], False: []}
                try:
                    for drafts in (True, False):  # warm both paths (graph replays, metadata buffers)
                        speculator.propose = propose if drafts else skip
                        self._step(bs, width, length)
                    for _ in range(repeats):
                        for drafts in (True, False):
                            speculator.propose = propose if drafts else skip
                            spans[drafts].append(
                                _elapsed_us(lambda bs=bs, length=length: self._step(bs, width, length))
                            )
                finally:
                    speculator.propose = propose
                step = (statistics.median(spans[True]) - statistics.median(spans[False])) / (width - 1)
                rows.append({"batch_size": bs, "query_length": 1, "cache_length": length, "time_us": max(step, 0.0)})
        return rows

    def _step(self, bs: int, q: int, length: int) -> None:
        """One dummy step of ``bs`` requests with ``q`` tokens each at cache length ``length``, on distinct state slots."""
        self.model_runner._dummy_run(q * bs, uniform_decode=q > 1, context_len=length, valid_dummy_state_slots=True)

    def _time_dummy(self, run: Any, repeats: int = REPEATS) -> float:
        run()
        torch.cuda.synchronize()
        return statistics.median(_elapsed_us(run) for _ in range(repeats))

    def calibrate_draft(self, rows_per_step: list[int], lengths: list[int]) -> list[dict]:
        """Time one tree step of the draft (``k`` rows of one request) at every cache length."""
        from eagle_harness.vllm.tree_drafter import TreeDrafter

        speculator = self.model_runner.speculator
        context = speculator.vllm_config.compilation_config.static_forward_context
        layer = context[speculator.draft_layer_name]
        group = next(
            g
            for g, spec in enumerate(speculator.kv_cache_config.kv_cache_groups)
            if speculator.draft_layer_name in spec.layer_names
        )
        block_size = speculator.kv_cache_config.kv_cache_groups[group].kv_cache_spec.block_size
        table = speculator.block_tables.input_block_tables[group][:1]
        table.copy_(
            torch.arange(table.shape[1], device=table.device) % layer.kv_cache.shape[0]
        )  # valid blocks, any content
        out = []
        for k in rows_per_step:
            drafter = TreeDrafter(
                speculator.model,
                layer,
                top_k=k,
                depth=2,
                max_reqs=1,
                device=speculator.device,
            )
            drafter.bind_cache(layer.kv_cache, block_size)
            hidden = torch.randn(
                1, speculator.model.config.hidden_size, device=speculator.device, dtype=drafter.tree.root_hidden.dtype
            )
            drafter.seed(
                1,
                torch.full((1, k), 1.0 / k, device=speculator.device),
                torch.arange(k, device=speculator.device)[None],
                hidden,
            )
            for length in lengths:
                committed = torch.full((1,), length, dtype=torch.int64, device=speculator.device)
                us = time_graph(lambda drafter=drafter, committed=committed: drafter.grow(1, committed, table))
                out.append({"query_length": k, "cache_length": length, "time_us": us})
        return out


def fit(
    data_path: str,
    target_rows: list[dict],
    draft_rows: list[dict],
    target: AttentionShape,
    draft: AttentionShape,
    tp: int,
    *,
    chain: bool = False,
    context_step: int = CONTEXT_STEP,
    context_stop: int,
) -> list[int]:
    """Run the core's fits for every timed batch size on the whole-step timings; returns the batch sizes."""

    def rows(timings, bs):
        return [
            StepTiming(r["query_length"], r["cache_length"], r["time_us"])
            for r in timings
            if r.get("batch_size", 1) == bs
        ]

    batch_sizes = sorted({r.get("batch_size", 1) for r in target_rows})
    for bs in batch_sizes:
        widest = max(r["query_length"] for r in target_rows if r.get("batch_size", 1) == bs)
        params = MeasureParams(
            context_step=context_step,
            context_stop=context_stop,
            batch_size=bs,
            verify_num=widest,
            node_top_k=1 if chain else max(DRAFT_ROWS),
            loop_top_m=widest,
            repeat_cnt=1,
        )
        fit_step_timings(
            CalibrationPaths(data_path),
            params,
            ComputeIntensity.CHAIN if chain else ComputeIntensity.HIGH,
            (draft, rows(draft_rows, bs)),
            (target, rows(target_rows, bs)),
            tp_size=tp,
        )
    return batch_sizes


def engine(model: str, draft: str, tp: int, width: int, chain: bool, max_num_seqs: int, **overrides: Any):
    """The serving engine of a verify width: a static chain (vLLM's chain loop) or a static tree, with the probe."""
    from vllm import LLM
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    if chain:
        method_config = {"mode": "static", "node_top_k": 1, "tree_depth": width - 1, "max_verify_width": width}
        backend = "FLASH_ATTN"
    else:
        method_config = {"mode": "static", "node_top_k": 8, "tree_depth": -(-width // 8) + 1, "max_verify_width": width}
        backend = "CUSTOM"
    args = {
        "model": model,
        "tensor_parallel_size": tp,
        "max_model_len": 4608,
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max(8192, max_num_seqs * width),
        "speculative_config": {
            "method": "eagle_harness",
            "model": draft,
            "method_config": method_config,
            **({"attention_backend": backend} if chain else {}),
        },
        **({"attention_backend": backend} if not chain else {}),
        "seed": 0,
        "worker_extension_cls": "eagle_harness.vllm.calibrate.CalibrationProbe",
    }
    args.update(overrides)
    return LLM(**args)


def measure(
    raw_path: str,
    width: int,
    chain: bool,
    batch_sizes: Sequence[int],
    context_step: int,
    repeats: int,
    *,
    model: str,
    draft: str,
    tp: int = 1,
    engine_args: dict | None = None,
) -> None:
    """Start an engine with verify width ``width``, time the target and the draft, write the raw timings to ``raw_path``."""
    if not chain and list(batch_sizes) != [1]:
        raise ValueError("tree calibration times batch size 1 only (the tree step is timed for one request)")
    if chain and not 2 <= width <= MAX_CHAIN_VERIFY_NUM:
        raise ValueError(f"a chain's width is 2..{MAX_CHAIN_VERIFY_NUM}, got {width}")
    llm = engine(model, draft, tp, width, chain, max(batch_sizes), **(engine_args or {}))
    config = llm.llm_engine.vllm_config
    # Save the resolved configs, including revisions and overrides used by the engine.
    target_shape = attention_shape(config.model_config.hf_config)
    draft_shape = attention_shape(config.speculative_config.draft_model_config.hf_config)
    context_stop = config.model_config.max_model_len
    lengths = sweep(context_step, context_stop=context_stop)
    if lengths[-1] + width > context_stop:
        raise ValueError("context_step leaves too little room for the verify width; increase context_step")
    sizes = sorted(batch_sizes)
    widths = [1, width] if width > 1 else [1]
    target_rows = llm.collective_rpc("calibrate_target", args=(widths, lengths, sizes, repeats))[0]
    if chain:
        draft_rows = llm.collective_rpc("calibrate_chain_draft", args=(width, lengths, sizes, repeats))[0]
    else:
        draft_rows = llm.collective_rpc("calibrate_draft", args=(list(DRAFT_ROWS), lengths))[0]
    raw = {
        "width": width,
        "tp": tp,
        "chain": chain,
        "model": model,
        "draft_model": draft,
        "engine_args": engine_args or {},
        "target_shape": asdict(target_shape),
        "draft_shape": asdict(draft_shape),
        "key": cost_key(model, draft, tp, config.model_config.dtype).to_dict(),
    }
    raw.update(context_step=context_step, context_stop=context_stop, target=target_rows, draft=draft_rows)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    print(f"width {width}: {len(target_rows)} target and {len(draft_rows)} draft timings -> {raw_path}")


def merge(raw_paths: list[str]) -> tuple[list[dict], list[dict], int, bool, int, int]:
    """The raw timings of several engines as one set: a (batch size, query length, cache length) timed by more than one
    engine takes the median. Returns target and draft rows, TP size, chain flag, context step and context stop."""
    raws = []
    for path in raw_paths:
        with open(path, encoding="utf-8") as source:
            raws.append(json.load(source))
    for key in (
        "tp",
        "chain",
        "context_step",
        "context_stop",
        "model",
        "draft_model",
        "engine_args",
        "target_shape",
        "draft_shape",
        "key",
    ):
        values = {
            json.dumps(raw.get(key, {"chain": False, "context_step": CONTEXT_STEP}.get(key)), sort_keys=True)
            for raw in raws
        }
        if len(values) != 1:
            raise ValueError(f"the raw timings differ in {key}: {sorted(values)}")

    def combined(key: str) -> list[dict]:
        groups: dict[tuple[int, int, int], list[float]] = {}
        for raw in raws:
            for r in raw[key]:
                groups.setdefault((r.get("batch_size", 1), r["query_length"], r["cache_length"]), []).append(
                    r["time_us"]
                )
        return [
            {"batch_size": bs, "query_length": q, "cache_length": length, "time_us": statistics.median(v)}
            for (bs, q, length), v in sorted(groups.items())
        ]

    first = raws[0]
    if first.get("context_stop") is None:
        raise ValueError("raw measurements lack context_stop; run calibration again with the served max_model_len")
    return (
        combined("target"),
        combined("draft"),
        first["tp"],
        first.get("chain", False),
        first.get("context_step", CONTEXT_STEP),
        first["context_stop"],
    )


def cost_key(model: str, draft: str, tp: int, dtype: Any) -> CostKey:
    """The setup the tables are measured on, checked by the server that loads them."""
    import vllm

    return CostKey(
        gpu=torch.cuda.get_device_name(),
        gpu_count=tp,
        model=config_digest(model),
        draft=config_digest(draft),
        engine=f"vllm {vllm.__version__}",
        dtype=str(dtype),
    )


def attention_shape(config: Any) -> AttentionShape:
    """Read attention counts from a text config, refusing unknown layer layouts."""
    config = getattr(config, "text_config", config)
    layers = getattr(config, "num_hidden_layers", 0)
    types = getattr(config, "layer_types", None)
    window = getattr(config, "sliding_window", None) or 0
    if types is None:
        if getattr(config, "full_attention_interval", None) or getattr(config, "hybrid_override_pattern", None):
            raise ValueError("hybrid calibration needs explicit layer_types in the model config")
        types = [
            "sliding_attention" if window and getattr(config, "use_sliding_window", True) else "full_attention"
        ] * layers
    known = {"full_attention", "sliding_attention", "linear_attention", "mamba"}
    if len(types) != layers or set(types) - known:
        raise ValueError(f"cannot infer attention layout for calibration: {types!r}")
    count = sum(kind in {"full_attention", "sliding_attention"} for kind in types)
    sliding = types.count("sliding_attention")
    if count < 1 or (sliding and window <= 0):
        raise ValueError("calibration needs attention layers and a positive window for sliding attention")
    return AttentionShape(count, sliding, window)


def validate_tables(data_path: str, *, context_stop: int | None = None) -> None:
    """Require complete, readable batch-one tables before publishing a serve command."""
    from eagle_harness.core.cost_table import load_cost_tables

    try:
        table = load_cost_tables(os.path.join(data_path, "cost"))[1]
        if context_stop is None:
            with open(os.path.join(data_path, "config", "measure_config.json"), encoding="utf-8") as source:
                context_stop = json.load(source)["context_stop"]
        if len(table.intervals) != (context_stop - table.cache_length_step) // table.cache_length_step:
            raise ValueError("missing context intervals")
        for interval in table.intervals:
            for curve, indices in ((interval.target, (0, 7, 15, 31, 63)), (interval.draft, (0, 1, 3, 7))):
                if curve.index != indices or len(curve.value) != len(indices):
                    raise ValueError("missing query widths")
                if any(not math.isfinite(value) or value <= 0 for value in curve.value):
                    raise ValueError("nonpositive or nonfinite timing")
    except (OSError, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"incomplete vLLM calibration under {data_path}/cost: {exc}") from exc


def main(argv: list[str] | None = None) -> None:
    """Run one measurement process or fit its saved timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--engine-args", type=json.loads, default={})
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("raw")
    m.add_argument("width", type=int)
    m.add_argument("--chain", action="store_true")
    m.add_argument("--batch-sizes", default="1")
    m.add_argument("--context-step", type=int, default=CONTEXT_STEP)
    m.add_argument("--repeats", type=int, default=REPEATS)
    f = sub.add_parser("fit")
    f.add_argument("data_path")
    f.add_argument("raws", nargs="+")
    args = parser.parse_args(argv)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    if args.command == "measure":
        measure(
            args.raw,
            args.width,
            args.chain,
            [int(x) for x in args.batch_sizes.split(",")],
            args.context_step,
            args.repeats,
            model=args.model,
            draft=args.draft,
            tp=args.tp,
            engine_args=args.engine_args,
        )
    else:
        target_rows, draft_rows, tp, chain, step, stop = merge(args.raws)
        with open(args.raws[0], encoding="utf-8") as source:
            geometry = json.load(source)
        if "target_shape" not in geometry or "draft_shape" not in geometry:
            raise ValueError("raw measurements lack resolved model geometry. Run calibration again.")
        target = AttentionShape(**geometry["target_shape"])
        draft = AttentionShape(**geometry["draft_shape"])
        sizes = fit(
            args.data_path,
            target_rows,
            draft_rows,
            target,
            draft,
            tp,
            chain=chain,
            context_step=step,
            context_stop=stop,
        )
        if geometry.get("key"):
            write_key(args.data_path, CostKey.from_dict(geometry["key"]))
        print(f"cost tables for batch sizes {sizes} under {args.data_path}/cost (TP{tp})")


if __name__ == "__main__":
    main()
