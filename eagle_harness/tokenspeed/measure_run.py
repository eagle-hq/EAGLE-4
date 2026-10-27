# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Measure TokenSpeed draft and verify costs for a serve-mode profile.

Run ``python -m eagle_harness.tokenspeed.measure_run --help`` for the driver.
The engine uses the harness calibration policy through generic drafter hooks.
Raw records, robust fits and the resulting profile belong to core.calibration.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from eagle_harness.core.calibration import (
    AttentionShape,
    CacheSweepPlan,
    CalibrationPaths,
    MeasureParams,
    attention_shape,
    fit_batch,
    load_jsonl,
    save_jsonl,
)
from eagle_harness.core.limits import ComputeIntensity, draft_query_lens, verify_buckets

# Longer than the recorder's 512 rounds, including its first untimed verify.
_MAX_NEW_TOKENS = 520
_PROMPT = (
    "Please explain how an operating system schedules threads, covering run queues, "
    "priorities, time slices, preemption and load balancing across CPU cores. "
) * 16


def attention_config(config):
    """Attention geometry and layer counts of a text or multimodal model."""
    text = getattr(config, "text_config", None) or config
    heads, kv_heads = text.num_attention_heads, text.num_key_value_heads
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // heads
    ids = getattr(text, "full_attention_layer_ids", None)
    # Hybrid configs that list their softmax layers by id may use other block names in layer_types.
    shape = AttentionShape(len(ids)) if ids is not None else attention_shape(text)
    if shape.attention_layers < 1:
        raise ValueError("TokenSpeed calibration requires at least one softmax attention layer")
    if shape.sliding_window_layers not in (0, shape.attention_layers):
        # One attention geometry is measured per model: either every layer is windowed or none is.
        raise ValueError("TokenSpeed calibration does not support mixed full and sliding-window attention layers")
    window = shape.sliding_window if shape.sliding_window_layers else 0
    shape = AttentionShape(shape.attention_layers, shape.sliding_window_layers, window)
    return (heads, kv_heads, head_dim, window), shape


def measure_batch(args, engine, input_ids, batch_size, target_cfg, draft_cfg, *, attention_samples=None):
    """Run one fixed-size batch, then fit its stage and attention measurements.

    The optional attention callable is the CPU fake-engine test boundary.
    Missing stage samples fail before any fitted cost is written.
    """
    if attention_samples is None:
        from eagle_harness.tokenspeed.attention_measurement import attention_samples

    intensity = ComputeIntensity(args.compute_intensity)
    paths = CalibrationPaths(args.run_dir)
    params = MeasureParams(args.context_step, args.context_stop, batch_size,
                           args.num_draft_tokens, args.topk, args.num_draft_tokens,
                           8 if args.fast else 16, args.fast)
    CacheSweepPlan.from_params(params)
    paths.write_measure_params(params)
    # A failed or short retry must not reuse the preceding attempt's timings.
    for model in ("slm", "llm"):
        raw = Path(paths.raw(batch_size, "model", model))
        if raw.exists():
            for file in raw.glob("model_qo_len_*.jsonl"):
                file.unlink()
    engine.generate(input_ids=[input_ids] * batch_size,
                    sampling_params={"temperature": 0.0, "max_new_tokens": _MAX_NEW_TOKENS,
                                     "ignore_eos": True})
    drafts = [w for w in draft_query_lens(batch_size, intensity) if w <= args.topk]
    verifies = sorted(w for w in verify_buckets(batch_size, intensity) if w <= args.num_draft_tokens)
    for model, widths in (("slm", drafts), ("llm", verifies)):
        for width in widths:
            file = os.path.join(paths.raw(batch_size, "model", model), f"model_qo_len_{width}.jsonl")
            records = load_jsonl(file)
            if not records or any(r["batch_size"] != batch_size or r["time_intervals"] <= 0 for r in records):
                raise RuntimeError(f"missing or invalid TokenSpeed measurements: {file}")

    target_geometry, target_shape = attention_config(target_cfg)
    draft_geometry, draft_shape = attention_config(draft_cfg)
    for model, widths, rows, window, geometry in (
        ("llm", verifies, args.num_draft_tokens, args.num_draft_tokens, target_geometry),
        ("slm", drafts, args.topk, (args.num_steps - 1) * args.topk, draft_geometry),
    ):
        heads, kv_heads, head_dim, sliding = geometry
        samples = attention_samples(
            batch_size=batch_size, rows=rows, window=window,
            num_q_heads=heads // args.tp_size, num_kv_heads=max(1, kv_heads // args.tp_size),
            head_dim=head_dim, sliding_window=sliding, context_step=args.context_step,
            context_stop=args.context_stop, fast=args.fast, repeat=params.repeat_cnt,
            widths=widths, category=f"cache_{model}",
        )
        groups = defaultdict(list)
        for sample in samples:
            groups[sample.query_length].append(sample)
        for width, records in groups.items():
            save_jsonl(os.path.join(paths.raw(batch_size, "cache", model), f"cache_qo_len_{width}.jsonl"), records)
    fit_batch(paths, params, intensity, draft_shape, target_shape, tp_size=args.tp_size)
    stable = True
    for model in ("slm", "llm"):
        for kind, limit in (("cache", 0.20 if args.fast else 0.25), ("query", 0.30)):
            for file in Path(paths.fitted(batch_size, kind, model)).glob("*.json"):
                with file.open() as f:
                    stable &= json.load(f)["instability_rate"] < limit
    return stable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--compute-intensity", choices=[i.value for i in ComputeIntensity], required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--num-draft-tokens", type=int, required=True)
    parser.add_argument("--max-bs", type=int, required=True)
    parser.add_argument("--mamba-ssm-dtype", choices=["float32", "bfloat16"], required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--context-step", type=int, default=64)
    parser.add_argument("--context-stop", type=int, default=8192)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args(argv)
    chain = args.compute_intensity == "chain_intensity"
    if not 1 <= args.max_bs <= (512 if chain else 32):
        parser.error("--max-bs must be 1..512 for chains or 1..32 for trees")
    if args.tp_size < 1 or args.topk not in ((1,) if chain else (2, 4, 8)):
        parser.error("positive --tp-size and topk 1 (chain) or 2/4/8 (tree) required")
    if not 2 <= args.num_steps <= (15 if chain else 32) or (args.num_steps - 1) * args.topk > 64:
        parser.error("draft depth or lane window exceeds TokenSpeed limits")
    if not 2 <= args.num_draft_tokens <= (16 if chain else 64):
        parser.error("verify width exceeds TokenSpeed limits")
    CacheSweepPlan.from_params(MeasureParams(args.context_step, args.context_stop, 1,
                                             args.num_draft_tokens, args.topk, args.num_draft_tokens, 16))
    run_dir = Path(args.run_dir)
    if (run_dir / "cost").exists():
        parser.error("--run-dir already contains costs, choose a fresh profile directory")

    from tokenspeed.runtime.configs.model_config import get_hf_text_config
    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.hf_transformers_utils import get_config
    from transformers import AutoConfig, AutoTokenizer

    target_cfg = get_hf_text_config(get_config(args.model, trust_remote_code=True))
    draft_cfg = AutoConfig.from_pretrained(args.draft_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = list(tokenizer(_PROMPT, add_special_tokens=False)["input_ids"])
    run_dir.mkdir(parents=True, exist_ok=True)
    # Initialize before startup can issue a warm-up prefill.
    CalibrationPaths(args.run_dir).write_measure_params(MeasureParams(
        args.context_step, args.context_stop, 1, args.num_draft_tokens, args.topk, args.num_draft_tokens, 16))
    with (run_dir / "config" / "tokenspeed.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    engine = Engine(
        model=args.model, attn_tp_size=args.tp_size, dtype="bfloat16",
        max_model_len=max(4096, len(input_ids) + _MAX_NEW_TOKENS + 2 * args.num_draft_tokens),
        max_num_seqs=args.max_bs, attention_backend="trtllm", sampling_backend="greedy",
        disable_kvstore=True, speculative_algorithm="EAGLE_HARNESS",
        speculative_draft_model_path=args.draft_model, speculative_eagle_topk=args.topk,
        speculative_num_steps=args.num_steps, speculative_num_draft_tokens=args.num_draft_tokens,
        speculative_algorithm_config=json.dumps({"mode": "calibrate", "data_path": args.run_dir,
                                                 "intensity": args.compute_intensity}),
        mamba_ssm_dtype=args.mamba_ssm_dtype, gpu_memory_utilization=args.gpu_memory_utilization,
        log_level="warning",
    )
    try:
        for batch_size in range(1, args.max_bs + 1):
            for attempt in range(4):
                if measure_batch(args, engine, input_ids, batch_size, target_cfg, draft_cfg):
                    print(f"batch {batch_size}: calibrated", flush=True)
                    break
                print(f"batch {batch_size}: unstable attempt {attempt + 1}/4", flush=True)
            else:
                raise RuntimeError(f"batch {batch_size} stayed unstable")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
