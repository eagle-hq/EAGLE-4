# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Tree calibration on TokenSpeed: measured rounds, attention sweeps and the core fits, into serve-mode cost tables.

usage: python -m eagle_harness.tokenspeed.calibrate --model TARGET --draft-model DRAFT --data-path DIR \\
           --tp-size 4 --topk 4 --num-steps 12 --num-draft-tokens 16 --max-bs 8 --mamba-ssm-dtype float32

For each batch size up to ``--max-bs`` the driver writes ``config/measure_config.json``
(:class:`~eagle_harness.core.calibration.MeasureParams`) and serves that many identical greedy requests on a
calibrate-mode engine, whose rounds record the draft and verify stage samples (:mod:`~eagle_harness.tokenspeed.
measurement`). It then times one attention layer of the target's verify and of the draft's lanes over the context
range with the kernels serving runs (:func:`attention_records`), and fits everything with
:func:`~eagle_harness.core.calibration.fit_batch` into ``cost/bs_<n>/model_{slm,llm}``. A batch whose averages do not
converge is measured again (three retries).

Serving reads the directory as ``{"mode": "serve", "data_path": DIR}`` with the same draft shape, TP size, attention
backend and ``--mamba-ssm-dtype``. Calibration needs TokenSpeed's ``BaseDrafter.rewrite_verify_outputs`` hook; the
tables it writes serve on any TokenSpeed this adapter supports.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

import torch

from eagle_harness.core.calibration import (
    AttentionShape,
    CacheSweepPlan,
    CalibrationPaths,
    MeasureParams,
    MeasureRecord,
    fit_batch,
    save_jsonl,
)
from eagle_harness.core.limits import MAX_BATCH_SIZE_TREE, ComputeIntensity, cost_batch_sizes
from eagle_harness.tokenspeed.config import ALGORITHM_NAME
from eagle_harness.tokenspeed.measurement import MEASURED_ROUNDS, swept_draft_widths, swept_verify_widths

_WORKSPACE_BYTES = 256 << 20
# Tokens a measured request may emit; the engine ends the batch with an EOS well before.
_MAX_NEW_TOKENS = 2 * MEASURED_ROUNDS
# The fixed prompt every measured request starts from.
_PROMPT = (
    "Please write a detailed, step-by-step explanation of how a modern operating system schedules threads across CPU "
    "cores, covering run queues, priorities, time slices, preemption, load balancing between cores, and how I/O-bound "
    "and CPU-bound workloads are treated differently. "
) * 8


def lane_mask(rows: int, window: int, *, causal: bool) -> list[int]:
    """Window-key bit masks of one request's ``rows`` query rows, as int64 words (bit ``j % 64`` of word ``j // 64``
    is window key ``j``): a verify chain (row ``r`` sees keys ``0..r``) or drafting lanes (row ``r`` sees its own slot
    at every depth, ``window // rows`` depths)."""
    if causal:
        bits = [(1 << (r + 1)) - 1 for r in range(rows)]
    else:
        bits = [sum(1 << (d * rows + r) for d in range(window // rows)) for r in range(rows)]
    words = [(b >> (64 * w)) & ((1 << 64) - 1) for b in bits for w in range((window + 63) // 64)]
    return [x - (1 << 64) if x >= 1 << 63 else x for x in words]


def attention_records(
    *,
    batch_size: int,
    rows: int,
    window: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int,
    plan: CacheSweepPlan,
    widths: list[int],
    category: str,
) -> list[MeasureRecord]:
    """One attention layer of the trtllm backend's tree cascade over the plan's cache lengths, ``plan.num_layers``
    times in a CUDA graph per length.

    The target verify (``sliding_window`` 0) is ``_tree_cascade``: a causal ``rows``-token decode over the prefix,
    then the ``window``-key tree window merged in; ``window`` 0 is a one-row verify, a plain decode over the prefix and
    the row's own key. The draft lanes of a sliding-window layer are ``_tree_lane_window_cascade``: each lane a one-row
    decode over the window ending at the prefix, then the lane window. ``widths`` are the swept widths whose forward
    runs these ``rows``, so one timing per length stands for each of them.
    """
    from tokenspeed.runtime.layers.attention.backends.paged.trtllm import canonicalize_stride
    from tokenspeed.runtime.layers.attention.kernel_page_sizes import TRTLLM_MHA_PAGE_SIZE
    from tokenspeed_kernel.ops.attention.mha.flashinfer import trtllm_batch_decode_with_kv_cache
    from tokenspeed_kernel.ops.attention.tree import tree_window_attention

    causal = sliding_window <= 0
    if window == 0 and (rows != 1 or not causal):
        raise ValueError("only a one-row full-attention verify runs without a window")
    device, dtype, page = torch.device("cuda"), torch.bfloat16, TRTLLM_MHA_PAGE_SIZE
    pages_per_req = math.ceil((plan.kv_stop + max(window, 1)) / page)
    num_pages = batch_size * pages_per_req
    k_rows = torch.randn((num_pages * page, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_rows = torch.randn_like(k_rows)

    def paged(rows_: torch.Tensor) -> torch.Tensor:
        cache = rows_.view(-1, page, num_kv_heads, head_dim).permute(0, 2, 1, 3)
        return canonicalize_stride(cache) if num_kv_heads == 1 else cache

    k_cache, v_cache = paged(k_rows), paged(v_rows)
    page_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(batch_size, pages_per_req)
    q = torch.randn((batch_size * rows, num_q_heads, head_dim), device=device, dtype=dtype)
    workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    scale = head_dim**-0.5
    mask = torch.tensor(lane_mask(rows, window, causal=causal) * batch_size, dtype=torch.int64, device=device)
    # A lane at draft step 1 sees the window's keys ending at its position (the layer's window_left is size - 1).
    window_left = -1 if causal else sliding_window - 2
    records = []
    for kv_len in plan.cache_lengths():
        if kv_len < plan.kv_step:  # warm-up lengths
            continue
        prefix = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
        seq_lens = prefix + window
        if causal:
            tables, prefix_lens, row_seq_lens, q_len, rows_per_req = page_table, prefix, seq_lens, rows, rows
        else:
            tables = page_table.repeat_interleave(rows, dim=0)
            prefix_lens, row_seq_lens = prefix.repeat_interleave(rows), seq_lens.repeat_interleave(rows)
            q_len, rows_per_req = 1, 1

        def decode() -> torch.Tensor:
            return trtllm_batch_decode_with_kv_cache(
                query=q, kv_cache=(k_cache, v_cache), workspace_buffer=workspace, block_tables=page_table,
                seq_lens=prefix + 1, max_seq_len=plan.kv_stop + 1, bmm1_scale=scale, bmm2_scale=1.0,
                out_dtype=dtype, q_len_per_req=1,
            )  # fmt: skip

        def layer() -> torch.Tensor:
            out, lse = trtllm_batch_decode_with_kv_cache(
                query=q, kv_cache=(k_cache, v_cache), workspace_buffer=workspace, block_tables=tables,
                seq_lens=prefix_lens, max_seq_len=plan.kv_stop + window, bmm1_scale=scale, bmm2_scale=1.0,
                window_left=window_left, out_dtype=dtype, q_len_per_req=q_len, return_lse=True,
            )  # fmt: skip
            return tree_window_attention(
                q, k_rows, v_rows, tables, row_seq_lens, mask, out, lse,
                rows_per_req=rows_per_req, window=window, page_size=page, sm_scale=scale,
            )  # fmt: skip

        step = decode if window == 0 else layer
        step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(plan.num_layers):
                step()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = max(int(start.elapsed_time(end) * 1000), 1)
        records += [MeasureRecord(elapsed, plan.num_layers, w, kv_len, category, batch_size) for w in widths]
    return records


def _attention_config(config) -> tuple[int, int, int, int]:
    """(query heads, key/value heads, head dim, sliding window) of a model's full-attention layers."""
    text = getattr(config, "text_config", None) or config
    heads = text.num_attention_heads
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // heads
    return heads, text.num_key_value_heads, head_dim, getattr(text, "sliding_window", None) or 0


def _full_attention_layers(config) -> int:
    text = getattr(config, "text_config", None) or config
    types = getattr(text, "layer_types", None)
    return text.num_hidden_layers if types is None else sum(t == "full_attention" for t in types)


def measure_params(args: argparse.Namespace, batch_size: int) -> MeasureParams:
    """One batch's settings: the target fits read the verify widths up to the window (``loop_top_m``)."""
    return MeasureParams(
        context_step=args.context_step,
        context_stop=args.context_stop,
        batch_size=batch_size,
        verify_num=args.num_draft_tokens,
        node_top_k=args.topk,
        loop_top_m=args.num_draft_tokens,
        repeat_cnt=16 if args.context_stop < 16384 else 8,
    )


def measure_batch(args, engine, input_ids, batch_size, target_cfg, draft_cfg) -> bool:
    """Measure and fit one batch size; whether its averages converged."""
    paths = CalibrationPaths(args.data_path)
    params = measure_params(args, batch_size)
    plan = CacheSweepPlan.from_params(params)
    paths.write_measure_params(params)
    paths.set_task_normal(True)
    for model in ("slm", "llm"):  # an earlier attempt's samples
        shutil.rmtree(paths.raw(batch_size, "model", model), ignore_errors=True)
    engine.generate(
        input_ids=[input_ids] * batch_size, sampling_params={"temperature": 0.0, "max_new_tokens": _MAX_NEW_TOKENS}
    )
    if not os.path.isdir(paths.raw(batch_size, "model", "llm")):
        raise RuntimeError(f"the engine wrote no measured rounds for batch {batch_size}: is calibrate mode on?")
    intensity = ComputeIntensity(args.intensity)
    heads, kv_heads, head_dim, _ = _attention_config(target_cfg)
    d_heads, d_kv_heads, d_head_dim, d_window = _attention_config(draft_cfg)
    tp, n = args.tp_size, args.num_draft_tokens
    # A tree verifies its whole window at every swept width; the draft forwards all K lanes at every width.
    shapes = (
        ("llm", swept_verify_widths(batch_size, intensity, n), n, n if n > 1 else 0, (heads, kv_heads, head_dim), 0),
        ("slm", swept_draft_widths(batch_size, intensity, args.topk), args.topk, (args.num_steps - 1) * args.topk,
         (d_heads, d_kv_heads, d_head_dim), d_window),
    )  # fmt: skip
    for model, widths, rows, window, (q_heads, k_heads, dim), sliding in shapes:
        records = attention_records(
            batch_size=batch_size, rows=rows, window=window, num_q_heads=q_heads // tp,
            num_kv_heads=max(1, k_heads // tp), head_dim=dim, sliding_window=sliding, plan=plan, widths=widths,
            category=f"cache_{model}",
        )  # fmt: skip
        for width in widths:
            save_jsonl(
                os.path.join(paths.raw(batch_size, "cache", model), f"cache_qo_len_{width}.jsonl"),
                [r for r in records if r.query_length == width],
            )
    draft = AttentionShape(1, 1, d_window) if d_window else AttentionShape(1)
    fit_batch(paths, params, intensity, draft, AttentionShape(_full_attention_layers(target_cfg)), tp_size=tp)
    return paths.task_normal()


def engine_args(args: argparse.Namespace, prompt_len: int) -> dict:
    """The calibrate-mode engine: serving's draft shape, TP size, attention backend and state dtype."""
    return dict(
        model=args.model,
        attn_tp_size=args.tp_size,
        dtype="bfloat16",
        max_model_len=max(4096, prompt_len + _MAX_NEW_TOKENS + 2 * args.num_draft_tokens),
        max_num_seqs=args.max_bs,
        attention_backend="trtllm",
        sampling_backend="greedy",
        disable_kvstore=True,
        enable_prefix_caching=False,
        speculative_algorithm=ALGORITHM_NAME,
        speculative_algorithm_config=json.dumps(
            {"mode": "calibrate", "intensity": args.intensity, "data_path": os.path.abspath(args.data_path)}
        ),
        speculative_draft_model_path=args.draft_model,
        speculative_eagle_topk=args.topk,
        speculative_num_steps=args.num_steps,
        speculative_num_draft_tokens=args.num_draft_tokens,
        mamba_ssm_dtype=args.mamba_ssm_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        log_level="warning",
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--data-path", required=True, help="the calibration directory serving reads as data_path")
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--intensity", required=True, choices=["high_intensity", "low_intensity"])
    parser.add_argument("--topk", type=int, required=True, help="--speculative-eagle-topk")
    parser.add_argument("--num-steps", type=int, required=True, help="--speculative-num-steps")
    parser.add_argument("--num-draft-tokens", type=int, required=True, help="--speculative-num-draft-tokens")
    parser.add_argument("--start-bs", type=int, default=1)
    parser.add_argument("--max-bs", type=int, required=True)
    parser.add_argument("--mamba-ssm-dtype", required=True, choices=["float32", "bfloat16"], help="as serving runs it")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85,
        help="the engine's share of each GPU; the rest is room for the attention sweeps on rank 0's GPU",
    )  # fmt: skip
    parser.add_argument("--context-step", type=int, default=64)
    parser.add_argument("--context-stop", type=int, default=8192)
    args = parser.parse_args(argv)
    if not 1 <= args.max_bs <= MAX_BATCH_SIZE_TREE:
        parser.error(f"--max-bs must be 1..{MAX_BATCH_SIZE_TREE} for a tree")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from tokenspeed.runtime.entrypoints.engine import Engine
    from transformers import AutoConfig, AutoTokenizer

    target_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    draft_cfg = AutoConfig.from_pretrained(args.draft_model, trust_remote_code=True)
    input_ids = list(AutoTokenizer.from_pretrained(args.model)(_PROMPT, add_special_tokens=False)["input_ids"])
    engine = Engine(**engine_args(args, len(input_ids)))
    try:
        for batch_size in cost_batch_sizes(args.max_bs, args.start_bs):
            for attempt in range(4):
                if measure_batch(args, engine, input_ids, batch_size, target_cfg, draft_cfg):
                    print(f"[calibrate] batch {batch_size}: done", flush=True)
                    break
                print(f"[calibrate] batch {batch_size}: unstable, retry {attempt + 1}/3", flush=True)
            else:
                print(f"[calibrate] batch {batch_size} stayed unstable", file=sys.stderr)
                return 1
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
