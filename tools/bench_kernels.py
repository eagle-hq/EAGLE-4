# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Micro-benchmark of the kernels on one GPU: microseconds per call (CUDA-graph replay) and output hashes.

    python tools/bench_kernels.py --out bench.json [--only candidate,gdn]

Every case builds fixed inputs (seeded), captures ``REPEAT`` calls into one CUDA graph and times the replay, so launch
overhead is excluded the way it is inside the serving graphs. ``hash`` is the md5 of the outputs: two revisions with the
same hash produce bit-identical results for these inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from typing import Callable, Dict, Tuple

import torch

REPEAT = 20
REPLAYS = 50


def _hash(*tensors: torch.Tensor) -> str:
    digest = hashlib.md5()
    for t in tensors:
        digest.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()[:12]


def _time(fn: Callable[[], object]) -> Tuple[float, object]:
    """``(microseconds per call of fn inside a CUDA graph, fn's result)``."""
    result = fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(REPEAT):
            fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(3):
        start.record()
        for _ in range(REPLAYS):
            graph.replay()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) * 1000 / (REPLAYS * REPEAT))
    return best, result


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


# ---------------------------------------------------------------------------------------------------------------------
# candidate head (Qwen3.8-27B draft: vocab 248320, TP4 shard 62080, C = 128 candidates, hidden 5120)
# ---------------------------------------------------------------------------------------------------------------------


def _record(results: Dict[str, dict], name: str, fn: Callable[[], object]) -> None:
    us, out = _time(fn)
    tensors = [t for t in (out if isinstance(out, (tuple, list)) else [out]) if isinstance(t, torch.Tensor)]
    results[name] = {"us": us, "hash": _hash(*tensors) if tensors else ""}


def candidate_cases(results: Dict[str, dict]) -> None:
    from eagle_harness.kernels.candidate_ops import candidate_tail, shard_topk
    from eagle_harness.kernels.gather_dot import gather_dot

    for rows in (4, 16):
        scores = torch.randn(rows, 62080, device="cuda", dtype=torch.bfloat16, generator=_gen(rows))
        for cuda in (True, False):
            _record(
                results,
                f"shard_topk rows={rows} {'cuda' if cuda else 'triton'}",
                lambda: shard_topk(scores, 128, 1000, cuda_kernels=cuda),
            )
        _record(results, f"shard_topk rows={rows} torch", lambda: torch.topk(scores, 128, dim=-1, sorted=False))

    weight = torch.randn(62080, 5120, device="cuda", dtype=torch.bfloat16, generator=_gen(1))
    for rows in (4, 16):
        hidden = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16, generator=_gen(2))
        ids = torch.randint(0, 62080, (rows, 128), device="cuda", generator=_gen(3))
        _record(results, f"gather_dot rows={rows}", lambda: gather_dot(weight, ids, hidden))

    for ranks, rows, dtype in ((1, 4, torch.bfloat16), (4, 4, torch.float32), (4, 16, torch.float32)):
        logits = torch.randn(ranks * rows, 128, device="cuda", dtype=torch.bfloat16, generator=_gen(4)).to(dtype)
        tokens = torch.randint(0, 248320, (ranks * rows, 128), device="cuda", generator=_gen(5))
        for cuda in (True, False):
            _record(
                results,
                f"candidate_tail ranks={ranks} rows={rows} {'cuda' if cuda else 'triton'}",
                lambda: _flat(candidate_tail(logits, tokens, rows, 4, cuda_kernels=cuda)),
            )


def candidate_head_cases(results: Dict[str, dict]) -> None:
    """The whole candidate head after the low-rank scores at TP4 (the all-gather is a local concatenation) and TP1."""
    from eagle_harness.kernels.candidate_ops import candidate_tail, shard_topk
    from eagle_harness.kernels.gather_dot import gather_dot

    try:
        from eagle_harness.kernels.candidate_ops import candidate_topk
    except ImportError:  # revisions before the fused head
        candidate_topk = None
    k, h, shard = 128, 5120, 62080
    weight = torch.randn(shard, h, device="cuda", dtype=torch.bfloat16, generator=_gen(1))
    for rows in (4, 16):
        scores = torch.randn(rows, shard, device="cuda", dtype=torch.bfloat16, generator=_gen(rows))
        hidden = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16, generator=_gen(2))

        def unfused(ranks):
            local, global_ids = shard_topk(scores, k, 1000)
            logits = gather_dot(weight, local, hidden)
            if ranks == 1:
                return candidate_tail(logits, global_ids, rows, 4)[2]
            packed = torch.cat([logits.float().view(torch.int32), global_ids.to(torch.int32)], dim=-1)
            packed = torch.cat([packed] * ranks)  # the all-gather
            return candidate_tail(packed[:, :k].view(torch.float32), packed[:, k:].long(), rows, 4)[2]

        for ranks in (1, 4):
            _record(results, f"candidate_head tp{ranks} rows={rows} unfused", lambda: unfused(ranks))
            if candidate_topk is not None:
                gather = (lambda own: torch.stack([own] * ranks)) if ranks > 1 else None
                _record(
                    results,
                    f"candidate_head tp{ranks} rows={rows} fused",
                    lambda: candidate_topk(
                        scores, weight, hidden, num_candidates=k, vocab_start=1000, top_k=4, all_gather=gather
                    ),
                )


def _flat(result):
    logits, tokens, topk = result
    return (logits, tokens, *topk)


def tree_step_case(results: Dict[str, dict]) -> None:
    """A draft-tree step at a delta slot (bookkeeping and the cost model's work deltas), at batch size 1 and 4."""
    from eagle_harness.core.draft_tree import DraftTree, TreeShape

    k, depth, h, cap = 4, 12, 5120, 64
    for bs in (1, 4):
        tree = DraftTree(TreeShape(k, depth, work_cap=cap), bs, h, torch.bfloat16, torch.device("cuda"), max_verify=cap)
        gen = _gen(7)
        tree.stage_root(
            bs,
            torch.rand(bs, k, device="cuda", generator=gen).softmax(-1),
            torch.randint(0, 1000, (bs, k), device="cuda", generator=gen),
            torch.randn(bs, h, device="cuda", generator=gen).to(torch.bfloat16),
        )
        tree.begin(bs)
        tree.set_step_slots(bs, 1, torch.randint(0, 10**6, (bs, k), device="cuda", generator=gen))
        p = torch.sort(torch.rand(bs * k, k, device="cuda", generator=gen).softmax(-1), dim=-1, descending=True).values
        ids = torch.randint(0, 248320, (bs * k, k), device="cuda", generator=gen)
        hidden = torch.randn(bs * k, h, device="cuda", generator=gen).to(torch.bfloat16)
        _record(results, f"tree_step bs={bs} k=4 with work deltas", lambda: tree.step(bs, 1, p, ids, hidden))


# ---------------------------------------------------------------------------------------------------------------------
# GDN tree verify (Qwen3.8-27B: K = V = 128; per rank H / HV = 16 / 48 at TP1, 4 / 12 at TP4)
# ---------------------------------------------------------------------------------------------------------------------


def _tree_links(width: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    parent = [-1] + [int(torch.randint(0, i, (1,), generator=gen)) for i in range(1, width)]
    first = [-1] * width
    sibling = [-1] * width
    last: Dict[int, int] = {}
    for node in range(1, width):
        p = parent[node]
        if p in last:
            sibling[last[p]] = node
        else:
            first[p] = node
        last[p] = node
    t = lambda x: torch.tensor([x], dtype=torch.int32, device="cuda")  # noqa: E731
    return t(first), t(sibling)


def gdn_cases(results: Dict[str, dict]) -> None:
    from eagle_harness.kernels import gdn_tree, gdn_tree_conv, gdn_tree_recurrence

    def ancestors(nxt, sib, parents, conv_idx, width, anc, qsl, paths):
        gdn_tree.gdn_tree_ancestors(nxt, sib, parents, conv_idx, width, anc, qsl, paths)
        return parents, anc, paths

    for name, (h, hv) in {"tp1": (16, 48), "tp2": (8, 24), "tp4": (4, 12), "tp8": (2, 6)}.items():
        kdim = vdim = 128
        for width in (8, 16):
            gen = _gen(width)
            nxt, sib = _tree_links(width, width)
            parents = torch.zeros(1, width, dtype=torch.int32, device="cuda")
            anc = torch.zeros(1, 3, width, dtype=torch.int32, device="cuda")
            paths = gdn_tree.gdn_tree_paths_buffer(1, width, "cuda")
            idx = torch.zeros(1, dtype=torch.int32, device="cuda")
            qsl = torch.tensor([0, width], dtype=torch.int32, device="cuda")
            _record(
                results,
                f"gdn ancestors {name} w={width}",
                lambda: ancestors(nxt, sib, parents, idx, width, anc, qsl, paths),
            )

            dim = 2 * h * kdim + hv * vdim
            x = torch.randn(1, dim, width, device="cuda", dtype=torch.bfloat16, generator=gen)
            conv_state = torch.randn(4, dim, 3, device="cuda", dtype=torch.bfloat16, generator=gen)
            weight = torch.randn(dim, 4, device="cuda", dtype=torch.bfloat16, generator=gen)
            window = torch.zeros(2, width, dim, 3, device="cuda", dtype=torch.bfloat16)
            _record(
                results,
                f"gdn conv1d {name} w={width}",
                lambda: (
                    gdn_tree_conv.gdn_tree_conv1d_update(x, conv_state, weight, None, True, idx, window, idx, anc),
                    window,
                ),
            )

            q = torch.randn(1, width, h, kdim, device="cuda", dtype=torch.bfloat16, generator=gen)
            k = torch.randn(1, width, h, kdim, device="cuda", dtype=torch.bfloat16, generator=gen)
            v = torch.randn(1, width, hv, vdim, device="cuda", dtype=torch.bfloat16, generator=gen)
            a = torch.randn(width, hv, device="cuda", dtype=torch.bfloat16, generator=gen)
            b = torch.randn(width, hv, device="cuda", dtype=torch.bfloat16, generator=gen)
            a_log = torch.randn(hv, device="cuda", dtype=torch.float32, generator=gen)
            dt_bias = torch.randn(hv, device="cuda", dtype=torch.bfloat16, generator=gen)
            for state_dtype in (torch.bfloat16, torch.float32):
                pool = (torch.randn(2, hv, vdim, kdim, device="cuda", generator=gen) * 0.1).to(state_dtype)
                scratch = torch.zeros(2, width, hv, vdim, kdim, device="cuda", dtype=state_dtype)
                bf16 = state_dtype == torch.bfloat16
                for variant, gluon, use_paths in (
                    [("triton", False, False), ("gluon", True, False)]
                    if bf16
                    else [("triton", False, False), ("triton paths", False, True)]
                ):
                    kwargs = dict(
                        initial_state_source=pool,
                        initial_state_indices=idx,
                        cu_seqlens=qsl,
                        intermediate_states_buffer=scratch,
                        intermediate_state_indices=idx,
                        retrieve_parent_token=parents,
                        tree_paths=paths if use_paths else None,
                        gluon=gluon,
                    )
                    _record(
                        results,
                        f"gdn recurrent {name} w={width} {str(state_dtype)[6:]} {variant}",
                        lambda: (
                            gdn_tree_recurrence.gdn_tree_recurrent_update(a_log, dt_bias, q, k, v, a, b, **kwargs),
                            scratch,
                        ),
                    )


# ---------------------------------------------------------------------------------------------------------------------
# Mamba2 tree verify, one layer (Nemotron 3 Nano: 64 heads x 64, 8 groups, state 128; per rank / TP)
# ---------------------------------------------------------------------------------------------------------------------


def mamba2_cases(results: Dict[str, dict]) -> None:
    """sglang's tree conv (links walked in the kernel) vs the harness tree conv (ancestor table built once per
    forward, timed apart), and sglang's selective state update for scale."""
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.kernels.ops.mamba.triton_ops.mamba_ssm import selective_state_update

    from eagle_harness.kernels import gdn_tree, gdn_tree_conv

    for name, (heads, groups) in {"tp4": (16, 2), "tp1": (64, 8), "super-tp4": (32, 2)}.items():
        head_dim, dstate = 64, 128
        dim = heads * head_dim + 2 * groups * dstate
        for width in (4, 8, 16):
            gen = _gen(width)
            nxt, sib = (t.long() for t in _tree_links(width, width))
            parents = torch.zeros(1, width, dtype=torch.int64, device="cuda")
            anc = torch.zeros(1, 3, width, dtype=torch.int32, device="cuda")
            idx = torch.zeros(1, dtype=torch.int32, device="cuda")
            x = torch.randn(1, width, dim, device="cuda", dtype=torch.bfloat16, generator=gen).transpose(1, 2)
            conv_state = torch.randn(4, dim, 3, device="cuda", dtype=torch.bfloat16, generator=gen)
            weight = torch.randn(dim, 4, device="cuda", dtype=torch.bfloat16, generator=gen)
            bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16, generator=gen)
            window = torch.zeros(2, width, dim, 3, device="cuda", dtype=torch.bfloat16)
            tag = f"{name} w={width}"
            _record(
                results,
                f"mamba2 conv1d sglang {tag}",
                lambda: causal_conv1d_update(
                    x,
                    conv_state,
                    weight,
                    bias,
                    "silu",
                    conv_state_indices=idx,
                    intermediate_conv_window=window,
                    intermediate_state_indices=idx,
                    retrieve_next_token=nxt,
                    retrieve_next_sibling=sib,
                    retrieve_parent_token=parents,
                ),
            )
            _record(
                results,
                f"mamba2 ancestors harness {tag}",
                lambda: gdn_tree.gdn_tree_ancestors(nxt, sib, parents, idx, width, anc) or anc,
            )
            _record(
                results,
                f"mamba2 conv1d harness {tag}",
                lambda: gdn_tree_conv.gdn_tree_conv1d_update(
                    x, conv_state, weight, bias, "silu", idx, window, idx, anc
                ),
            )
            xs = torch.randn(1, width, heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=gen)
            dt = torch.randn(1, width, heads, device="cuda", dtype=torch.bfloat16, generator=gen)
            bc = torch.randn(2, 1, width, groups, dstate, device="cuda", dtype=torch.bfloat16, generator=gen)
            a = -torch.rand(heads, device="cuda", generator=gen)
            dt_bias = torch.randn(heads, device="cuda", generator=gen)
            out = torch.empty_like(xs)
            for state_dtype in (torch.bfloat16, torch.float32):
                pool = torch.randn(2, heads, head_dim, dstate, device="cuda", generator=gen).to(state_dtype)
                scratch = torch.zeros(1, width, heads, head_dim, dstate, device="cuda", dtype=state_dtype)
                _record(
                    results,
                    f"mamba2 ssu sglang {tag} {str(state_dtype)[6:]}",
                    lambda: (
                        selective_state_update(
                            pool,
                            xs,
                            dt[..., None].expand(-1, -1, -1, head_dim),
                            a[:, None, None].expand(-1, head_dim, dstate),
                            bc[0],
                            bc[1],
                            dt_bias[:, None].expand(-1, head_dim),
                            z=None,
                            dt_bias=dt_bias[:, None].expand(-1, head_dim),
                            dt_softplus=True,
                            state_batch_indices=idx,
                            out=out,
                            disable_state_update=True,
                            intermediate_states_buffer=scratch,
                            cache_steps=width,
                            retrieve_parent_token=parents,
                            intermediate_state_indices=idx,
                        )
                        or out
                    ),
                )


GROUPS: Dict[str, Callable[[Dict[str, dict]], None]] = {
    "candidate": candidate_cases,
    "candidate_head": candidate_head_cases,
    "tree_step": tree_step_case,
    "gdn": gdn_cases,
    "mamba2": mamba2_cases,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=",".join(GROUPS))
    args = ap.parse_args()
    results: Dict[str, dict] = {}
    for group in args.only.split(","):
        GROUPS[group](results)
    props = torch.cuda.get_device_properties(0)
    import triton

    meta = {
        "device": props.name,
        "capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
    }
    json.dump({"meta": meta, "results": results}, open(args.out, "w"), indent=1)
    width = max(map(len, results))
    for name, r in results.items():
        print(f"{name:<{width}}  {r['us']:9.2f} us  {r.get('hash', '')}")


if __name__ == "__main__":
    main()
