# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The Triton tree step, work deltas, ranking and selection against DraftTree's torch reference (skipped
without CUDA).

Run in a GPU job: ``pytest -q tests/test_tree_step_gpu.py``.
"""

import pytest
import torch

from eagle_harness.core.draft_tree import DraftTree, TreeShape

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

BUFFERS = (
    "scores",
    "tokens",
    "parents",
    "front_scores",
    "front_tokens",
    "front_hidden",
    "ancestors",
    "best",
    "best_prev",
    "layer_top",
)


def _tree(k, depth, bs, h, cap):
    return DraftTree(TreeShape(k, depth, work_cap=cap), bs, h, torch.bfloat16, torch.device("cuda"), max_verify=cap)


@pytest.mark.parametrize(
    "k,depth,bs,h,cap",
    [(4, 12, 1, 5120, 64), (4, 12, 3, 5120, 64), (8, 6, 2, 256, 32), (1, 6, 1, 256, 8)]
    + [(4, 6, bs, 256, 32) for bs in (33, 64, 128, 256, 512)],
)
def test_tree_step_kernel_matches_reference(k, depth, bs, h, cap):
    gen = torch.Generator(device="cuda").manual_seed(k * 1000 + bs)
    trees = [_tree(k, depth, bs, h, cap) for _ in range(2)]
    root_p = torch.rand(bs, k, device="cuda", generator=gen).softmax(-1)
    root_ids = torch.randint(0, 1000, (bs, k), device="cuda", generator=gen)
    root_h = torch.randn(bs, h, device="cuda", generator=gen).to(torch.bfloat16)
    for tree in trees:
        tree.stage_root(bs, root_p, root_ids, root_h)
        tree.begin(bs)
    for step in range(1, depth):
        p = torch.sort(torch.rand(bs * k, k, device="cuda", generator=gen).softmax(-1), dim=-1, descending=True).values
        ids = torch.randint(0, 248320, (bs * k, k), device="cuda", generator=gen)
        hidden = torch.randn(bs * k, h, device="cuda", generator=gen).to(torch.bfloat16)
        slots = torch.randint(0, 10**6, (bs, k), device="cuda", generator=gen)
        for tree in trees:
            tree.set_step_slots(bs, step, slots)
        trees[0].step(bs, step, p, ids, hidden)  # CUDA: the Triton kernels
        trees[1].step_reference(bs, step, p, ids, hidden)
        trees[1]._store_deltas_reference(bs, step)
        for name in BUFFERS:
            got, want = getattr(trees[0], name)[:bs], getattr(trees[1], name)[:bs]
            assert torch.equal(got, want), (step, name)
        # the kernel sums in another order than torch.cumsum: equal to fp32 rounding
        torch.testing.assert_close(trees[0].deltas[step], trees[1].deltas[step], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "k,bs,h,cap",
    [(4, 1, 5120, 64), (4, 5, 5120, 64), (8, 3, 256, 32)] + [(4, bs, 256, 32) for bs in (33, 64, 128, 256, 512)],
)
def test_tree_begin_kernel_matches_reference(k, bs, h, cap):
    gen = torch.Generator(device="cuda").manual_seed(k * 100 + bs)
    trees = [_tree(k, 12, bs, h, cap) for _ in range(2)]
    root_p = torch.rand(bs, k, device="cuda", generator=gen).softmax(-1)
    root_p[0, 1] = root_p[0, 2]  # a tie
    root_ids = torch.randint(0, 248320, (bs, k), device="cuda", generator=gen)
    root_h = torch.randn(bs, h, device="cuda", generator=gen).to(torch.bfloat16)
    for tree in trees:
        tree.stage_root(bs, root_p, root_ids, root_h)
    trees[0].begin(bs)  # CUDA: the Triton kernels
    trees[1].begin_reference(bs)
    trees[1]._store_deltas_reference(bs, 0)
    for name in BUFFERS:
        got, want = getattr(trees[0], name)[:bs], getattr(trees[1], name)[:bs]
        assert torch.equal(got, want), name
    torch.testing.assert_close(trees[0].deltas[0], trees[1].deltas[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "k,depth,bs,count,width",
    [(4, 12, 1, 15, 16), (4, 4, 3, 63, 8), (8, 6, 2, 32, 2), (4, 2, 1, 1, 2)]
    + [(4, 6, bs, 31, 16) for bs in (33, 64, 128, 256, 512)],
)
def test_rank_and_select_kernels_match_reference(k, depth, bs, count, width):
    gen = torch.Generator(device="cuda").manual_seed(k * 10 + depth)
    tree = _tree(k, 12, bs, 64, 64)
    tree.scores.copy_(torch.rand(tree.scores.shape, device="cuda", generator=gen))
    tree.scores[:, 1] = tree.scores[:, 2]  # a tie: the lower node first
    tree.tokens.copy_(torch.randint(0, 248320, tree.tokens.shape, device="cuda", generator=gen))
    count = min(count, tree.shape.num_nodes(depth))
    ranked, cum = tree.rank(bs, depth, count)  # CUDA: the Triton kernel
    want_ranked, want_cum = tree.rank_reference(bs, depth, count)
    assert torch.equal(ranked, want_ranked)
    torch.testing.assert_close(cum, want_cum, rtol=1e-6, atol=1e-6)  # another summation order
    got = tree.select(bs, depth, ranked, width)
    index = torch.sort(want_ranked[:, : width - 1], dim=-1).values
    assert torch.equal(got.top_scores_index, index)
    assert torch.equal(got.draft_tokens, torch.gather(tree.tokens[:bs], 1, index))


_STEP_AT_FULL_BATCH = """
import torch
from eagle_harness.core.draft_tree import DraftTree, TreeShape

k, depth, bs, h = {k}, 4, 32, 256
tree = DraftTree(TreeShape(k, depth, work_cap=64), bs, h, torch.bfloat16, torch.device("cuda"), max_verify=32)
tree.stage_root(bs, torch.rand(bs, k, device="cuda").softmax(-1), torch.zeros(bs, k, dtype=torch.int64,
                device="cuda"), torch.zeros(bs, h, device="cuda", dtype=torch.bfloat16))
tree.begin(bs)
tree.set_step_slots(bs, 1, torch.zeros(bs, k, dtype=torch.int64, device="cuda"))
p = torch.rand(bs * k, k, device="cuda").softmax(-1).sort(-1, descending=True).values
tree.step(bs, 1, p, torch.zeros(bs * k, k, dtype=torch.int64, device="cuda"),
          torch.zeros(bs * k, h, device="cuda", dtype=torch.bfloat16))
torch.cuda.synchronize()
"""


@pytest.mark.parametrize("k", [2, 4, 8])
def test_tree_step_work_deltas_read_only_their_rows_layer_scores(tmp_path, k):
    """The step's work-delta merge loaded next_pow2(K + 1) - 1 layer scores per row, past the row's K: the last row
    of a full batch read past ``layer_top``, an illegal address where that tensor ended its allocation (one GPU,
    batch size 32, --disable-radix-cache, at startup). compute-sanitizer reports any such read."""
    import os
    import shutil
    import subprocess
    import sys

    sanitizer = shutil.which("compute-sanitizer") or os.path.join(
        os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "compute-sanitizer"
    )
    if not os.path.exists(sanitizer):
        pytest.skip("needs compute-sanitizer")
    script = tmp_path / "step.py"
    script.write_text(_STEP_AT_FULL_BATCH.format(k=k))
    run = subprocess.run(
        [sanitizer, "--tool", "memcheck", "--error-exitcode", "3", sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        # One cudaMalloc per tensor, so the sanitizer sees each tensor's own bounds.
        env=dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1"),
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]
