# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-Harness draft-tree kernels against PyTorch references."""

import sys

import numpy as np
import pytest
import torch

pytest.importorskip("tvm_ffi", reason="TokenSpeed round kernels require apache-tvm-ffi")

from eagle_harness.tokenspeed.tree_round.ops.tree import (
    eagle_harness_draft_depth_update,
    eagle_harness_stable_topk_desc,
    eagle_harness_update_node_data,
    eagle_harness_work_deltas,
    eagle_harness_write_verify_tree,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU is required"
)

_DEVICE = "cuda"


@pytest.mark.parametrize("k", [2, 4, 8])
def test_update_root_children(k):
    bs, columns = 3, 65
    generator = torch.Generator().manual_seed(k)
    scores = torch.rand((bs, k), generator=generator).to(_DEVICE)
    tokens = torch.randint(0, 1000, (bs, k), generator=generator).to(_DEVICE)
    # Row strides may differ between the pool and the work/sort lists.
    pool_scores = torch.full((bs, columns), -1.0, device=_DEVICE)
    pool_tokens = torch.full((bs, columns), -1, dtype=torch.int64, device=_DEVICE)
    work = torch.full((bs, 64 + k), -1.0, device=_DEVICE)
    sort = torch.full((bs, 64), -1.0, device=_DEVICE)
    eagle_harness_update_node_data(pool_scores, pool_tokens, work, sort, scores, tokens)
    for target in (pool_scores, work, sort):
        assert torch.equal(target[:, 1 : k + 1], scores)
        assert (target[:, 0] == -1).all()
        assert (target[:, k + 1 :] == -1).all()
    assert torch.equal(pool_tokens[:, 1 : k + 1], tokens)


def _count(value: int) -> torch.Tensor:
    """A device int32 ``[1]`` count, as the drafter's round state holds them."""
    return torch.tensor([value], dtype=torch.int32, device=_DEVICE)


def _distinct_scores(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    """Scores in (0, 1) whose values and products almost surely never tie."""
    return (torch.rand((rows, cols), generator=generator) * 0.98 + 0.01).to(_DEVICE)


@pytest.mark.parametrize("n,k", [(5, 5), (1024, 64)])
def test_stable_topk_desc(n: int, k: int) -> None:
    generator = torch.Generator().manual_seed(n)
    # Few distinct values: ties must keep the lower index first. A padded row
    # and columns past n (as in a captured graph) are never read or written.
    cols = min(n + 3, 1024)
    scores = (torch.randint(0, 8, (4, cols), generator=generator).float() / 8).to(
        _DEVICE
    )
    values = torch.full((4, k + 2), -1.0, device=_DEVICE)
    indices = torch.full((4, k + 2), -1, dtype=torch.int64, device=_DEVICE)
    eagle_harness_stable_topk_desc(
        scores, _count(3), _count(n), _count(k), values, indices
    )
    expected = torch.sort(scores[:3, :n], dim=-1, descending=True, stable=True)
    torch.testing.assert_close(values[:3, :k], expected.values[:, :k], rtol=0, atol=0)
    torch.testing.assert_close(indices[:3, :k], expected.indices[:, :k], rtol=0, atol=0)
    assert (values[3] == -1).all() and (values[:, k:] == -1).all()


@pytest.mark.parametrize("node_count,verify_num", [(1, 16), (7, 16), (14, 16)])
def test_work_deltas(node_count: int, verify_num: int) -> None:
    bs, topk = 3, 4
    generator = torch.Generator().manual_seed(node_count)
    # One padded row past the live ones, as in a captured graph.
    work = torch.ones((bs + 1, verify_num + topk), device=_DEVICE)
    work[:bs] = 0.0
    work[:bs, :node_count] = (
        _distinct_scores(bs, node_count, generator).sort(dim=-1, descending=True).values
    )
    work[:bs, node_count : node_count + topk] = (
        _distinct_scores(bs, topk, generator).sort(dim=-1, descending=True).values
    )
    deltas = torch.empty(topk + 1, device=_DEVICE)
    live_rows = torch.tensor([bs], dtype=torch.int32, device=_DEVICE)
    counts = _count(node_count), _count(verify_num)
    eagle_harness_work_deltas(work, deltas, live_rows, counts[0], topk, counts[1])
    unpadded = torch.empty(topk + 1, device=_DEVICE)
    eagle_harness_work_deltas(
        work[:bs], unpadded, live_rows, counts[0], topk, counts[1]
    )
    # A zero node count queues no deltas.
    untouched = torch.full((topk + 1,), -1.0, device=_DEVICE)
    eagle_harness_work_deltas(work, untouched, live_rows, _count(0), topk, counts[1])
    eagle_harness_work_deltas(work, untouched, _count(0), counts[0], topk, counts[1])

    assert torch.equal(deltas, unpadded) and (untouched == -1).all()
    expected = []
    for j in range(topk + 1):
        merged = work[:bs, : node_count + j].sort(dim=-1, descending=True).values
        expected.append(merged[:, : min(node_count + j, verify_num)].sum(-1).mean())
    torch.testing.assert_close(deltas, torch.stack(expected), rtol=1e-6, atol=1e-6)


def _work_deltas_reference(
    work: np.ndarray, bs: int, node_count: int, verify_num: int, topk: int, rows: int
) -> np.ndarray:
    """The work deltas in the kernel's float32 order: each row's merge summed
    in turn, then a halving tree over 256 threads, or 512 above 256 rows,
    whose padded rows hold zeros."""
    threads = 256 if rows <= 256 else 512
    out = np.zeros(topk + 1, dtype=np.float32)
    for j in range(topk + 1):
        sums = np.zeros(threads, dtype=np.float32)
        for row in range(bs):
            head, tail = work[row, :node_count], work[row, node_count:]
            a = b = 0
            total = np.float32(0.0)
            for _ in range(min(node_count + j, verify_num)):
                va = head[a] if a < node_count else np.float32(0.0)
                vb = tail[b] if b < j else np.float32(0.0)
                if va >= vb:
                    total, a = np.float32(total + va), a + 1
                else:
                    total, b = np.float32(total + vb), b + 1
            sums[row] = total
        half = threads // 2
        while half:
            sums[:half] = sums[:half] + sums[half : 2 * half]
            half //= 2
        out[j] = np.float32(sums[0] / np.float32(bs))
    return out


@pytest.mark.parametrize(
    "bs,rows",
    [(1, 1), (3, 4), (16, 32), (31, 32), (32, 32)],
)
def test_work_deltas_sum_order(bs: int, rows: int) -> None:
    """Bit for bit the kernel's summation order up to 512 rows (padded rows
    past the live ones, as in a captured graph); up to 256 live rows the
    256- and 512-thread halving trees give the same bits."""
    node_count, verify_num, topk = 7, 16, 4
    generator = torch.Generator().manual_seed(bs)
    work = torch.zeros((rows, verify_num + topk))
    work[:bs, :node_count] = _distinct_scores(bs, node_count, generator).cpu()
    work[:bs, :node_count] = work[:bs, :node_count].sort(-1, descending=True).values
    tail = _distinct_scores(bs, topk, generator).cpu().sort(-1, descending=True).values
    work[:bs, node_count : node_count + topk] = tail
    work[bs:] = 1.0
    deltas = torch.empty(topk + 1, device=_DEVICE)
    eagle_harness_work_deltas(
        work.to(_DEVICE),
        deltas,
        _count(bs),
        _count(node_count),
        topk,
        _count(verify_num),
    )
    expected = _work_deltas_reference(
        work.numpy(), bs, node_count, verify_num, topk, rows
    )
    assert np.array_equal(deltas.cpu().numpy(), expected)
    if bs <= 256:
        other = _work_deltas_reference(
            work.numpy(), bs, node_count, verify_num, topk, 512 if rows <= 256 else 256
        )
        assert np.array_equal(other, expected)


def _bitonic_sort_64(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernels' descending sort of each row's ``n <= 64`` scores: a
    bitonic network over 64 positions that skips every pair reaching past
    ``n`` and never swaps ties, so its order (not always fully sorted when
    ``n`` is not a power of two) is a fixed function of the inputs. Returns
    the values and their original positions."""
    rows, n = scores.shape
    values = scores.clone()
    order = torch.arange(n, device=scores.device).repeat(rows, 1)
    size = 2
    while size <= 64:
        stride = size // 2
        while stride:
            lower = [p for p in range(64) if not p & stride and p + stride < n]
            if lower:
                i = torch.tensor(lower, device=scores.device)
                j = i + stride
                vi, vj, oi, oj = values[:, i], values[:, j], order[:, i], order[:, j]
                swap = torch.where((i & size) == 0, vi < vj, vi > vj)
                values[:, i] = torch.where(swap, vj, vi)
                values[:, j] = torch.where(swap, vi, vj)
                order[:, i] = torch.where(swap, oj, oi)
                order[:, j] = torch.where(swap, oi, oj)
            stride //= 2
        size *= 2
    return values, order


def _lane_mask_words(mask: torch.Tensor, lane_mask: torch.Tensor) -> None:
    """Bool ``[rows, K, window]`` lane masks as int64 words: window slot ``j``
    at bit ``j % 64`` of word ``j // 64``, later words zero."""
    window = mask.shape[-1]
    bits = 1 << (torch.arange(window, device=mask.device) % 64)
    for word in range(lane_mask.shape[-1]):
        lo, hi = 64 * word, min(64 * word + 64, window)
        if lo >= hi:
            lane_mask[..., word].zero_()
            continue
        torch.sum(mask[..., lo:hi] * bits[lo:hi], dim=-1, out=lane_mask[..., word])


def _unfused_depth(
    state: dict[str, torch.Tensor],
    masks: list[torch.Tensor],
    child_p: torch.Tensor,
    child_tokens: torch.Tensor,
    expanded: torch.Tensor,
    child_hidden: torch.Tensor,
    depth: int,
    cumu_count: int,
    verify_num: int,
    num_slots: int,
) -> None:
    """The sequence the fused update replaces (the layer-gen and pool-update
    kernels, the next depth's bool ancestor masks ``masks[depth + 1]`` and
    their lane-mask words, the lane increments) on ``state``'s first rows."""
    rows, width, k = child_p.shape
    n = width * k
    row = torch.arange(rows, device=_DEVICE)[:, None]
    path = (child_p * state["scores"][:rows, :width, None]).reshape(rows, n)
    tokens = child_tokens.reshape(rows, n)
    layer, order = _bitonic_sort_64(path)
    best = order[:, :k]
    parent = best // k
    state["frontier_index"][depth % 2][:rows] = cumu_count + best
    state["scores"][:rows] = layer[:, :k]
    state["tokens"][:rows] = tokens.gather(1, best)
    state["hidden"][:rows] = child_hidden[row, parent]
    # Device powf: CUDA pow with a tensor exponent.
    decay = torch.pow(
        torch.tensor(0.9999, device=_DEVICE), torch.tensor(depth + 1.0, device=_DEVICE)
    )
    new = slice(cumu_count, cumu_count + n)
    child = torch.arange(n, device=_DEVICE)
    state["pool_tokens"][:rows, new] = tokens
    state["pool_scores"][:rows, new] = path * decay
    state["pool_depths"][:rows, new] = depth + 1
    state["pool_prev"][:rows, new] = expanded[:, :width].repeat_interleave(k, dim=1)
    state["pool_next"][:rows, new] = -1
    state["pool_side"][:rows, new] = torch.where(
        child % k == k - 1, -1, cumu_count + child + 1
    )
    first_child = cumu_count + k * torch.arange(width, device=_DEVICE)
    state["pool_next"][row, expanded[:, :width]] = first_child
    # Work list, then the sort list's stable rank merge with the children.
    head, kept = min(verify_num, cumu_count), min(verify_num, cumu_count + n)
    merged = torch.cat([state["sort"][:rows, :head], layer], dim=1)
    pos = torch.arange(merged.shape[1], device=_DEVICE)
    above = (merged[:, None, :] > merged[:, :, None]) | (
        (merged[:, None, :] == merged[:, :, None]) & (pos[None, :] < pos[:, None])
    )
    rank = above.sum(-1)
    state["work"][:rows, :head] = state["sort"][:rows, :head]
    state["work"][:rows, head : head + k] = layer[:, :k]
    keep = rank < kept
    state["sort"][row.expand_as(rank)[keep], rank[keep]] = merged[keep]
    # The next depth's ancestors: the parents' rows, then each lane's own slot.
    if (depth + 1) * k <= num_slots:
        window = depth * k
        masks[depth + 1][:rows, :, :window] = masks[depth][row, parent, :window]
        _lane_mask_words(masks[depth + 1][:rows], state["lane_mask"][:rows])
    state["positions"][:rows] += 1
    state["slot_starts"][:rows] += k


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, name: str):
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), name


# K=2/4/8, including the sign bit of the 64-slot lane mask.
@pytest.mark.parametrize(
    "rows,max_rows,k,steps",
    [(3, 5, 2, 32), (3, 5, 4, 17), (3, 5, 8, 9), (32, 32, 8, 4)],
)
def test_draft_depth_update_matches_the_unfused_sequence(
    rows: int, max_rows: int, k: int, steps: int, disabled: bool = False
) -> None:
    """A round of depths at random widths: from the same forward outputs the
    fused update leaves every buffer bit-identical to the unfused sequence,
    ties included (few distinct probabilities and path scores); rows past
    ``rows`` (graph padding of the buffers) stay untouched."""
    gen = torch.Generator(device=_DEVICE).manual_seed(1000 * rows + 10 * k + steps)
    hidden_size = 1032  # 129 16-byte vectors per row
    nodes, verify = max(1024, k * k * steps), max(64, k * (steps + 1) + 1)
    num_slots = (steps - 1) * k
    words = (num_slots + 63) // 64

    def ints(shape, high=1000):
        return torch.randint(-1, high, shape, generator=gen, device=_DEVICE)

    def ties(shape):
        return torch.randint(1, 5, shape, generator=gen, device=_DEVICE) / 4

    state = {
        "pool_tokens": ints((max_rows, nodes)),
        "pool_scores": torch.rand((max_rows, nodes), generator=gen, device=_DEVICE),
        "pool_depths": ints((max_rows, nodes)),
        "pool_prev": ints((max_rows, nodes)),
        "pool_next": ints((max_rows, nodes)),
        "pool_side": ints((max_rows, nodes)),
        "work": torch.rand((max_rows, verify + k), generator=gen, device=_DEVICE),
        "sort": ties((max_rows, verify)).sort(dim=-1, descending=True).values,
        "tokens": ints((max_rows, k)),
        "scores": ties((max_rows, k)),
        "hidden": torch.randn(
            (max_rows, k, hidden_size), generator=gen, device=_DEVICE
        ).to(torch.bfloat16),
        "frontier_index": ints((2, max_rows, k)),
        "positions": ints((max_rows, k)),
        "slot_starts": ints((max_rows,)),
        "lane_mask": ints((max_rows, k, words), 1 << 62),
    }
    # Depth 1's lane masks: lane r sees its own slot.
    state["lane_mask"][:rows] = 0
    state["lane_mask"][:rows, :, :1] = (1 << torch.arange(k, device=_DEVICE))[:, None]
    fused = {name: t.clone() for name, t in state.items()}
    # The unfused sequence's bool masks per depth: stale ancestors, own slots.
    masks = [torch.empty(0)]
    for s in range(1, steps):
        mask = torch.rand((max_rows, k, s * k), generator=gen, device=_DEVICE) < 0.5
        mask[:, :, (s - 1) * k :] = torch.eye(k, dtype=torch.bool, device=_DEVICE)
        masks.append(mask)
    root_children = torch.arange(1, k + 1, device=_DEVICE).repeat(rows, 1)

    def fused_update(
        child_p, child_tokens, expanded, child_hidden, width, depth, cumu_count
    ):
        eagle_harness_draft_depth_update(
            child_p, child_tokens, expanded, child_hidden, fused["tokens"][:rows],
            fused["scores"][:rows], fused["hidden"][:rows],
            fused["frontier_index"][depth % 2][:rows], fused["pool_tokens"][:rows],
            fused["pool_scores"][:rows], fused["pool_depths"][:rows],
            fused["pool_prev"][:rows], fused["pool_next"][:rows], fused["pool_side"][:rows],
            fused["work"][:rows], fused["sort"][:rows], fused["positions"][:rows],
            fused["slot_starts"][:rows], fused["lane_mask"][:rows], _count(width),
            depth, _count(cumu_count), _count(verify_num), num_slots,
        )  # fmt: skip

    cumu_count = 1 + k
    verify_num = int(torch.randint(1, 65, (1,), generator=gen, device=_DEVICE))
    for depth in range(1, steps):
        width = int(torch.randint(1, k + 1, (1,), generator=gen, device=_DEVICE))
        # The lane forward's outputs: K children per lane, the first width expanded.
        child_p = ties((rows * k, k)).view(rows, k, k)
        child_tokens = ints((rows * k, k)).view(rows, k, k)
        child_hidden = torch.randn(
            (rows * k, hidden_size), generator=gen, device=_DEVICE
        ).to(torch.bfloat16)
        expanded = [
            root_children if depth == 1 else s["frontier_index"][(depth + 1) % 2][:rows]
            for s in (state, fused)
        ]
        if disabled:
            fused_update(
                child_p,
                child_tokens,
                expanded[1],
                child_hidden.view(rows, k, -1),
                0,
                depth,
                0,
            )
            torch.cuda.synchronize()
            for name, expected in state.items():
                _assert_bitwise_equal(fused[name], expected, name)
            return
        _unfused_depth(
            state, masks, child_p[:, :width], child_tokens[:, :width], expanded[0],
            child_hidden.view(rows, k, -1), depth, cumu_count, verify_num, num_slots,
        )  # fmt: skip
        fused_update(
            child_p, child_tokens, expanded[1], child_hidden.view(rows, k, -1), width,
            depth, cumu_count,
        )  # fmt: skip
        cumu_count += width * k
        for name, expected in state.items():
            _assert_bitwise_equal(fused[name], expected, f"{name} at depth {depth}")


def test_disabled_chain_depth_preserves_pool_and_frontier():
    test_draft_depth_update_matches_the_unfused_sequence(3, 5, 1, 7, disabled=True)


@pytest.mark.parametrize("verify_num", [1, 16])
def test_write_verify_tree(verify_num: int) -> None:
    """Ranked nodes' tokens and parent slots, the rest the root with parent -1,
    in each request's row of the next round's maps."""
    bs, nodes, slots = 3, 64, 16
    generator = torch.Generator().manual_seed(verify_num)
    prev = torch.full((bs, nodes), -1, dtype=torch.int64)
    for node in range(1, nodes):
        prev[:, node] = torch.randint(0, node, (bs,), generator=generator)
    tokens = torch.randint(0, 1000, (bs, nodes), generator=generator)
    # An ancestor-first ranking: each node's parent is ranked before it.
    # A padded row past the live ones is never written.
    ranked = torch.zeros((bs + 1, 16), dtype=torch.int64)
    for b in range(bs):
        chosen = [0]
        while len(chosen) < 16:
            node = int(torch.randint(1, nodes, (1,), generator=generator))
            if node not in chosen and int(prev[b, node]) in chosen:
                chosen.append(node)
        ranked[b] = torch.tensor(chosen)
    root = torch.tensor([7, 8, 9, 6], dtype=torch.int32)
    rows = torch.tensor([5, 2, 9, 11], dtype=torch.int64)
    prev, tokens = (torch.cat([t, t[:1]]) for t in (prev, tokens))
    input_map = torch.zeros((12, slots), dtype=torch.int32, device=_DEVICE)
    parent_map = torch.zeros((12, slots), dtype=torch.int32, device=_DEVICE)
    eagle_harness_write_verify_tree(
        ranked.to(_DEVICE), tokens.to(_DEVICE), prev.to(_DEVICE), root.to(_DEVICE),
        rows.to(_DEVICE), input_map, parent_map, _count(bs), _count(verify_num),
    )  # fmt: skip
    assert (input_map[11] == 0).all() and (parent_map[11] == 0).all()

    for b in range(bs):
        want_tokens = [int(root[b])] * slots
        want_parents = [-1] * slots
        for i in range(verify_num):
            node = int(ranked[b, i])
            want_tokens[i] = int(tokens[b, node])
            parent = int(prev[b, node])
            want_parents[i] = ranked[b, :i].tolist().index(parent) if i else -1
        assert input_map[rows[b]].tolist() == want_tokens
        assert parent_map[rows[b]].tolist() == want_parents


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
