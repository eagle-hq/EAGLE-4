# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The draft tree's stop rule (:mod:`eagle_harness.vllm.depth_stop`): exact for ``min_gain = 0``, a step priced by the
cost tables, kernel == reference, and a captured growth with conditional steps == the full-depth growth (GPU)."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from eagle_harness.core.calibration import AttentionShape, CalibrationPaths, MeasureParams, StepTiming, fit_step_timings
from eagle_harness.core.cost_table import BatchCostTable, CostCurve, IntervalCost, load_cost_tables
from eagle_harness.core.draft_tree import DEPTH_DECAY, DraftTree, TreeShape
from eagle_harness.core.limits import ComputeIntensity, verify_buckets
from tests.triton_probe import needs_triton

K, DEPTH, H = 4, 8, 8


class _Draft(torch.nn.Module):
    """The slice of the EAGLE-4 draft model TreeDrafter reads at construction."""

    def __init__(self, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(qkv_proj=torch.nn.Linear(hidden, hidden)))]
        )


def _decay(d: int) -> float:
    return float(torch.tensor(DEPTH_DECAY**d, dtype=torch.float32))


def _tree(bs: int) -> DraftTree:
    return DraftTree(TreeShape(top_k=K, max_depth=DEPTH), bs, H, torch.float32, torch.device("cpu"), max_verify=1)


def _probs(gen: torch.Generator, rows: int, sharpness: torch.Tensor) -> torch.Tensor:
    """Top-``K`` probabilities of a random vocabulary of 50 per row (``sharpness``: per row, high = confident)."""
    logits = torch.randn(rows, 50, generator=gen) * sharpness[:, None]
    return torch.topk(logits.softmax(-1), K, dim=-1).values


def keep_growing_reference(
    scores: torch.Tensor,
    decay: torch.Tensor,
    front: torch.Tensor,
    count: int,
    next_decay: float,
    min_gain: float,
    parent_layer: tuple[int, int],
    step_share: float = 0.0,
) -> torch.Tensor:
    """The rule of :func:`~eagle_harness.vllm.depth_stop.keep_growing` in torch: ``scores`` ``[bs, nodes]`` (0 where no
    node was grown), ``decay`` ``[nodes]``, ``front`` ``[bs, K]`` the frontier's scores (the newest layer's best),
    ``parent_layer`` the node range of the layer above the newest, ``step_share`` :meth:`RoundCosts.step_share`; a
    scalar bool."""
    ranked = torch.sort(scores * decay, dim=-1, descending=True).values
    threshold = ranked[:, count - 1 : count]
    reach = front * torch.tensor(next_decay, dtype=torch.float32, device=front.device)
    parent_best = scores[:, parent_layer[0] : parent_layer[1]].max(-1, keepdim=True).values
    ratio = front.max(-1, keepdim=True).values / parent_best.clamp(min=1e-30)
    gain = (reach * ratio - threshold).clamp(min=0).sum(-1)
    expected = 1.0 + ranked[:, :count].sum(-1)
    keep = (reach.max(-1).values >= threshold[:, 0]) & (gain >= min_gain + step_share * expected)
    return keep.any()


def _grow(bs: int, count: int, seed: int, min_gain: float | None):
    """Grow a tree on CPU with the reference step; with ``min_gain`` stop as the rule says. Returns
    ``(verify node set per request, steps run)``."""
    gen = torch.Generator().manual_seed(seed)
    sharp = torch.rand(bs, generator=gen) * 6  # some requests confident, some not
    tree = _tree(bs)
    tree.root_probs[:bs] = _probs(gen, bs, sharp)
    tree.root_tokens[:bs] = torch.randint(0, 1000, (bs, K), generator=gen)
    tree.begin_reference(bs)
    tree.scores[:bs, tree.shape.layer_start(3) :] = 0
    steps = 0
    for step in range(1, DEPTH):
        # the draws do not depend on whether an earlier step was skipped: same tree up to where it stops
        probs = _probs(gen, bs * K, sharp.repeat_interleave(K))
        tokens = torch.randint(0, 1000, (bs * K, K), generator=gen)
        if step > 1 and min_gain is not None:
            front = tree.front_scores[:bs]
            parent = (tree.shape.layer_start(step - 1), tree.shape.layer_start(step))
            if not keep_growing_reference(
                tree.scores[:bs], tree.decay, front, count, _decay(step + 1), min_gain, parent
            ):
                break
        tree.step_reference(bs, step, probs, tokens, torch.zeros(bs * K, H))
        steps += 1
    ranked, _ = tree.rank(bs, DEPTH, count)
    return [set(row.tolist()) for row in ranked], steps


@pytest.mark.parametrize("bs, count", [(1, 15), (1, 7), (3, 15), (2, 20)])
def test_zero_gain_stop_keeps_the_full_depth_verify_set(bs, count):
    stopped = 0
    for seed in range(40):
        full, full_steps = _grow(bs, count, seed, None)
        early, steps = _grow(bs, count, seed, 0.0)
        assert early == full, f"seed {seed}: the early stop changed the verify set"
        assert full_steps == DEPTH - 1
        stopped += steps < full_steps
    assert stopped > 0  # the rule does fire on these trees


def test_a_positive_gain_stops_no_later():
    for seed in range(20):
        _, exact = _grow(1, 15, seed, 0.0)
        _, lossy = _grow(1, 15, seed, 0.05)
        assert lossy <= exact


def test_reference_rule_on_a_hand_made_tree():
    def keep(front, gain, parent=(1, 2), scores=torch.tensor([[0.9, 0.5, 0.2, 0.1, 0.0]])):
        return bool(keep_growing_reference(scores, torch.ones(5), torch.tensor(front), 3, 1.0, gain, parent))

    # 3rd best is 0.2: a frontier at 0.25 can still place a child, one at 0.15 cannot (whatever the ratio)
    assert keep([[0.25, 0.1]], 0.0) and keep([[0.25, 0.1]], 0.0, parent=(0, 1))
    assert not keep([[0.15, 0.1]], 0.0)
    # gain: the children are expected at their parent's score x the last step's best ratio. Layer above at 0.5 and a
    # frontier at 0.5: ratio 1, excess 0.3 + 0.25. Layer above at 0.9: ratio 0.5 / 0.9, excess 0.078 + 0.05.
    assert keep([[0.5, 0.45]], 0.5)
    assert not keep([[0.5, 0.45]], 0.5, parent=(0, 1))
    assert keep([[0.5, 0.45]], 0.1, parent=(0, 1))
    # a batch keeps growing if any request wants to
    two = torch.tensor([[0.9, 0.5, 0.2, 0.1, 0.0]] * 2)
    assert keep([[0.15, 0.1], [0.3, 0.0]], 0.0, scores=two)


def test_reference_prices_a_step_by_the_expected_accept():
    scores = torch.tensor([[0.9, 0.5, 0.2, 0.1, 0.0]])

    def keep(share, gain=0.0):
        front = torch.tensor([[0.5, 0.45]])  # ratio 1 (layer above at 0.5): excess 0.3 + 0.25 = 0.55
        return bool(keep_growing_reference(scores, torch.ones(5), front, 3, 1.0, gain, (1, 2), share))

    # expected accept of the verify set (0.9, 0.5, 0.2) plus the root's token: 2.6; 0.55 / 2.6 = 0.2115
    assert keep(0.21) and not keep(0.212)
    assert keep(0.0, 0.55) and not keep(0.0, 0.551)
    assert not keep(0.1, 0.3)  # the fixed gain and the priced step add up: 0.3 + 0.26 > 0.55


@needs_triton
@pytest.mark.parametrize("share, keeps", [(0.21, True), (0.212, False)])
def test_kernel_prices_a_step_by_the_expected_accept(share, keeps):
    """The hand-made case above through the kernel: the expected accept counts the root and exactly ``count`` rows."""
    from eagle_harness.vllm.depth_stop import keep_growing

    device = "cuda" if torch.cuda.is_available() else "cpu"
    go = torch.zeros((), dtype=torch.bool, device=device)
    stats = torch.zeros(2, dtype=torch.int64, device=device)
    scores = torch.tensor([[0.9, 0.5, 0.2, 0.1, 0.0]], device=device)
    front = torch.tensor([[0.5, 0.45]], device=device)
    keep_growing(
        go, stats, scores, torch.ones(5, device=device), front, 3, 1.0, 0.0, (1, 2), steps_before=0, step_share=share
    )
    assert bool(go) is keeps


@needs_triton
@pytest.mark.parametrize(
    "bs, count, gain, share",
    [
        (1, 15, 0.0, 0.0),
        (3, 15, 0.0, 0.0),
        (2, 7, 0.02, 0.0),
        (1, 20, 0.1, 0.0),
        (1, 15, 0.0, 0.04),
        (3, 7, 0.01, 0.02),
    ],
)
def test_kernel_matches_the_reference(bs, count, gain, share):
    from eagle_harness.vllm.depth_stop import keep_growing

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = TreeShape(top_k=K, max_depth=DEPTH)
    decay = shape.decay(torch.device(device))
    stats = torch.zeros(2, dtype=torch.int64, device=device)
    go = torch.zeros((), dtype=torch.bool, device=device)
    gen = torch.Generator().manual_seed(bs * 100 + count)
    agree, ran = set(), 0
    for trial in range(12):
        scores = torch.rand(bs, shape.capacity, generator=gen) ** 3
        scores[:, shape.layer_start(3 + trial % 4) :] = 0
        front = torch.rand(bs, K, generator=gen) * scores[:, :count].mean() * 2
        parent = (shape.layer_start(2), shape.layer_start(3))
        scores[:, parent[1]] = 1.0  # the next layer's first node: a parent range one too wide changes the ratio
        scores, front = scores.to(device), front.to(device)
        want = keep_growing_reference(scores, decay, front, count, _decay(4), gain, parent, share)
        keep_growing(
            go,
            stats,
            scores,
            decay,
            front,
            count,
            _decay(4),
            gain,
            parent,
            steps_before=2 if trial % 2 == 0 else 0,
            step_share=share,
        )
        ran += (2 if trial % 2 == 0 else 0) + int(want)
        assert bool(go) == bool(want), trial
        agree.add(bool(want))
    assert agree == {True, False}
    assert stats.tolist() == [6, ran]


@needs_triton
@pytest.mark.parametrize("gain, keeps", [(1.0, True), (6.8, False)])
def test_decision_scales_by_the_layer_above_the_newest(gain, keeps):
    """After step 2 (layers 1-3) the ratio is the frontier's best over layer 2's best: 0.6 / 0.2. Layer 1 (0.9) or
    layer 3 (0.18) in its place gives an expected gain of 0.8 or 7.2 instead of 6.4."""
    pytest.importorskip("vllm")
    from eagle_harness.vllm.tree_drafter import TreeDrafter

    attn = SimpleNamespace(
        impl=SimpleNamespace(scale=1.0), sliding_window=None, num_heads=2, num_kv_heads=2, head_size=4
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # the kernel runs where the tree lives
    drafter = TreeDrafter(_Draft(), attn, top_k=4, depth=6, max_reqs=1, device=device, min_gain=gain)
    tree, shape = drafter.tree, drafter.tree.shape
    tree.scores.zero_()
    tree.scores[0, : shape.layer_start(2)] = 0.9
    tree.scores[0, shape.layer_start(2) : shape.layer_start(3)] = 0.2
    tree.scores[0, shape.layer_start(3) : shape.layer_start(4)] = 0.18
    tree.front_scores[0] = 0.6
    drafter._decide(1, 2, 15, 1)
    assert bool(drafter._go) is keeps


def _table(draft_us: float, verify_us: float) -> dict[int, BatchCostTable]:
    """bs-1 cost tables: a draft step of 1, 2, 4 or 8 rows and a target forward of 1 or 16 tokens, the same in two
    context intervals and 1.5x in a third (the median, not the mean)."""

    def curve(index, value):
        return CostCurve(
            value=value, ratio=tuple(v / value[0] for v in value), index=index, array_query=(), array_cache=()
        )

    rows = tuple(
        IntervalCost(
            target=curve((0, 15), (verify_us * s / 2, verify_us * s)),
            draft=curve((0, 1, 3, 7), (draft_us * s / 2, draft_us * s * 0.8, draft_us * s, draft_us * s * 1.4)),
        )
        for s in (1.0, 1.0, 1.5)
    )
    return {1: BatchCostTable(1, 256, 256, 64, 16, 8, rows, tuple(2.0 for _ in rows))}


def test_round_costs_read_the_tables_at_the_round_shape():
    from eagle_harness.vllm.depth_stop import RoundCosts, round_costs

    costs = round_costs(_table(100.0, 1300.0), batch_size=1, top_k=4, count=15)
    assert costs == RoundCosts(draft_step=100.0, rest=1300.0)  # 4 rows: index 3; 15 drafts + root: index 15
    assert round_costs(_table(100.0, 1300.0), batch_size=3, top_k=4, count=15) == costs  # nearest measured batch size
    assert costs.step_share(2) == pytest.approx(100 / 1500)
    assert costs.step_share(5) < costs.step_share(2)  # deeper rounds price a step lower
    # fallback: shapes the tables did not time, or no tables
    assert round_costs(_table(100.0, 1300.0), 1, 3, 15) is None
    assert round_costs(_table(100.0, 1300.0), 1, 4, 23) is None
    assert round_costs({}, 1, 4, 15) is None


def test_round_costs_from_the_calibration_fits(tmp_path):
    """Tables written by the core's fits from whole-step timings (what a vLLM calibration run writes)."""
    from eagle_harness.vllm.depth_stop import round_costs

    lengths = [256 * (1 + i) + off for i in range(15) for off in (64, 192)]
    draft = [StepTiming(q, n, 90.0 + 8.0 * q + 0.002 * n * q) for q in (1, 2, 4, 8) for n in lengths]
    widths = sorted(verify_buckets(1, ComputeIntensity.HIGH))  # the fits read every bucket
    target = [StepTiming(q, n, 4400.0 + 20.0 * q + 0.02 * n * q) for q in widths for n in lengths]
    params = MeasureParams(
        context_step=256, context_stop=4096, batch_size=1, verify_num=64, node_top_k=8, loop_top_m=64, repeat_cnt=1
    )
    paths = CalibrationPaths(str(tmp_path))
    fit_step_timings(
        paths, params, ComputeIntensity.HIGH, (AttentionShape(1, 1, 4096), draft), (AttentionShape(16), target)
    )
    costs = round_costs(load_cost_tables(paths.cost), 1, top_k=4, count=15)
    middle = lengths[len(lengths) // 2]
    assert costs.draft_step == pytest.approx(90.0 + 32.0 + 0.008 * middle, rel=0.05)
    assert costs.rest == pytest.approx(4400.0 + 320.0 + 0.32 * middle, rel=0.05)


def test_derived_cost_tables_fall_back_to_lossless(tmp_path, caplog):
    from eagle_harness.vllm.config import parse_method_config
    from eagle_harness.vllm.depth_stop import derived_cost_tables

    shape = {"mode": "static", "node_top_k": 4, "tree_depth": 14, "max_verify_width": 16}
    assert derived_cost_tables(parse_method_config(shape)) is None  # no data_path
    assert derived_cost_tables(parse_method_config({**shape, "data_path": str(tmp_path)})) is None  # no tables there
    assert "lossless" in caplog.text


def test_derived_serving_refuses_missing_prices(tmp_path):
    from eagle_harness.vllm.config import parse_method_config
    from eagle_harness.vllm.depth_stop import derived_cost_tables

    config = parse_method_config({"data_path": str(tmp_path)})
    with pytest.raises(ValueError, match="calibrate.*depth_stop_gain"):
        derived_cost_tables(config)


def test_derived_serving_checks_every_scheduled_shape(tmp_path):
    from eagle_harness.vllm.config import parse_method_config
    from eagle_harness.vllm.depth_stop import derived_cost_tables
    from tests.test_chain_width import write_tables

    # Batch one is priced, but the larger batch is missing the required draft width.
    tables = _table(100.0, 1300.0)
    from dataclasses import replace

    tables[64] = replace(
        tables[1],
        batch_size=64,
        intervals=tuple(replace(i, draft=replace(i.draft, index=(0, 1, 2, 3))) for i in tables[1].intervals),
    )
    write_tables(str(tmp_path), tables)
    config = parse_method_config({"data_path": str(tmp_path), "max_verify_width": 16})
    with pytest.raises(ValueError, match="batch size"):
        derived_cost_tables(config)


def test_derived_serving_prices_only_the_active_user_schedule(tmp_path):
    from eagle_harness.vllm.config import parse_method_config
    from eagle_harness.vllm.depth_stop import derived_cost_tables, round_costs
    from tests.test_chain_width import write_tables

    write_tables(str(tmp_path), _table(100.0, 1300.0))
    config = parse_method_config({"data_path": str(tmp_path)})  # reserve width 32, serve width 16
    tables = derived_cost_tables(config, schedule=[(1, 8, 15), (9, 512, 31)], max_batch_size=8)
    assert round_costs(tables, 8, 8, 15).rest == 1300.0
    with pytest.raises(ValueError, match="batch size 9"):
        derived_cost_tables(config, schedule=[(1, 8, 15), (9, 512, 31)], max_batch_size=9)


def _decide_after_step_2(cost_tables, min_gain=0.0) -> bool:
    """One decision after step 2 on a hand-made tree: layer 1 at 0.9, layer 2 at 0.2, a frontier at 0.25.
    Expected accept 1 + 4 x 0.9 + 11 x 0.2 = 6.8; the step's estimated gain 4 x (0.25 x 0.25 / 0.2 - 0.2) = 0.45."""
    pytest.importorskip("vllm")
    from eagle_harness.vllm.tree_drafter import TreeDrafter

    attn = SimpleNamespace(
        impl=SimpleNamespace(scale=1.0), sliding_window=None, num_heads=2, num_kv_heads=2, head_size=4
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drafter = TreeDrafter(
        _Draft(), attn, top_k=4, depth=6, max_reqs=1, device=device, min_gain=min_gain, cost_tables=cost_tables
    )
    tree, shape = drafter.tree, drafter.tree.shape
    tree.scores.zero_()
    tree.scores[0, : shape.layer_start(2)] = 0.9
    tree.scores[0, shape.layer_start(2) : shape.layer_start(3)] = 0.2
    tree.front_scores[0] = 0.25
    drafter._decide(1, 2, 15, 1)
    return bool(drafter._go)


@needs_triton
def test_drafter_prices_a_step_by_the_steps_run():
    # 2 steps run: 6.8 x 100 / (1300 + 2 x 100) = 0.453 > 0.45 stops; priced as after 3 steps (0.425) it would grow
    assert not _decide_after_step_2(_table(100.0, 1300.0))
    assert _decide_after_step_2(_table(100.0, 1600.0))  # 6.8 x 100 / 1800 = 0.378
    assert _decide_after_step_2(None)  # no tables: lossless
    assert not _decide_after_step_2(None, min_gain=0.46)  # a fixed gain (the config override)


def test_drafter_stops_early_only_with_a_gain_and_enough_nodes():
    pytest.importorskip("vllm")
    from eagle_harness.vllm.tree_drafter import TreeDrafter

    attn = SimpleNamespace(
        impl=SimpleNamespace(scale=1.0), sliding_window=None, num_heads=2, num_kv_heads=2, head_size=4
    )

    def drafter(**kw):
        return TreeDrafter(_Draft(), attn, top_k=4, max_reqs=1, device=torch.device("cpu"), **kw)

    assert drafter(depth=6, min_gain=0.0).stops_early(15)
    assert drafter(depth=6, min_gain=0.0)._first_decision(20) == 1  # 4 + 16 nodes after the first step
    assert drafter(depth=6, min_gain=0.0)._first_decision(21) == 2
    assert drafter(depth=6, min_gain=0.0).stops_early(31)
    assert not drafter(depth=3, min_gain=0.0).stops_early(21)  # no step left after the tree holds 21 nodes
    assert not drafter(depth=6).stops_early(15)
    assert not drafter(depth=2, min_gain=0.0).stops_early(3)


# ---------------------------------------------------------------------------------------------- GPU: captured growth
class _Model:
    """A tiny draft with the interface TreeDrafter drives: rows attend through ``attend``; ``temp`` (a device tensor)
    sets how confident the candidates are, so a replayed graph sees it change."""

    def __init__(self, heads: int, dim: int, vocab: int, device: str):
        gen = torch.Generator().manual_seed(0)
        hd = heads * dim

        def w(*shape):
            return (torch.randn(*shape, generator=gen) * 0.3).to(device, torch.bfloat16)

        self.config = SimpleNamespace(hidden_size=hd)
        self.model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(qkv_proj=SimpleNamespace(weight=w(1))))]
        )
        self.wq, self.wk, self.wv, self.wo, self.emb, self.head = (
            w(hd, hd),
            w(hd, hd),
            w(hd, hd),
            w(hd, hd),
            w(vocab, hd),
            w(hd, vocab),
        )
        self.temp = torch.ones((), device=device)

    def forward_rows(self, ids, positions, hidden, attend):
        x = hidden + self.emb[ids]
        h = x + attend(x @ self.wq, x @ self.wk, x @ self.wv) @ self.wo
        return h, h

    def candidate_topk(self, normed, k):
        probs = ((normed.float() @ self.head.float()) * self.temp).softmax(-1)
        values, ids = torch.topk(probs, k, dim=-1)
        return values, ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph conditional nodes need a GPU")
@pytest.mark.parametrize("bs, count", [(1, 15), (2, 15), (1, 23), (64, 15), (128, 15), (256, 15), (512, 15)])
def test_captured_growth_with_conditional_steps_equals_the_full_depth(bs, count, monkeypatch):
    pytest.importorskip("vllm")
    import vllm.distributed.parallel_state as ps

    from eagle_harness.vllm.tree_drafter import TreeDrafter

    @contextlib.contextmanager
    def graph_capture(device):  # no TP group in a unit test
        yield SimpleNamespace(stream=torch.cuda.Stream())

    monkeypatch.setattr(ps, "graph_capture", graph_capture)
    dev, heads, dim, block, depth = "cuda", 2, 64, 16, 8
    model = _Model(heads, dim, 300, dev)
    attn = SimpleNamespace(
        impl=SimpleNamespace(scale=dim**-0.5), sliding_window=None, num_heads=heads, num_kv_heads=heads, head_size=dim
    )
    kv = torch.randn(max(64, bs * 16), heads, block, 2 * dim, generator=torch.Generator().manual_seed(1)).to(
        dev, torch.bfloat16
    )
    table = torch.arange(bs * 16, dtype=torch.int32, device=dev).view(bs, 16)
    committed = torch.full((bs,), 40, dtype=torch.int64, device=dev)
    committed[1::2] = 77

    def rounds(graphs: bool, min_gain):
        drafter = TreeDrafter(
            model, attn, top_k=K, depth=depth, max_reqs=bs, device=torch.device(dev), min_gain=min_gain
        )
        drafter.bind_cache(kv, block)
        drafter.graphs = graphs
        out = []
        # temp 0: every candidate has the same probability, so no child can outrank the verify set: a round that must stop
        for r, temp in enumerate([1.0, 30.0, 1.0, 0.3, 30.0, 0.0]):
            model.temp.fill_(temp)
            gen = torch.Generator().manual_seed(r)
            hidden = (torch.randn(bs, heads * dim, generator=gen) * 0.5).to(dev, torch.bfloat16)
            probs, tokens = model.candidate_topk(hidden, K)
            drafter.seed(bs, probs, tokens, hidden)
            tokens, parents = drafter.grow_and_select(bs, count, committed, table)
            out.append((tokens.clone(), parents.clone()))
        torch.cuda.synchronize()
        return out, drafter

    full, _ = rounds(False, None)
    early, drafter = rounds(True, 0.0)
    for r, ((t0, p0), (t1, p1)) in enumerate(zip(full, early)):
        assert torch.equal(t0, t1) and torch.equal(p0, p1), f"round {r}"
    steps = drafter.mean_steps()
    assert steps is not None and steps <= depth - 1
    if count == 15:  # the temp-0 round stops at the first decision (at count 23 the other trees need every step)
        assert steps < depth - 1
    lossy, drafter = rounds(True, 0.2)
    assert drafter.mean_steps() <= steps
