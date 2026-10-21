# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""DynamicDrafter wiring checks that run on CPU (the round itself needs CUDA graphs)."""

from types import SimpleNamespace

import pytest
import torch

from eagle_harness.core.cost_model import CostModel, CostModelParams, FixedShapePolicy
from eagle_harness.core.draft_tree import DraftTree, TreeShape
from eagle_harness.core.limits import ComputeIntensity

pytest.importorskip("sglang")
from eagle_harness.sglang.drafter import DynamicDrafter  # noqa: E402

K, DEPTH, H, WIDTH = 4, 12, 8, 16


def _drafter(tree: DraftTree, cost: CostModel) -> DynamicDrafter:
    return DynamicDrafter(
        SimpleNamespace(tree=tree),
        cost,
        fixed=FixedShapePolicy(ComputeIntensity.HIGH, K, DEPTH, WIDTH),
        build_trivial_input=lambda batch: None,
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(model_is_mrope=False),
                kv_index_translator=SimpleNamespace(reads_are_translated=True),
            )
        ),
        algorithm=SimpleNamespace(target_verify_graphs=None),
        widths=lambda bs: [WIDTH, 8, 1],
        tree_mask_mode=None,
        device=torch.device("cpu"),
    )


def _cost() -> CostModel:
    return CostModel(
        tables={},
        params=CostModelParams(),
        intensity=ComputeIntensity.HIGH,
        node_top_k=K,
        tree_depth=DEPTH,
        verify_num=WIDTH,
    )


def test_drafter_refuses_deltas_sized_by_the_verify_width():
    """The configuration of that regression (deltas for budgets 1..16, history budget 32) fails at startup."""
    cost = _cost()
    tree = DraftTree(TreeShape(K, DEPTH), 1, H, torch.float32, torch.device("cpu"), max_verify=WIDTH)
    with pytest.raises(ValueError, match="budgets up to 32"):
        _drafter(tree, cost)


def test_drafter_refuses_missing_launch_depths():
    cost = _cost()
    tree = DraftTree(TreeShape(K, DEPTH), 1, H, torch.float32, torch.device("cpu"), max_verify=32, delta_slots=(0, 2))
    with pytest.raises(ValueError, match="at depths"):
        _drafter(tree, cost)


class _Event:
    def __init__(self):
        self.records = 0

    def record(self):
        self.records += 1

    def synchronize(self):
        pass

    def query(self):
        return True


class _Steps(SimpleNamespace):
    def run(self, bs, step):
        self.ran.append(step)


class _StopAt:
    """The real cost model's step kinds; DECIDE stops at ``stop``."""

    def __init__(self, cost: CostModel, stop: int):
        self.cost, self.stop = cost, stop

    def __getattr__(self, name):
        return getattr(self.cost, name)

    def tree_width(self, depth, bs, past, row, deltas):
        return SimpleNamespace(stop=depth >= self.stop)


def _sequenced(tiny: bool, stop: int, policy=None):
    cost = _cost()
    cost.tier.tiny = tiny
    tree = DraftTree(
        TreeShape(K, DEPTH), 1, H, torch.float32, torch.device("cpu"), max_verify=32, delta_slots=cost.launch_depths
    )
    steps = _Steps(tree=tree, ran=[], max_step=DEPTH - 1, host_deltas=torch.zeros(DEPTH, 32, K + 1))
    drafter = DynamicDrafter.__new__(DynamicDrafter)
    drafter.steps, drafter.tree, drafter.cost = steps, tree, _StopAt(cost, stop)
    drafter.fixed = FixedShapePolicy(ComputeIntensity.HIGH, K, DEPTH, WIDTH)
    drafter.live, drafter.timer, drafter._seeding = None, None, None
    drafter._deltas_ready, drafter._deltas_slot = _Event(), 0
    drafter._policy = drafter.fixed if policy == "fixed" else drafter.cost
    drafter.policy = lambda bs: drafter._policy
    return drafter, steps


@pytest.mark.parametrize("tiny,first", [(True, 3), (False, 1)])
def test_queue_runs_the_steps_before_the_first_decision(tiny, first):
    """Batch size 1 (tiny): depths 0-1 keep, depth 2 launches, depth 3 decides; otherwise depth 1 decides."""
    drafter, steps = _sequenced(tiny, stop=99)
    assert drafter._queue_until_decision(1, drafter.cost) == first
    assert steps.ran == list(range(1, first + 1))
    assert drafter._deltas_slot == first - 1 and drafter._deltas_ready.records == 1


@pytest.mark.parametrize("stop", [3, 5, 7])
def test_prelaunched_and_direct_rounds_run_the_same_steps(stop):
    direct, steps = _sequenced(True, stop)
    policy = direct.cost
    depth = direct._grow(1, direct._queue_until_decision(1, policy), None, 32, policy)
    resumed, resumed_steps = _sequenced(True, stop)
    policy = resumed.cost
    first = resumed._queue_until_decision(1, policy)  # at the end of the previous round
    assert resumed._grow(1, first, None, 32, policy) == depth == stop + 1
    assert steps.ran == resumed_steps.ran == list(range(1, stop + 1))


@pytest.mark.parametrize("stop", [3, 5, 99])
def test_sampled_depth_uses_the_previous_round_plan(stop):
    drafter, steps = _sequenced(True, stop)
    steps.sampled = SimpleNamespace(active=True)
    drafter._sampled_depth = 6
    first = drafter._queue_until_decision(1, drafter.cost)
    depth = drafter._grow(1, first, None, 32, drafter.cost)
    assert depth == 6
    assert steps.ran == list(range(1, 6))
    assert drafter._sampled_depth == (stop + 1 if stop <= 5 else 8)


def test_grow_stops_at_the_deepest_level():
    drafter, steps = _sequenced(True, stop=99)
    assert drafter._grow(1, drafter._queue_until_decision(1, drafter.cost), None, 32, drafter.cost) == DEPTH
    assert steps.ran == list(range(1, DEPTH))


def test_fixed_shape_runs_every_step_without_host_decisions():
    """mode static / the live seed sweep: the whole tree is queued ahead, no work deltas are copied or waited on."""
    drafter, steps = _sequenced(True, stop=0, policy="fixed")
    fixed = drafter.fixed
    first = drafter._queue_until_decision(1, fixed)
    assert first == DEPTH - 1 and steps.ran == list(range(1, DEPTH))
    assert drafter._deltas_ready.records == 0
    assert drafter._grow(1, first, None, WIDTH, fixed) == DEPTH
    assert steps.ran == list(range(1, DEPTH))
    assert fixed.verify_num_paged([], 40, 1, 100, lagged=True) == (WIDTH, False)
    fixed.force_width = 8
    assert fixed.verify_num_paged([], 40, 1, 100, lagged=True) == (8, False)
    assert fixed.verify_count(1, 5) == 5 and fixed.history_verify_num(1) == WIDTH


CHAIN_DEPTH, CHAIN_WIDTH = 15, 16


class _ChainStopAt:
    """A chain cost model whose chain rule stops at depth ``stop``; records the deltas each decision reads."""

    def __init__(self, stop: int):
        self.cost = CostModel(
            tables={},
            params=CostModelParams(),
            intensity=ComputeIntensity.CHAIN,
            node_top_k=1,
            tree_depth=CHAIN_DEPTH,
            verify_num=CHAIN_WIDTH,
        )
        self.stop, self.read = stop, []
        self.cost.chain_stop = self._chain_stop

    def _chain_stop(self, depth, bs, deltas):
        self.read.append((depth, deltas))
        return depth >= self.stop


def _chain_sequenced(stop: int):
    policy = _ChainStopAt(stop)
    cost = policy.cost
    slots = cost.launch_depths
    tree = DraftTree(
        TreeShape(1, CHAIN_DEPTH), 1, H, torch.float32, torch.device("cpu"), max_verify=CHAIN_WIDTH, delta_slots=slots
    )
    # Host deltas: slot d, budget row v - 1 holds (d, v) so a test can tell which row a decision read.
    host = torch.tensor([[[float(d), float(v + 1)] for v in range(CHAIN_WIDTH)] for d in range(CHAIN_DEPTH)])
    steps = _Steps(tree=tree, ran=[], max_step=CHAIN_DEPTH - 1, host_deltas=host)
    drafter = DynamicDrafter.__new__(DynamicDrafter)
    drafter.steps, drafter.tree, drafter.cost = steps, tree, cost
    drafter.live, drafter.timer, drafter._seeding = None, None, None
    drafter._deltas_ready, drafter._deltas_slot = _Event(), 0
    drafter.policy = lambda bs: cost
    return drafter, steps, policy


@pytest.mark.parametrize("stop", [2, 6, 12])
def test_chain_rounds_stop_with_the_chain_rule(stop):
    """A top-1 chain at chain intensity asks the chain rule (CostModel.chain_stop), not the tree rule:
    depth 0 keeps, odd depths launch their deltas, even depths decide from the whole-chain budget's deltas."""
    drafter, steps, policy = _chain_sequenced(stop)
    cost = policy.cost
    assert cost.launch_depths == (1, 3, 5, 7, 9, 11, 13)
    first = drafter._queue_until_decision(1, cost)
    assert first == 2 and steps.ran == [1, 2] and drafter._deltas_slot == 1
    for _ in range(16):  # a history of narrow verifies: the tree rule's budget would read another deltas row
        cost.record_verify_num(1, 4)
    past = cost.history_verify_num(1)
    assert past < CHAIN_WIDTH
    assert drafter._grow(1, first, None, past, cost) == stop + 1
    assert steps.ran == list(range(1, stop + 1))
    assert policy.read == [(d, [float(d - 1), float(CHAIN_WIDTH)]) for d in range(2, stop + 1, 2)]


def test_chain_rounds_draft_the_whole_chain_when_the_rule_never_stops():
    drafter, steps, policy = _chain_sequenced(stop=99)
    cost = policy.cost
    first = drafter._queue_until_decision(1, cost)
    assert drafter._grow(1, first, None, cost.history_verify_num(1), cost) == CHAIN_DEPTH
    assert steps.ran == list(range(1, CHAIN_DEPTH))
    assert [d for d, _ in policy.read] == list(range(2, CHAIN_DEPTH - 1, 2))


def test_chain_rounds_verify_every_drafted_token():
    """A chain verifies every drafted token: the paged verify budget is not asked; the history is not touched."""
    drafter, _, policy = _chain_sequenced(stop=4)
    cost = policy.cost

    def refuse(*args, **kwargs):
        raise AssertionError("a chain round asked for the paged verify budget")

    cost.verify_num_paged = refuse
    history = [list(h) for h in cost._history]
    assert cost.verify_count(1, 6) == 6
    assert drafter._verify_width(cost, [0.0] * 6, 6, 1, 100) == 6
    assert cost._history == history


def test_narrow_chains_keep_the_tree_rule():
    """The chain rule needs the 16-token chain's cost index; a width-8 chain (live costs index 0, 1, 3, 5, 7) keeps the
    tree rule instead of failing its first round."""
    cost = CostModel(
        tables={}, params=CostModelParams(), intensity=ComputeIntensity.CHAIN, node_top_k=1, tree_depth=7, verify_num=8
    )
    assert not cost.chain_rounds and not cost.verifies_whole_draft
    assert cost.launch_depths == (0, 2, 4)
    assert _ChainStopAt(stop=2).cost.chain_rounds


def test_tree_rounds_keep_the_tree_rule():
    cost = _cost()
    cost.tier.tiny = True
    assert not cost.verifies_whole_draft and cost.deltas_budget(32) == 32
    assert [cost.width_step(d).value for d in range(4)] == ["keep", "keep", "launch", "decide"]
    assert cost.launch_depths == tuple(range(0, DEPTH - 1, 2))


def test_batch_key_is_the_block_table_rows():
    from eagle_harness.sglang.drafter import batch_key

    batch = SimpleNamespace(reqs=[SimpleNamespace(kv=SimpleNamespace(req_pool_idx=i)) for i in (3, 1)])
    assert batch_key(batch) == (3, 1)


class _ChainSteps(SimpleNamespace):
    max_step = 6  # --speculative-num-steps 7: the widest chain verifies 8 tokens

    def run(self, bs, step):
        assert 1 <= step <= self.max_step  # DraftTree.step's range
        self.ran.append(step)

    def load_batch(self, bs, seq_lens, req_pool_indices):
        self.loaded.append(bs)

    def begin(self, bs, probs, tokens, hidden):
        self.begun.append(bs)


class _Tail:
    def __init__(self):
        self.built = []

    def ranked_verify_input(self, bs, depth, count, width, on_rank):
        self.built.append((bs, depth, count, width))
        return SimpleNamespace(prepared_mrope_positions=None)


def _chain_drafter(widths):
    """A drafter with a fixed width per round (``widths`` handed out in order) and recording fakes."""
    drafter = DynamicDrafter.__new__(DynamicDrafter)
    plan = iter(widths)
    drafter._chain_width = lambda bs: next(plan)
    drafter.steps = _ChainSteps(ran=[], loaded=[], begun=[])
    drafter.tail = _Tail()
    drafter._trivial = lambda batch: "trivial"
    drafter._bonus = torch.zeros(4, dtype=torch.int64)
    drafter._device = torch.device("cpu")
    drafter._pending = None
    return drafter


def _chain_batch(bs):
    return SimpleNamespace(
        seq_lens=torch.full((bs,), 100),
        req_pool_indices=torch.arange(bs),
        reqs=[SimpleNamespace(kv=SimpleNamespace(req_pool_idx=i)) for i in range(bs)],
        spec_info=SimpleNamespace(topk_p=None, topk_index=None, hidden_states=None, bonus_tokens=torch.ones(bs)),
        multimodal_inputs=[None] * bs,
    )


def test_fixed_width_chain_drafts_width_minus_one_tokens_and_verifies_them():
    """The begin drafts depth 1, so width 4 (3 drafted tokens) runs steps 1 and 2."""
    drafter = _chain_drafter([4])
    drafter.draft(_chain_batch(2))
    assert drafter.steps.ran == [1, 2] and drafter.steps.begun == [2]
    assert drafter.tail.built == [(2, 3, 3, 4)]
    assert (drafter.last_depth, drafter.last_width) == (3, 4)


def test_the_widest_chain_ends_at_the_last_draft_step():
    """Width 8 with --speculative-num-steps 7: 7 drafted tokens, steps 1..6 (a step 7 was out of the tree's range)."""
    drafter = _chain_drafter([8, 2])
    drafter.draft(_chain_batch(4))
    assert drafter.steps.ran == [1, 2, 3, 4, 5, 6]
    assert drafter.tail.built == [(4, 7, 7, 8)]
    drafter.draft(_chain_batch(4))  # width 2: the begin's token only
    assert drafter.steps.ran == [1, 2, 3, 4, 5, 6] and drafter.steps.begun == [4, 4]
    assert drafter.tail.built[-1] == (4, 1, 1, 2)


def test_fixed_width_one_is_plain_decoding():
    """Width 1 drafts nothing but still stages the batch: the draft extend after the verify reads it from the steps
    (a stale batch there wrote the draft KV of other requests and broke acceptance)."""
    drafter = _chain_drafter([1])
    assert drafter.draft(_chain_batch(2)) == "trivial"
    assert drafter.steps.loaded == [2]
    assert drafter.steps.ran == [] and drafter.steps.begun == [] and drafter.tail.built == []
    assert drafter.last_width == 1


def test_prelaunched_chain_keeps_the_width_it_was_queued_with():
    """The plan is asked once per round: by the prelaunch, which queues the whole chain, not again by the draft."""
    drafter = _chain_drafter([6, 2])
    batch = _chain_batch(1)
    drafter._prelaunch_stream = None
    drafter._take_pending = lambda: drafter.__dict__.pop("_pending_round", None)
    width = drafter._chain_width(1)
    from eagle_harness.sglang.drafter import Prelaunch, batch_key

    drafter._pending_round = Prelaunch(batch_key(batch), drafter._queue_chain(1, width), width, stream=None)
    drafter.draft(batch)
    assert drafter.steps.ran == [1, 2, 3, 4] and drafter.steps.begun == []
    assert drafter.tail.built == [(1, 5, 5, 6)]
    drafter.draft(_chain_batch(3))  # no prelaunch for this batch: the plan's next width
    assert drafter.tail.built[-1] == (3, 1, 1, 2)


@pytest.mark.parametrize("cycles, offset, sampled", [(1, -1, False), (1, 0, True), (1, 1, False), (2, 0, True)])
def test_live_draft_costs_continue_sampling_after_seeding(cycles, offset, sampled):
    from eagle_harness.core.live_costs import LiveCosts
    from eagle_harness.sglang.drafter import LIVE_REFRESH_ROUNDS

    drafter, _ = _sequenced(True, stop=0)
    drafter.live = LiveCosts(widths=lambda bs: [1, 8])
    drafter._rounds = LIVE_REFRESH_ROUNDS * cycles + offset
    drafter._bucket = lambda bs: bs
    events = []
    drafter.timer = SimpleNamespace(start=lambda: "start", stop=lambda *args: events.append(args))
    assert drafter._timed_steps(4, lambda: 3) == 3
    assert events == ([("start", 4, 0, 0, 3)] if sampled else [])


def test_prelaunched_draft_timing_uses_the_incoming_batch_and_context():
    from eagle_harness.core.live_costs import LiveCosts

    drafter, _ = _sequenced(True, stop=0)
    drafter.live = LiveCosts(widths=lambda bs: [1, 8])
    drafter._seeding = (4, 100)  # The previous batch is shrinking before its next prelaunch.
    drafter._bucket = lambda bs: bs
    events = []
    drafter.timer = SimpleNamespace(start=lambda: "start", stop=lambda *args: events.append(args))
    assert drafter._timed_steps(2, lambda: 3, context=2048) == 3
    assert events == [("start", 2, 0, 2048, 3)]


def test_cached_or_refreshed_unusable_live_costs_keep_fixed_shape():
    from eagle_harness.core.live_costs import DRAFT_WIDTH, LiveCosts
    from eagle_harness.sglang.drafter import LIVE_REFRESH_ROUNDS

    drafter, _ = _sequenced(True, stop=0)
    del drafter.policy  # Exercise the production policy, not the step-order fixture's override.
    drafter.cost = _cost()
    drafter.live = LiveCosts(widths=lambda bs: [1, 8, 16], seed_samples=1)
    for width in (1, 8, 16):
        drafter.live.record(1, width, 100, 4000.0)
    drafter.live.record(1, DRAFT_WIDTH, 100, 2000.0)  # target/draft ratio 2: not priced by this model
    drafter.cost.set_tables(drafter.live.tables())
    drafter._bucket = lambda bs: bs
    drafter.timer = SimpleNamespace(drain=lambda **kw: None)
    drafter._sync_cells = drafter._save_cells = None
    drafter._unpriced = set()
    batch = SimpleNamespace(reqs=[None], seq_lens_cpu=torch.tensor([100]))
    for previous_rounds in (0, LIVE_REFRESH_ROUNDS - 1):
        drafter._rounds = previous_rounds
        drafter._unpriced.clear()
        drafter._begin_round(batch)
        assert drafter.policy(1) is drafter.fixed
    # A new measurement can recover the bucket on a later refresh.
    drafter.live.cells[(1, DRAFT_WIDTH, 0)].us = 170.0
    drafter._rounds = LIVE_REFRESH_ROUNDS - 1
    drafter._begin_round(batch)
    assert drafter.policy(1) is drafter.cost


def test_a_bucket_the_cost_model_refuses_keeps_the_fixed_shape(caplog):
    """Live costs outside the cost model's ratio checks must not raise mid-serving."""
    drafter, _ = _sequenced(True, stop=0)
    drafter._unpriced = set()
    with caplog.at_level("WARNING"):
        drafter._check_priced(1)  # the model has no table for bucket 1
    assert drafter._unpriced == {1} and "keeps the fixed draft shape" in caplog.text


def test_average_context_reads_request_lengths_when_the_overlap_scheduler_drops_host_mirrors():
    """Under the overlap scheduler a decode batch has neither seq_lens_cpu nor seq_lens_sum (first live-cost round)."""
    reqs = [SimpleNamespace(seqlen=10), SimpleNamespace(seqlen=21)]
    overlap = SimpleNamespace(reqs=reqs, seq_lens=torch.tensor([10, 21]), seq_lens_cpu=None, seq_lens_sum=None)
    assert DynamicDrafter.average_context(overlap) == 15
    synchronous = SimpleNamespace(reqs=reqs, seq_lens=torch.tensor([10, 21]), seq_lens_cpu=torch.tensor([12, 22]))
    assert DynamicDrafter.average_context(synchronous) == 17
