# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The served-round chain policy of a serve-mode chain without data_path (:class:`ServedChainPolicy`), on CPU."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from eagle_harness.core.config import HarnessConfig, Mode
from eagle_harness.core.limits import ComputeIntensity
from eagle_harness.sglang import chain_rounds
from eagle_harness.sglang.chain_rounds import ChainRounds, ServedChainPolicy, ServedTreePolicy, served_chain, served_tree


class Clock:
    """``time.perf_counter_ns`` under test control."""

    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def advance(self, ms: float) -> None:
        self.ns += int(ms * 1e6)


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(chain_rounds.time, "perf_counter_ns", clock)
    return clock


def _policy(
    tmp_path=None, *, widths=(1, 2, 4, 8), native=False, writer=True, sync=lambda step: step, key=None, packaged=None
):
    return ServedChainPolicy(
        widths,
        torch.device("cpu"),
        max_batch_size=512,
        native_decode=native,
        writer=writer,
        sync=sync,
        cache_path=None if tmp_path is None else str(tmp_path / "chain.json"),
        key=key,
        packaged=packaged,
    )


def _reqs(*ids):
    return [SimpleNamespace(rid=i) for i in ids]


def _serve(policy, clock, bs, period_ms, accept, rounds=10_000):
    """Serve rounds of ``bs`` requests until their batch key is decided; ``period_ms[w]`` is the whole period of a
    width-``w`` round (0: native decode) and ``accept[w]`` its accept length. Returns the widths served."""
    requests = _reqs(*range(bs))
    served = []
    for _ in range(rounds):
        width = policy.width(bs)
        policy.round_start(bs, width, requests)
        if not policy.exploring(bs):
            return served
        served.append(width)
        if width:
            policy.accepted(width, torch.full((bs,), accept[width]))
        clock.advance(period_ms[width])
    raise AssertionError("no decision")


def test_a_prefill_between_two_rounds_is_part_of_the_first_rounds_period(clock):
    policy = _policy()
    width = policy.width(64)
    policy.round_start(64, width, _reqs(1, 2))
    clock.advance(10)
    # A prefill admits request 3 (no decode round): its time stays in the previous round's period.
    clock.advance(25)
    policy.round_start(64, policy.width(64), _reqs(1, 2, 3))
    assert policy._cells[(64, width)] == [35_000.0, 64, 1]  # us, requests (the batch size), periods


def test_a_complete_turnover_does_not_count_as_a_served_interval(clock):
    policy = _policy()
    width = policy.width(8)
    policy.round_start(8, width, _reqs(1, 2))
    clock.advance(5_000)  # the batch drained: the server may have idled before the next requests came
    policy.round_start(8, policy.width(8), _reqs(3, 4))
    assert policy._cells.get((8, width), [0.0, 0, 0])[2] == 0


def test_a_burst_of_arrivals_counts_toward_the_comparison_without_being_priced(clock):
    policy = _policy()
    width = policy.width(64)
    policy.round_start(64, width, _reqs(*range(64)))
    clock.advance(200)  # a large prefill admits 9 requests (more than 1/8 of the batch)
    policy.round_start(64, policy.width(64), _reqs(*range(9, 73)))
    assert policy._cells[(64, width)] == [0.0, 0, 1]
    clock.advance(10)  # one refill: priced
    policy.round_start(64, policy.width(64), _reqs(*range(10, 74)))
    assert policy._cells[(64, width)] == [10_000.0, 64, 2]


def test_a_turnover_measures_native_decode_and_its_best_width_again_together(clock):
    policy = _policy(native=True)
    policy._pair[64] = (4, 0)
    requests = _reqs(*range(64))
    while policy.width(64) == 4:
        policy.round_start(64, 4, requests)
        clock.advance(10)
    policy.round_start(64, 0, requests)  # native decode's first round: the best width has its block
    clock.advance(5_000)
    policy.round_start(64, policy.width(64), _reqs(*range(100, 164)))  # a new load after the batch drained
    assert policy._cells[(64, 4)] == [0.0, 0, 0] and policy._cells[(64, 0)] == [0.0, 0, 0]
    assert policy.width(64) == 4


def test_the_search_widens_while_the_wider_chain_wins(clock):
    policy = _policy()
    # Accept lengths 1, 2, 3, 3 at widths 1, 2, 4, 8: tokens per ms 0.1, 0.167, 0.2, 0.1 per request.
    served = _serve(policy, clock, 64, {1: 10, 2: 12, 4: 15, 8: 30}, {1: 1, 2: 2, 4: 3, 8: 3})
    assert policy.decided == {64: 4} and set(served) == {1, 2, 4, 8}
    # Another batch key searches on its own: compute-bound rounds keep the root-only verify.
    _serve(policy, clock, 256, {1: 20, 2: 45, 4: 80, 8: 160}, {1: 1, 2: 2, 4: 3, 8: 3})
    assert policy.decided == {64: 4, 256: 1}


def test_a_larger_batch_never_searches_wider_than_a_smaller_one_chose(clock):
    policy = _policy()
    _serve(policy, clock, 16, {1: 10, 2: 12, 4: 15, 8: 30}, {1: 1, 2: 2, 4: 3, 8: 3})
    # At batch 64 width 8 would win, but width 4 won at batch 16: the search stops there.
    served = _serve(policy, clock, 64, {1: 10, 2: 12, 4: 15, 8: 16}, {1: 1, 2: 2, 4: 3, 8: 5})
    assert 8 not in served and policy.decided == {16: 4, 64: 4}


def test_native_decode_is_measured_last_and_spreads_to_larger_batches(clock):
    policy = _policy(native=True)
    # Width 2 beats width 1 (2/15 > 1/10) and width 4 (2/18); native decode (1/7) beats width 2.
    served = _serve(policy, clock, 64, {0: 7, 1: 10, 2: 15, 4: 18, 8: 25}, {1: 1, 2: 2, 4: 2, 8: 2})
    first_native = served.index(0)
    assert 0 not in served[:first_native] and set(served[first_native:]) == {0}  # once native, the batch stays
    assert policy.decided[64] == 0
    assert all(policy.decided[bs] == 0 for bs in (96, 128, 192, 256, 384, 512))  # larger batch keys: unmeasured
    assert policy.exploring(48)


def test_a_losing_native_decode_leaves_the_best_width(clock):
    policy = _policy(native=True)
    _serve(policy, clock, 16, {0: 9, 1: 10, 2: 12, 4: 15, 8: 30}, {1: 1, 2: 2, 4: 3, 8: 3})
    assert policy.decided == {16: 4}


def test_no_smaller_batch_measures_native_decode_once_speculation_won(clock):
    policy = _policy(native=True)
    periods, accept = {0: 9, 1: 10, 2: 12, 4: 15, 8: 30}, {1: 1, 2: 2, 4: 3, 8: 3}
    assert 0 in _serve(policy, clock, 64, periods, accept) and policy.decided == {64: 4}
    assert 0 not in _serve(policy, clock, 16, periods, accept) and policy.decided[16] == 4


def test_native_decode_is_not_chosen_without_the_engine_route(clock):
    policy = _policy(native=False)
    served = _serve(policy, clock, 64, {0: 1, 1: 10, 2: 15, 4: 18, 8: 25}, {1: 1, 2: 2, 4: 2, 8: 2})
    assert 0 not in served and policy.decided == {64: 2}


def test_comparisons_interleave_the_two_widths_in_blocks(clock):
    policy = _policy()
    plan = []
    for _ in range(4 * chain_rounds.BLOCK_ROUNDS):
        width = policy.width(64)
        plan.append(width)
        policy.round_start(64, width, _reqs(1))
    block = chain_rounds.BLOCK_ROUNDS
    assert plan == ([1] * block + [2] * block) * 2


def test_decisions_survive_a_restart_on_the_same_setup_only(tmp_path, clock):
    key = {"gpu": "NVIDIA B300", "gpu_count": 4}
    policy = _policy(tmp_path, key=key)
    _serve(policy, clock, 64, {1: 10, 2: 12, 4: 15, 8: 30}, {1: 1, 2: 2, 4: 3, 8: 3})
    restarted = _policy(tmp_path, key=key, writer=False)
    assert restarted.decided == {64: 4} and restarted.width(60) == 4 and not restarted.exploring(64)
    assert _policy(tmp_path, key={**key, "gpu_count": 8}).decided == {}  # another setup explores again
    assert _policy(tmp_path, key=key, native=True).decided == {}  # another action set too
    (tmp_path / "chain.json").write_text(
        json.dumps({**json.loads((tmp_path / "chain.json").read_text()), "chain": {"64": 3}})
    )
    assert _policy(tmp_path, key=key).decided == {}  # a width without a graph is refused


def test_a_first_start_serves_the_packaged_choices_of_its_setup(clock):
    policy = _policy(native=True, packaged={16: 4, 64: 4, 256: 0})
    assert {k: w for k, w in policy.decided.items() if w} == {16: 4, 64: 4}
    assert all(policy.decided[k] == 0 for k in (256, 384, 512))  # native decode serves every larger batch key
    assert policy.width(60) == 4 and policy.width(300) == 0 and not policy.exploring(64)
    assert policy.exploring(128) and policy.exploring(8)  # unmeasured keys are searched on the served rounds
    # A search between packaged keys stays within the width a smaller key chose.
    served = _serve(policy, clock, 128, {0: 9, 1: 10, 2: 12, 4: 15, 8: 30}, {1: 1, 2: 2, 4: 3, 8: 3})
    assert 8 not in served and policy.decided[128] == 4


def test_the_servers_own_choices_take_precedence_over_the_packaged_ones(tmp_path, clock):
    key = {"gpu": "NVIDIA B300 SXM6 AC", "gpu_count": 4}
    periods, accept = {0: 9, 1: 10, 2: 12, 4: 30, 8: 60}, {1: 1, 2: 2, 4: 3, 8: 3}
    _serve(_policy(tmp_path, key=key, native=True), clock, 64, periods, accept)
    assert _policy(tmp_path, key=key, native=True, packaged={16: 4, 64: 4, 256: 0}).decided == {64: 2}


def test_packaged_choices_that_are_not_actions_of_the_server_are_not_served():
    assert _policy(native=False, packaged={16: 4, 64: 4, 256: 0}).decided == {}  # no native decode route
    assert _policy(native=True, widths=(1, 2, 8), packaged={16: 4}).decided == {}  # no graph for width 4


def test_other_ranks_apply_rank_zeros_step_without_reading_the_gpu(clock):
    class NoDeviceCounts:
        def __getitem__(self, width):
            raise AssertionError("a non-writer rank counted accept lengths")

        def tolist(self):
            raise AssertionError("a non-writer rank read the accept counts")

    received = []
    policy = _policy(writer=False, sync=lambda step: received.append(step) or ("decide", 2, None))
    policy._accept = NoDeviceCounts()
    rounds = ChainRounds(None, None, lambda batch: 0, policy)
    requests = _reqs(1)
    while policy.exploring(8):
        width = policy.width(8)
        policy.round_start(8, width, requests)
        rounds.accepted(width, torch.ones(8, dtype=torch.int64))
        clock.advance(1)
    assert received == [None] and policy.decided == {8: 2}


def test_batch_sizes_share_the_nearest_calibration_batch_size():
    policy = _policy()
    policy.decided = {64: 4, 256: 0}
    rounds = ChainRounds(None, None, lambda batch: 0, policy)
    assert rounds.draft_width(57) == rounds.draft_width(70) == 4
    assert rounds.plain_width(256) == 0 and rounds.draft_width(256) == 1  # a draft never asks for width 0
    assert policy.exploring(8) and not policy.exploring(250)


def test_only_a_serve_mode_chain_without_data_path_is_served_by_the_policy():
    chain = HarnessConfig(intensity=ComputeIntensity.CHAIN)
    assert served_chain(chain, 1)
    assert not served_chain(chain, 4)
    assert not served_chain(HarnessConfig(intensity=ComputeIntensity.CHAIN, data_path="/cost"), 1)
    assert not served_chain(HarnessConfig(intensity=ComputeIntensity.CHAIN, mode=Mode.STATIC), 1)
    assert not served_chain(HarnessConfig(), 1)


def _decide_width_4_or_native(clock, period_ms):
    """Width 4 (1.05 tokens per request and round) against native decode at batch 64; ``period_ms(i)`` is the i-th
    period of either."""
    policy = _policy(native=True)
    policy._pair[64] = (4, 0)
    policy.profile = lambda: chain_rounds.AcceptProfile((0.05, 0.0, 0.0))
    requests = _reqs(*range(64))
    i = 0
    while policy.exploring(64):
        policy.round_start(64, policy.width(64), requests)
        clock.advance(period_ms(i))
        i += 1
    return policy.decided[64]


def test_speculation_beats_native_decode_only_beyond_the_noise_of_the_two_rates(clock):
    # Width 4 commits 5% more per period, but every other period of both takes twice as long: 32 periods each cannot
    # tell them apart, and native decode serves.
    assert _decide_width_4_or_native(clock, lambda i: 20 if i % 2 else 10) == 0
    # Without the noise the same 5% decides for speculation.
    assert _decide_width_4_or_native(clock, lambda i: 10) == 4


def _tree(tmp_path=None, key=None, writer=True, sync=lambda step: step, native_us=None):
    return ServedTreePolicy(
        16,
        torch.device("cpu"),
        max_batch_size=512,
        writer=writer,
        sync=sync,
        cache_path=None if tmp_path is None else str(tmp_path / "tree.json"),
        key=key,
        native_us=native_us,
    )


def _serve_tree(policy, clock, bs, spec_ms, native_ms, accept, rounds=10_000, seed_rounds=0, widths=(16, 7, 3)):
    """Serve ``bs`` requests until their batch key is decided: speculative rounds (the cost model's widths, cycling
    through ``widths``) commit ``accept`` tokens per request in ``spec_ms``, native decode one in ``native_ms``. The
    first ``seed_rounds`` speculative rounds seed the live costs. Returns the actions served (16: speculation)."""
    requests = _reqs(*range(bs))
    served = []
    for i in range(rounds):
        action = policy.width(bs)
        width = widths[i % len(widths)] if action else 0
        settled = not (action and i < seed_rounds)
        policy.round_start(bs, width, requests, settled=settled)
        if not policy.exploring(bs):
            return served
        served.append(action)
        if width:
            policy.accepted(width, torch.full((bs,), accept, dtype=torch.int32))
        clock.advance(spec_ms if action and settled else (100 * spec_ms if action else native_ms))
    raise AssertionError("no decision")


def test_a_tree_decodes_natively_where_its_rounds_commit_fewer_tokens_per_time(clock):
    policy = _tree()
    # 2.5 tokens per request in 30 ms (0.083 per ms) against native decode's 1 in 10 ms (0.1 per ms).
    served = _serve_tree(policy, clock, 256, spec_ms=30, native_ms=10, accept=2.5)
    first_native = served.index(0)
    assert set(served[:first_native]) == {16} and set(served[first_native:]) == {0}  # native decode last, sticky
    assert all(policy.decided[k] == 0 for k in (256, 384, 512)) and policy.exploring(192)


@pytest.mark.parametrize("stalled, spec_ms, winner", [(0, 30, 0), (16, 15, 16)])
def test_a_one_time_stall_does_not_decide_the_comparison(clock, stalled, spec_ms, winner):
    """A kernel compile or graph capture in one period (3 s against 10-30 ms rounds) is counted, not priced: the
    action that commits more tokens per time (2.5 in ``spec_ms`` against 1 in 10 ms) wins although it stalled."""
    policy, requests, rounds = _tree(), _reqs(*range(256)), {0: 0, 16: 0}
    while policy.exploring(256):
        action = policy.width(256)
        policy.round_start(256, action, requests)
        if action:
            policy.accepted(action, torch.full((256,), 2.5, dtype=torch.int32))
        rounds[action] += 1
        clock.advance(3000 if action == stalled and rounds[action] == 2 else spec_ms if action else 10)
    assert rounds[stalled] > 2 and policy.decided[256] == winner
    assert chain_rounds.STALL_FACTOR * 30 < 3000


def test_the_first_round_after_a_switch_of_action_is_counted_not_priced(clock):
    """Native decode's first round after speculation also finishes speculation's work (a prelaunched draft): its
    period (95 ms here, under the stall factor) is counted toward the comparison but not priced."""
    policy, requests, compared = _tree(), _reqs(*range(256)), []
    policy._compare = lambda key, pair: compared.append(policy._rates(key, pair)) or policy._decide(key, 0)
    native = 0
    while policy.exploring(256):
        action = policy.width(256)
        policy.round_start(256, action, requests)
        if action:
            policy.accepted(action, torch.full((256,), 2.5, dtype=torch.int32))
        native += not action
        clock.advance(30 if action else 95 if native == 1 else 10)
    (rates,) = compared
    assert rates[0][0] == pytest.approx(256 / 10_000) and native == chain_rounds.EXPLORE_ROUNDS + 1


def test_a_tree_keeps_speculating_where_it_wins_and_so_does_every_smaller_batch(clock):
    policy = _tree()
    _serve_tree(policy, clock, 64, spec_ms=15, native_ms=10, accept=2.5)
    assert policy.decided[64] == 16
    assert all(policy.decided[k] == 16 for k in (1, 2, 8, 16, 48)) and policy.exploring(96)
    rounds = ChainRounds(None, None, lambda batch: 0, policy)
    assert rounds.plain_width(1) == 16 and rounds.plain_width(64) == 16  # never native below a speculating key


def test_rounds_that_seed_the_live_costs_are_not_counted(clock):
    policy = _tree()
    # The seed rounds force widths the cost model would not choose (100x the settled period here): were they counted,
    # native decode would win.
    served = _serve_tree(policy, clock, 64, spec_ms=15, native_ms=10, accept=2.5, seed_rounds=40)
    assert policy.decided[64] == 16
    assert served[:40] == [16] * 40 and served.count(0) == chain_rounds.EXPLORE_ROUNDS


def test_speculation_far_ahead_of_native_decodes_estimate_never_decodes_natively(clock):
    # 4 tokens per request in 10 ms (0.4 per ms) against an estimated 1 in 10 ms: native decode would hold the batch.
    priced = []
    policy = _tree(native_us=lambda bs: priced.append(bs) or 10_000.0)
    served = _serve_tree(policy, clock, 8, spec_ms=10, native_ms=10, accept=4)
    assert 0 not in served and len(served) == chain_rounds.EXPLORE_ROUNDS
    assert set(priced) == {8} and all(policy.decided[k] == 16 for k in (1, 2, 8))


@pytest.mark.parametrize("native_us", [lambda bs: 10_000.0, lambda bs: None])
def test_speculation_near_native_decodes_estimate_or_without_one_measures_native_decode(clock, native_us):
    # 2 tokens per request in 15 ms against an estimated 1 in 10 ms (1.33x, inside the skip ratio), or no estimate.
    policy = _tree(native_us=native_us)
    served = _serve_tree(policy, clock, 8, spec_ms=15, native_ms=10, accept=2)
    assert served.count(0) == chain_rounds.EXPLORE_ROUNDS and policy.decided[8] == 16


def test_noisy_speculation_measures_native_decode_even_ahead_of_its_estimate(clock):
    # Periods of 1 and 49 ms: 1.6x the estimated 0.8 tokens per ms on average, but not beyond the noise.
    policy = _tree(native_us=lambda bs: 10_000.0)
    requests, served = _reqs(*range(8)), []
    while policy.exploring(8):
        action = policy.width(8)
        policy.round_start(8, action, requests)
        served.append(action)
        if action:
            policy.accepted(action, torch.full((8,), 4, dtype=torch.int32))
        clock.advance((1, 49)[len(served) % 2] if action else 10)
    assert 0 in served


def test_a_chain_without_a_native_estimate_never_checks_one(clock):
    policy = _policy(native=True)
    policy._check_native = lambda key, best: pytest.fail("checked native decode without an estimate")
    _serve(policy, clock, 64, {0: 9, 1: 10, 2: 12, 4: 15, 8: 30}, {1: 1, 2: 2, 4: 3, 8: 3})
    assert 64 in policy.decided


def test_other_ranks_take_rank_zeros_native_check_without_pricing(clock):
    received = []

    def sync(step):
        received.append(step)
        return ("decide", 16, None)

    policy = _tree(writer=False, sync=sync, native_us=lambda bs: pytest.fail("a non-writer rank priced native decode"))
    served = _serve_tree(policy, clock, 8, spec_ms=10, native_ms=10, accept=4)
    assert received == [None] and 0 not in served and policy.decided[8] == 16


def test_a_first_tree_start_serves_the_packaged_choices_of_its_setup():
    policy = ServedTreePolicy(
        16, torch.device("cpu"), max_batch_size=512, writer=True, sync=lambda step: step, packaged={8: 1, 64: 0}
    )
    assert all(policy.decided[k] == 16 for k in (1, 2, 8)) and all(policy.decided[k] == 0 for k in (64, 96, 512))
    assert policy.exploring(16) and policy.exploring(48)  # unmeasured keys are searched on the served rounds


def test_tree_decisions_survive_a_restart_on_the_same_setup(tmp_path, clock):
    key = {"gpu": "NVIDIA B300 SXM6 AC", "gpu_count": 4}
    _serve_tree(_tree(tmp_path, key=key), clock, 128, spec_ms=30, native_ms=10, accept=2.5)
    restarted = _tree(tmp_path, key=key, writer=False)
    assert restarted.width(130) == 0 and restarted.width(512) == 0 and restarted.exploring(64)
    assert _tree(tmp_path, key={**key, "gpu_count": 8}).decided == {}


def test_other_ranks_never_read_the_trees_tokens(clock):
    policy = _tree(writer=False, sync=lambda step: ("decide", 0, None))
    requests = _reqs(1)
    while policy.exploring(256):
        width = policy.width(256)
        policy.round_start(256, width, requests)
        policy.accepted(width, torch.ones(256, dtype=torch.int32))
        clock.advance(1)
    assert policy._tokens is None and policy.decided[256] == 0


def test_every_serve_mode_tree_chooses_native_decode_on_its_served_rounds():
    assert served_tree(HarnessConfig(), 4)
    assert served_tree(HarnessConfig(data_path="/cost"), 4)
    assert not served_tree(HarnessConfig(), 1)
    assert not served_tree(HarnessConfig(mode=Mode.STATIC), 4)
