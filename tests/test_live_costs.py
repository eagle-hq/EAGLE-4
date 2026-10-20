# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Live costs: served-round timings become the cost model's tables; keys bind costs to a setup."""

import json

import pytest

from eagle_harness.core.cost_model import CostModel, CostModelParams
from eagle_harness.core.limits import ComputeIntensity
from eagle_harness.core.live_costs import DRAFT_WIDTH, CostKey, LiveCosts, check_key, read_key, write_key

WIDTHS = {1: [64, 32, 16, 8, 1], 4: [32, 16, 8, 4, 1]}
KEY = CostKey("NVIDIA B300", 4, "qwen:abc", "eagle4:def", "sglang 0.5.23", "torch.bfloat16")


def _live(**kw) -> LiveCosts:
    return LiveCosts(widths=lambda bs: WIDTHS[bs], key=KEY, seed_samples=2, **kw)


def _seed(live: LiveCosts, bucket: int, verify_us, draft_us=170.0, context=500) -> None:
    for width, us in verify_us.items():
        for sample in (us * 1.1, us):  # the seed keeps the fastest sample
            live.record(bucket, width, context, sample)
            live.record(bucket, DRAFT_WIDTH, context, draft_us)


def test_seed_sweep_orders_width_one_first_then_widest_and_ends_when_sampled():
    live = _live()
    assert live.seed_width(1) == 1 and not live.ready(1)
    live.record(1, 1, 500, 4000.0)
    live.record(1, 1, 500, 4000.0)
    assert live.seed_width(1) == 64
    _seed(live, 1, {1: 4000.0, 64: 7200.0, 32: 5700.0, 16: 4800.0, 8: 4500.0})
    assert live.seed_width(1) is None and live.ready(1)
    assert not live.ready(4) and live.seed_width(4) == 1


def test_tables_hold_the_served_curve_and_fill_unmeasured_intervals():
    live = _live()
    _seed(live, 1, {1: 4000.0, 64: 7200.0, 32: 5700.0, 16: 4800.0, 8: 4500.0}, context=3000)
    tables = live.tables()
    table = tables[1]
    assert table.batch_size == 1 and len(table.intervals) == live.context_intervals
    target = table.intervals[0].target
    assert target.index == (0, 7, 15, 31, 63)
    assert target.value == (4000.0, 4500.0, 4800.0, 5700.0, 7200.0)
    assert target.ratio[0] == 1.0 and target.ratio[-1] == 1.8
    # Interval 2 (context 3000) was measured; every other interval takes the nearest measured one.
    assert table.intervals[live.interval(3000)].target.value == table.intervals[15].target.value
    assert table.intervals[0].draft.value == (170.0,) and table.model_ratio[0] == round(4000.0 / 170.0, 4)
    # The cost model consumes the live table like an offline one.
    cost = CostModel(tables=tables, params=CostModelParams(), intensity=ComputeIntensity.HIGH, node_top_k=8, tree_depth=12, verify_num=64)
    assert cost.verify_cost(1, 3000) is not None
    row = cost.draft_cost_row(1, 3000)
    assert row.model_cost_ratio == table.model_ratio[0]


def test_refinement_is_an_ema_after_the_seed():
    live = _live(alpha=0.5)
    live.record(1, 8, 100, 4000.0)
    live.record(1, 8, 100, 5000.0)  # still seeding: min wins
    cell = live.cells[(1, 8, 0)]
    assert cell.us == 4000.0 and cell.samples == 2
    live.record(1, 8, 100, 6000.0)
    assert cell.us == 5000.0 and cell.samples == 3


def test_set_tables_refreshes_the_cost_model_in_place():
    live = _live()
    cost = CostModel(tables={}, params=CostModelParams(), intensity=ComputeIntensity.HIGH, node_top_k=8, tree_depth=12, verify_num=64)
    assert cost.table(1) is None
    _seed(live, 4, {1: 4300.0, 32: 6000.0, 16: 5100.0, 8: 4700.0, 4: 4500.0})
    cost.set_tables(live.tables())
    assert cost.table(1).batch_size == 4 and cost.table(32).batch_size == 4


def test_cache_round_trips_and_rejects_another_setup(tmp_path):
    live = _live()
    _seed(live, 1, {1: 4000.0, 64: 7200.0, 32: 5700.0, 16: 4800.0, 8: 4500.0})
    path = str(tmp_path / "costs.json")
    live.save(path)
    again = _live()
    assert again.load(path) and again.cells == live.cells and again.ready(1)
    other = LiveCosts(widths=lambda bs: WIDTHS[bs], key=CostKey("NVIDIA H100", 4, "qwen:abc", "eagle4:def", "sglang 0.5.23", "torch.bfloat16"))
    assert not other.load(path) and not other.cells
    assert not _live().load(str(tmp_path / "missing.json"))


@pytest.mark.parametrize("broken", [None, [], {}, {"key": {}}, {"cells": [[1, 8, 0, 0.0, 3]]},
                                       {"cells": [[1, 8, 0, float("nan"), 3]]},
                                       {"cells": [[1, 8, 0, 4000.0, -1]]},
                                       {"cells": [[1, 8, 16, 4000.0, 3]]},
                                       {"cells": [[1, 8, 0, "bad", 3]]},
                                       {"cells": [[1, 8, 0, 4000.0]]}])
def test_invalid_cache_is_a_miss_without_partially_replacing_cells(tmp_path, broken):
    live = _live()
    _seed(live, 1, {1: 4000.0, 64: 7200.0, 32: 5700.0, 16: 4800.0, 8: 4500.0})
    before = live.cells.copy()
    data = live.to_dict() | broken if isinstance(broken, dict) and broken else broken
    path = tmp_path / "costs.json"
    path.write_text(json.dumps(data))
    assert not live.load(str(path))
    assert live.cells == before


def test_cache_reseeds_when_the_draft_step_shape_changes(tmp_path):
    live = _live(draft_shape=(4, 12))
    _seed(live, 1, {1: 4000.0, 64: 7200.0, 32: 5700.0, 16: 4800.0, 8: 4500.0})
    path = str(tmp_path / "costs.json")
    live.save(path)
    assert _live(draft_shape=(4, 12)).load(path)
    for shape in ((8, 12), (4, 6)):
        changed = _live(draft_shape=shape)
        assert not changed.load(path) and not changed.cells


@pytest.mark.parametrize("us", [float("nan"), float("inf"), -1.0, 0.0])
def test_nonfinite_or_nonpositive_timings_do_not_seed_a_bucket(us):
    live = _live()
    live.record(1, 8, 100, us)
    assert live.cells == {}


def test_offline_tables_are_refused_on_another_setup(tmp_path, caplog):
    data_path = str(tmp_path)
    with caplog.at_level("WARNING"):
        check_key(data_path, KEY)  # unkeyed: accepted with a warning
    assert "carry no key.json" in caplog.text
    write_key(data_path, KEY)
    assert read_key(data_path) == KEY
    check_key(data_path, KEY)
    with pytest.raises(ValueError, match="gpu_count"):
        check_key(data_path, CostKey("NVIDIA B300", 1, "qwen:abc", "eagle4:def", "sglang 0.5.23", "torch.bfloat16"))
    assert json.load(open(tmp_path / "key.json"))["gpu"] == "NVIDIA B300"


def test_every_calibration_reader_refuses_another_setup(tmp_path):
    """A directory measured on B300/TP4 is refused by an H200/TP8 server on each path that reads it, as the cost
    tables are: the served chain widths, the vLLM tree rounds, and the vLLM worker's re-read of both."""
    from types import SimpleNamespace

    from eagle_harness.core.chain_width import (
        chain_widths,
        load_tree_rounds,
        merge_tree_rounds,
        save_accept_profile,
        save_round_times,
    )
    from eagle_harness.vllm.config import check_data_key

    data_path = str(tmp_path)
    b300 = CostKey("NVIDIA B300", 4, "qwen:abc", "eagle4:def", "vllm 0.11.1", "torch.bfloat16")
    h200 = CostKey("NVIDIA H200", 8, "qwen:abc", "eagle4:def", "vllm 0.11.1", "torch.bfloat16")
    write_key(data_path, b300)
    save_accept_profile(data_path, 100, [90, 80, 70])
    rows = [{"batch_size": 1, "width": w, "round_us": us, "count": 10} for w, us in ((1, 900.0), (4, 1000.0))]
    save_round_times(data_path, rows)
    merge_tree_rounds(data_path, [{"batch_size": 64, "width": 16, "round_us": 48000.0, "accept": 4.0, "count": 10}])
    chain = SimpleNamespace(method_config={"intensity": "chain_intensity", "data_path": data_path})
    tree = SimpleNamespace(method_config={"mode": "static", "max_verify_width": 16, "data_path": data_path})
    assert chain_widths(data_path, 4, 8, key=b300) == [(1, 8, 4)]
    assert load_tree_rounds(data_path, b300) == {64: {16: (48000.0, 4.0)}}
    check_data_key(chain, b300)
    check_data_key(tree, b300)
    for read in (
        lambda: chain_widths(data_path, 4, 8, key=h200),
        lambda: load_tree_rounds(data_path, h200),
        lambda: check_data_key(chain, h200),
        lambda: check_data_key(tree, h200),
    ):
        with pytest.raises(ValueError, match="another setup.*'gpu_count': \\(4, 8\\)"):
            read()


def test_offline_tables_are_validated_and_keyed_at_load(tmp_path):
    """A table the cost model could not decide from fails at load with the directory, not in a decode round."""
    from eagle_harness.core.config import HarnessConfig, build_cost_model
    from eagle_harness.core.cost_table import cost_model_problem, load_cost_tables, table_problem

    live = _live()
    _seed(live, 1, {1: 4000.0, 64: 7200.0, 32: 5700.0, 16: 4800.0, 8: 4500.0})
    table = live.tables()[1]
    assert table_problem(table) is None and cost_model_problem(table) is None
    bad = _live()
    _seed(bad, 1, {1: 4000.0, 64: 7200.0, 32: 5700.0, 16: 4800.0, 8: 4500.0}, draft_us=2000.0)  # ratio 2
    assert table_problem(bad.tables()[1]) is None and "cost ratio" in cost_model_problem(bad.tables()[1])
    bad = _live()
    _seed(bad, 1, {1: 4000.0, 64: 7200.0, 32: 5700.0, 16: 4800.0, 8: 1000.0})  # width 8 cheaper than width 1
    assert "8 tokens" in cost_model_problem(bad.tables()[1])
    # On disk: a cost-table directory with a zero draft cost is refused at load, with the path.
    for model, values in (("llm", [4000.0, 4500.0]), ("slm", [0.0])):
        d = tmp_path / "cost" / "bs_1" / f"model_{model}"
        d.mkdir(parents=True)
        (d / "model_interval_0.json").write_text(json.dumps({
            "value": values, "ratio": [1.0] * len(values), "index": list(range(0, 8 * len(values) - 7, 7)),
            "array_query": [], "array_cache": [], "cache_length_start": 64, "cache_length_step": 64,
            "max_query_length_query": 8, "max_query_length_cache": 8}))
    with pytest.raises(RuntimeError, match="bs_1.*draft cost"):
        load_cost_tables(str(tmp_path / "cost"))
    (tmp_path / "cost/bs_1/model_slm/model_interval_0.json").write_text(json.dumps({
        "value": [3000.0], "ratio": [1.0], "index": [0], "array_query": [], "array_cache": [],
        "cache_length_start": 64, "cache_length_step": 64, "max_query_length_query": 8, "max_query_length_cache": 8}))
    assert list(load_cost_tables(str(tmp_path / "cost"))) == [1]  # a table; the depth stop can read it
    with pytest.raises(ValueError, match="bs_1.*cost ratio"):  # the cost model cannot decide from it
        build_cost_model(HarnessConfig(data_path=str(tmp_path)), node_top_k=8, tree_depth=12, verify_num=64, target_is_moe=False)
    write_key(str(tmp_path), KEY)
    with pytest.raises(ValueError, match="another setup"):
        load_cost_tables(str(tmp_path / "cost"), CostKey("H100", 4, "qwen:abc", "eagle4:def", "sglang 0.5.23", "torch.bfloat16"))
    assert load_cost_tables(str(tmp_path / "missing"), KEY) == {}
