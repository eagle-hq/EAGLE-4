# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Full deferred chain rounds with a weight-free candidate-head forward."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("tvm_ffi", reason="TokenSpeed round kernels require apache-tvm-ffi")

from tests.tokenspeed_round.chain_cost_records import records
from tests.tokenspeed_round.test_dynamic_gpu import _drafter

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from eagle_harness.tokenspeed.hooks import sampled_hook_problems

if sampled_hook_problems():
    pytest.skip("engine lacks recorded-proposal hooks", allow_module_level=True)

from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.flashinfer import FlashInferSamplingBackend
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

from eagle_harness.tokenspeed.models import CandidateRows
from eagle_harness.tokenspeed.tree_round.device_policy import DevicePolicy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _chain(capacity, scheduled):
    drafter = _drafter()
    del drafter._draft_depth
    device = drafter.device = torch.device("cuda")
    drafter.spec_num_steps, drafter.spec_num_tokens = 7, 8
    drafter.draft_reads_token_history = False
    drafter._depth_graphs, drafter._finalize_graphs = {}, {}
    drafter._sampled = {}
    drafter._sampling_backend = None
    drafter.token_to_kv_pool = None
    drafter.input_buffers = InputBuffers(capacity, capacity * 8, 0, device=device)
    slots = torch.arange(1, capacity + 1, device=device)
    drafter.input_buffers.req_pool_indices_buf.copy_(slots)
    drafter.input_buffers.state_write_req_pool_indices_buf.copy_(slots)
    drafter.input_buffers.seq_lens_buf.fill_(128)
    drafter.padded_gather_ids_offsets_buf = (
        torch.arange(capacity, device=device) * 8 - 1
    )
    drafter.attn_backend = SimpleNamespace(
        bind_tree_draft=lambda tree: None,
        publish_draft_step_locations=lambda **kw: None,
        override_num_extends=lambda n: contextlib.nullcontext(),
        begin_tree_lanes=lambda rows, live_rows: None,
    )
    head_logits = -torch.arange(4, dtype=torch.float32, device=device)
    head_ids = torch.arange(2, 6, device=device)

    def forward(input_ids, captured_hidden_states, **kw):
        rows = input_ids.numel()
        return CandidateRows(
            next_token_logits=head_logits.expand(rows, 4).contiguous(),
            candidate_ids=head_ids.expand(rows, 4).contiguous(),
            hidden_states=captured_hidden_states * 0.5 + 1,
        )

    drafter.draft_model_runner = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(hidden_size=8),
            logits_processor=SimpleNamespace(
                do_argmax=True, config=SimpleNamespace(vocab_size=32)
            ),
        ),
        model_config=SimpleNamespace(dtype=torch.float32),
        forward=forward,
    )
    drafter.round_candidates = lambda out: (
        out.next_token_logits.softmax(-1)[:, :1].clone(),
        out.candidate_ids[:, :1].clone(),
    )
    # The target/step-0 model forward is the only part replaced by this fixture.
    drafter._run_first_step = lambda bs, inputs, narrowing: (
        forward(
            input_ids=inputs.base_model_output[:bs],
            captured_hidden_states=torch.ones((bs, 8), device=device),
        ),
        None,
    )
    drafter.runtime_states = RuntimeStates(capacity, 32, 8, device)
    drafter.runtime_states.init_draft_trees(8)
    spec = SimpleNamespace(
        config=SimpleNamespace(topk=1, num_steps=7, num_nodes=8),
        chain_parent=torch.arange(-1, 7, dtype=torch.int32, device=device),
        draft_parent_buf=torch.zeros((capacity, 8), dtype=torch.int32, device=device),
    )
    policy = DevicePolicy(
        None if scheduled else records(),
        verify_num=8,
        node_top_k=1,
        tree_depth=7,
        compute_intensity="chain_intensity",
        is_moe_target=False,
        exec_mode=0,
        benchmark_config="",
        chain_width_table=((1, 4),) if scheduled else (),
        device=device,
    )
    drafter.bind_dynamic_tree(spec, policy)
    backend = FlashInferSamplingBackend(SamplingBackendConfig(
        synthetic_acceptance_length=None, max_bs=capacity, max_draft_tokens_per_req=8,
        max_req_pool_size=capacity + 1, vocab_size=32, device="cuda",
        enable_speculative_sampling=True, enable_output_logprobs=False,
        sampling_stream="batch", logprob_order="torch",
    ))
    drafter.bind_sampling_backend(backend)
    drafter.enable_sampled_trees(num_candidates=4, seed=17)
    return drafter, backend


@pytest.mark.parametrize("capacity", [64, 512])
@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("graphs", [False, True])
def test_sampled_chain_rounds_after_idle_warmup(capacity, scheduled, graphs):
    drafter, backend = _chain(capacity, scheduled)
    rows = capacity
    inputs = SimpleNamespace(
        num_extends=rows,
        accept_lengths=torch.zeros(rows, dtype=torch.int32, device="cuda"),
        base_model_output=torch.ones(rows, dtype=torch.int32, device="cuda"),
    )
    drafter.prepare_round_sampling([])
    for _ in range(3):
        drafter.draft(inputs)
    if graphs:
        drafter.capture_round_graphs(
            rows, lo_bs=0, stream=torch.cuda.Stream(), pool=None
        )
        decode = torch.cuda.CUDAGraph()
        with torch.cuda.graph(decode):
            drafter.draft(inputs)
    try:
        for seq, (live, sampled) in enumerate(
            ((3, True), (1, False), (33, True), (capacity, True)), 1
        ):
            params = [
                SamplingParams(
                    temperature=1.0 if sampled and i % 2 == 0 else 0.0, seed=97 + i
                )
                for i in range(live)
            ]
            for p in params:
                p.normalize(None)
            slots = list(range(1, live + 1))
            backend.prepare_step(
                [f"{seq}-{i}" for i in slots], slots, params, num_tokens_per_req=8
            )
            drafter.prepare_round_sampling(params)
            drafter.input_buffers.req_pool_indices_buf[live:].zero_()
            drafter.input_buffers.state_write_req_pool_indices_buf[live:].zero_()
            drafter.input_buffers.req_pool_indices_buf[:live].copy_(
                torch.arange(1, live + 1, device="cuda")
            )
            drafter.input_buffers.state_write_req_pool_indices_buf[:live].copy_(
                torch.arange(1, live + 1, device="cuda")
            )
            ctx = SimpleNamespace(
                bs=live,
                num_extends=0,
                global_num_tokens=None,
                global_bs=None,
                all_decode_or_idle=True,
            )
            if graphs:
                drafter.replay_decode_graph(decode, bs=live, graph_bs=rows)
            else:
                eager = SimpleNamespace(
                    num_extends=live,
                    accept_lengths=inputs.accept_lengths[:live],
                    base_model_output=inputs.base_model_output[:live],
                )
                drafter.draft(eager)
            drafter.finish_round(ctx, rows if graphs else None)
            drafter.complete_round()
            torch.cuda.synchronize()
            width = drafter.policy.verify_num.item()
            if seq == 1:
                assert width == (4 if scheduled else 2)
            expected = torch.arange(-1, width - 1, dtype=torch.int32, device="cuda")
            torch.testing.assert_close(
                drafter.runtime_states.future_parent_map[1 : live + 1, :width],
                expected.expand(live, width),
            )
            assert (
                drafter.policy.go_ready(drafter.policy.last_depth.item() + 1, seq)
                is False
            )
            if sampled:
                assert drafter.runtime_states.future_draft_q.slot[1, 1].item() == 0
    finally:
        drafter.release_round_graphs()


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("graph_phase", [False, True])
def test_forward_runner_warmup_policy_frontier(monkeypatch, scheduled, graph_phase):
    """Exercise real warmup buffers without loading target/draft weights."""
    from tokenspeed.runtime.execution import forward_step

    drafter, _backend = _chain(64, scheduled)
    inputs = SimpleNamespace(
        num_extends=0,
        accept_lengths=torch.ones(4, dtype=torch.int32, device="cuda"),
        base_model_output=torch.ones(
            4 * drafter.spec_num_tokens, dtype=torch.int32, device="cuda"
        ),
    )
    observed = []
    begin_round = drafter.policy.begin_round

    def inspect_begin_round(*args):
        signature = [(str(t.dtype), str(t.device)) for t in args]
        observed.append(signature)
        print("warmup policy operands (live_rows, frontier, sampled):", signature)
        return begin_round(*args)

    monkeypatch.setattr(drafter.policy, "begin_round", inspect_begin_round)
    monkeypatch.setattr(forward_step.dist, "barrier", lambda: None)

    def forward(*, bs, ctx, sampling_info):
        assert bs == 4 and sampling_info.valid_cache_lengths is not None
        inputs.base_ctx = ctx
        drafter.draft(inputs)

    runner = SimpleNamespace(
        _forward_func=forward,
        _prepare_request_token_history_graph_inputs=lambda **kwargs: None,
        _prepare_sampling_capture=lambda **kwargs: None,
        _init_capture_metadata=lambda bs: None,
        attn_backend=None,
        token_to_kv_pool=None,
        max_tokens_per_req=drafter.spec_num_tokens,
        drafter=drafter,
        dp_size=1,
        input_buffers=drafter.input_buffers,
        runtime_states=drafter.runtime_states,
        vocab_size=4,
        device="cuda",
        device_module=torch.cuda,
        eager_grammar_buffers=None,
        capturable_grammar=None,
        grammar_backend=None,
        sampling_backend=None,
    )
    drafter.prepare_round_sampling([])
    forward_step.ForwardStepRunner.warmup_decode_path(
        runner, (4,), graph_phase=graph_phase
    )
    device = str(torch.device("cuda", torch.cuda.current_device()))
    assert observed == [
        [("torch.int32", device), ("torch.int64", device), ("torch.bool", device)]
    ]
    assert drafter._round_seq == 0
    assert drafter.policy._round_seq.item() == 0
