# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The draft steps' layer input (:meth:`LlamaForCausalLMEagle4._layer_input`, :func:`stack_layer_input`): the two-launch
path gives the same bits as the embedding + two norms + cat it replaces."""

from types import SimpleNamespace

import pytest
import torch

from tests.triton_probe import needs_triton

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@needs_triton
@pytest.mark.parametrize("rows, hidden", [(4, 5120), (3, 1000)])
def test_stack_layer_input_puts_the_token_row_beside_the_hidden_row(rows, hidden):
    from eagle_harness.vllm.draft_rows import stack_layer_input

    gen = torch.Generator().manual_seed(rows)
    table = torch.randn(50, hidden, generator=gen).to(torch.bfloat16).to(DEV)
    ids = torch.randint(0, 50, (rows,), generator=gen).to(DEV)
    states = torch.randn(rows, 2 * hidden, generator=gen).to(torch.bfloat16).to(DEV)[:, :hidden]  # a strided view
    got = stack_layer_input(table, ids, states)
    assert got.shape == (rows, 2, hidden) and got.is_contiguous()
    assert torch.equal(got, torch.stack([table[ids], states], dim=1))


def _draft(hidden: int, vocab: int, *, rank_start: int = 0):
    """A ``LlamaForCausalLMEagle4`` shell with the parts ``_layer_input`` touches (TP 1)."""
    pytest.importorskip("vllm")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    from eagle_harness.vllm import models

    torch.manual_seed(0)
    with set_current_vllm_config(VllmConfig()):
        norms = [RMSNorm(hidden, eps=1e-6).to(DEV, torch.bfloat16) for _ in range(2)]
    for norm in norms:
        norm.weight.data.normal_(1.0, 0.1)
    layer = SimpleNamespace(input_layernorm=norms[0], hidden_norm=norms[1])
    weight = torch.randn(vocab, hidden, device=DEV, dtype=torch.bfloat16)
    shard = SimpleNamespace(num_added_elements=0, num_org_vocab_padding=0, org_vocab_start_index=rank_start)
    embed = SimpleNamespace(weight=weight, shard_indices=shard)
    model = SimpleNamespace(layers=[layer], embed_tokens=embed, embed_input_ids=lambda ids: weight[ids])
    draft = object.__new__(models.LlamaForCausalLMEagle4)
    draft.__dict__.update(
        model=model,
        tp_size=1,
        _max_rows=8,
        _full_embed=None,
        _input_norm_weights=None,
        _norm_before_residual=False,
        _modules={},
        _parameters={},
    )
    from eagle_harness.vllm import model_rows

    return model_rows, draft


def test_layer_input_falls_back_off_cuda_or_with_a_padded_vocab(monkeypatch):
    models, draft = _draft(64, 32, rank_start=7)
    monkeypatch.setattr(models, "get_tensor_model_parallel_rank", lambda: 0)
    states = torch.randn(4, 64, device=DEV, dtype=torch.bfloat16)
    assert not draft._fused_layer_input(draft.model.layers[0], states)
    assert draft._full_embed is None  # nothing was built
    out, residual = draft._layer_input(torch.tensor([1, 2, 3, 4], device=DEV), states)
    assert out.shape == (4, 128) and residual is states


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM's RMSNorm kernel needs a GPU")
@pytest.mark.parametrize("rows", [1, 4, 8])
def test_two_launch_layer_input_is_bit_identical(rows, monkeypatch):
    models, draft = _draft(5120, 1000)
    monkeypatch.setattr(models, "get_tensor_model_parallel_rank", lambda: 0)
    ids = torch.randint(0, 1000, (rows,), device=DEV)
    states = torch.randn(rows, 5120, device=DEV, dtype=torch.bfloat16)
    with models._kernel_ops():
        layer = draft.model.layers[0]
        want = torch.cat([layer.input_layernorm(draft.model.embed_input_ids(ids)), layer.hidden_norm(states)], dim=-1)
        got, residual = draft._layer_input(ids, states)
    assert draft._full_embed is not None  # the two-launch path ran
    assert residual is states and torch.equal(got, want)
