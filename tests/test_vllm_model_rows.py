# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Release integration of Nemotron structures with the shared EAGLE3 row forward."""

import contextlib
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
import torch  # noqa: E402

from eagle_harness.vllm import model_rows, models  # noqa: E402


@pytest.mark.parametrize("draft_kind", ["native", "dense", "latent"])
@pytest.mark.parametrize("attn_norm", [False, True])
@pytest.mark.parametrize("norm_output", [False, True])
def test_shared_rows_preserve_attention_projection_and_mlp(monkeypatch, draft_kind, attn_norm, norm_output):
    monkeypatch.setattr(model_rows, "_kernel_ops", contextlib.nullcontext)
    monkeypatch.setattr(model_rows, "_kernel", lambda op: op)
    calls = []
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    residual = torch.tensor([[5.0, 6.0]])

    def qkv(value):
        # Both the 2H dense input and H projected input reach a real numeric assertion.
        value = value.reshape(1, -1, 2).sum(1)
        return torch.cat([value, value * 2, value * 3], dim=-1), None

    def dense(value):
        calls.append("dense")
        return value * 7, None

    def latent(mlp, value):
        calls.append("latent")
        return value * 11

    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            qkv_proj=qkv,
            q_size=2,
            kv_size=2,
            rotary_emb=lambda positions, q, k: (q, k),
            o_proj=lambda value: (value, None),
        ),
        post_attention_layernorm=lambda value, res: (value - res, value + res),
        mlp=SimpleNamespace(gate_up_proj=dense, act_fn=lambda value: value, down_proj=lambda value: (value, None)),
    )
    if attn_norm:
        layer.eh_proj = lambda value: value[:, :2] + value[:, 2:]
        layer.pre_attention_layernorm = lambda value: value * 5
    draft = SimpleNamespace(
        model=SimpleNamespace(layers=[layer], norm=lambda value, res: (value - res, value + res), norm_output=norm_output),
        _layer_input=lambda ids, hidden: (x, residual),
        _moe_rows=latent,
    )
    if draft_kind != "native":
        draft.latent_moe = draft_kind == "latent"
    normed, hidden = model_rows.DraftRowsMixin.forward_rows(
        draft, torch.tensor([0]), torch.tensor([0]), x, lambda q, k, v: q + k + v
    )
    projected = x[:, :2] + x[:, 2:]
    attention_input = projected * 5 if attn_norm else projected
    expected_residual = projected if attn_norm else residual
    attention = attention_input + attention_input * 2 + attention_input * 3
    mlp_output = (attention - expected_residual) * (11 if draft_kind == "latent" else 7)
    summed_residual = attention + expected_residual
    torch.testing.assert_close(normed, mlp_output - summed_residual)
    torch.testing.assert_close(hidden, normed if norm_output else mlp_output + summed_residual)
    assert calls == ["latent" if draft_kind == "latent" else "dense"]


@pytest.mark.parametrize("context_exists", [False, True])
def test_latent_rows_enter_engine_context_only_when_missing(monkeypatch, context_exists):
    events = []
    config = object()
    monkeypatch.setattr(models, "is_forward_context_available", lambda: context_exists)

    @contextlib.contextmanager
    def context(attention_metadata, vllm_config):
        assert attention_metadata is None and vllm_config is config
        events.append("enter")
        yield
        events.append("exit")

    monkeypatch.setattr(models, "set_forward_context", context)
    x = torch.ones(1)

    def mlp(value):
        events.append("mlp")
        return value * 2

    out = models.LlamaForCausalLMEagle4._moe_rows(SimpleNamespace(vllm_config=config), mlp, x)
    torch.testing.assert_close(out, x * 2)
    assert events == (["mlp"] if context_exists else ["enter", "mlp", "exit"])
