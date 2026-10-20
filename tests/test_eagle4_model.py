# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""EAGLE-4 draft model port: supported structures and checkpoint names (CPU, no weights)."""

from __future__ import annotations

import logging
import os
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("sglang.srt.models.llama")

import torch  # noqa: E402

from eagle_harness.core.config import harness_config_from_dict  # noqa: E402
from eagle_harness.sglang.models import llama_eagle4  # noqa: E402
from eagle_harness.sglang.models.llama_eagle4 import (  # noqa: E402
    Eagle4DecoderLayer,
    LlamaForCausalLMEagle4,
    _base_layer_config,
    _checkpoint_name,
    _NoRotary,
    build_latent_moe,
    check_supported,
    replicate_embedding,
)

transformers = pytest.importorskip("transformers")

# config.json of the Nemotron 3.5 Super EAGLE-4 drafts (sep27-r128-k128-exported; oct08-r128-k128-iter1001 is identical).
NEMOTRON_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "golden", "nemotron35_super_eagle4_draft_config.json")

# Tensor names of EAGLE4-Qwen3.8-27B-final-v3/model.safetensors.
CHECKPOINT = {
    "efficient_lm_head.weight": "efficient_lm_head.weight",
    "efficient_lm_head_down_proj.weight": "efficient_lm_head_down_proj.weight",
    "embed_tokens.weight": "model.embed_tokens.weight",
    "fc.weight": "model.fc.weight",
    "fc_norm.0.weight": "model.fc_norm.0.weight",
    "fc_norm.2.weight": "model.fc_norm.2.weight",
    "layers.0.hidden_norm.weight": "model.layers.0.hidden_norm.weight",
    "layers.0.input_layernorm.weight": "model.layers.0.input_layernorm.weight",
    "layers.0.mlp.down_proj.weight": "model.layers.0.mlp.down_proj.weight",
    "layers.0.mlp.gate_proj.weight": "model.layers.0.mlp.gate_proj.weight",
    "layers.0.post_attention_layernorm.weight": "model.layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
    "lm_head.weight": "lm_head.weight",
    "norm.weight": "model.norm.weight",
    # legacy spellings
    "midlayer.mlp.up_proj.weight": "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.self_attn.o_proj.weight": "model.layers.0.self_attn.o_proj.weight",
    "fc.linear.weight": "model.fc.weight",
    # Nemotron 3.5 Super drafts (attn_norm, latent MoE)
    "midlayer.eh_proj.weight": "model.layers.0.eh_proj.weight",
    "midlayer.pre_attention_layernorm.weight": "model.layers.0.pre_attention_layernorm.weight",
    "midlayer.mlp.experts.511.up_proj.weight": "model.layers.0.mlp.experts.511.up_proj.weight",
    "midlayer.mlp.gate.e_score_correction_bias": "model.layers.0.mlp.gate.e_score_correction_bias",
    "midlayer.mlp.fc1_latent_proj.weight": "model.layers.0.mlp.fc1_latent_proj.weight",
    "midlayer.mlp.shared_experts.down_proj.weight": "model.layers.0.mlp.shared_experts.down_proj.weight",
}


@pytest.mark.parametrize("name, param", sorted(CHECKPOINT.items()))
def test_checkpoint_names(name, param):
    assert _checkpoint_name(name) == param


def test_replicated_embedding_is_the_unpadded_table_of_the_gathered_shards():
    vocab, hidden, tp = 100, 8, 4
    table = torch.randn(104, hidden)  # four shards of 26 rows; the last 4 rows pad the vocab
    shards = table.chunk(tp)
    full = replicate_embedding(shards[2], vocab, lambda shard: torch.cat(shards, dim=0))
    assert full.shape == (vocab, hidden) and full.is_contiguous()
    assert torch.equal(full, table[:vocab])
    tokens = torch.tensor([0, 25, 26, 99])
    assert torch.equal(torch.nn.functional.embedding(tokens, full), table[tokens])


def test_the_draft_keeps_its_own_lm_head():
    """sglang's EAGLE draft setup hands a draft the target's embedding and LM head; EAGLE-4 takes only the embedding."""
    calls = []
    draft = type("Draft", (), {"set_embed": lambda self, embed: calls.append(embed), "lm_head": "own head"})()
    LlamaForCausalLMEagle4.set_embed_and_head(draft, "target embed", "target head")
    assert calls == ["target embed"]
    assert draft.lm_head == "own head"


# The structure keys of the Nemotron 3.5 Super EAGLE-4 drafts (oct08-r128-k128-iter1001, sep27-r128-k128-exported).
NEMOTRON_DRAFT = dict(
    architectures=["LlamaForCausalLMEagle4"],
    attn_norm=True,
    draft_vocab_size=1000,
    eagle_config={"eagle_aux_hidden_state_layer_ids": [7, 47, 78]},
    efficient_lm_head_config={"num_candidates": 128, "rank": 128, "type": "low_rank"},
    enable_moe=True,
    fc_norm=True,
    head_dim=64,
    hidden_act="relu2",
    hidden_size=256,
    intermediate_size=512,
    layer_types=["sliding_attention"],
    max_position_embeddings=1048576,
    moe_architecture="nemotron_latent_relu2",
    moe_intermediate_size=96,
    moe_latent_size=64,
    moe_shared_expert_intermediate_size=192,
    n_group=1,
    n_routed_experts=16,
    n_shared_experts=1,
    norm_output=True,
    norm_topk_prob=True,
    num_attention_heads=4,
    num_experts_per_tok=4,
    num_hidden_layers=1,
    num_key_value_heads=4,
    routed_scaling_factor=5.0,
    scoring_func="sigmoid",
    sliding_window=4096,
    target_hidden_size=256,
    topk_group=1,
    topk_method="noaux_tc",
    use_rope=False,
    vocab_size=1000,
)


def test_the_nemotron_draft_structures_are_supported():
    check_supported(transformers.LlamaConfig(**NEMOTRON_DRAFT))


@pytest.mark.parametrize(
    "extra",
    [
        {"moe_architecture": "sparse_swiglu", "hidden_act": "silu"},
        {"enable_post_mlp_norm": True, "norm_output": False},
        {"share_draft_embedding_from_target": False},
    ],
)
def test_structures_sglang_does_not_build_are_refused(extra):
    with pytest.raises(NotImplementedError):
        check_supported(transformers.LlamaConfig(**{**NEMOTRON_DRAFT, **extra}))


def test_the_base_layer_gets_a_silu_mlp_and_no_long_rope_table_for_a_nope_draft():
    config = transformers.LlamaConfig(**NEMOTRON_DRAFT)
    base = _base_layer_config(config)
    assert (base.hidden_act, base.max_position_embeddings) == ("silu", 1)
    assert (config.hidden_act, config.max_position_embeddings) == ("relu2", 1048576)
    roped = _base_layer_config(transformers.LlamaConfig(**{**NEMOTRON_DRAFT, "use_rope": True}))
    assert roped.max_position_embeddings == 1048576


def test_nope_leaves_q_and_k_unchanged():
    q, k = torch.randn(3, 8), torch.randn(3, 4)
    out_q, out_k = _NoRotary()(torch.arange(3), q, k)
    assert out_q is q and out_k is k


class _Param:
    def __init__(self):
        self.calls = []

    def weight_loader(self, param, weight, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_latent_moe_experts_load_into_the_ungated_expert_banks():
    params = {name: _Param() for name in ("m.mlp.experts.w13_weight", "m.mlp.experts.w2_weight", "m.mlp.gate.weight")}
    load = LlamaForCausalLMEagle4._load_latent_moe_weight
    w = torch.zeros(1)
    assert load(params, "m.mlp.experts.7.up_proj.weight", w) == "m.mlp.experts.w13_weight"
    assert load(params, "m.mlp.experts.7.down_proj.weight", w) == "m.mlp.experts.w2_weight"
    assert load(params, "m.mlp.gate.weight", w) == "m.mlp.gate.weight"
    assert params["m.mlp.experts.w13_weight"].calls == [
        (("m.mlp.experts.w13_weight",), dict(shard_id="w1", expert_id=7))
    ]
    assert params["m.mlp.experts.w2_weight"].calls == [(("m.mlp.experts.w2_weight",), dict(shard_id="w2", expert_id=7))]
    assert params["m.mlp.gate.weight"].calls == [((), {})]
    with pytest.raises(KeyError):
        load(params, "m.mlp.shared_experts.up_proj.weight", w)


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_latent_expert_is_rejected_even_when_another_is_repeated(duplicate):
    draft = object.__new__(LlamaForCausalLMEagle4)
    torch.nn.Module.__init__(draft)
    draft.config = transformers.LlamaConfig(**{**NEMOTRON_DRAFT, "n_routed_experts": 2})
    prefix = "model.layers.0.mlp.experts"
    params = {f"{prefix}.{bank}_weight": _Param() for bank in ("w13", "w2")}
    params.update({name: _Param() for name in ("efficient_lm_head.weight", "efficient_lm_head_down_proj.weight")})
    draft.named_parameters = lambda: params.items()
    names = [f"{prefix}.{expert}.{proj}_proj.weight" for expert in range(2) for proj in ("up", "down")]
    weights = [(name, torch.zeros(1)) for name in params if name.startswith("efficient_")]
    weights += [(name, torch.zeros(1)) for name in names[:-1]]
    if duplicate:
        weights.append((names[0], torch.zeros(1)))
    with pytest.raises(RuntimeError, match="missing"):
        draft.load_weights(weights)
    # The complete stream must load both expert banks and the candidate head.
    draft.load_weights(weights[:2] + [(name, torch.zeros(1)) for name in names])


def _layer(attn_norm: bool, reduce_mlp: bool):
    """An ``Eagle4DecoderLayer`` shell whose parts are affine maps, to follow the data through ``forward``."""
    layer = object.__new__(Eagle4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.attn_norm, layer.reduce_mlp = attn_norm, reduce_mlp
    layer.input_layernorm = lambda x: x + 1
    layer.hidden_norm = lambda x: x * 2
    layer.eh_proj = lambda x: (x[:, :2] + x[:, 2:], None)  # 2H -> H
    layer.pre_attention_layernorm = lambda x: x * 10
    # q/k/v of 2H or H inputs -> H outputs: the sum of the H-wide chunks, plus 100
    layer.self_attn = lambda positions, hidden_states, forward_batch: hidden_states.view(1, -1, 2).sum(1) + 100
    layer.post_attention_layernorm = lambda x, residual: (x - residual, x + residual)
    layer.mlp = lambda x: x * 3
    return layer


@pytest.mark.parametrize("attn_norm", [True, False])
def test_the_layer_attends_over_the_fused_input_and_reduces_a_moe_partial_sum(monkeypatch, attn_norm):
    embeds, hidden = torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])
    monkeypatch.setattr(llama_eagle4, "tensor_model_parallel_all_reduce", lambda x: x * 1000)
    fused = (embeds + 1) + hidden * 2  # eh_proj(cat(input_layernorm(embeds), hidden_norm(hidden)))
    # attn_norm: eh_proj's output is the residual and its norm feeds attention; else the cat feeds it, the raw
    # hidden is the residual.
    attn, res = (fused * 10 + 100, fused) if attn_norm else (fused + 100, hidden)
    for reduce_mlp, scale in ((False, 1), (True, 1000)):
        mlp, residual = _layer(attn_norm, reduce_mlp)(None, embeds, hidden, None)
        assert torch.equal(residual, attn + res)
        assert torch.equal(mlp, (attn - res) * 3 * scale)


# --- The real Nemotron draft config, the latent MoE block and its checkpoint experts ---


def nemotron_draft_config():
    return transformers.LlamaConfig.from_json_file(NEMOTRON_CONFIG_PATH)


def test_the_real_nemotron_draft_config_is_supported():
    config = nemotron_draft_config()
    check_supported(config)
    assert (config.n_routed_experts, config.num_experts_per_tok, config.moe_latent_size) == (512, 22, 1024)
    assert (config.attn_norm, config.use_rope, config.hidden_act, config.sliding_window) == (True, False, "relu2", 4096)


@pytest.mark.parametrize(
    "field", ["n_routed_experts", "num_experts_per_tok", "moe_intermediate_size", "moe_latent_size"]
)
def test_a_latent_moe_draft_missing_an_expert_field_is_refused_by_name(field):
    """The engine block reads these from the checkpoint; a config without them must fail here, not deep in sglang."""
    config = nemotron_draft_config()
    delattr(config, field)
    with pytest.raises(ValueError, match=field):
        check_supported(config)


class _FakeNemotronHMoE(torch.nn.Module):
    """Stands in for sglang's block, which needs a CUDA MoE runner: records what the port hands it."""

    def __init__(self, config, layer_idx, quant_config, prefix):
        super().__init__()
        self.config, self.layer_idx, self.quant_config, self.prefix = config, layer_idx, quant_config, prefix


def _latent_moe_host(monkeypatch, moe_gate_acc="fp32", ep_size=1, a2a_none=True):
    """One TP rank with no expert parallelism, the cleaned harness config and the fake engine block."""
    monkeypatch.setattr(
        llama_eagle4, "get_parallel", lambda: SimpleNamespace(tp_size=1, moe_tp_size=1, moe_ep_size=ep_size)
    )
    monkeypatch.setattr(llama_eagle4, "get_moe_a2a_backend", lambda: SimpleNamespace(is_none=lambda: a2a_none))
    options = {"mode": "static", "draft_model": {"moe_gate_acc": moe_gate_acc}}
    monkeypatch.setattr(llama_eagle4, "harness_config", lambda: harness_config_from_dict(options))
    module = types.ModuleType("sglang.srt.models.nemotron_h")
    module.NemotronHMoE = _FakeNemotronHMoE
    monkeypatch.setitem(sys.modules, "sglang.srt.models.nemotron_h", module)


def test_the_latent_moe_is_the_engine_block_over_the_checkpoint_experts(monkeypatch, caplog):
    """Expert count, top-k and widths come from the draft config; the harness has no MoE knob for them."""
    _latent_moe_host(monkeypatch)
    config = nemotron_draft_config()
    with caplog.at_level(logging.WARNING, logger=llama_eagle4.logger.name):
        moe = build_latent_moe(config, "model.layers.0.mlp")
    assert isinstance(moe, _FakeNemotronHMoE)
    assert (moe.layer_idx, moe.quant_config, moe.prefix) == (0, None, "model.layers.0.mlp")
    assert (moe.config.n_routed_experts, moe.config.num_experts_per_tok, moe.config.moe_latent_size) == (512, 22, 1024)
    assert moe.config.moe_intermediate_size == 2688
    assert (moe.config.n_shared_experts, moe.config.routed_scaling_factor) == (1, 5.0)
    assert (moe.config.mlp_hidden_act, moe.config.mlp_bias, moe.config.norm_topk_prob) == ("relu2", False, True)
    assert moe.config is not config and not hasattr(config, "mlp_hidden_act")
    assert caplog.text == ""


def test_a_non_fp32_gate_accumulation_is_kept_as_a_knob_but_ignored_with_a_warning(monkeypatch, caplog):
    _latent_moe_host(monkeypatch, moe_gate_acc="bf16")
    with caplog.at_level(logging.WARNING, logger=llama_eagle4.logger.name):
        build_latent_moe(nemotron_draft_config(), "m")
    assert "moe_gate_acc" in caplog.text and "latent MoE ignores it" in caplog.text


@pytest.mark.parametrize("host", [dict(ep_size=2), dict(a2a_none=False)])
def test_expert_parallel_layouts_are_refused_before_the_block_is_built(monkeypatch, host):
    _latent_moe_host(monkeypatch, **host)
    with pytest.raises(ValueError, match="moe EP 1"):
        build_latent_moe(nemotron_draft_config(), "m")


class _ExpertBank:
    """A FusedMoE expert bank as sglang loads an ungated expert (``w1`` fills the whole ``w13`` row)."""

    def __init__(self, shape):
        self.data = torch.zeros(shape, dtype=torch.float64)

    def weight_loader(self, param, weight, name, shard_id, expert_id):
        assert param is self and shard_id in ("w1", "w2")
        self.data[expert_id] = weight


def _reference_latent_moe(x, ckpt, top_k, scale):
    """The block as sglang's ``NemotronHMoE`` computes it, expert by expert.

    Sigmoid router; top-k over the bias-corrected scores; the unbiased scores, renormalized, weight the routed
    ReLU^2 experts in the latent space; ``routed_scaling_factor`` scales the routed sum before ``fc2_latent_proj``;
    a ReLU^2 shared expert is added after.
    """
    scores = torch.sigmoid(x @ ckpt["gate.weight"].T)
    chosen = torch.topk(scores + ckpt["gate.e_score_correction_bias"], top_k, dim=-1).indices
    weights = torch.gather(scores, -1, chosen)
    weights = weights / weights.sum(-1, keepdim=True)
    latent = x @ ckpt["fc1_latent_proj.weight"].T
    routed = torch.zeros_like(latent)
    for token in range(x.shape[0]):
        for k in range(top_k):
            expert = int(chosen[token, k])
            up = torch.relu(latent[token] @ ckpt[f"experts.{expert}.up_proj.weight"].T) ** 2
            routed[token] += weights[token, k] * (up @ ckpt[f"experts.{expert}.down_proj.weight"].T)
    shared = torch.relu(x @ ckpt["shared_experts.up_proj.weight"].T) ** 2 @ ckpt["shared_experts.down_proj.weight"].T
    return (scale * routed) @ ckpt["fc2_latent_proj.weight"].T + shared


def test_the_packed_expert_banks_compute_the_reference_latent_moe():
    """Tiny forward: the checkpoint's per-expert tensors, loaded through the port into packed ``w13``/``w2`` banks
    and applied as the fused ungated kernel does, equal the expert-by-expert reference."""
    experts, hidden, latent, inter, shared, top_k, scale = 6, 16, 8, 12, 20, 3, 5.0
    g = torch.Generator().manual_seed(0)
    rnd = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64)  # noqa: E731
    ckpt = {
        "gate.weight": rnd(experts, hidden),
        "gate.e_score_correction_bias": rnd(experts),
        "fc1_latent_proj.weight": rnd(latent, hidden),
        "fc2_latent_proj.weight": rnd(hidden, latent),
        "shared_experts.up_proj.weight": rnd(shared, hidden),
        "shared_experts.down_proj.weight": rnd(hidden, shared),
    }
    for e in range(experts):
        ckpt[f"experts.{e}.up_proj.weight"] = rnd(inter, latent)
        ckpt[f"experts.{e}.down_proj.weight"] = rnd(latent, inter)

    prefix = "model.layers.0.mlp"
    w13, w2 = _ExpertBank((experts, inter, latent)), _ExpertBank((experts, latent, inter))
    params = {f"{prefix}.experts.w13_weight": w13, f"{prefix}.experts.w2_weight": w2}
    load = LlamaForCausalLMEagle4._load_latent_moe_weight
    routed_tensors = {name: weight for name, weight in ckpt.items() if name.startswith("experts.")}
    assert len(routed_tensors) == 2 * experts
    loaded = {load(params, f"{prefix}.{name}", weight) for name, weight in routed_tensors.items()}
    assert loaded == set(params)

    x = rnd(5, hidden)
    scores = torch.sigmoid(x @ ckpt["gate.weight"].T)
    chosen = torch.topk(scores + ckpt["gate.e_score_correction_bias"], top_k, dim=-1).indices
    weights = torch.gather(scores, -1, chosen)
    weights = weights / weights.sum(-1, keepdim=True)
    latent_x = x @ ckpt["fc1_latent_proj.weight"].T
    shared_out = (
        torch.relu(x @ ckpt["shared_experts.up_proj.weight"].T) ** 2 @ ckpt["shared_experts.down_proj.weight"].T
    )

    def packed_forward(w13_bank, w2_bank):
        """The fused ungated kernel: every chosen expert's bank row on the token's latent input, weighted and summed."""
        up = torch.relu(torch.einsum("tl,tkil->tki", latent_x, w13_bank[chosen])) ** 2
        routed = (torch.einsum("tki,tkli->tkl", up, w2_bank[chosen]) * weights[..., None]).sum(1)
        return (scale * routed) @ ckpt["fc2_latent_proj.weight"].T + shared_out

    packed = packed_forward(w13.data, w2.data)
    torch.testing.assert_close(packed, _reference_latent_moe(x, ckpt, top_k, scale))
    # Shifting every expert's bank by one changes the function: the expert index mapping is load-bearing.
    assert not torch.allclose(packed_forward(w13.data.roll(1, 0), w2.data), packed)


def test_the_candidate_gather_skips_the_entry_barrier(monkeypatch):
    """Consecutive candidate gathers are separated by the draft's all-reduces, so the gatherer drops its entry
    barrier (sglang's ``skip_entry_sync``); the output stays the rank-major view of the last-dim gather."""
    from eagle_harness.sglang.models import llama_eagle4 as module

    built = []

    class Gatherer:
        def __init__(self, max_tokens, **kwargs):
            built.append((max_tokens, kwargs))

        def __call__(self, x):
            return torch.cat([x, x + 1], dim=-1)  # a two-rank gather along the last dim

    monkeypatch.setattr(module, "MultimemAllGatherer", Gatherer)
    monkeypatch.setattr(module, "recommended_max_tokens", lambda **kwargs: 128)
    draft = type("Draft", (), {"_candidate_gatherer": None, "_tp_size": 2})()
    packed = torch.arange(6, dtype=torch.int32).view(2, 3)
    out = LlamaForCausalLMEagle4._gather_packed_candidates(draft, packed)
    assert built == [(128, {"skip_entry_sync": True})]
    want = torch.stack([packed, (packed.view(torch.bfloat16) + 1).view(torch.int32)])
    assert out.shape == (2, 2, 3) and torch.equal(out, want)
    assert LlamaForCausalLMEagle4._gather_packed_candidates(draft, packed).shape == (2, 2, 3) and len(built) == 1
