# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Adapter admission and engine-owned Nemotron draft structures."""

import pytest
import torch
from transformers import LlamaConfig

try:
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.utils.env import global_server_args_dict
    from eagle_harness.tokenspeed.models import LlamaForCausalLMEagle4
except ImportError as exc:
    pytest.skip(f"TokenSpeed runtime unavailable: {exc}", allow_module_level=True)
except RuntimeError as exc:
    if torch.cuda.is_available():
        raise
    pytest.skip(f"TokenSpeed needs GPU platform discovery: {exc}", allow_module_level=True)


@pytest.fixture
def config():
    return LlamaConfig(
        architectures=["LlamaForCausalLMEagle4"],
        hidden_size=128, vocab_size=512, draft_vocab_size=512,
        intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, head_dim=32,
        max_position_embeddings=512, attn_norm=True, use_rope=False,
        layer_types=["sliding_attention"], sliding_window=64,
        enable_moe=True, moe_architecture="nemotron_latent_relu2",
        hidden_act="relu2", n_routed_experts=8, num_experts_per_tok=2,
        moe_latent_size=64, moe_intermediate_size=128,
        n_shared_experts=1, moe_shared_expert_intermediate_size=256,
        n_group=1, topk_group=1, norm_topk_prob=True,
        routed_scaling_factor=5.0, scoring_func="sigmoid", topk_method="noaux_tc",
        moe_router_dtype="float32", norm_output=True, fc_norm=True,
        efficient_lm_head_config={"type": "low_rank", "rank": 16, "num_candidates": 32},
    )


@pytest.fixture
def mapping():
    original = global_server_args_dict.copy()
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    mapping = Mapping(rank=0, world_size=1)
    global_server_args_dict.update(mapping=mapping, max_model_len=512)
    yield mapping
    torch.set_default_dtype(original_dtype)
    global_server_args_dict.clear()
    global_server_args_dict.update(original)


def test_latent_draft_retains_the_adapter_candidate_head(config, mapping):
    with torch.device("cpu"):
        model = LlamaForCausalLMEagle4(config, mapping)
    block = model.model.midlayer
    assert block.self_attn.qkv_proj.weight.shape[1] == config.hidden_size
    assert block.self_attn.rotary_emb is None
    assert block.eh_proj.weight.shape == (128, 256)
    assert block.mlp.__class__.__module__ == "tokenspeed.runtime.models.nemotron_h"
    assert model.efficient_lm_head_down_proj.weight.shape == (16, 128)
    assert model.efficient_lm_head.weight.shape == (512, 16)
    assert model.num_candidates == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 8, 16])
def test_normalized_attention_input_on_gpu(config, mapping, rows):
    generator = torch.Generator(device="cuda").manual_seed(781)
    with torch.device("cuda"), torch.inference_mode():
        model = LlamaForCausalLMEagle4(config, mapping)
        block = model.model.midlayer
        for parameter in block.parameters():
            parameter.normal_(mean=0, std=0.1, generator=generator)
        embeds = torch.randn(rows, config.hidden_size, generator=generator)
        hidden = torch.randn(rows, config.hidden_size, generator=generator)

        def norm(x, weight):
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + config.rms_norm_eps)
            return (normalized * weight.float()).to(x.dtype)

        joined = torch.cat((norm(embeds, block.input_layernorm.weight), norm(hidden, block.hidden_norm.weight)), dim=-1)
        residual = torch.nn.functional.linear(joined, block.eh_proj.weight)
        expected = norm(residual, block.pre_attention_layernorm.weight)
        actual, actual_residual = block.prepare_attention_input(embeds, hidden, fuse_embed_reduce=False)
        torch.testing.assert_close(actual_residual, residual, atol=0.002, rtol=0.02)
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)


def test_extended_draft_refuses_an_older_engine(config, mapping, monkeypatch):
    from eagle_harness.tokenspeed.models import Eagle3DecoderLayer

    monkeypatch.delattr(Eagle3DecoderLayer, "prepare_attention_input")
    with pytest.raises(RuntimeError, match="requires TokenSpeed"):
        LlamaForCausalLMEagle4(config, mapping)


def test_post_mlp_norm_is_still_refused(config, mapping):
    config.enable_post_mlp_norm = True
    with pytest.raises(NotImplementedError, match="enable_post_mlp_norm"):
        LlamaForCausalLMEagle4(config, mapping)


def test_sparse_swiglu_is_not_silently_loaded_as_latent_moe(config, mapping):
    config.moe_architecture = "sparse_swiglu"
    config.hidden_act = "silu"
    with pytest.raises((ValueError, NotImplementedError), match="nemotron_latent_relu2|moe_architecture"):
        with torch.device("cpu"):
            LlamaForCausalLMEagle4(config, mapping)
