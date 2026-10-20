# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The worker must retain the learned EAGLE3 head and map its tokens."""

import pytest

pytest.importorskip("sglang.srt.speculative.spec_registry")
from eagle_harness.sglang.algorithm import EagleHarnessSpecAlgo


def test_harness_uses_native_eagle3_head_initialization():
    from types import SimpleNamespace

    import torch
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    from eagle_harness.sglang.models.llama_eagle3 import Eagle3LlamaForCausalLM

    draft = object.__new__(Eagle3LlamaForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.config = SimpleNamespace(hidden_size=4)
    draft.model = torch.nn.Module()
    draft.model.embed_tokens = torch.nn.Embedding(8, 4)
    draft.model.fc = torch.nn.Linear(12, 4, bias=False)
    draft.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    draft.model.layers[0].hidden_norm = torch.nn.LayerNorm(4, bias=False)
    draft.lm_head = torch.nn.Linear(4, 3, bias=False)
    head = torch.arange(12, dtype=torch.float32).view(3, 4)
    fc = torch.arange(48, dtype=torch.float32).view(4, 12)
    draft.load_weights(
        [
            ("d2t", torch.tensor([1, 2, 5])),
            ("t2d", torch.zeros(8, dtype=torch.bool)),
            ("lm_head.weight", head),
            ("fc.weight", fc),
            ("midlayer.hidden_norm.weight", torch.full((4,), 2.0)),
        ]
    )
    draft.load_lm_head_from_target = False
    embed = torch.nn.Parameter(torch.ones(8, 4))
    target_head = torch.nn.Parameter(torch.ones(8, 4))
    worker = SimpleNamespace(
        speculative_algorithm=EagleHarnessSpecAlgo("EAGLE_HARNESS", False),
        draft_runner=SimpleNamespace(model=draft),
        target_worker=SimpleNamespace(model_runner=SimpleNamespace(model=torch.nn.Module())),
        _resolve_shared_embed_and_head=lambda: (embed, target_head),
        hot_token_id=None,
    )
    EagleDraftWorker.init_lm_head(worker)
    assert draft.model.embed_tokens.weight is embed
    torch.testing.assert_close(draft.lm_head.weight, head)
    torch.testing.assert_close(worker.hot_token_id, torch.tensor([1, 3, 7]))
    torch.testing.assert_close(draft.model.fc.weight, fc)
    torch.testing.assert_close(draft.model.layers[0].hidden_norm.weight, torch.full((4,), 2.0))
