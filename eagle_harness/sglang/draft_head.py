# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Candidate selection for native EAGLE3 and low-rank EAGLE4 heads."""

import torch
from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def draft_topk(model, hidden: torch.Tensor, k: int):
    if hasattr(model, "candidate_topk"):
        return model.candidate_topk(hidden, k)
    output = model.logits_processor(None, hidden, model.lm_head, LogitsMetadata(forward_mode=ForwardMode.DECODE))
    probs, tokens = torch.topk(torch.softmax(output.next_token_logits, dim=-1), k, dim=-1)
    if model.hot_token_id is not None:
        model.hot_token_id = model.hot_token_id.to(tokens.device)
        tokens = model.hot_token_id[tokens]
    return probs, tokens
