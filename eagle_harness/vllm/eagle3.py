# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Native vLLM EAGLE3 loading and logits, with harness row forwards."""

import torch
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

from eagle_harness.vllm.model_rows import DraftRowsMixin


class Eagle3ForHarness(DraftRowsMixin, Eagle3LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._init_row_forward(8 * vllm_config.scheduler_config.max_num_seqs)

    def candidate_topk(self, hidden: torch.Tensor, k: int):
        # Native compute_logits scatters d2t offsets into the target vocabulary.
        return torch.topk(torch.softmax(self.compute_logits(hidden), dim=-1), k, dim=-1)
