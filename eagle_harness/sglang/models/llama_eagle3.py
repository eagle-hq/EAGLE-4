# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Public vLLM-style architecture spelling for SGLang's native EAGLE3 loader."""
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3


class Eagle3LlamaForCausalLM(LlamaForCausalLMEagle3):
    pass


EntryClass = [Eagle3LlamaForCausalLM]
