# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""vLLM plugin: the EAGLE-Harness speculative method on vLLM's Model Runner V2.

Registered through the ``vllm.general_plugins`` entry point (see pyproject.toml)
and the generic hooks of vLLM branch ``eagle-hooks``; nothing in vLLM is patched.

Modules (``cpu`` = tested without a GPU, ``gpu`` = tested on a GPU):

- ``plugin``          cpu  entry point: registers the method and the EAGLE-4 draft architecture
- ``config``          cpu  ``method_config`` parsing, the batch-size verify schedule, lookahead, aux layers
- ``adapter``         cpu  which vLLM hook carries which EngineAdapter need (checked against vLLM)
- ``models``          gpu  the EAGLE-4 draft model for vLLM (candidate head, rows with caller-supplied attention)
- ``speculator``      gpu  ``HarnessSpeculator``: chain mode on vLLM's EAGLE loop, tree mode on the verify hooks
- ``tree_drafter``    gpu  the draft tree's growth and verify-row selection (one CUDA graph per batch size)
- ``verify_layout``   gpu  the verify tree in graph-stable buffers (depth positions, ancestor masks)
- ``tree_attention``  gpu  ``CUSTOM`` attention backend: FlashAttention + tree part, merged by log-sum-exp
- ``draft_attention`` gpu  per-row key-list attention of the draft tree's steps
- ``tree_accept``     gpu  tree walk as the speculator's rejection sampler
- ``round_commit``    gpu  KV compaction of the accepted rows, draft inputs in path order (``chainize``)
- ``gdn_tree_verify`` gpu  GDN tree verify extension (tree conv, the shared recurrence of ``kernels.gdn_tree_recurrence``,
  linear-state commit)
"""
