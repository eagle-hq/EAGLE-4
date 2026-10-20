# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""vLLM plugin entry point (group ``vllm.general_plugins``, see pyproject.toml).

vLLM calls :func:`register` in every process (API server, engine core, workers)
before it resolves the speculative config. It registers the ``eagle_harness``
speculative method, the EAGLE-4 draft architecture and the tree attention backend (``CUSTOM``) through
vLLM's public registries; it patches nothing.

On a vLLM that lacks a hook (:func:`eagle_harness.vllm.adapter.missing_hooks`, including the speculative-method
registry of branch ``eagle-hooks``) it registers nothing and logs an error with the vLLM version, the missing hooks
and the fix (:mod:`eagle_harness.compat`); it never raises, so installing this package can never break a vLLM install.
"""

from __future__ import annotations

import logging

from eagle_harness.vllm.config import METHOD_NAME

logger = logging.getLogger(__name__)

SPECULATOR = "eagle_harness.vllm.speculator:HarnessSpeculator"
GDN_TREE_VERIFY = "eagle_harness.vllm.gdn_tree_verify:create_extension"
TREE_ATTENTION = "eagle_harness.vllm.tree_attention.TreeFlashAttentionBackend"
DRAFT_ARCHITECTURE = "LlamaForCausalLMEagle4"
DRAFT_MODEL = "eagle_harness.vllm.models:LlamaForCausalLMEagle4"


def register() -> bool:
    """Register the method and the draft model (idempotent). False if vLLM lacks the hooks."""
    from eagle_harness.compat import check_for_registration

    _log_through_vllm()
    if not check_for_registration("vllm"):
        return False

    from vllm.v1.spec_decode.registry import SpeculativeMethodSpec, register_speculative_method

    from eagle_harness.vllm.config import configure_speculative_config, num_lookahead_tokens

    register_speculative_method(
        METHOD_NAME,
        SpeculativeMethodSpec(
            speculator_cls=SPECULATOR,
            configure=configure_speculative_config,
            uses_target_hidden_states=True,
            uses_aux_hidden_states=True,
            num_lookahead_tokens=num_lookahead_tokens,
            supports_async_scheduling=True,  # a round is device-only: no host read between verify and the next drafts
            linear_attn_verify_extension=GDN_TREE_VERIFY,
        ),
    )

    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(AttentionBackendEnum.CUSTOM, TREE_ATTENTION)  # ``--attention-backend CUSTOM``

    from vllm.model_executor.models.registry import ModelRegistry

    if DRAFT_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(DRAFT_ARCHITECTURE, DRAFT_MODEL)
    if "Eagle3ForHarness" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Eagle3ForHarness", "eagle_harness.vllm.eagle3:Eagle3ForHarness")
    logger.info("eagle-harness: registered speculative method %s and draft %s", METHOD_NAME, DRAFT_ARCHITECTURE)
    return True


def _log_through_vllm() -> None:
    """The package's loggers write where vLLM's do (vLLM configures only its own ``vllm`` logger, which does not
    propagate, so ``eagle_harness.*`` records were dropped in vLLM processes)."""
    ours, theirs = logging.getLogger("eagle_harness"), logging.getLogger("vllm")
    if theirs.handlers and not ours.handlers:
        for handler in theirs.handlers:
            ours.addHandler(handler)
        ours.setLevel(theirs.level)
        ours.propagate = False
