# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""sglang plugin entry point (group ``sglang.srt.plugins``, see pyproject.toml).

``sglang.srt.plugins.load_plugins()`` calls :func:`register` once per process,
before server arguments are resolved. It registers the EAGLE_HARNESS algorithm
and the EAGLE-4 draft model class; it patches nothing. On an sglang whose hooks
differ from :data:`eagle_harness.sglang.hooks.HOOKS` it registers nothing and
logs an error with the sglang version, the mismatches and the fix
(:mod:`eagle_harness.compat`), so ``--speculative-algorithm EAGLE_HARNESS`` is
then unknown; sglang itself keeps working.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


MODEL_PACKAGE = "eagle_harness.sglang.models"
DRAFT_ARCHITECTURE = "LlamaForCausalLMEagle4"


def register() -> bool:
    """Register the EAGLE_HARNESS speculative algorithm and the draft models (idempotent).

    Returns False, registering nothing, when the installed sglang lacks a hook the plugin needs.
    """
    from eagle_harness.compat import check_for_registration

    if not check_for_registration("sglang"):
        return False

    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    from eagle_harness.sglang.algorithm import ALGORITHM_NAME, EagleHarnessSpecAlgo

    try:
        SpeculativeAlgorithm.from_string(ALGORITHM_NAME)
        return True  # already registered in this process
    except ValueError:
        pass

    if DRAFT_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register(MODEL_PACKAGE, strict=True)

    @SpeculativeAlgorithm.register(ALGORITHM_NAME, supports_overlap=True, spec_class=EagleHarnessSpecAlgo)
    def _worker_class(server_args):
        from eagle_harness.sglang.worker import EagleHarnessWorker

        return EagleHarnessWorker

    logger.info("eagle-harness: registered speculative algorithm %s and models from %s", ALGORITHM_NAME, MODEL_PACKAGE)
    return True
