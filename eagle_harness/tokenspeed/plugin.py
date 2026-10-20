# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""TokenSpeed plugin entry point (group ``tokenspeed.plugins``, see pyproject.toml).

TokenSpeed calls :func:`register` once per process (frontend, scheduler, workers) before it resolves a model config.
It registers the ``EAGLE_HARNESS`` drafter (:class:`~eagle_harness.tokenspeed.drafter.HarnessDrafter`, reading the
target's aux hidden states) and the EAGLE-4 draft architecture through TokenSpeed's plugin registries; it patches
nothing.

On a TokenSpeed that lacks a hook (:func:`eagle_harness.tokenspeed.hooks.hook_problems`) it registers nothing and logs
an error with the TokenSpeed version, the missing hooks and the fix (:mod:`eagle_harness.compat`); it never raises, so
installing this package can never break a TokenSpeed install (``--speculative-algorithm EAGLE_HARNESS`` is then
unknown).
"""

from __future__ import annotations

import logging

from eagle_harness.models.eagle4_config import EAGLE4_ARCH
from eagle_harness.tokenspeed.config import ALGORITHM_NAME

logger = logging.getLogger(__name__)


def register() -> bool:
    """Register the drafter and the draft model. False, registering nothing, if TokenSpeed lacks the hooks."""
    from eagle_harness.compat import check_for_registration

    if not check_for_registration("tokenspeed"):
        return False

    from tokenspeed.runtime.plugins.registry import register_drafter, register_model

    from eagle_harness.tokenspeed.drafter import HarnessDrafter
    from eagle_harness.tokenspeed.models import LlamaForCausalLMEagle4

    register_drafter(ALGORITHM_NAME, HarnessDrafter, captures_aux_hidden_states=True, uses_tree_layout=True)
    register_model(LlamaForCausalLMEagle4, architectures=(EAGLE4_ARCH,))
    logger.info("eagle-harness: registered drafter %s and draft %s", ALGORITHM_NAME, EAGLE4_ARCH)
    return True
