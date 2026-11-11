# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Find the draft-round megakernel: ``$EAGLE_HARNESS_DRAFTMK_LIB`` if set, else the ``eagle-harness-draftmk`` wheel's
library and cubins (``$DRAFTMK_CUBIN_DIR``: beside an explicit library unless set, always the wheel's own).

:func:`library` returns the library's path, or None (logged once) where neither is there or the GPU is not one its
cubins are built for; the draft rounds then run on the harness's own graphs (:mod:`eagle_harness.sglang.draft_round`).
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Optional

from eagle_harness.kernels import arch

logger = logging.getLogger(__name__)


def _explicit() -> str:
    """Its cubins beside it unless ``$DRAFTMK_CUBIN_DIR`` names theirs."""
    path = Path(os.environ["EAGLE_HARNESS_DRAFTMK_LIB"])
    if not path.is_file():
        raise RuntimeError(f"$EAGLE_HARNESS_DRAFTMK_LIB={path}: no such file")
    os.environ.setdefault("DRAFTMK_CUBIN_DIR", str(path.parent))
    return str(path)


def _wheel() -> str:
    try:
        import eagle_harness_draftmk
    except ImportError as e:
        raise RuntimeError("eagle-harness-draftmk is not installed") from e
    os.environ["DRAFTMK_CUBIN_DIR"] = str(eagle_harness_draftmk.CUBINS)  # its own build's, never another's
    return str(eagle_harness_draftmk.LIBRARY)


@functools.lru_cache(maxsize=None)
def library() -> Optional[str]:
    """The library's path, or None where the draft rounds run on the graphs."""
    if not arch.draftmk():
        logger.info("eagle-harness: no draft-round megakernel for this GPU; draft rounds run on the graphs")
        return None
    load = _explicit if "EAGLE_HARNESS_DRAFTMK_LIB" in os.environ else _wheel
    try:
        path = load()
    except RuntimeError as e:
        logger.warning("eagle-harness: draft-round megakernel unavailable (%s); draft rounds run on the graphs", e)
        return None
    logger.info("eagle-harness: draft-round megakernel from %s", path)
    return path
