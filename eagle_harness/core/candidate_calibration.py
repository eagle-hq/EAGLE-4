# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Load an offline-measured candidate selection crossover, never time at startup."""

import json
from pathlib import Path


def load_candidate_crossover(data_path, identity):
    """Return a measured score-count limit for an exact environment, or None.

    Calibration identity includes device, torch/Triton versions, dtype, vocabulary
    shard and candidate count. A missing or different profile retains the normal
    dispatcher; malformed matching data fails rather than selecting arbitrary code.
    """
    if data_path is None:
        return None
    path = Path(data_path) / "candidate_topk.json"
    if not path.is_file():
        return None
    profile = json.loads(path.read_text())
    if profile.get("identity") != identity:
        return None
    limit = profile["max_triton_elements"]
    if type(limit) is not int or limit < 0:
        raise ValueError(f"invalid candidate top-k crossover in {path}: {limit!r}")
    return limit
