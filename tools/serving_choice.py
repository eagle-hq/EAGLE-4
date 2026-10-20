# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Choose an engine configuration before startup from measured served-round costs.

A zero-draft speculative engine is a separate candidate from a true no-spec
engine: it still owns draft KV and speculative recurrent state. This policy
never translates a no-spec choice into an in-engine width-one schedule.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


def serving_choice(no_spec_us: float, candidates: Mapping[str, tuple[float, float]]) -> str | None:
    """Return the fastest candidate, or None for the independently timed no-spec engine.

    Each candidate is ``(expected tokens per request per round, round microseconds)``
    at the same batch size and workload. Only measured candidates are considered;
    a tie keeps no speculation. This is a startup choice, not live request migration.
    """
    if not math.isfinite(no_spec_us) or no_spec_us <= 0:
        raise ValueError("no-spec round time must be positive and finite")
    best, best_rate = None, 1.0 / no_spec_us
    for name, (tokens, us) in sorted(candidates.items()):
        if not math.isfinite(tokens) or tokens < 1 or not math.isfinite(us) or us <= 0:
            raise ValueError(f"invalid served cost for {name}: tokens={tokens}, round_us={us}")
        rate = tokens / us
        if rate > best_rate:
            best, best_rate = name, rate
    return best
