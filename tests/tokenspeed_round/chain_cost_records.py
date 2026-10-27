# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Structural chain records; values are intentionally distinct, not a cost model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StopTerms:
    gain_scale: float
    cost: float
    threshold: float
    cliff_gain_scale: float
    cliff_cost: float


@dataclass(frozen=True)
class BatchCosts:
    cache_length_step: int
    intervals: tuple[tuple[StopTerms, ...], ...]


@dataclass(frozen=True)
class PolicyCosts:
    batches: tuple[BatchCosts, ...]


def records(depth: int = 7) -> PolicyCosts:
    zero = StopTerms(0.0, 0.0, 0.0, 0.0, 0.0)
    return PolicyCosts(
        tuple(
            BatchCosts(
                512 if live % 2 else 1024,
                tuple(
                    tuple(
                        (
                            StopTerms(
                                1.123456789012345 + d,
                                0.25 + live,
                                1.5 + interval,
                                2.123456789012345 + d,
                                0.5 + live,
                            )
                            if d >= 2 and d % 2 == 0 and d < depth - 1
                            else zero
                        )
                        for d in range(depth)
                    )
                    for interval in range(1 + live % 2)
                ),
            )
            for live in range(1, 513)
        )
    )
