# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Host-only structural records for packing tests, with no policy implementation."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class IntervalCosts:
    verify_widths: tuple[int, ...]
    verify_cost_deltas: tuple[tuple[float, ...], ...]
    draft_cost: float
    model_cost_ratio: float
    depth_thresholds: tuple[tuple[float, ...], ...]
    verify_tier: int
    verify_thresholds: tuple[float, float]


@dataclass(frozen=True)
class BatchCosts:
    cost_batch_size: int
    cache_length_step: int
    intervals: tuple[IntervalCosts, ...]


@dataclass(frozen=True)
class PolicyCosts:
    batches: tuple[BatchCosts, ...]
    history_floors: tuple[int, ...]
    initial_widths: tuple[int, ...]
    async_layer_ratio: float


def records() -> PolicyCosts:
    first = IntervalCosts(
        verify_widths=(0, 3),
        # A value below the producer's usual cutoff must be packed unchanged.
        verify_cost_deltas=((0.0, 0.01234567890123456), (0.0, 0.0)),
        draft_cost=1.2345678901234567,
        model_cost_ratio=11.123456789012345,
        depth_thresholds=tuple((0.125 + tier, 0.75 + tier) for tier in range(5)),
        verify_tier=0,
        verify_thresholds=(0.9123456789012345, 1.1234567890123456),
    )
    second = replace(
        first,
        verify_widths=(0, 1, 7),
        verify_cost_deltas=((0.0, 0.5, 0.75), (0.0, 0.0, 0.25), (0.0, 0.0, 0.0)),
    )
    # First-use IDs are 7 then 2; repeated measurements have distinct live decisions.
    batches = tuple(
        (
            BatchCosts(
                7,
                512,
                (
                    replace(
                        first, verify_tier=i % 4, verify_thresholds=(0.5 + i, 0.75 + i)
                    ),
                ),
            )
            if i % 2 == 0
            else BatchCosts(2, 1024, (second, first))
        )
        for i in range(32)
    )
    return PolicyCosts(batches, (16, 8, 4, 2, 32), (2, 2, 2, 2), 0.73)
