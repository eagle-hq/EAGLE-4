# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""``--speculative-algorithm-config`` of the ``EAGLE_HARNESS`` algorithm on TokenSpeed.

TokenSpeed hands the parsed JSON object to the drafter (``BaseDrafter.configure_algorithm``, hook H1); the keys are
:class:`~eagle_harness.core.config.HarnessConfig`'s, with sglang's strict parsing. The tree shape is TokenSpeed's own:
``--speculative-eagle-topk`` (children per node), ``--speculative-num-steps`` (tree depth) and
``--speculative-num-draft-tokens`` (verify width, root included).

Serve mode, the default, uses the calibrated GPU-side policy for draft depth and the number of meaningful verify
nodes; the physical verify window stays fixed and unused slots are unconnected roots. It needs ``data_path`` unless a
complete chain schedule sets every width. Static mode (``{"mode": "static"}``) keeps a fixed tree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from eagle_harness.core.config import HarnessConfig, Mode, harness_config_from_dict
from eagle_harness.core.limits import ComputeIntensity

ALGORITHM_NAME = "EAGLE_HARNESS"


@dataclass(frozen=True)
class TokenSpeedConfig(HarnessConfig):
    """Core policy settings plus an optional explicit per-batch chain schedule."""

    chain_width_table: tuple[tuple[int, int], ...] = ()
    sampled_proposals: bool = False

    @property
    def fully_scheduled(self) -> bool:
        """A chain schedule from batch size 1 up: it sets every verify width, so serve mode needs no cost data."""
        return bool(self.chain_width_table) and self.chain_width_table[0][0] == 1


def parse_algorithm_config(algorithm_config: Mapping[str, Any] | None) -> TokenSpeedConfig:
    """The TokenSpeedConfig of a launch (None: no ``--speculative-algorithm-config``, read as ``{}``). Serve is the
    default mode, as in :class:`HarnessConfig`; unknown keys and modes TokenSpeed cannot serve raise."""
    values = dict(algorithm_config or {})
    sampled = values.pop("sampled_proposals", False)
    if type(sampled) is not bool:
        raise ValueError("sampled_proposals must be a boolean")  # noqa: TRY004 — configuration validation
    table = values.pop("chain_width_table", ())
    if not isinstance(table, (list, tuple)):
        raise ValueError("chain_width_table must contain [minimum batch size, verify width] pairs")  # noqa: TRY004 — configuration validation
    entries = []
    for entry in table:
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 2
            or any(type(value) is not int or value < 1 for value in entry)
            or entry[1] < 2
            or (entries and entry[0] <= entries[-1][0])
        ):
            raise ValueError("chain_width_table needs increasing positive batch sizes and verify widths >= 2")
        entries.append(tuple(entry))
    core = harness_config_from_dict(values)
    config = TokenSpeedConfig(**{field.name: getattr(core, field.name) for field in fields(core)},
                              chain_width_table=tuple(entries), sampled_proposals=sampled)
    if sampled and config.mode is not Mode.SERVE:
        raise ValueError("sampled_proposals requires serve mode")
    if config.mode not in (Mode.STATIC, Mode.SERVE, Mode.CALIBRATE):
        raise NotImplementedError(f"{ALGORITHM_NAME} on TokenSpeed supports static, serve and calibrate modes")
    if config.mode is Mode.CALIBRATE and not config.data_path:
        raise ValueError("TokenSpeed calibrate mode needs data_path")
    if entries and (config.intensity is not ComputeIntensity.CHAIN or config.mode is not Mode.SERVE):
        raise ValueError("chain_width_table requires serve mode and chain_intensity")
    if config.mode is Mode.SERVE and not config.fully_scheduled and not config.data_path:
        raise ValueError(
            "TokenSpeed serve mode (the default) needs data_path or a complete explicit chain schedule "
            '(chain_intensity with a chain_width_table from batch size 1); live costs are unsupported. '
            'Pass {"mode": "static"} for a fixed tree.'
        )
    return config
