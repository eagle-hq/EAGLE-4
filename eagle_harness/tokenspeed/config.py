# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""``--speculative-algorithm-config`` of the ``EAGLE_HARNESS`` algorithm on TokenSpeed.

TokenSpeed hands the parsed JSON object to the drafter (``BaseDrafter.configure_algorithm``, hook H1); the keys are
:class:`~eagle_harness.core.config.HarnessConfig`'s, with sglang's strict parsing. The tree shape is TokenSpeed's own:
``--speculative-eagle-topk`` (children per node), ``--speculative-num-steps`` (tree depth) and
``--speculative-num-draft-tokens`` (verify width, root included).

Static mode keeps a fixed tree. Serve mode uses the calibrated GPU-side policy
for draft depth and the number of meaningful verify nodes; the physical verify
window stays fixed and unused slots are unconnected roots.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

from eagle_harness.core.config import HarnessConfig, Mode, harness_config_from_dict

ALGORITHM_NAME = "EAGLE_HARNESS"
@dataclass(frozen=True)
class TokenSpeedConfig(HarnessConfig):
    """Core policy settings plus an optional explicit per-batch chain schedule."""

    chain_width_table: tuple[tuple[int, int], ...] = ()


STATIC = TokenSpeedConfig(mode=Mode.STATIC)


def parse_algorithm_config(algorithm_config: Mapping[str, Any] | None) -> TokenSpeedConfig:
    """The HarnessConfig of a launch (None: no ``--speculative-algorithm-config``); unknown keys and modes TokenSpeed
    cannot serve raise."""
    if algorithm_config is None:
        return STATIC
    values = dict(algorithm_config)
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
    fully_scheduled = bool(entries and entries[0][0] == 1 and values.get("mode") == "serve"
                           and values.get("intensity") == "chain_intensity")
    # A complete explicit schedule makes no cost decisions and needs no timing files.
    core = harness_config_from_dict({**values, "mode": "static"} if fully_scheduled else values)
    if fully_scheduled:
        core = replace(core, mode=Mode.SERVE)
    if core.mode is Mode.SERVE and not fully_scheduled and not core.data_path:
        raise ValueError("TokenSpeed serve mode needs data_path or a complete explicit chain schedule; live costs are unsupported")
    config = TokenSpeedConfig(**{field.name: getattr(core, field.name) for field in fields(core)},
                              chain_width_table=tuple(entries))
    if entries:
        from eagle_harness.core.limits import ComputeIntensity

        if config.intensity is not ComputeIntensity.CHAIN or config.mode is not Mode.SERVE:
            raise ValueError("chain_width_table requires serve mode and chain_intensity")
    if config.mode not in (Mode.STATIC, Mode.SERVE):
        raise NotImplementedError(f"{ALGORITHM_NAME} on TokenSpeed supports static and serve modes")
    return config
