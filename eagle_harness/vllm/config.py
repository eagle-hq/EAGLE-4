# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""``--speculative-config`` options of the ``eagle_harness`` method on vLLM.

vLLM does not interpret ``SpeculativeConfig.method_config``; this module does.
The keys are :class:`~eagle_harness.core.config.HarnessConfig` (same names and
strict parsing as sglang's ``--speculative-algorithm-config``) plus three that
sglang takes from its own server arguments, and one of the vLLM tree drafter:

- ``max_verify_width``: the largest verify width (tree nodes including the root);
  vLLM's ``num_speculative_tokens`` is this minus one.
- ``node_top_k`` / ``tree_depth``: the draft tree shape, which sizes the draft's KV
  lookahead (``req_to_token_span`` in sglang terms).
- ``depth_stop_gain``: what one more draft step must add to the verify set's score sum (its
  expected accepted length) for a round to run it (:mod:`eagle_harness.vllm.depth_stop`).
  ``"derived"`` (default): the step's measured cost in accepted tokens, from the cost tables
  under ``data_path`` (:func:`~eagle_harness.vllm.depth_stop.round_costs`: the round's expected
  accept x draft step / (verify forward + steps run x draft step)); without tables that time
  this shape, automatic serving refuses startup. Static mode keeps a lossless stop.
  A number overrides it with a fixed gain;
  ``0.0`` stops only when no deeper node can enter the verify set, so the verify rows equal
  those of the full ``tree_depth``; ``null``: always ``tree_depth``.
- ``record_rounds`` (tree, default false): time the served decode rounds and merge them into
  ``<data_path>/rounds/tree.json`` (:mod:`eagle_harness.vllm.round_recorder`). A tree whose
  ``data_path`` holds those rounds verifies fewer rows from the first batch size where that
  measured faster (:func:`~eagle_harness.core.chain_width.served_tree_schedule`); smaller batches
  keep ``max_verify_width``.

Example::

    --speculative-config '{"method": "eagle_harness", "model": "eagle-ai/EAGLE-4-Qwen3.8-27B",
        "method_config": {"data_path": "/data/cost/qwen38-tp4", "max_verify_width": 32,
                          "node_top_k": 8, "tree_depth": 12}}'

Pure Python over ``core``: no GPU and no vLLM import at module load.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Tuple

from eagle_harness.core.chain_width import (
    PROFILE_FILE,
    TREE_ROUNDS_FILE,
    chain_widths,
    load_tree_rounds,
    merge_ranges,
    served_tree_schedule,
)
from eagle_harness.core.config import HarnessConfig, harness_config_from_dict
from eagle_harness.core.limits import (
    MAX_CHAIN_VERIFY_NUM,
    MAX_VERIFY_NUM,
    ComputeIntensity,
    check_shape,
    cost_batch_sizes,
    max_batch_size,
    req_to_token_span,
    verify_buckets,
)
from eagle_harness.core.live_costs import CostKey

METHOD_NAME = "eagle_harness"
# Keys of ``method_config`` that are not HarnessConfig fields.
VLLM_KEYS = ("max_verify_width", "node_top_k", "tree_depth", "depth_stop_gain", "record_rounds")
DEFAULT_NODE_TOP_K = 8
DEFAULT_TREE_DEPTH = 12
DERIVED = "derived"  # depth_stop_gain from the measured costs
NATIVE_DECODE_UNSUPPORTED = (
    "eagle_harness: the chain schedule selects native plain decode (width 0), "
    "which requires vLLM's suspend_drafting speculator hook or its native_decode_batch_sizes field. "
    "K=0 still extends draft KV. "
    "Use a compatible host, run without speculative_config, or calibrate root-only width 1."
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VllmMethodConfig:
    """A parsed ``method_config``."""

    harness: HarnessConfig
    max_verify_width: int  # tree nodes including the root; K = max_verify_width - 1
    node_top_k: int
    tree_depth: int
    depth_stop_gain: float | Literal["derived"] | None = DERIVED
    record_rounds: bool = False  # time served rounds into <data_path>/rounds/tree.json

    @property
    def num_speculative_tokens(self) -> int:
        return self.max_verify_width - 1

    @property
    def draft_kv_lookahead(self) -> int:
        """KV slots past the scheduled query that a round writes (draft or verify)."""
        return max(
            self.num_speculative_tokens, req_to_token_span(self.node_top_k, self.tree_depth, self.max_verify_width, 1)
        )


def default_max_verify_width(intensity: ComputeIntensity) -> int:
    return MAX_CHAIN_VERIFY_NUM if intensity is ComputeIntensity.CHAIN else 32


def parse_method_config(method_config: Mapping[str, Any]) -> VllmMethodConfig:
    """Parse and validate ``SpeculativeConfig.method_config``. Unknown keys raise."""
    data = dict(method_config)
    extra = {key: data.pop(key) for key in VLLM_KEYS if key in data}
    harness = harness_config_from_dict(data)
    chain = harness.intensity is ComputeIntensity.CHAIN
    width = extra.get("max_verify_width", default_max_verify_width(harness.intensity))
    top_k = extra.get("node_top_k", 1 if chain else DEFAULT_NODE_TOP_K)
    depth = extra.get("tree_depth", MAX_CHAIN_VERIFY_NUM - 1 if chain else DEFAULT_TREE_DEPTH)
    for name, value in (("max_verify_width", width), ("node_top_k", top_k), ("tree_depth", depth)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"method_config.{name} must be an integer, got {value!r}.")
    if width > (MAX_CHAIN_VERIFY_NUM if chain else MAX_VERIFY_NUM):
        raise NotImplementedError(f"max_verify_width={width} is not supported for {harness.intensity.value}.")
    check_shape(harness.intensity, top_k, depth, width, 1)
    gain = extra.get("depth_stop_gain", DERIVED)
    if gain not in (None, DERIVED) and (isinstance(gain, bool) or not isinstance(gain, (int, float)) or gain < 0):
        raise ValueError(f'method_config.depth_stop_gain must be "derived", a number >= 0 or null, got {gain!r}.')
    record = extra.get("record_rounds", False)
    if not isinstance(record, bool):
        raise ValueError(f"method_config.record_rounds must be true or false, got {record!r}.")
    if record and not harness.data_path:
        raise ValueError("method_config.record_rounds needs a data_path to write rounds/tree.json under.")
    return VllmMethodConfig(
        harness=harness,
        max_verify_width=width,
        node_top_k=top_k,
        tree_depth=depth,
        depth_stop_gain=gain if gain in (None, DERIVED) else float(gain),
        record_rounds=record,
    )


def verify_width_schedule(
    intensity: ComputeIntensity, max_verify_width: int, data_path: str | None = None, *, allow_unpriced: bool = False
) -> List[Tuple[int, int, int]]:
    """The draft-count schedule for callers without native-decode metadata.

    Trees schedule the widest verify bucket minus its root. Chains use the
    canonical priced width minus one. A selected width zero is rejected here:
    only configure_speculative_config can pass its separate native ranges to a
    capable host. Unpriced chains require an explicit fixed width.
    """
    widths = _verify_widths(intensity, max_verify_width, data_path, allow_unpriced=allow_unpriced)
    if any(width == 0 for _, _, width in widths):
        raise ValueError(NATIVE_DECODE_UNSUPPORTED)
    return _draft_token_schedule(widths, max_batch_size(intensity))


def _verify_widths(
    intensity: ComputeIntensity,
    max_verify_width: int,
    data_path: str | None,
    *,
    allow_unpriced: bool,
    key: CostKey | None = None,
) -> List[Tuple[int, int, int]]:
    limit = max_batch_size(intensity)
    chain = intensity is ComputeIntensity.CHAIN
    widths = derived_chain_widths(data_path, max_verify_width, limit, key) if chain else None
    if widths is None and chain and not allow_unpriced:
        raise ValueError(
            f"eagle_harness: no priced chain schedule with an acceptance profile under {data_path} "
            f"({PROFILE_FILE}); calibrate the chain or explicitly set max_verify_width for a fixed unpriced chain."
        )
    if widths is None:
        widths = [(bs, bs, min(max(verify_buckets(bs, intensity)), max_verify_width)) for bs in cost_batch_sizes(limit)]
    return widths


def _draft_token_schedule(widths: List[Tuple[int, int, int]], limit: int) -> List[Tuple[int, int, int]]:
    # each range starts after the previous one: tree widths come per calibrated batch size, with gaps between them
    firsts = [1] + [last + 1 for _, last, _ in widths[:-1]]
    ranges = merge_ranges((first, last, max(0, width - 1)) for first, (_, last, width) in zip(firsts, widths))
    return ranges + [(limit + 1, 1 << 16, 0)]


def derived_chain_widths(
    data_path: str | None, max_verify_width: int, limit: int, key: CostKey | None = None
) -> List[Tuple[int, int, int]] | None:
    """The chain widths per batch size range priced by the served round times (else the cost tables) and the acceptance
    profile under ``data_path`` (:func:`eagle_harness.core.chain_width.chain_widths`); None when missing."""
    if not data_path:
        return None
    widths = chain_widths(data_path, max_verify_width, limit, allow_no_spec=True, key=key)

    if widths is not None:
        logger.info("eagle_harness: chain widths per batch size from %s: %s", data_path, widths)
    return widths


def aux_hidden_state_layers(draft_hf_config: Any) -> Tuple[int, ...]:
    """vLLM layer ids whose input the target captures for the draft (``()``: none named).

    The checkpoint names the layers whose *output* the draft was trained on (the sglang convention);
    vLLM captures the input of the layer it is given, one layer later.
    """
    named = (getattr(draft_hf_config, "eagle_config", None) or {}).get("eagle_aux_hidden_state_layer_ids") or ()
    return tuple(layer + 1 for layer in named)


def configure_speculative_config(spec_config: Any) -> None:
    """``SpeculativeMethodSpec.configure``: runs at the end of ``SpeculativeConfig.__post_init__``.

    ``num_speculative_tokens`` defaults to the widest verify the batch-size schedule runs, not to
    ``max_verify_width - 1``: vLLM reserves ``1 + num_speculative_tokens`` linear-attention state slots per request
    for the whole request, so a K no batch size verifies only costs running requests.
    """
    parsed = parse_method_config(spec_config.method_config)
    if parsed.node_top_k > 1 and getattr(spec_config, "native_decode_batch_sizes", None):
        raise ValueError("eagle_harness: native decode ranges require chain mode (node_top_k=1).")
    if spec_config.draft_model_config is not None:
        draft = spec_config.draft_model_config.hf_config
        if "LlamaForCausalLMEagle4" in getattr(draft, "architectures", ()):
            draft.eagle_aux_hidden_state_layer_ids = aux_hidden_state_layers(draft)
        elif set(getattr(draft, "architectures", ())) & {
            "Eagle3LlamaForCausalLM",
            "LlamaForCausalLMEagle3",
            "LlamaForCausalLM",
        }:
            # Native EAGLE3 top-level layer ids already name layer inputs.
            draft.architectures = ["Eagle3ForHarness"]
            spec_config.update_arch_()
    if parsed.harness.mode.uses_cost_model and spec_config.num_speculative_tokens_per_batch_size is None:
        widths = _verify_widths(
            parsed.harness.intensity,
            parsed.max_verify_width,
            parsed.harness.data_path,
            allow_unpriced="max_verify_width" in spec_config.method_config,
        )
        native_ranges = _native_ranges(widths)
        if native_ranges:
            from eagle_harness.vllm.batch_policy import host_speculator_hooks

            if "suspend_drafting" in host_speculator_hooks():
                pass  # HarnessSpeculator.suspend_drafting answers from the same widths (native_decode_ranges)
            elif hasattr(spec_config, "native_decode_batch_sizes"):  # the 0.1.0 release engine
                spec_config.native_decode_batch_sizes = native_ranges
            else:
                raise ValueError(NATIVE_DECODE_UNSUPPORTED)
        spec_config.num_speculative_tokens_per_batch_size = _draft_token_schedule(
            widths, max_batch_size(parsed.harness.intensity)
        )
    elif parsed.node_top_k > 1 and spec_config.num_speculative_tokens_per_batch_size is None:
        tree = measured_tree_schedule(parsed)
        if tree is not None:
            spec_config.num_speculative_tokens_per_batch_size = tree
    schedule = spec_config.num_speculative_tokens_per_batch_size
    widest = parsed.num_speculative_tokens if schedule is None else max(1, max(k for _, _, k in schedule))
    if spec_config.num_speculative_tokens is None:
        spec_config.num_speculative_tokens = widest
    elif spec_config.num_speculative_tokens not in (widest, parsed.num_speculative_tokens):
        raise ValueError(
            f"num_speculative_tokens={spec_config.num_speculative_tokens} disagrees with "
            f"method_config.max_verify_width={parsed.max_verify_width} (they must differ by one); "
            "set only one of them."
        )


def _native_ranges(widths: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    return [(first, last) for first, last, width in widths if width == 0]


def native_decode_ranges(spec_config: Any) -> List[Tuple[int, int]]:
    """The batch-size ranges where the schedule :func:`configure_speculative_config` installed decodes natively
    (width 0); empty when it installed none or the user set the schedule. Deterministic, so the speculator in each
    worker recomputes what configure saw."""
    parsed = parse_method_config(spec_config.method_config)
    if not parsed.harness.mode.uses_cost_model:
        return []
    widths = _verify_widths(
        parsed.harness.intensity,
        parsed.max_verify_width,
        parsed.harness.data_path,
        allow_unpriced="max_verify_width" in spec_config.method_config,
    )
    installed = [tuple(entry) for entry in spec_config.num_speculative_tokens_per_batch_size or ()]
    if installed != _draft_token_schedule(widths, max_batch_size(parsed.harness.intensity)):
        return []
    return _native_ranges(widths)


def measured_tree_schedule(parsed: VllmMethodConfig, key: CostKey | None = None) -> List[Tuple[int, int, int]] | None:
    """The static tree's draft-count schedule from served rounds under ``data_path`` (None: one width throughout)."""
    data_path = parsed.harness.data_path
    rounds = load_tree_rounds(data_path, key) if data_path else None
    if not rounds:
        return None
    limit = max_batch_size(parsed.harness.intensity)
    widths = served_tree_schedule(rounds, parsed.max_verify_width, limit)
    if widths is None:
        return None
    path = os.path.join(data_path, TREE_ROUNDS_FILE)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    logger.info("eagle_harness: tree verify width per batch size from %s (sha256 %s): %s", path, digest, widths)
    return _draft_token_schedule(widths, limit)


def check_data_key(spec_config: Any, key: CostKey) -> None:
    """Refuse a ``data_path`` measured on another setup: re-read what :func:`configure_speculative_config` sized the
    schedule from, with this server's ``key``. Configure runs in the API process, which cannot name the GPU, so every
    worker checks before it serves."""
    parsed = parse_method_config(spec_config.method_config)
    harness = parsed.harness
    if harness.mode.uses_cost_model:
        _verify_widths(harness.intensity, parsed.max_verify_width, harness.data_path, allow_unpriced=True, key=key)
    elif parsed.node_top_k > 1:
        measured_tree_schedule(parsed, key)


def num_lookahead_tokens(spec_config: Any) -> int:
    """``SpeculativeMethodSpec.num_lookahead_tokens`` (sglang hook H5's role)."""
    return parse_method_config(spec_config.method_config).draft_kv_lookahead
