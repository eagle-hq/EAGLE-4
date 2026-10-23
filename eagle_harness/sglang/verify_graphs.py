# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Target-verify CUDA graphs per verify width (hook H7).

The cost model picks the verify width of each round from the cost table's
buckets (:func:`eagle_harness.core.limits.verify_buckets`). The target's
decode runner captures one TARGET_VERIFY graph per (batch size, width) through
sglang's ``DecodeGraphVariants`` hook; a round replays the graph of its width.

At batch size 1 every tree width gets a second graph, keyed ``a<width>``, captured
with a :class:`~.verify_input.HarnessVerifyInput` whose ``greedy_tokens_only`` is
set: sglang's logits processor then returns the rows' greedy tokens from the vocab
shards instead of gathering the logits. A round replays it when its verify input
is greedy (a greedy round of the fused accept tail). The batch-size-1 graphs are
captured with the tree-tail mask attention (:class:`~.verify_plan.TreeTailMask`).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.decode_graph_variants import DecodeGraphVariant, DecodeGraphVariants

from eagle_harness.core.limits import ComputeIntensity, verify_buckets
from eagle_harness.sglang.verify_input import HarnessVerifyInput
from eagle_harness.sglang.verify_plan import TreeTailMask


def native_decode_graphs() -> bool:
    """Whether the engine captures a speculative graph variant in its own forward mode (native DECODE graphs)."""
    return hasattr(DecodeGraphVariant, "forward_mode")


def verify_widths(batch_size: int, intensity: ComputeIntensity, max_width: int) -> list:
    """Verify widths with a graph at ``batch_size``, widest first (``max_width`` and 1 always included)."""
    widths = {w for w in verify_buckets(batch_size, intensity) if w <= max_width}
    widths.update((1, max_width))
    return sorted(widths, reverse=True)


def round_up_width(width: int, widths: Sequence[int]) -> int:
    """The smallest captured width ``>= width`` (``widths`` must contain one)."""
    fitting = [w for w in widths if w >= width]
    if not fitting:
        raise ValueError(f"verify width {width} exceeds every captured width {list(widths)}")
    return min(fitting)


class VerifyWidthGraphs(DecodeGraphVariants):
    """One target-verify graph per verify width, keyed ``w<width>``, and at batch size 1 one greedy graph
    per tree width, keyed ``a<width>``."""

    def __init__(
        self,
        intensity: ComputeIntensity,
        max_width: int,
        tail_mask: TreeTailMask,
        greedy: bool = True,
        *,
        plain_decode: bool = False,
    ):
        self.intensity = intensity
        self.max_width = max_width
        self.tail_mask = tail_mask
        self.greedy = greedy
        if plain_decode and not native_decode_graphs():
            raise RuntimeError("Native no-spec fallback requires SGLang's per-variant forward-mode graph contract.")
        self._plain = DecodeGraphVariant("decode", 1, ForwardMode.DECODE) if plain_decode else None
        self._variants: Dict[int, DecodeGraphVariant] = {}
        self._argmax_variants: Dict[int, DecodeGraphVariant] = {}

    def _variant(self, width: int, argmax: bool = False) -> DecodeGraphVariant:
        table = self._argmax_variants if argmax else self._variants
        variant = table.get(width)
        if variant is None:
            variant = table[width] = DecodeGraphVariant(f"{'a' if argmax else 'w'}{width}", width)
        return variant

    def capture_variants(self, bs: int) -> Sequence[DecodeGraphVariant]:
        """The verify-width graphs to capture for a batch size (the argmax graphs last)."""
        widths = verify_widths(bs, self.intensity, self.max_width)
        variants = [self._variant(w) for w in widths]
        if self.greedy and bs == 1:
            variants += [self._variant(w, argmax=True) for w in widths if w > 1]
        if self._plain is not None:
            variants.append(self._plain)
        return variants

    def replay_variant(self, forward_batch) -> Optional[DecodeGraphVariant]:
        """The graph variant of this forward's verify width (its greedy graph for a greedy input), None for
        other forwards."""
        if self._plain is not None and forward_batch.forward_mode.is_decode():
            return self._plain
        spec = forward_batch.spec_info
        table = self._argmax_variants if getattr(spec, "greedy_tokens_only", False) else self._variants
        return table.get(getattr(spec, "draft_token_num", None))

    def begin_capture(self, variant, forward_batch, attn_backend, stream_idx):
        """Capture the variant with the harness's verify input at the variant's width (greedy for a greedy
        variant, with the tail-mask attention); return the attention backend."""
        if variant is self._plain:
            forward_batch.spec_info = None
            return attn_backend
        greedy = self._argmax_variants.get(variant.num_tokens_per_req) is variant
        spec = forward_batch.spec_info = HarnessVerifyInput.from_eagle(
            forward_batch.spec_info, greedy_tokens_only=greedy, tail_mask=self.tail_mask
        )
        spec.draft_token_num = variant.num_tokens_per_req
        spec.num_tokens_per_req = variant.num_tokens_per_req
        spec.num_tokens_for_logprob_per_req = variant.num_tokens_per_req
        return attn_backend
