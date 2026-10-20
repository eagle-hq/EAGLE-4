# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Weight L2 warm-ahead in the target-verify graphs.

A verify at a few tokens streams every weight of the target from HBM once, but
the layers leave HBM idle between their GEMMs: during the mixer (GDN tree conv
+ recurrence + gated norm, or qk-norm / rope / attention) and during the
all-reduces. An L2 prefetch kernel forked onto a side stream in such a window
warms the tail of a later GEMM's weight (the head is left to the GEMM's own
k-tile prefetch); it only prefetches, so results are unchanged.

Plain PyTorch module hooks do it, only while a verify of at most
:data:`MAX_TOKENS` tokens is captured: a forward pre-hook on the mixer's core
(its ``attn`` module: the GDN conv + recurrence, or the attention kernel) forks
the prefetch of the layer's MLP ``gate_up`` weight (warm-ahead, tuned on B300 at 16 tokens:
GDN layers pace it over the whole mixer from few CTAs, attention layers issue a short burst before the attention
kernel launches, whose CTAs fill every SM); optional hooks after the mixer's
output projection and after the MLP's ``down`` projection prefetch the ``down``
weight and the next layer's input projection
(:class:`~eagle_harness.core.config.L2Warm`). A forward hook on the last layer
joins the side stream once. The graph records the stream launches and the join; replays
cost nothing on the host. A weight larger than the L2 is skipped: its own head
reads would evict the warmed part.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
from torch import nn

from eagle_harness.core.config import L2Warm
from eagle_harness.kernels.l2_prefetch import l2_prefetch

logger = logging.getLogger(__name__)

MAX_TOKENS = 16  # wider verifies keep the mixers busy enough, and their GEMMs are compute-bound
PAGE = 4096  # prefetched ranges are page aligned


class _Layer(NamedTuple):
    module: nn.Module  # the decoder layer
    attention: bool  # full attention (else a GDN layer)
    in_proj: nn.Module  # the mixer's input projection
    core: nn.Module  # the mixer's attention / recurrence module (``attn``), whose start opens the HBM-idle window
    out_proj: nn.Module  # the mixer's output projection
    gate_up: nn.Module
    down: nn.Module


def _prefetchable(linear: nn.Module, l2_bytes: int) -> bool:
    weight = getattr(linear, "weight", None)
    return (
        weight is not None
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and weight.numel() * weight.element_size() <= l2_bytes
    )


def _decoder_layers(model: nn.Module, l2_bytes: int) -> list[_Layer]:
    """The decoder layers (in order) whose projections and MLP have plain bf16 weights that fit in L2."""
    layers = []
    for module in model.modules():
        mlp = getattr(module, "mlp", None)
        # A GDN layer keeps its projections in ``linear_attn``; a full-attention layer owns them.
        for mixer, in_proj, out_proj, attention in (
            ("linear_attn", "in_proj_qkvz", "out_proj", False),
            (None, "qkv_proj", "o_proj", True),
        ):
            mixer_module = module if mixer is None else getattr(module, mixer, None)
            parts = (
                getattr(mixer_module, in_proj, None),
                getattr(mixer_module, out_proj, None),
                getattr(mlp, "gate_up_proj", None),
                getattr(mlp, "down_proj", None),
            )
            core = getattr(mixer_module, "attn", None)
            if core is None or any(part is None for part in parts):
                continue
            if all(_prefetchable(part, l2_bytes) for part in parts):
                layers.append(_Layer(module, attention, parts[0], core, *parts[1:]))
            break
    return layers


def _tokens(args: tuple, kwargs: dict) -> int | None:
    """Rows of the first tensor a module is called with (a fused all-reduce hands down a tuple); None: no tensor."""
    for arg in (*args, *kwargs.values()):
        if isinstance(arg, tuple) and arg:
            arg = arg[0]
        if isinstance(arg, torch.Tensor):
            return arg.shape[0]
    return None


class MlpL2WarmAhead:
    """Forward hooks that fork and join the prefetches while a small target verify is captured."""

    def __init__(self, layers: list[_Layer], plan: L2Warm, device: torch.device):
        self._stream = torch.cuda.Stream(device)
        self._forked = False
        self._handles = []
        for i, layer in enumerate(layers):
            gate_up_plan = plan.gate_up_attention if layer.attention else plan.gate_up_gdn
            # On the core, not after the input projection: a packed input projection does not call that module.
            self._hook(layer.core, layer.gate_up.weight, gate_up_plan, before=True)
            self._hook(layer.out_proj, layer.down.weight, plan.down_proj)
            if i + 1 < len(layers):
                self._hook(layer.down, layers[i + 1].in_proj.weight, plan.next_input)
        self._handles.append(layers[-1].module.register_forward_hook(self._join))

    @classmethod
    def install(cls, model: nn.Module, plan: L2Warm, device: torch.device) -> "MlpL2WarmAhead | None":
        """Hook ``model``'s decoder layers, or None when none has suitable weights (e.g. MoE)."""
        layers = _decoder_layers(model, torch.cuda.get_device_properties(device).L2_cache_size)
        if not layers:
            return None
        logger.info("L2 warm-ahead on %d decoder layers: %s", len(layers), plan)
        return cls(layers, plan, device)

    def remove(self) -> None:
        """Remove the hooks."""
        for handle in self._handles:
            handle.remove()

    def _hook(
        self, trigger: nn.Module, weight: torch.Tensor, prefetch: tuple[float, int] | None, before: bool = False
    ) -> None:
        """Prefetch ``prefetch = (share of the weight's bytes, CTAs)`` of ``weight`` when ``trigger`` starts
        (``before``) or after it runs."""
        if prefetch is None:
            return
        share, num_ctas = prefetch
        total = weight.numel() * weight.element_size()
        nbytes = int(total * share) // PAGE * PAGE

        def fork(_module: nn.Module, args: tuple, kwargs: dict, _output=None) -> None:
            """Module hook: prefetch the weight's warm share on the side stream while the captured graph goes on."""
            tokens = _tokens(args, kwargs)
            if not torch.cuda.is_current_stream_capturing() or tokens is None or tokens > MAX_TOKENS:
                return
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                l2_prefetch(weight, total - nbytes, nbytes, num_ctas)
            self._forked = True

        register = trigger.register_forward_pre_hook if before else trigger.register_forward_hook
        self._handles.append(register(fork, with_kwargs=True))

    def _join(self, _module: nn.Module, _args: tuple, _output) -> None:
        if self._forked:
            torch.cuda.current_stream().wait_stream(self._stream)
            self._forked = False
