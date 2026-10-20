# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""L2 warm-ahead: hooks fork the prefetches only while a small verify is captured, results unchanged.

Run in a GPU job: ``pytest -q tests/test_l2_warm_gpu.py``.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eagle_harness.core.config import L2Warm
from eagle_harness.kernels.l2_prefetch import l2_prefetch
from eagle_harness.sglang.l2_warm import MAX_TOKENS, MlpL2WarmAhead

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

HIDDEN = 256
EVERYTHING = L2Warm(down_proj=(0.5, 8), next_input=(0.5, 8))


class Core(nn.Module):
    """The mixer's attention / recurrence (sglang's ``attn``): the GDN one takes the batch, then keyword tensors."""

    def forward(self, *args, **kwargs):
        x = next(a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor))
        return 2 * x


class Layer(nn.Module):
    """A decoder layer in the shape the hooks look for: a mixer (input projection, core, output projection), then an
    MLP."""

    def __init__(self, mixer: str | None, in_proj: str, out_proj: str, packed: bool = False):
        """``mixer``: the submodule holding the projections (a GDN layer), None: the layer itself (attention).
        ``packed``: the input projection runs on its weight without calling the module (sglang's packed GDN GEMM)."""
        super().__init__()
        holder = self if mixer is None else nn.Module()
        setattr(holder, in_proj, nn.Linear(HIDDEN, HIDDEN, bias=False))
        holder.attn = Core()
        setattr(holder, out_proj, nn.Linear(HIDDEN, HIDDEN, bias=False))
        if mixer is not None:
            setattr(self, mixer, holder)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = nn.Linear(HIDDEN, 4 * HIDDEN, bias=False)
        self.mlp.down_proj = nn.Linear(4 * HIDDEN, HIDDEN, bias=False)
        self.mixer, self.names, self.packed = mixer, (in_proj, out_proj), packed

    def forward(self, x):
        in_proj, out_proj = self.names
        holder = self if self.mixer is None else getattr(self, self.mixer)
        proj = getattr(holder, in_proj)
        x = nn.functional.linear(x, proj.weight) if self.packed else proj(x)
        x = holder.attn(x) if self.mixer is None else holder.attn(None, mixed_qkv=x)
        x = getattr(holder, out_proj)(x)
        return self.mlp.down_proj(self.mlp.gate_up_proj(x))


def _model(device) -> nn.Module:
    kinds = [
        ("linear_attn", "in_proj_qkvz", "out_proj"),
        ("linear_attn", "in_proj_qkvz", "out_proj", True),
        (None, "qkv_proj", "o_proj"),
        ("linear_attn", "in_proj_qkvz", "out_proj"),
    ]
    return nn.Sequential(*(Layer(*kind) for kind in kinds)).to(device, torch.bfloat16)


def _replay_prefetch_kernels(model: nn.Module, x: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Capture ``model(x)``; the number of prefetch kernels in one replay and the graph's output."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = model(x)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    return sum("_l2_prefetch_kernel" in event.name for event in prof.events()), out


def test_prefetch_leaves_data_unchanged():
    weight = torch.randn(1 << 20, device="cuda").to(torch.bfloat16)
    before = weight.clone()
    l2_prefetch(weight, 4096, weight.numel(), 16)
    torch.cuda.synchronize()
    assert torch.equal(weight, before)


@pytest.mark.parametrize("plan,kernels", [(L2Warm(), 4), (EVERYTHING, 11)])
def test_only_a_captured_small_verify_forks_the_prefetches(plan, kernels):
    device = torch.device("cuda")
    model = _model(device)
    x = torch.randn(MAX_TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)
    reference = model(x)
    warm = MlpL2WarmAhead.install(model, plan, device)
    assert warm is not None
    assert torch.equal(model(x), reference)  # eager: the hooks do nothing

    count, out = _replay_prefetch_kernels(model, x)
    # gate_up per layer (the packed one too); with EVERYTHING also down per layer and next_input between layers
    assert count == kernels
    assert torch.equal(out, reference)

    wide = torch.randn(MAX_TOKENS + 1, HIDDEN, device=device, dtype=torch.bfloat16)
    count, out = _replay_prefetch_kernels(model, wide)
    assert count == 0  # wider verifies are not warmed
    assert torch.equal(out, model(wide))
    warm.remove()


def test_models_without_plain_bf16_weights_are_not_hooked():
    device = torch.device("cuda")
    assert MlpL2WarmAhead.install(_model(device).float(), L2Warm(), device) is None
    assert MlpL2WarmAhead.install(SimpleNamespace(modules=list), L2Warm(), device) is None
