# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The TokenSpeed plugin against the installed TokenSpeed: hook check, registration, the draft model's profile.

No GPU work, but TokenSpeed imports only on a host with a GPU (its kernel package refuses otherwise), so this runs in
the GPU test jobs and is skipped elsewhere. On a TokenSpeed without the ``pr-spec-plugin-hooks`` hooks the registration tests
check that the plugin switches itself off instead.
"""

from __future__ import annotations

import sys
from importlib.metadata import entry_points

import pytest

try:
    import tokenspeed.runtime.plugins.registry  # noqa: F401
except (ImportError, RuntimeError) as exc:  # RuntimeError: tokenspeed-kernel on a host without a GPU
    pytest.skip(f"TokenSpeed is not importable here: {exc}", allow_module_level=True)

from tokenspeed.runtime.configs.model_profile import ModelProfile  # noqa: E402
from tokenspeed.runtime.execution import drafter as drafter_registry  # noqa: E402
from tokenspeed.runtime.plugins import registry  # noqa: E402

from eagle_harness import compat  # noqa: E402
from eagle_harness.models.eagle4_config import EAGLE4_ARCH  # noqa: E402
from eagle_harness.tokenspeed import plugin  # noqa: E402
from eagle_harness.tokenspeed.config import ALGORITHM_NAME  # noqa: E402
from eagle_harness.tokenspeed.hooks import hook_problems  # noqa: E402

HAS_HOOKS = not hook_problems()
needs_hooks = pytest.mark.skipif(
    not HAS_HOOKS, reason=f"TokenSpeed without the pr-spec-plugin-hooks hooks: {hook_problems()}"
)


@pytest.fixture
def registered():
    drafters, models = dict(drafter_registry._PLUGIN_DRAFTERS), dict(registry._MODELS)
    assert plugin.register() is True
    yield
    drafter_registry._PLUGIN_DRAFTERS.clear()
    drafter_registry._PLUGIN_DRAFTERS.update(drafters)
    registry._MODELS.clear()
    registry._MODELS.update(models)


def test_entry_point_targets_register():
    (ep,) = [ep for ep in entry_points(group="tokenspeed.plugins") if ep.name == "eagle_harness"] or [None]
    if ep is None:
        pytest.skip("eagle-harness is not installed with its entry points here")
    assert ep.value == "eagle_harness.tokenspeed.plugin:register"


@needs_hooks
def test_register_adds_the_drafter_and_the_draft_model(registered):
    from eagle_harness.tokenspeed.drafter import HarnessDrafter
    from eagle_harness.tokenspeed.models import LlamaForCausalLMEagle4

    assert drafter_registry.get_drafter_impl(ALGORITHM_NAME, None) is HarnessDrafter
    assert drafter_registry.drafter_captures_aux_hidden_states(ALGORITHM_NAME)
    assert registry.registered_model(EAGLE4_ARCH).cls is LlamaForCausalLMEagle4


@needs_hooks
def test_draft_profile_is_a_valid_mha_profile():
    from eagle_harness.tokenspeed.models import LlamaForCausalLMEagle4

    profile = LlamaForCausalLMEagle4.model_profile(None)
    assert isinstance(profile, ModelProfile) and profile.cache_family == "mha" and profile.linear_attention is None


@needs_hooks
def test_compat_reports_the_hooks_ok():
    assert compat.check("tokenspeed").ok


def test_register_is_off_without_the_hooks(monkeypatch):
    # The registry module fails to import, as on a TokenSpeed without plugins: hook_problems names it.
    monkeypatch.setitem(sys.modules, "tokenspeed.runtime.plugins.registry", None)
    assert any("registry" in p for p in hook_problems())
    assert plugin.register() is False


def test_hook_problems_name_a_missing_parameter(monkeypatch):
    def register_drafter(algorithm, drafter_cls, *, model_cls=None):  # main's signature without H2
        raise AssertionError("never called")

    monkeypatch.setattr(registry, "register_drafter", register_drafter)
    assert any("captures_aux_hidden_states" in p for p in hook_problems())


@needs_hooks
def test_candidate_gather_rides_the_last_dim_as_16_bit_words(monkeypatch):
    """The TP candidate exchange asks the engine for a last-dim gather of 16-bit rows (its barrier-free Lamport
    path, which keeps the draft's device-launched round graphs free of NCCL) and unpacks rank-major columns."""
    import torch

    from eagle_harness.tokenspeed import models

    tp, rows, n = 4, 6, 8
    g = torch.Generator().manual_seed(7)
    shards = [torch.randint(-(2**31), 2**31 - 1, (rows, n), dtype=torch.int32, generator=g) for _ in range(tp)]
    calls = []

    def all_gather(tensor, group, dim):
        calls.append((tensor.dtype, tensor.shape, group, dim))
        return torch.cat([s.view(tensor.dtype) for s in shards], dim=-1)

    monkeypatch.setattr(models, "all_gather", all_gather)
    head = models.LlamaForCausalLMEagle4.__new__(models.LlamaForCausalLMEagle4)
    head.tp_size, head.tp_group = tp, "tp"
    out = head._gather_candidates(shards[0])
    assert calls == [(torch.int16, (rows, 2 * n), "tp", -1)]
    assert out.shape == (tp, rows, n) and out.dtype == torch.int32
    assert all(torch.equal(out[r], shards[r]) for r in range(tp))
