# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Contract of the sglang hooks the plugin uses (eagle_harness/sglang/hooks.py).

Needs sglang importable. Fails when the installed sglang drops or changes a hook, or when a package override drifts
from the signature the table pins.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sglang.srt.speculative.spec_registry")

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.srt.speculative.spec_registry import _REGISTRY  # noqa: E402

from eagle_harness.sglang import hooks, plugin  # noqa: E402
from eagle_harness.sglang.algorithm import ALGORITHM_NAME  # noqa: E402


@pytest.fixture
def clean_registry():
    snapshot = _REGISTRY.copy()
    _REGISTRY.pop(ALGORITHM_NAME, None)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def test_installed_sglang_has_every_hook():
    assert hooks.hook_problems() == []


@pytest.mark.parametrize("hook", [h for h in hooks.HOOKS if h.kind == hooks.OVERRIDE], ids=lambda h: h.path)
def test_every_override_has_the_pinned_signature(hook):
    owner = hooks.resolve(hook.owner)
    name = hook.path.rpartition(".")[2]
    assert name in vars(owner), f"{hook.owner} does not override {name}"
    assert hooks.parameter_names(getattr(owner, name)) == hook.params


def test_overrides_in_the_package_are_all_listed():
    """A new override of an sglang method must be added to HOOKS (and so to the load-time check)."""
    listed = {(h.owner, h.path.rpartition(".")[2]) for h in hooks.HOOKS if h.kind == hooks.OVERRIDE}
    bases = {h.owner: hooks.resolve(h.path.rpartition(".")[0]) for h in hooks.HOOKS if h.kind == hooks.OVERRIDE}
    for owner_path, base in bases.items():
        owner = hooks.resolve(owner_path)
        for name, value in vars(owner).items():
            # Dunder methods (dataclass-generated __init__ / __repr__ / __eq__) are not hooks.
            if callable(value) and not name.startswith("__") and hasattr(base, name):
                assert (owner_path, name) in listed, f"{owner_path}.{name} overrides sglang but is not in HOOKS"


def test_a_missing_or_changed_hook_is_reported(monkeypatch):
    algo = "sglang.srt.speculative.spec_registry:CustomSpecAlgo"
    monkeypatch.setattr(
        hooks,
        "HOOKS",
        (
            hooks.Hook(f"{algo}.no_such_hook", (), hooks.OVERRIDE),
            hooks.Hook(f"{algo}.req_to_token_extra_len", ("page_size", "extra"), hooks.OVERRIDE),
            hooks.Hook("sglang.srt.speculative.spec_info:SpeculativeAlgorithm.register", ("name", "x"), hooks.CALL),
            hooks.Hook("sglang.srt.arg_groups.fields.spec:Spec.no_such_field", (), hooks.FIELD),
            hooks.Hook("sglang.srt.speculative.spec_info:SpecInput.no_such_flag", (), hooks.FIELD),
        ),
    )
    assert hooks.hook_problems() == [
        f"{algo}.no_such_hook is missing",
        f"{algo}.req_to_token_extra_len('page_size',) is not ('page_size', 'extra')",
        "sglang.srt.speculative.spec_info:SpeculativeAlgorithm.register"
        f"{hooks.parameter_names(SpeculativeAlgorithm.register)} lacks ('x',)",
        "sglang.srt.arg_groups.fields.spec:Spec.no_such_field is missing",
        "sglang.srt.speculative.spec_info:SpecInput.no_such_flag is missing",
    ]


def test_optional_hook_accepts_absence_but_not_signature_drift(monkeypatch):
    algo = "sglang.srt.speculative.spec_registry:CustomSpecAlgo"
    monkeypatch.setattr(
        hooks,
        "HOOKS",
        (
            hooks.Hook(f"{algo}.no_such_hook", (), hooks.OVERRIDE, optional=True),
            hooks.Hook(f"{algo}.req_to_token_extra_len", ("wrong",), hooks.OVERRIDE, optional=True),
        ),
    )
    assert hooks.hook_problems() == [f"{algo}.req_to_token_extra_len('page_size',) is not ('wrong',)"]


def test_register_refuses_an_sglang_without_the_hooks(monkeypatch, clean_registry, caplog):
    import sglang

    monkeypatch.setattr(hooks, "hook_problems", lambda: ["X is missing"])
    assert plugin.register() is False
    assert f"sglang {sglang.__version__} lacks 1 of the hooks it needs (X is missing)" in caplog.text
    assert "Fix: install sglang with the eagle-harness hooks" in caplog.text
    with pytest.raises(ValueError):
        SpeculativeAlgorithm.from_string(ALGORITHM_NAME)


def test_register_on_a_matching_sglang(clean_registry):
    assert plugin.register() is True
    assert plugin.register() is True  # idempotent
    assert SpeculativeAlgorithm.from_string(ALGORITHM_NAME).is_eagle()
