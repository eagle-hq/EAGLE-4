# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The load-time host check (eagle_harness/compat.py) against fake hosts: runs without sglang or vLLM."""

from __future__ import annotations

import logging
import sys
import types

import pytest
from packaging.specifiers import SpecifierSet

from eagle_harness import __version__, compat

FAKE = compat.Host("fakehost", SpecifierSet(">=1.2.dev0"), "fakehost main abc123", "fakehost_hooks:problems")


@pytest.fixture
def fake_host(monkeypatch):
    """Installs a host module ``fakehost`` and its hook check; returns a setter for (version, problems)."""
    host_module, hooks_module = types.ModuleType("fakehost"), types.ModuleType("fakehost_hooks")
    monkeypatch.setitem(sys.modules, "fakehost", host_module)
    monkeypatch.setitem(sys.modules, "fakehost_hooks", hooks_module)
    monkeypatch.setattr(compat, "HOSTS", (FAKE,))

    def install(version, problems):
        host_module.__version__ = version
        hooks_module.problems = lambda: list(problems)

    return install


def test_matching_host_registers_quietly(fake_host, caplog):
    fake_host("1.3.0", [])
    report = compat.check("fakehost")
    assert report.ok and report.tested_version
    with caplog.at_level(logging.WARNING):
        assert compat.check_for_registration("fakehost") is True
    assert caplog.text == ""


def test_missing_hook_refuses_with_version_problems_and_fix(fake_host, caplog):
    fake_host("1.1.0", ["fakehost:Worker.step is missing"])
    assert compat.check_for_registration("fakehost") is False
    message = caplog.text
    assert f"eagle-harness {__version__} is disabled" in message
    assert "fakehost 1.1.0 lacks 1 of the hooks" in message and "fakehost:Worker.step is missing" in message
    assert "fakehost>=1.2.dev0" in message and "tested: fakehost main abc123" in message


def test_untested_version_with_matching_hooks_warns_and_registers(fake_host, caplog):
    fake_host("0.0.0.dev0", [])  # a source checkout without git tags
    assert compat.check_for_registration("fakehost") is True
    assert "fakehost 0.0.0.dev0 is outside the tested versions" in caplog.text


@pytest.mark.parametrize("version", ["dev", "unknown"])
def test_unparsable_version_is_untested_not_an_error(fake_host, version):
    fake_host(version, [])
    report = compat.check("fakehost")
    assert report.ok and not report.tested_version


def test_cli_exit_codes(fake_host, capsys):
    fake_host("1.3.0", [])
    assert compat.main([]) == 0
    assert "fakehost 1.3.0: ok, every hook matches" in capsys.readouterr().out
    fake_host("1.3.0", ["fakehost:Worker.step is missing"])
    assert compat.main([]) == 1
    assert "fakehost 1.3.0: FAILED" in capsys.readouterr().out


def test_cli_fails_without_any_host(monkeypatch, capsys):
    monkeypatch.setattr(compat, "HOSTS", (compat.Host("nosuchhost", SpecifierSet(">=1"), "-", "nosuchhost:x"),))
    assert compat.main([]) == 1
    out = capsys.readouterr().out
    assert "nosuchhost: not installed" in out and "no supported host installed" in out


def test_every_host_check_resolves():
    for host in compat.HOSTS:
        assert callable(compat._resolve(host.problems))


@pytest.mark.parametrize("name", [h.name for h in compat.HOSTS])
def test_host_range_matches_its_extra(name):
    from importlib.metadata import PackageNotFoundError, requires

    from packaging.requirements import Requirement

    try:
        declared = requires("eagle-harness") or []
    except PackageNotFoundError:
        pytest.skip("eagle-harness is not installed")
    (extra,) = [r for r in map(Requirement, declared) if r.name == name and r.marker is not None]
    assert extra.specifier == compat.host(name).versions
