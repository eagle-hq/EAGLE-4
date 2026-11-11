# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Where the draft-round megakernel serves: found on a B300 with its wheel or ``$EAGLE_HARNESS_DRAFTMK_LIB``, the
default backend then; anywhere else the draft rounds run on the graphs."""

import os
import sys
from types import SimpleNamespace

import pytest

from eagle_harness.kernels import arch, draftmk


@pytest.fixture
def megakernel(monkeypatch, tmp_path):
    """A B300 and an installed eagle-harness-draftmk wheel at ``tmp_path``."""
    monkeypatch.setattr(arch, "draftmk", lambda device=None: True)
    monkeypatch.setitem(
        sys.modules, "eagle_harness_draftmk", SimpleNamespace(CUBINS=tmp_path, LIBRARY=tmp_path / "libdraftmk.so")
    )
    monkeypatch.delenv("EAGLE_HARNESS_DRAFTMK_LIB", raising=False)
    monkeypatch.delenv("DRAFTMK_CUBIN_DIR", raising=False)
    draftmk.library.cache_clear()
    yield tmp_path
    draftmk.library.cache_clear()


def test_the_wheel_serves_with_its_own_cubins(megakernel):
    assert draftmk.library() == str(megakernel / "libdraftmk.so")
    assert os.environ["DRAFTMK_CUBIN_DIR"] == str(megakernel)


def test_an_explicit_library_wins_over_the_wheel_with_its_cubins_beside_it(megakernel, monkeypatch, tmp_path_factory):
    lib = tmp_path_factory.mktemp("build") / "libdraftmk.so"
    lib.touch()
    monkeypatch.setenv("EAGLE_HARNESS_DRAFTMK_LIB", str(lib))
    assert draftmk.library() == str(lib)
    assert os.environ["DRAFTMK_CUBIN_DIR"] == str(lib.parent)


def test_a_missing_explicit_library_is_no_megakernel(megakernel, monkeypatch):
    monkeypatch.setenv("EAGLE_HARNESS_DRAFTMK_LIB", "/nonexistent/libdraftmk.so")
    assert draftmk.library() is None


def test_the_wheel_pairs_its_library_with_its_own_cubins(megakernel, monkeypatch):
    monkeypatch.setenv("DRAFTMK_CUBIN_DIR", "/some/other/build")
    draftmk.library()
    assert os.environ["DRAFTMK_CUBIN_DIR"] == str(megakernel)


def test_no_wheel_means_no_megakernel(megakernel, monkeypatch):
    monkeypatch.setitem(sys.modules, "eagle_harness_draftmk", None)  # import raises ImportError
    assert draftmk.library() is None


def test_other_gpus_have_no_megakernel(megakernel, monkeypatch):
    monkeypatch.setattr(arch, "draftmk", lambda device=None: False)
    assert draftmk.library() is None


@pytest.mark.parametrize(
    "env, found, backend",
    [(None, True, "draftmk"), (None, False, None), ("off", True, None), ("graphs", True, "graphs"),
     ("draftmk", True, "draftmk")],
)
def test_draftmk_is_the_default_backend_where_it_is_found(monkeypatch, env, found, backend):
    pytest.importorskip("sglang")
    from eagle_harness.sglang import draft_round

    monkeypatch.setattr(draftmk, "library", lambda: "libdraftmk.so" if found else None)
    if env is None:
        monkeypatch.delenv("EAGLE_DRAFT_ROUND", raising=False)
    else:
        monkeypatch.setenv("EAGLE_DRAFT_ROUND", env)
    assert draft_round.round_backend() == backend


def test_asking_for_draftmk_without_it_fails(monkeypatch):
    pytest.importorskip("sglang")
    from eagle_harness.sglang import draft_round

    monkeypatch.setattr(draftmk, "library", lambda: None)
    monkeypatch.setenv("EAGLE_DRAFT_ROUND", "draftmk")
    with pytest.raises(RuntimeError, match="eagle-harness-draftmk"):
        draft_round.round_backend()
