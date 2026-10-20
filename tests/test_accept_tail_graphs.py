# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""AcceptTailGraphs on CPU: a key's first round and keys past the cap run eagerly."""

from unittest import mock

import pytest

pytest.importorskip("sglang")
from eagle_harness.sglang import verify_step  # noqa: E402


def test_a_new_key_runs_eagerly_once():
    graphs = verify_step.AcceptTailGraphs()
    body = mock.Mock(return_value="out")
    assert graphs.run(("k",), body) == "out"
    body.assert_called_once_with()


def test_keys_past_the_cap_stay_eager():
    graphs = verify_step.AcceptTailGraphs()
    body = mock.Mock(return_value="out")
    with mock.patch.object(verify_step, "MAX_TAIL_GRAPHS", 0):
        for _ in range(3):
            assert graphs.run(("k",), body) == "out"
    assert body.call_count == 3
