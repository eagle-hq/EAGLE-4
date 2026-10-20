# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
"""Exercise the publication guard with registry responses, without a Docker daemon."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "ghcr.io/eagle-hq/eagle-4-sglang:0.1.0"


@pytest.mark.parametrize(
    ("status", "message", "absent"),
    [
        (0, "Name: " + IMAGE, False),
        (1, "ERROR: " + IMAGE + ": not found", True),
        (1, "ERROR: unexpected status: 401 Unauthorized", False),
        (1, "ERROR: unexpected status: 403 Forbidden", False),
        (1, "ERROR: unexpected status: 429 Too Many Requests", False),
        (1, "ERROR: unexpected status: 500 Internal Server Error", False),
        (1, "ERROR: unexpected status: 503 Service Unavailable", False),
        (1, "ERROR: failed to fetch oauth token: 404 Not Found", False),
        (1, "ERROR: dial tcp: network is unreachable", False),
        (1, "ERROR: context deadline exceeded", False),
        (1, "ERROR: TLS handshake timeout", False),
        (1, "ERROR: another/image:0.1.0: not found", False),
        (1, "ERROR: " + IMAGE + ": not found\nERROR: authentication failed", False),
        (1, "", False),
        (127, "docker: command not found", False),
    ],
)
def test_only_confirmed_tag_absence_allows_publication(tmp_path, status, message, absent):
    docker = tmp_path / "docker"
    docker.write_text(
        '#!/bin/sh\n'
        'test "$1 $2 $3 $4" = "buildx imagetools inspect $EXPECTED_IMAGE" || exit 98\n'
        'printf "%s\\n" "$RESPONSE" >&2\n'
        'exit "$STATUS"\n'
    )
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "tools/check_image_tag.sh"), IMAGE],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "EXPECTED_IMAGE": IMAGE,
            "RESPONSE": message,
            "STATUS": str(status),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is absent, result.stdout + result.stderr
