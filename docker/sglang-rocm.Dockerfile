# syntax=docker/dockerfile:1
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

# Sketch only. Supply an approved Python 3.10 / ROCm 7.2 MI355X base
# with matching PyTorch, Triton, AITER and FlyDSL already installed.
# No default tag is invented. This file has not been built or served.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY pyproject.toml README.md LICENSE NOTICE TRADEMARKS.md THIRD_PARTY_NOTICES.md /src/
COPY LICENSES/ /src/LICENSES/
COPY eagle_harness/ /src/eagle_harness/
COPY docs/release.env /opt/eagle-release.env
RUN . /opt/eagle-release.env \
    && python3 -c 'import sys, torch; assert sys.version_info[:2] == (3, 10); assert torch.version.hip' \
    && python3 -m pip install --no-cache-dir --no-deps --extra-index-url "$EAGLE_ROCM_INDEX" \
      "sglang==$EAGLE_ROCM_VERSION" \
    && python3 -m pip install --no-cache-dir --no-deps --force-reinstall /src \
    && eagle-harness calibrate --help
# Run eagle-harness-check with /dev/kfd and /dev/dri available, then follow
# docs/quickstart-amd.md. A CPU build-time import is not GPU qualification.
