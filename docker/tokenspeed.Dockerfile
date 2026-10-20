# syntax=docker/dockerfile:1
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

# Sketch only. Supply an approved Python 3.12 / CUDA 13.0 TokenSpeed base.
# Scheduler and kernel wheels must be built from the matching release source.
# This file has not been built or served.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY pyproject.toml README.md LICENSE NOTICE TRADEMARKS.md THIRD_PARTY_NOTICES.md /src/
COPY LICENSES/ /src/LICENSES/
COPY eagle_harness/ /src/eagle_harness/
COPY docs/release.env /opt/eagle-release.env
# Keep the release closure separate from unrelated base-image Python packages.
RUN python3 -m venv --without-pip /opt/eagle \
    && python3 -m pip --python /opt/eagle install --no-cache-dir --upgrade pip
ENV VIRTUAL_ENV=/opt/eagle
ENV PATH="/opt/eagle/bin:${PATH}"
RUN . /opt/eagle-release.env \
    && python3 -c 'import sys; assert sys.version_info[:2] == (3, 12)' \
    && test -n "$EAGLE_TOKENSPEED_AUX_REQUIREMENTS" \
    && python3 -m pip install --no-cache-dir --extra-index-url "$EAGLE_CUDA_INDEX" \
      --extra-index-url "$EAGLE_TORCH_CUDA_INDEX" \
      "tokenspeed==$EAGLE_TOKENSPEED_VERSION" $EAGLE_TOKENSPEED_AUX_REQUIREMENTS \
    && python3 -m pip install --no-cache-dir --no-deps --force-reinstall /src \
    && python3 -m pip check
# Run eagle-harness-check with NVIDIA GPU access, then follow
# docs/quickstart-tokenspeed.md. Hook imports may require a visible GPU.
