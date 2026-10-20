# syntax=docker/dockerfile:1
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

# Sketch only. Base and final image need clean-build qualification.
FROM lmsysorg/sglang:nightly-dev-cu13-20261018-b234b5c3
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
    && python3 -m pip install --no-cache-dir --extra-index-url "$EAGLE_CUDA_INDEX" \
      --extra-index-url "$EAGLE_TORCH_CUDA_INDEX" "sglang==$EAGLE_SGLANG_VERSION" 'flashinfer-python>=0.7.2' \
    && python3 -m pip install --no-cache-dir --no-deps --force-reinstall /src \
    && python3 -m pip check \
    && eagle-harness calibrate --help
