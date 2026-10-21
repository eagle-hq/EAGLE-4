# syntax=docker/dockerfile:1
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

# The base is pinned by digest, and docker/sglang/requirements.lock pins every runtime dependency by wheel hash.
# The engine and eagle-harness install on top of that closure without further dependency resolution.
FROM lmsysorg/sglang:nightly-dev-cu13-20261018-b234b5c3@sha256:bfef46e5a45a815f8c2f88a722980baf47b71a8689b89c6cc738f596ceeaf47c
COPY pyproject.toml README.md LICENSE NOTICE TRADEMARKS.md THIRD_PARTY_NOTICES.md /src/
COPY LICENSES/ /src/LICENSES/
COPY eagle_harness/ /src/eagle_harness/
COPY docs/release.env /opt/eagle-release.env
COPY docker/sglang/requirements.lock /opt/eagle-requirements.lock
# Keep the release closure separate from unrelated base-image Python packages.
RUN python3 -m venv --without-pip /opt/eagle \
    && python3 -m pip --python /opt/eagle install --no-cache-dir --upgrade pip
ENV VIRTUAL_ENV=/opt/eagle
ENV PATH="/opt/eagle/bin:${PATH}"
RUN . /opt/eagle-release.env \
    && python3 -m pip install --no-cache-dir --require-hashes -r /opt/eagle-requirements.lock \
    && python3 -m pip install --no-cache-dir --no-deps --index-url "$EAGLE_CUDA_INDEX" \
      "sglang==$EAGLE_SGLANG_VERSION" \
    && python3 -m pip install --no-cache-dir --no-deps --force-reinstall /src \
    && python3 -m pip check \
    && eagle-harness calibrate --help
