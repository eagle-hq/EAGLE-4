# syntax=docker/dockerfile:1
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

# The base is pinned by digest: the published image's base layers.
FROM vllm/vllm-openai:nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294@sha256:a8d84c552462d1c60da4d7fc97f8f8f48a8fbfbf6820f73477fa0efe2571bf9d
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
      --extra-index-url "$EAGLE_TORCH_CUDA_INDEX" "vllm==$EAGLE_VLLM_VERSION" \
    && python3 -m pip install --no-cache-dir --no-deps --force-reinstall /src \
    && python3 -m pip check \
    && eagle-harness calibrate --help
