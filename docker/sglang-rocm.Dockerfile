# syntax=docker/dockerfile:1
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

# lmsysorg/sglang:v0.5.20-rocm720-mi35x, Ubuntu 22.04 / CPython 3.10.
# Build with --build-context rocm-wheels=/path/to/rocm/wheels.
# Image construction/runtime still require separate container qualification.
FROM lmsysorg/sglang@sha256:ef84c44150da0f90495953458e7386350ffc50a37b2b9ac6802d9a94c454c6dd
COPY docker/rocm/constraints.txt docker/rocm/requirements.lock /opt/eagle/rocm/
# Isolate the locked closure from the base image's editable SGLang and packages.
RUN --mount=type=bind,from=rocm-wheels,target=/wheels \
    python3 -c 'import sys; assert sys.version_info[:2] == (3, 10)' \
    && python3 -m venv /opt/eagle/venv \
    && /opt/eagle/venv/bin/python -m pip install --no-cache-dir --no-index \
      --find-links /wheels --require-hashes -r /opt/eagle/rocm/requirements.lock \
    && /opt/eagle/venv/bin/python -m pip check
ENV PATH="/opt/eagle/venv/bin:${PATH}" \
    PYTHONPATH="" \
    PYTHONNOUSERSITE=1
WORKDIR /opt/eagle
CMD ["bash"]
# Run eagle-harness-check with /dev/kfd and /dev/dri available, then follow
# docs/quickstart-amd.md. A CPU build-time install is not GPU qualification.
