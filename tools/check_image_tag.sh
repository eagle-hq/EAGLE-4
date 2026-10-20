#!/usr/bin/env bash
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
set -euo pipefail

image=${1:?usage: check_image_tag.sh IMAGE:TAG}
if output=$(docker buildx imagetools inspect "$image" 2>&1); then
  echo "Image tag already exists: $image; publish a new version instead of replacing it" >&2
  exit 1
else
  status=$?
fi

# Buildx's resolver reports a manifest HTTP 404 with this exact image-scoped error.
# Unknown formats fail closed; a token-endpoint 404 is not tag absence.
if [[ "$status" == 1 && "$output" == "ERROR: $image: not found" ]]; then
  printf 'Confirmed absent image tag: %s\n' "$image"
  exit 0
fi
printf 'Cannot confirm image tag absence: %s\n%s\n' "$image" "$output" >&2
exit 1
