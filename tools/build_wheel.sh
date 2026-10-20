#!/bin/bash
# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

# Build the eagle-harness sdist and wheels from a source tree; nothing is uploaded.
#   tools/build_wheel.sh <source tree> <out dir> <cuda home>...
# One native library per CUDA major (each <cuda home> gives _native/cu<major>/, see eagle_harness/kernels/build.py)
# goes into the binary wheel, which auditwheel retags manylinux. The sdist holds the CUDA source only; the pure wheel
# built from it has no native library, so nvcc builds one on first use (the JIT fallback).
# Out: <out>/eagle_harness-<v>.tar.gz, eagle_harness-<v>-py3-none-manylinux_*.whl, pure/eagle_harness-<v>-py3-none-any.whl,
# SHA256SUMS, build.txt (toolkits, glibc symbols, wheel contents). Needs python with `build`, `auditwheel`, `patchelf`.
set -euo pipefail
SRC=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
[ $# -ge 1 ] || { echo "usage: $0 <source tree> <out dir> <cuda home>..." >&2; exit 2; }
PY=${PYTHON:-python3}
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/pure"
cp -r "$SRC/." "$WORK/src"; rm -rf "$WORK/src/eagle_harness/kernels/_native" "$WORK/src/build" "$WORK/src"/*.egg-info
cd "$WORK/src"
{
  echo "source: $SRC $(cat REVISION 2>/dev/null || true)"
  for cuda in "$@"; do
    "$cuda/bin/nvcc" --version | tail -2
    "$PY" -m eagle_harness.kernels.build prebuild --cuda-home "$cuda"
  done
  ls -l eagle_harness/kernels/_native/*/
  for lib in eagle_harness/kernels/_native/*/*.so; do
    echo "$lib: glibc $(objdump -T "$lib" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)," \
      "needs $(objdump -p "$lib" | awk '/NEEDED/ {print $2}' | tr '\n' ' ')"
  done
} > "$OUT/build.txt" 2>&1
# The sdist and the binary wheel are both built from this tree (`build` alone would build the wheel from the sdist).
"$PY" -m build --sdist --outdir "$WORK/dist" . >> "$OUT/build.txt" 2>&1
"$PY" -m build --wheel --outdir "$WORK/dist" . >> "$OUT/build.txt" 2>&1
"$PY" -m auditwheel show "$WORK"/dist/*.whl >> "$OUT/build.txt" 2>&1
"$PY" -m auditwheel repair --wheel-dir "$OUT" "$WORK"/dist/*.whl >> "$OUT/build.txt" 2>&1
cp "$WORK"/dist/*.tar.gz "$OUT/"
mkdir "$WORK/sdist" && tar xzf "$OUT"/eagle_harness-*.tar.gz -C "$WORK/sdist"
"$PY" -m build --wheel --outdir "$OUT/pure" "$WORK"/sdist/eagle_harness-* >> "$OUT/build.txt" 2>&1
cd "$OUT"
{
  echo "== sdist: native libraries (expect none)"; tar tzf eagle_harness-*.tar.gz | grep '\.so$' || echo none
  echo "== wheel contents"; "$PY" -m zipfile -l eagle_harness-*manylinux*.whl | grep -E '\.so|\.cu|entry_points|METADATA'
  echo "== pure wheel: native libraries (expect none)"; "$PY" -m zipfile -l pure/*.whl | grep '\.so' || echo none
  ls -l eagle_harness-* pure/
} >> build.txt 2>&1
sha256sum eagle_harness-* pure/* > SHA256SUMS
cat SHA256SUMS
