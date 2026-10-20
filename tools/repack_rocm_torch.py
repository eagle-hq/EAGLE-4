# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Correct the MI35x upstream image's Torch/Triton wheel metadata.

The upstream ROCm image uses these Torch binaries with Triton 3.7, but its
metadata rewrite drops the Requires-Dist header. Preserve that runtime pairing
with a real requirement, a distinct wheel build tag, and complete RECORD hashes.
No Python module or native library is changed.
"""

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

TORCH = "torch-2.9.1+rocm7.2.0.lw.git7e1940d4"
OLD = "Requires-Dist: triton==3.5.1+rocm7.2.0.gita272dfa8;"
NEW = "Requires-Dist: triton==3.7.0+amd.rocm7.2.0.git89002410;"


def repack(source: Path, destination: Path) -> dict:
    expected = f"{TORCH}-cp310-cp310-linux_x86_64.whl"
    if source.name != expected:
        raise ValueError(f"expected {expected}, got {source.name}")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{TORCH}-1eagle-cp310-cp310-linux_x86_64.whl"
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(source) as original:
        names = original.namelist()
        metadata = next(n for n in names if n.endswith(".dist-info/METADATA"))
        record = next(n for n in names if n.endswith(".dist-info/RECORD"))
        wheel = next(n for n in names if n.endswith(".dist-info/WHEEL"))
        text = original.read(metadata).decode()
        if text.count(OLD) != 1:
            raise ValueError("expected exactly one original Torch Triton requirement")
        changes = {
            metadata: text.replace(OLD, NEW).encode(),
            wheel: original.read(wheel).rstrip() + b"\nBuild: 1eagle\n",
        }
        rows = []
        payload = {}
        attributes = {info.filename: (info.external_attr, info.date_time) for info in original.infolist()}
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as rebuilt:
            for name in names:
                if name == record:
                    continue
                data = changes[name] if name in changes else original.read(name)
                digest = hashlib.sha256(data).digest()
                encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
                rows.append([name, f"sha256={encoded}", str(len(data))])
                if name not in changes:
                    payload[name] = digest.hex()
                rebuilt.writestr(copy.copy(original.getinfo(name)), data, compresslevel=1)
            rows.append([record, "", ""])
            buffer = io.StringIO(newline="")
            csv.writer(buffer).writerows(rows)
            rebuilt.writestr(copy.copy(original.getinfo(record)), buffer.getvalue(), compresslevel=1)
    with zipfile.ZipFile(output) as rebuilt:
        for info in rebuilt.infolist():
            if (info.external_attr, info.date_time) != attributes[info.filename]:
                raise ValueError(f"archive attributes changed: {info.filename}")
        for name, digest in payload.items():
            if hashlib.sha256(rebuilt.read(name)).hexdigest() != digest:
                raise ValueError(f"payload changed: {name}")
        for name, digest, size in csv.reader(io.StringIO(rebuilt.read(record).decode())):
            if name == record:
                continue
            data = rebuilt.read(name)
            actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            if digest != f"sha256={actual}" or size != str(len(data)):
                raise ValueError(f"RECORD mismatch: {name}")

    def sha256(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    receipt = {
        "input": source.name,
        "input_sha256": sha256(source),
        "output": output.name,
        "output_sha256": sha256(output),
        "old_requirement": OLD,
        "new_requirement": NEW,
        "changed_members": [metadata, wheel, record],
        "unchanged_payload_members": len(payload),
        "record_verified": True,
        "archive_modes_and_timestamps_preserved": True,
    }
    (destination / "torch-repack.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(repack(args.source, args.destination), indent=2))
