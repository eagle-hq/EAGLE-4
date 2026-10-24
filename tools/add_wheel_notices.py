# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Add license or notice files to a built third-party wheel under a new build tag.

Use it when a wheel we built from upstream source omits notices its source tree
carries (sglang-kernel's THIRDPARTYNOTICES.txt). The files go to
``<dist-info>/licenses/`` with ``License-File`` metadata. WHEEL gains the build
tag and RECORD is regenerated. Every other member keeps its bytes, mode and
timestamp; the result is verified before the receipt is written.

    python tools/add_wheel_notices.py <wheel> <destination> --build 1eagle NAME=PATH...
"""

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def add_notices(source: Path, destination: Path, build: str, notices: dict[str, Path]) -> dict:
    parts = source.name.removesuffix(".whl").split("-")
    if len(parts) != 5:
        raise ValueError(f"expected a wheel without a build tag: {source.name}")
    if not re.fullmatch(r"\d[A-Za-z0-9_.]*", build):
        raise ValueError(f"a build tag starts with a digit: {build!r}")
    if not notices or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in notices):
        raise ValueError("give at least one NAME=PATH with a plain file name")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / ("-".join([*parts[:2], build, *parts[2:]]) + ".whl")
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(source) as original:
        names = original.namelist()
        info_dir = f"{parts[0]}-{parts[1]}.dist-info"
        metadata, wheel, record = (f"{info_dir}/{name}" for name in ("METADATA", "WHEEL", "RECORD"))
        if not {metadata, wheel, record} <= set(names):
            raise ValueError(f"missing {info_dir} METADATA, WHEEL or RECORD")
        added = {f"{info_dir}/licenses/{name}": path.read_bytes() for name, path in notices.items()}
        if added.keys() & set(names):
            raise ValueError(f"wheel already contains {sorted(added.keys() & set(names))}")
        wheel_text = original.read(wheel).decode()
        if re.search(r"^Build:", wheel_text, re.M):
            raise ValueError("wheel already has a build tag")
        meta_text = original.read(metadata).decode()
        head, separator, body = meta_text.partition("\n\n")
        # A METADATA without a description has no blank line: keep the new headers out of a body.
        head = head.rstrip("\n") + "".join(f"\nLicense-File: {name}" for name in notices)
        changes = {
            metadata: (head + (separator + body if separator else "\n")).encode(),
            wheel: (wheel_text.rstrip("\n") + f"\nBuild: {build}\n").encode(),
        }
        payload, rows = {}, []
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as rebuilt:
            for name in names:
                if name == record:
                    continue
                data = changes.get(name, None)
                if data is None:
                    data = original.read(name)
                    payload[name] = hashlib.sha256(data).hexdigest()
                rows.append([name, record_hash(data), str(len(data))])
                rebuilt.writestr(copy.copy(original.getinfo(name)), data)
            template = original.getinfo(metadata)
            for name, data in added.items():
                info = zipfile.ZipInfo(name, date_time=template.date_time)
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                rows.append([name, record_hash(data), str(len(data))])
                rebuilt.writestr(info, data)
            rows.append([record, "", ""])
            buffer = io.StringIO(newline="")
            csv.writer(buffer).writerows(rows)
            rebuilt.writestr(copy.copy(original.getinfo(record)), buffer.getvalue())
        attributes = {info.filename: (info.external_attr, info.date_time) for info in original.infolist()}
    with zipfile.ZipFile(output) as rebuilt:
        for info in rebuilt.infolist():
            if info.filename in attributes and (info.external_attr, info.date_time) != attributes[info.filename]:
                raise ValueError(f"archive attributes changed: {info.filename}")
        for name, digest in payload.items():
            if hashlib.sha256(rebuilt.read(name)).hexdigest() != digest:
                raise ValueError(f"payload changed: {name}")
        listed = set()
        for name, digest, size in csv.reader(io.StringIO(rebuilt.read(record).decode())):
            listed.add(name)
            if name != record and (digest, size) != (record_hash(rebuilt.read(name)), str(len(rebuilt.read(name)))):
                raise ValueError(f"RECORD mismatch: {name}")
        if listed != set(rebuilt.namelist()):
            raise ValueError("RECORD does not list exactly the archive members")
    receipt = {
        "input": source.name,
        "input_sha256": sha256(source),
        "output": output.name,
        "output_sha256": sha256(output),
        "build": build,
        "added_members": {name: hashlib.sha256(data).hexdigest() for name, data in added.items()},
        "changed_members": [metadata, wheel, record],
        "unchanged_payload_members": len(payload),
        "record_verified": True,
        "archive_modes_and_timestamps_preserved": True,
    }
    (destination / f"{output.name}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--build", required=True, help="wheel build tag, for example 1eagle")
    parser.add_argument("notices", nargs="+", metavar="NAME=PATH")
    args = parser.parse_args()
    notices = {}
    for item in args.notices:
        name, separator, path = item.partition("=")
        if not separator:
            parser.error(f"expected NAME=PATH: {item}")
        notices[name] = Path(path)
    print(json.dumps(add_notices(args.source, args.destination, args.build, notices), indent=2))


if __name__ == "__main__":
    main()
