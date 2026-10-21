# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Write a static PEP 503 index: <root>/files/<artifact> and <root>/simple/<project>/index.html.

    python mk_index.py <root> <artifact>...

Links default to ../../files/<name>#sha256=<hex>. With --base-url, links instead
point to that HTTPS release-asset directory. Only simple/ needs deploying in this
mode; files/ is a local staging cache, not part of the Pages site.

Each project page lists every staged artifact of that project. Requires-Python
comes from the artifact metadata. Re-running adds artifacts; released filenames
must stay byte-identical.
"""

import argparse
import hashlib
import html
import re
import shutil
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path
from urllib.parse import quote, urlsplit


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def metadata(path: Path) -> dict:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as z:
            text = z.read(next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))).decode()
    else:
        with tarfile.open(path) as t:
            member = next(m for m in t.getmembers() if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO"))
            text = t.extractfile(member).read().decode()
    msg = Parser().parsestr(text, headersonly=True)
    return {"name": msg["Name"], "version": msg["Version"], "requires_python": msg.get("Requires-Python")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(root: Path, artifacts: list[Path], base_url: str | None = None) -> None:
    if base_url is not None:
        url = urlsplit(base_url)
        if (
            url.scheme != "https"
            or not url.netloc
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
        ):
            sys.exit("--base-url must be an HTTPS URL without credentials, query or fragment")
        base_url = base_url.rstrip("/")
    files = root / "files"
    files.mkdir(parents=True, exist_ok=True)
    for a in artifacts:
        dst = files / a.name
        if dst.exists() and sha256(dst) != sha256(a):
            sys.exit(f"{dst} exists with other content: a release artifact never changes; bump the version")
        if not dst.exists():
            shutil.copyfile(a, dst)
    projects: dict[str, list[tuple[Path, dict, str]]] = {}
    for f in sorted(files.iterdir()):
        meta = metadata(f)
        projects.setdefault(normalize(meta["name"]), []).append((f, meta, sha256(f)))
    simple = root / "simple"
    simple.mkdir(parents=True, exist_ok=True)  # the root page is written even before any artifact is staged
    for project, entries in projects.items():
        lines = [
            f"<!DOCTYPE html>\n<html><head><title>Links for {project}</title></head><body>",
            f"<h1>Links for {project}</h1>",
        ]
        for f, meta, sha in entries:
            rp = f' data-requires-python="{html.escape(meta["requires_python"])}"' if meta["requires_python"] else ""
            href = f"{base_url or '../../files'}/{quote(f.name)}#sha256={sha}"
            lines.append(f'<a href="{html.escape(href)}"{rp}>{html.escape(f.name)}</a><br/>')
            print(f"{project} {meta['version']} {f.name} {f.stat().st_size} sha256={sha}")
        (simple / project).mkdir(parents=True, exist_ok=True)
        (simple / project / "index.html").write_text("\n".join(lines + ["</body></html>\n"]))
    links = "\n".join(f'<a href="{p}/">{p}</a><br/>' for p in sorted(projects))
    (simple / "index.html").write_text(f"<!DOCTYPE html>\n<html><body>\n{links}\n</body></html>\n")
    (root / "SHA256SUMS").write_text("".join(f"{sha}  files/{f.name}\n" for e in projects.values() for f, _, sha in e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("artifacts", type=Path, nargs="*")
    parser.add_argument("--base-url", help="HTTPS directory containing the release assets")
    args = parser.parse_args()
    main(args.root, args.artifacts, args.base_url)
