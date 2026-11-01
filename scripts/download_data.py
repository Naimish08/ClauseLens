#!/usr/bin/env python3
"""Download and unpack the LegalBench-RAG corpus and benchmark files.

The upstream project publishes its data as a Dropbox folder, not as a pip
package or an HF dataset, so this script is the single entry point for getting
the data onto a fresh machine.

    python scripts/download_data.py            # download (idempotent) + unpack
    python scripts/download_data.py --offline  # use an existing zip only

After a successful run the layout is:

    data/raw/legalbenchrag.zip     archived download (gitignored)
    data/raw/MANIFEST.json         url, sha256, size, extracted file counts
    data/benchmarks/*.json         cuad, maud, privacy_qa, contractnli
    data/corpus/<benchmark>/*.txt  raw contract text
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_PATH = RAW_DIR / "legalbenchrag.zip"
MANIFEST_PATH = RAW_DIR / "MANIFEST.json"

URL = (
    "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/"
    "AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=1"
)
USER_AGENT = "ClauseLens/0.1 (research repro)"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(dest: Path, url: str = URL) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"downloading {url}")
    with urllib.request.urlopen(request, timeout=300) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        next_mark = 10
        while chunk := response.read(1 << 20):
            out.write(chunk)
            written += len(chunk)
            if total:
                pct = 100 * written / total
                if pct >= next_mark:
                    print(f"  {written / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)")
                    next_mark += 10
            elif written // (10 << 20) != (written - len(chunk)) // (10 << 20):
                print(f"  {written / 1e6:6.1f} MB")
    print()
    tmp.replace(dest)


def _safe_parts(name: str) -> tuple[str, ...]:
    """Normalise an archive member name and reject traversal.

    The upstream Dropbox archive contains a literal ``"/"`` root entry, so
    absolute-looking names are stripped of leading separators rather than
    rejected outright; ``..`` segments are still fatal.
    """
    name = name.replace("\\", "/")
    if ".." in name.split("/"):
        raise RuntimeError(f"refusing to extract path with '..': {name!r}")
    return tuple(part for part in name.split("/") if part not in ("", "."))


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> int:
    """Extract members under ``destination``; returns the file count."""
    destination = destination.resolve()
    count = 0
    for member in archive.infolist():
        parts = _safe_parts(member.filename)
        if not parts:
            continue
        target = destination.joinpath(*parts)
        if not target.resolve().is_relative_to(destination):
            raise RuntimeError(f"refusing to extract outside data dir: {member.filename}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        count += 1
    return count


def unpack(archive_path: Path, destination: Path) -> int:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for required in ("corpus/", "benchmarks/"):
            if not any(name.lstrip("/").startswith(required) for name in names):
                raise RuntimeError(f"archive is missing {required}; got {names[:5]}")
        return safe_extract(archive, destination)


def count_files(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        counts[directory.name] = sum(1 for _ in directory.rglob("*.txt"))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--offline", action="store_true", help="fail instead of downloading")
    parser.add_argument("--force", action="store_true", help="re-download even if the zip exists")
    args = parser.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists() and not args.force:
        print(f"using cached archive {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")
    else:
        if args.offline:
            print(f"error: {ZIP_PATH} not found and --offline was set", file=sys.stderr)
            return 1
        download(ZIP_PATH)

    if not zipfile.is_zipfile(ZIP_PATH):
        print(f"error: {ZIP_PATH} is not a valid zip archive", file=sys.stderr)
        return 1

    unpack(ZIP_PATH, DATA_DIR)

    manifest = {
        "source_url": URL,
        "downloaded_zip": str(ZIP_PATH.relative_to(REPO_ROOT)),
        "zip_sha256": sha256_of(ZIP_PATH),
        "zip_bytes": ZIP_PATH.stat().st_size,
        "unpacked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus_files_per_benchmark": count_files(DATA_DIR / "corpus"),
        "benchmarks": sorted(p.name for p in (DATA_DIR / "benchmarks").glob("*.json")),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total = sum(manifest["corpus_files_per_benchmark"].values())
    print(f"unpacked {total} corpus files -> {DATA_DIR / 'corpus'}")
    print(f"benchmarks: {', '.join(manifest['benchmarks'])}")
    print(f"zip sha256: {manifest['zip_sha256']}")
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
