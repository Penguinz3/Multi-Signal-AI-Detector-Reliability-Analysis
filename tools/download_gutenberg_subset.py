from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MIRROR = "https://gutenberg.pglaf.org/cache/epub/{book_id}/pg{book_id}.txt"


def select_books(catalog: Path, limit: int) -> list[dict]:
    with gzip.open(catalog, "rt", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["Type"] == "Text"
            and row["Language"] == "en"
            and row["Authors"] not in {"", "Anonymous", "Various"}
        ]
    rows.sort(key=lambda row: hashlib.sha256(row["Text#"].encode()).digest())
    selected, authors = [], set()
    for row in rows:
        author = row["Authors"].strip()
        if author in authors:
            continue
        authors.add(author)
        selected.append(row)
        if len(selected) == limit:
            break
    return selected


def download(row: dict, output: Path) -> dict:
    book_id = row["Text#"]
    destination = output / f"pg{book_id}.txt"
    if destination.is_file() and destination.stat().st_size:
        return {"book_id": book_id, "status": "existing", "bytes": destination.stat().st_size}
    request = urllib.request.Request(
        MIRROR.format(book_id=book_id),
        headers={"User-Agent": "FPRINT academic corpus preparation"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
        destination.write_bytes(payload)
        return {"book_id": book_id, "status": "downloaded", "bytes": len(payload)}
    except Exception as error:
        return {"book_id": book_id, "status": "failed", "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--books", type=int, default=2_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_books(args.catalog, args.books)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, row, args.output_dir) for row in selected]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 100 == 0:
                print(f"{len(results)}/{len(selected)}", flush=True)
    manifest = {
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "mirror": MIRROR,
        "selection": "lowest SHA-256-ranked book ID; one English text per non-anonymous author",
        "requested": len(selected),
        "downloaded_or_existing": sum(result["status"] != "failed" for result in results),
        "failed": [result for result in results if result["status"] == "failed"],
        "books": selected,
    }
    (args.output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print({key: manifest[key] for key in ("requested", "downloaded_or_existing")})


if __name__ == "__main__":
    main()
