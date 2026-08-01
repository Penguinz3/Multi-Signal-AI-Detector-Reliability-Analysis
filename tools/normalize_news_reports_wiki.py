from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(]*[A-Z0-9])")
ARTICLE_HEADER = re.compile(r"^\s*=\s+([^=].*?)\s+=\s*$")
SOURCE_REVISIONS = {
    "cnn_dailymail": "abisee/cnn_dailymail@690bb95a2ac2c5a99d7bde63ac1401539ddd3967",
    "govreport": "launch/gov_report@c0b3f7bd48f480f34a572beff5f110fc6c0f11c4",
    "wikitext_103": "Salesforce/wikitext@3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e",
}


def clean_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def passage_chunks(text: object, minimum: int = 100, target: int = 220, maximum: int = 300) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]
    if len(sentences) == 1:
        words = text.split()
        return [" ".join(words[start:start + maximum]) for start in range(0, len(words), maximum)
                if len(words[start:start + maximum]) >= minimum]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if current and count + len(words) > maximum:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.extend(words)
        count += len(words)
        if count >= target:
            chunks.append(" ".join(current))
            current, count = [], 0
    if count >= minimum:
        chunks.append(" ".join(current))
    elif current and chunks and len(chunks[-1].split()) + count <= maximum:
        chunks[-1] += " " + " ".join(current)
    return [chunk for chunk in chunks if minimum <= len(chunk.split()) <= maximum]


def parquet_rows(paths: Iterable[Path], columns: list[str]) -> Iterator[dict]:
    import pyarrow.parquet as parquet

    for path in paths:
        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(columns=columns, batch_size=2048):
            yield from batch.to_pylist()


def deterministic_sample(rows: Iterable[dict], key: str, limit: int) -> list[dict]:
    ranked = sorted(rows, key=lambda row: hashlib.sha256(str(row[key]).encode()).digest())
    return ranked[:limit]


def normalize_cnn(path: Path, limit: int) -> list[dict]:
    rows = []
    for row in parquet_rows([path], ["id", "article"]):
        chunks = passage_chunks(row["article"])
        if not chunks:
            continue
        article_id = str(row["id"])
        index = int.from_bytes(hashlib.sha256(article_id.encode()).digest()[:4], "big") % len(chunks)
        rows.append({"record_id": f"cnn:{article_id}", "article_id": article_id, "text": chunks[index]})
    return deterministic_sample(rows, "record_id", limit)


def normalize_govreport(path: Path, limit: int) -> list[dict]:
    rows = []
    for row in parquet_rows([path], ["id", "document"]):
        chunks = passage_chunks(row["document"])
        if not chunks:
            continue
        report_id = str(row["id"])
        index = int.from_bytes(hashlib.sha256(report_id.encode()).digest()[:4], "big") % len(chunks)
        rows.append({"record_id": f"gov:{report_id}", "report_id": report_id, "text": chunks[index]})
    return deterministic_sample(rows, "record_id", limit)


def wikitext_articles(paths: Iterable[Path]) -> Iterator[tuple[str, str]]:
    title = ""
    body: list[str] = []
    for row in parquet_rows(paths, ["text"]):
        line = str(row["text"] or "").strip()
        match = ARTICLE_HEADER.fullmatch(line)
        if match:
            if title and body:
                yield title, "\n".join(body)
            title, body = match.group(1).strip(), []
        elif title and line:
            body.append(line)
    if title and body:
        yield title, "\n".join(body)


def normalize_wikitext(paths: Iterable[Path], limit: int) -> list[dict]:
    rows = []
    used: set[str] = set()
    for ordinal, (title, text) in enumerate(wikitext_articles(paths)):
        chunks = passage_chunks(text)
        if not chunks:
            continue
        identity = hashlib.sha256(f"{title}:{ordinal}".encode()).hexdigest()[:20]
        if identity in used:
            raise RuntimeError(f"Duplicate WikiText article identity: {identity}")
        used.add(identity)
        index = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big") % len(chunks)
        rows.append({"record_id": f"wiki:{identity}", "article_id": identity,
                     "title": title, "text": chunks[index]})
    return deterministic_sample(rows, "record_id", limit)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No eligible rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_all(raw: Path, output: Path, limit: int = 8_000) -> Mapping[str, int]:
    inputs = {
        "cnn_dailymail": [raw / "cnn_dailymail_3.0.0_test.parquet"],
        "govreport": [raw / "gov_report_train_0000.parquet"],
        "wikitext_103": [
            raw / "wikitext_103_raw_train_0000.parquet",
            raw / "wikitext_103_raw_train_0001.parquet",
        ],
    }
    missing = [str(path) for paths in inputs.values() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw inputs: " + ", ".join(missing))
    normalized = {
        "cnn_dailymail": normalize_cnn(inputs["cnn_dailymail"][0], limit),
        "govreport": normalize_govreport(inputs["govreport"][0], limit),
        "wikitext_103": normalize_wikitext(inputs["wikitext_103"], limit),
    }
    for corpus, rows in normalized.items():
        if len(rows) < 6_000:
            raise RuntimeError(f"{corpus} yielded only {len(rows)} eligible records")
        write_csv(output / f"{corpus}.csv", rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "word_range": [100, 300],
        "selection": "SHA-256 rank; one deterministically selected passage per source document",
        "sources": {
            corpus: {
                "revision": SOURCE_REVISIONS[corpus],
                "files": [{"name": path.name, "sha256": file_sha256(path)} for path in paths],
                "normalized_records": len(normalized[corpus]),
            }
            for corpus, paths in inputs.items()
        },
    }
    (output / "news_reports_wiki.provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {corpus: len(rows) for corpus, rows in normalized.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8_000)
    args = parser.parse_args()
    print(dict(normalize_all(args.raw_dir, args.output_dir, args.limit)))


if __name__ == "__main__":
    main()
