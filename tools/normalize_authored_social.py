"""Normalize the three authored/social corpora into FPRINT CSV inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator

MIN_WORDS = 100
MAX_WORDS = 350
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
SPACE_RE = re.compile(r"[^\S\r\n]+")
PG_FILE_RE = re.compile(r"^(?:pg)?(\d+)(?:-\d+)?\.txt(?:\.utf-8)?$", re.I)
PG_START_RE = re.compile(r"\*{3}\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*{3}", re.I)
PG_END_RE = re.compile(r"\*{3}\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*{3}", re.I)
SCHEMAS = {
    "gutenberg": ("record_id", "text", "author_id", "book_id"),
    "blog_authorship": ("record_id", "text", "author_id"),
    "stack_exchange": ("record_id", "text", "site_id", "user_id", "post_id"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_space(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", SPACE_RE.sub(" ", text)).strip()


def chunks(text: str, counts: Counter[str]) -> Iterator[str]:
    """Yield sentence-preserving chunks; ambiguous short/oversize tails are dropped."""
    pending: list[str] = []
    pending_words = 0
    for sentence in (part.strip() for part in SENTENCE_RE.split(_clean_space(text))):
        size = len(sentence.split())
        if not size:
            continue
        if size > MAX_WORDS:
            if pending_words >= MIN_WORDS:
                yield " ".join(pending)
            elif pending:
                counts["short_fragments_dropped"] += 1
            counts["oversize_sentences_dropped"] += 1
            pending, pending_words = [], 0
            continue
        if pending_words + size > MAX_WORDS:
            if pending_words >= MIN_WORDS:
                yield " ".join(pending)
            elif pending:
                counts["short_fragments_dropped"] += 1
            pending, pending_words = [sentence], size
        else:
            pending.append(sentence)
            pending_words += size
    if pending_words >= MIN_WORDS:
        yield " ".join(pending)
    elif pending:
        counts["short_fragments_dropped"] += 1


def _select(rows: Iterable[dict[str, str]], cap: int, seed: int) -> list[dict[str, str]]:
    if cap <= 0:
        raise ValueError("cap must be positive")

    def key(row: dict[str, str]) -> tuple[str, str]:
        record_id = row["record_id"]
        digest = hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()
        return digest, record_id

    return sorted(heapq.nsmallest(cap, rows, key=key), key=lambda row: row["record_id"])


def _source(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write(
    output: Path,
    corpus: str,
    rows: Iterable[dict[str, str]],
    counts: Counter[str],
    sources: list[Path],
    cap: int,
    seed: int,
    options: dict[str, object] | None = None,
) -> Path:
    selected = _select(rows, cap, seed)
    counts["selected_chunks"] = len(selected)
    if len({row["record_id"] for row in selected}) != len(selected):
        raise RuntimeError("normalizer produced duplicate record_id values")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMAS[corpus], extrasaction="raise")
        writer.writeheader()
        writer.writerows(selected)
    temporary.replace(output)

    provenance = {
        "tool": "tools/normalize_authored_social.py",
        "version": 1,
        "corpus": corpus,
        "schema": list(SCHEMAS[corpus]),
        "chunk_words": [MIN_WORDS, MAX_WORDS],
        "cap": cap,
        "seed": seed,
        "options": options or {},
        "counts": dict(sorted(counts.items())),
        "sources": [_source(path) for path in sorted(set(sources))],
        "output": _source(output),
    }
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance_path


def _strip_gutenberg(text: str) -> str | None:
    start = PG_START_RE.search(text)
    end = PG_END_RE.search(text, start.end() if start else 0)
    if not start or not end or end.start() <= start.end():
        return None
    return text[start.end() : end.start()]


def _gutenberg_files(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        match = PG_FILE_RE.fullmatch(path.name)
        if match:
            indexed.setdefault(str(int(match.group(1))), path)
    return indexed


def normalize_gutenberg(
    catalog: Path, text_root: Path, output: Path, cap: int, seed: int
) -> Path:
    counts: Counter[str] = Counter()
    files = _gutenberg_files(text_root)
    used_sources: list[Path] = [catalog]

    def rows() -> Iterator[dict[str, str]]:
        opener = gzip.open if catalog.suffix.casefold() == ".gz" else Path.open
        with opener(catalog, "rt", encoding="utf-8-sig", newline="") as handle:
            for entry in csv.DictReader(handle):
                counts["catalog_rows"] += 1
                book_id = str(entry.get("Text#") or "").strip()
                authors = SPACE_RE.sub(" ", str(entry.get("Authors") or "")).strip()
                if entry.get("Type") != "Text" or entry.get("Language") != "en":
                    counts["non_english_or_non_text"] += 1
                    continue
                if (
                    not authors
                    or authors.casefold() in {"anonymous", "various"}
                    or ";" in authors
                ):
                    counts["ambiguous_author"] += 1
                    continue
                path = files.get(book_id)
                if not path:
                    counts["missing_text_file"] += 1
                    continue
                text = _strip_gutenberg(path.read_text(encoding="utf-8-sig", errors="replace"))
                if text is None:
                    counts["missing_boilerplate_markers"] += 1
                    continue
                used_sources.append(path)
                counts["books_read"] += 1
                for index, chunk in enumerate(chunks(text, counts)):
                    counts["candidate_chunks"] += 1
                    yield {
                        "record_id": f"{book_id}:{index:05d}",
                        "text": chunk,
                        "author_id": authors,
                        "book_id": book_id,
                    }

    return _write(
        output, "gutenberg", rows(), counts, used_sources, cap, seed,
        {"catalog": str(catalog.resolve()), "text_root": str(text_root.resolve())},
    )


def normalize_blog(parquet: Path, output: Path, cap: int, seed: int) -> Path:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError("Install requirements-data.txt for Parquet support") from exc

    dataset = ds.dataset(parquet, format="parquet")
    author_column = "author_id" if "author_id" in dataset.schema.names else "id"
    if author_column not in dataset.schema.names or "text" not in dataset.schema.names:
        raise ValueError("Blog Parquet requires id/author_id and text columns")
    counts: Counter[str] = Counter()

    def rows() -> Iterator[dict[str, str]]:
        source_order = 0
        scanner = dataset.scanner(columns=[author_column, "text"], batch_size=4096)
        for batch in scanner.to_batches():
            for entry in batch.to_pylist():
                counts["source_rows"] += 1
                author_id = str(entry.get(author_column) or "").strip()
                text = str(entry.get("text") or "").strip()
                if not author_id or author_id.casefold() in {"none", "null", "nan"} or not text:
                    counts["missing_author_or_text"] += 1
                    source_order += 1
                    continue
                for index, chunk in enumerate(chunks(text, counts)):
                    counts["candidate_chunks"] += 1
                    yield {
                        "record_id": f"{author_id}:{source_order:07d}:{index:03d}",
                        "text": chunk,
                        "author_id": author_id,
                    }
                source_order += 1

    sources = sorted(parquet.rglob("*.parquet")) if parquet.is_dir() else [parquet]
    return _write(output, "blog_authorship", rows(), counts, sources, cap, seed)


class _StackText(HTMLParser):
    SKIP = {"blockquote", "code", "pre", "script", "style"}
    BREAK = {"br", "div", "h1", "h2", "h3", "h4", "li", "p"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag in self.SKIP:
            self.depth += 1
        elif not self.depth and tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self.SKIP and self.depth:
            self.depth -= 1
        elif not self.depth and tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.depth:
            self.parts.append(data)

    def text(self) -> str:
        return _clean_space(html.unescape("".join(self.parts)))


def _stack_text(body: str) -> str:
    parser = _StackText()
    parser.feed(body)
    parser.close()
    return parser.text()


def normalize_stack(
    posts: Path, site_id: str, output: Path, cap: int, seed: int
) -> Path:
    if not site_id.strip():
        raise ValueError("site_id is required")
    counts: Counter[str] = Counter()

    def rows() -> Iterator[dict[str, str]]:
        for _, element in ET.iterparse(posts, events=("end",)):
            if element.tag != "row":
                continue
            counts["source_rows"] += 1
            values = element.attrib
            user_id = str(values.get("OwnerUserId") or "").strip()
            post_id = str(values.get("Id") or "").strip()
            if values.get("PostTypeId") not in {"1", "2"}:
                counts["non_question_answer"] += 1
            elif not user_id or user_id in {"-1", "0"} or not post_id:
                counts["missing_user_or_post"] += 1
            elif values.get("CommunityOwnedDate"):
                counts["community_owned"] += 1
            else:
                license_name = str(values.get("ContentLicense") or "unspecified")
                counts[f"license:{license_name}"] += 1
                text = _stack_text(str(values.get("Body") or ""))
                for index, chunk in enumerate(chunks(text, counts)):
                    counts["candidate_chunks"] += 1
                    yield {
                        "record_id": f"{post_id}:{index:03d}",
                        "text": chunk,
                        "site_id": site_id,
                        "user_id": user_id,
                        "post_id": post_id,
                    }
            element.clear()

    return _write(
        output, "stack_exchange", rows(), counts, [posts], cap, seed,
        {"site_id": site_id},
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    gutenberg = commands.add_parser("gutenberg")
    gutenberg.add_argument("--catalog", type=Path, required=True)
    gutenberg.add_argument("--text-root", type=Path, required=True)
    _common(gutenberg)

    blog = commands.add_parser("blog")
    blog.add_argument("--parquet", type=Path, required=True)
    _common(blog)

    stack = commands.add_parser("stack")
    stack.add_argument("--posts", type=Path, required=True)
    stack.add_argument("--site-id", required=True)
    _common(stack)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "gutenberg":
        provenance = normalize_gutenberg(
            args.catalog, args.text_root, args.output, args.cap, args.seed
        )
    elif args.command == "blog":
        provenance = normalize_blog(args.parquet, args.output, args.cap, args.seed)
    else:
        provenance = normalize_stack(
            args.posts, args.site_id, args.output, args.cap, args.seed
        )
    print(provenance)


if __name__ == "__main__":
    main()
