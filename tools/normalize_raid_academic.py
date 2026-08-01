"""Normalize the official RAID, ASAP 2.0, BAWE, and PMC sources for FPRINT.

This tool is deliberately download-agnostic.  It consumes the official files
verbatim, writes the small CSV schema accepted by ``fprint prepare``, and emits
a provenance manifest beside each output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

RAID_URL = "https://dataset.raid-bench.xyz/train_none.csv"
ASAP_URL = (
    "https://github.com/scrosseye/ASAP_2.0/raw/refs/heads/main/"
    "ASAP_2_Final_github_train.zip"
)
BAWE_URL = (
    "https://llds.ling-phil.ox.ac.uk/llds/xmlui/bitstream/handle/"
    "20.500.14106/2539/2539.zip?sequence=4&isAllowed=y"
)
PMC_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PMC_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_QUERY = (
    '"open access"[filter] AND ("cc by license"[filter] OR '
    '"cc0 license"[filter]) AND 1900/01/01:2019/12/31[pubdate]'
)

SENTENCE_RE = re.compile(r"(?<=[.!?])(?:[\"'”’)\]]*)\s+|\n+")
SPACE_RE = re.compile(r"\s+")
CC_BY_RE = re.compile(r"creativecommons\.org/licenses/by/[0-9]", re.I)
CC0_RE = re.compile(r"creativecommons\.org/publicdomain/zero/[0-9]", re.I)
BLOCKED_JATS = {
    "ack", "boxed-text", "disp-formula", "fig", "ref-list",
    "supplementary-material", "table-wrap",
}
BLOCKED_BAWE = {"bibl", "figure", "formula", "note", "quote", "ref", "table"}


def word_count(text: str) -> int:
    return len(text.split())


def _split_oversize(unit: str, maximum: int) -> Iterator[str]:
    tokens = unit.split()
    start = 0
    while start < len(tokens):
        end = start
        count = 0
        while end < len(tokens):
            token_words = word_count(tokens[end])
            if count and count + token_words > maximum:
                break
            count += token_words
            end += 1
        if end == start:
            end += 1
        yield " ".join(tokens[start:end])
        start = end


def passage_chunks(text: str, minimum: int = 100, maximum: int = 350) -> list[str]:
    """Return sentence-aligned passages; never pad or independently truncate."""
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Require 0 < minimum <= maximum")
    units: list[str] = []
    for raw in SENTENCE_RE.split(str(text)):
        unit = SPACE_RE.sub(" ", raw).strip()
        if not unit:
            continue
        units.extend(_split_oversize(unit, maximum))

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        count = word_count(unit)
        if current and current_words + count > maximum:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(unit)
        current_words += count
    if current:
        tail = " ".join(current)
        if word_count(tail) >= minimum:
            chunks.append(tail)
        elif chunks and word_count(chunks[-1]) + word_count(tail) <= maximum:
            chunks[-1] = f"{chunks[-1]} {tail}"
    return [chunk for chunk in chunks if minimum <= word_count(chunk) <= maximum]


def normalize_raid(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    required = {"id", "source_id", "model", "attack", "domain", "generation"}
    stats = {"input_rows": 0, "not_verified_human": 0, "length_rejected": 0}
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"RAID input is missing columns: {sorted(missing)}")
        for row in reader:
            stats["input_rows"] += 1
            if row["model"].strip().casefold() != "human" or row["attack"].strip().casefold() != "none":
                stats["not_verified_human"] += 1
                continue
            passages = passage_chunks(row["generation"])
            if not passages:
                stats["length_rejected"] += 1
                continue
            rows.append({
                "record_id": row["id"].strip(),
                "text": passages[0],
                "source_id": row["source_id"].strip() or row["id"].strip(),
                "domain": row["domain"].strip(),
            })
    _require_unique(rows)
    rows.sort(key=lambda row: _stable_key(row["record_id"]))
    stats["output_rows"] = len(rows)
    return rows, stats


def normalize_raid_ai(
    path: Path, cap: int = 200,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    required = {"id", "source_id", "model", "attack", "domain", "generation"}
    stats = {"input_rows": 0, "not_verified_ai": 0, "length_rejected": 0}
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"RAID input is missing columns: {sorted(missing)}")
        for row in reader:
            stats["input_rows"] += 1
            if (
                row["model"].strip().casefold() == "human"
                or row["attack"].strip().casefold() != "none"
            ):
                stats["not_verified_ai"] += 1
                continue
            passages = passage_chunks(row["generation"])
            if not passages:
                stats["length_rejected"] += 1
                continue
            rows.append({
                "record_id": row["id"].strip(),
                "text": passages[0],
                "source_id": row["source_id"].strip() or row["id"].strip(),
                "domain": row["domain"].strip(),
                "model": row["model"].strip(),
            })
    _require_unique(rows)
    rows.sort(key=lambda row: _stable_key(row["record_id"]))
    rows = rows[:cap]
    stats["output_rows"] = len(rows)
    return rows, stats


def _decode_csv_archive(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.casefold().endswith(".csv")
            and not name.startswith("__MACOSX/")
            and not Path(name).name.startswith("._")
        )
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in ASAP archive, found {names}")
        data = archive.read(names[0])
    try:
        return names[0], data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return names[0], data.decode("cp1252")


def normalize_asap(path: Path) -> tuple[list[dict[str, str]], dict[str, int | str]]:
    member, text = _decode_csv_archive(path)
    reader = csv.DictReader(io.StringIO(text, newline=""))
    required = {"essay_id", "full_text", "prompt_name"}
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"ASAP 2.0 input is missing columns: {sorted(missing)}")
    stats: dict[str, int | str] = {
        "archive_member": member, "input_rows": 0, "length_rejected": 0,
    }
    rows: list[dict[str, str]] = []
    for source in reader:
        stats["input_rows"] = int(stats["input_rows"]) + 1
        essay_id = source["essay_id"].strip()
        prompt = source["prompt_name"].strip()
        if not essay_id or not prompt:
            raise ValueError("ASAP 2.0 requires non-empty essay_id and prompt_name")
        chunks = passage_chunks(source["full_text"])
        if not chunks:
            stats["length_rejected"] = int(stats["length_rejected"]) + 1
            continue
        for index, chunk in enumerate(chunks):
            rows.append({
                "record_id": f"{essay_id}:chunk{index:02d}",
                "text": chunk,
                "essay_id": essay_id,
                "student_id": "",
                "essay_set": prompt,
                "prompt_name": prompt,
            })
    _require_unique(rows)
    rows.sort(key=lambda row: _stable_key(row["record_id"]))
    stats["output_rows"] = len(rows)
    return rows, stats


def _bawe_metadata(root: ET.Element, name: str) -> str:
    for element in root.iter():
        if _tag(element) == "p" and element.attrib.get("n") == name:
            return SPACE_RE.sub(" ", "".join(element.itertext())).strip()
    return ""


def _bawe_paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []

    def collect(node: ET.Element) -> None:
        if _tag(node) in BLOCKED_BAWE:
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            collect(child)
            if child.tail:
                parts.append(child.tail)

    collect(paragraph)
    return SPACE_RE.sub(" ", " ".join(parts)).strip()


def _bawe_body_paragraphs(root: ET.Element) -> list[str]:
    body = next((element for element in root.iter() if _tag(element) == "body"), None)
    if body is None:
        return []
    paragraphs: list[str] = []

    def visit(node: ET.Element) -> None:
        tag = _tag(node)
        if tag in BLOCKED_BAWE:
            return
        if tag.startswith("div") and node.attrib.get("type", "").casefold() in {
            "appendix", "bibliography", "front-back-matter", "toc",
        }:
            return
        if tag == "p":
            text = _bawe_paragraph_text(node)
            if text:
                paragraphs.append(text)
            return
        for child in node:
            visit(child)

    visit(body)
    return paragraphs


def normalize_bawe(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    stats = {
        "input_documents": 0, "parse_rejected": 0, "metadata_rejected": 0,
        "length_rejected": 0,
    }
    rows: list[dict[str, str]] = []
    writer_groups: dict[str, set[str]] = {}
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.startswith("download/CORPUS_UTF-8/") and name.endswith(".xml")
        )
        if not members:
            raise ValueError("BAWE archive has no CORPUS_UTF-8 XML documents")
        for member in members:
            stats["input_documents"] += 1
            try:
                root = ET.fromstring(archive.read(member))
            except ET.ParseError:
                stats["parse_rejected"] += 1
                continue
            document_id = root.attrib.get("id", "").lstrip("_") or Path(member).stem
            student_id = _bawe_metadata(root, "student ID")
            group = _bawe_metadata(root, "disciplinary group")
            if not document_id or not student_id or not group:
                stats["metadata_rejected"] += 1
                continue
            writer_groups.setdefault(student_id, set()).add(group)
            chunks = passage_chunks(" ".join(_bawe_body_paragraphs(root)))
            if not chunks:
                stats["length_rejected"] += 1
                continue
            metadata = {
                "document_id": document_id,
                "student_id": student_id,
                "disciplinary_group": group,
                "discipline": _bawe_metadata(root, "discipline"),
                "genre_family": _bawe_metadata(root, "genre family"),
                "level": _bawe_metadata(root, "level"),
            }
            for index, chunk in enumerate(chunks):
                rows.append({
                    "record_id": f"{document_id}:chunk{index:02d}",
                    "text": chunk,
                    **metadata,
                })
    for row in rows:
        row["writer_stratum"] = "|".join(sorted(writer_groups[row["student_id"]]))
    stats["multi_group_students"] = sum(
        len(groups) > 1 for groups in writer_groups.values()
    )
    _require_unique(rows)
    rows.sort(key=lambda row: _stable_key(row["record_id"]))
    stats["output_rows"] = len(rows)
    return rows, stats


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _first_text(element: ET.Element, tag: str, attribute: str | None = None,
                value: str | None = None) -> str:
    for child in element.iter():
        if _tag(child) != tag:
            continue
        if attribute and child.attrib.get(attribute) != value:
            continue
        text = SPACE_RE.sub(" ", "".join(child.itertext())).strip()
        if text:
            return text
    return ""


def _article_id(article: ET.Element, kind: str | Sequence[str]) -> str:
    kinds = {kind} if isinstance(kind, str) else set(kind)
    for child in article.iter():
        if _tag(child) == "article-id" and child.attrib.get("pub-id-type") in kinds:
            return (child.text or "").strip()
    return ""


def _license(article: ET.Element) -> str:
    for child in article.iter():
        if _tag(child) != "license":
            continue
        href = next((value for key, value in child.attrib.items() if key.endswith("href")), "")
        text = SPACE_RE.sub(" ", " ".join(child.itertext())).strip()
        combined = f"{href} {text}".strip()
        if CC_BY_RE.search(combined) or CC0_RE.search(combined):
            return combined
    return ""


def _body_paragraphs(article: ET.Element) -> list[str]:
    bodies = [child for child in article.iter() if _tag(child) == "body"]
    paragraphs: list[str] = []

    def visit(node: ET.Element) -> None:
        if _tag(node) in BLOCKED_JATS:
            return
        if _tag(node) == "p":
            text = SPACE_RE.sub(" ", "".join(node.itertext())).strip()
            if text:
                paragraphs.append(text)
            return
        for child in node:
            visit(child)

    for body in bodies:
        visit(body)
    return paragraphs


def _author_cluster(article: ET.Element) -> str:
    for contributor in article.iter():
        if _tag(contributor) != "contrib" or contributor.attrib.get("contrib-type") != "author":
            continue
        orcid = _first_text(contributor, "contrib-id", "contrib-id-type", "orcid")
        if orcid:
            return f"orcid:{orcid.casefold()}"
        surname = _first_text(contributor, "surname")
        given = _first_text(contributor, "given-names")
        if surname:
            normalized = re.sub(r"[^a-z0-9]+", "", f"{surname}{given[:1]}".casefold())
            return f"name:{normalized}"
    return ""


def _year(article: ET.Element) -> int | None:
    article_meta = next(
        (child for child in article.iter() if _tag(child) == "article-meta"), None
    )
    if article_meta is None:
        return None
    publication_dates = [
        child for child in article_meta
        if _tag(child) == "pub-date"
    ]
    publication_dates.sort(
        key=lambda child: {"epub": 0, "ppub": 1, "collection": 2}.get(
            child.attrib.get("pub-type", ""), 3
        )
    )
    for publication_date in publication_dates:
        for child in publication_date:
            text = (child.text or "").strip()
            if _tag(child) == "year" and text.isdigit():
                value = int(text)
                if 1800 <= value <= 2100:
                    return value
    return None


def _xml_paths(inputs: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(path for path in item.rglob("*.xml") if path.is_file())
        elif item.is_file():
            paths.append(item)
        else:
            raise FileNotFoundError(item)
    return sorted(set(path.resolve() for path in paths))


def normalize_pmc(inputs: Sequence[Path]) -> tuple[list[dict[str, str]], dict[str, int]]:
    stats = {
        "input_articles": 0, "missing_article_id": 0, "date_rejected": 0,
        "license_rejected": 0, "article_type_rejected": 0, "body_rejected": 0,
    }
    rows: list[dict[str, str]] = []
    seen_articles: set[str] = set()
    for path in _xml_paths(inputs):
        for _, article in ET.iterparse(path, events=("end",)):
            if _tag(article) != "article":
                continue
            stats["input_articles"] += 1
            article_type = article.attrib.get("article-type", "").casefold()
            if any(term in article_type for term in ("preprint", "correction", "retraction")):
                stats["article_type_rejected"] += 1
                article.clear()
                continue
            pmcid = _article_id(article, ("pmc", "pmcid", "pmcaid"))
            if not pmcid:
                stats["missing_article_id"] += 1
                article.clear()
                continue
            pmcid = pmcid.upper()
            if not pmcid.startswith("PMC"):
                pmcid = f"PMC{pmcid}"
            if pmcid in seen_articles:
                article.clear()
                continue
            seen_articles.add(pmcid)
            year = _year(article)
            if year is None or year > 2019:
                stats["date_rejected"] += 1
                article.clear()
                continue
            license_text = _license(article)
            if not license_text:
                stats["license_rejected"] += 1
                article.clear()
                continue
            chunks = passage_chunks("\n\n".join(_body_paragraphs(article)))
            if not chunks:
                stats["body_rejected"] += 1
                article.clear()
                continue
            pmid = _article_id(article, "pmid")
            author_cluster = _author_cluster(article)
            for index, chunk in enumerate(chunks):
                rows.append({
                    "record_id": f"{pmcid}:chunk{index:03d}",
                    "text": chunk,
                    "article_id": pmcid,
                    "pmid": pmid,
                    "author_cluster": author_cluster,
                    "publication_year": str(year),
                    "license": license_text,
                })
            article.clear()
    _require_unique(rows)
    rows.sort(key=lambda row: _stable_key(row["record_id"]))
    stats["output_rows"] = len(rows)
    return rows, stats


def _stable_key(value: str, seed: int = 20260729) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _require_unique(rows: Sequence[Mapping[str, str]]) -> None:
    identifiers = [row["record_id"] for row in rows]
    if any(not identifier for identifier in identifiers):
        raise ValueError("Empty record_id")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate record_id in normalized output")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty normalized corpus")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest(args: argparse.Namespace, inputs: Sequence[Path],
              stats: Mapping[str, object]) -> dict[str, object]:
    source_urls = {
        "raid": RAID_URL, "raid-ai": RAID_URL,
        "asap": ASAP_URL, "bawe": BAWE_URL, "pmc": PMC_ESEARCH_URL,
    }
    payload: dict[str, object] = {
        "normalizer": Path(__file__).name,
        "corpus": args.command,
        "source_url": source_urls[args.command],
        "input_sha256": {str(path.resolve()): _sha256(path) for path in inputs},
        "filters": {
            "word_count": [100, 350],
            "verified_human": args.command == "raid",
            "verified_ai": args.command == "raid-ai",
            "pmc_query": PMC_QUERY if args.command == "pmc" else None,
            "bawe_research_only": args.command == "bawe",
        },
        "counts": dict(stats),
        "output_sha256": _sha256(args.output),
    }
    if args.command == "pmc":
        payload["efetch_url"] = PMC_EFETCH_URL
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("raid", "raid-ai", "asap", "bawe", "pmc"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--input", type=Path, nargs="+" if command == "pmc" else None,
                               required=True)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--manifest", type=Path)
        if command == "raid-ai":
            subparser.add_argument("--cap", type=int, default=200)
        subparser.add_argument(
            "--minimum", type=int,
            default=(
                10000 if command == "raid" else
                50 if command == "raid-ai" else
                6000 if command == "bawe" else
                5064
            ),
            help="Fail closed if fewer normalized passages remain.",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "raid":
        inputs = [args.input]
        rows, stats = normalize_raid(args.input)
    elif args.command == "raid-ai":
        inputs = [args.input]
        rows, stats = normalize_raid_ai(args.input, args.cap)
    elif args.command == "asap":
        inputs = [args.input]
        rows, stats = normalize_asap(args.input)
    elif args.command == "bawe":
        inputs = [args.input]
        rows, stats = normalize_bawe(args.input)
    else:
        inputs = _xml_paths(args.input)
        rows, stats = normalize_pmc(args.input)
    if len(rows) < args.minimum:
        raise RuntimeError(
            f"{args.command} produced {len(rows)} passages; minimum is {args.minimum}"
        )
    write_csv(args.output, rows)
    manifest = args.manifest or args.output.with_suffix(".provenance.json")
    write_manifest(manifest, _manifest(args, inputs, stats))
    print(json.dumps({"corpus": args.command, "rows": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
