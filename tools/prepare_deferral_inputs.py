"""Prepare the text-free topic and token inputs for the deferral pilot.

This is deliberately separate from the scoring workflow.  It reads the
already-normalized human rows, optionally intersects them with the grouped
study database, and recovers generation topics from source metadata.  A
passage is never used as its own generation topic.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fprint.core import grouping_key
from fprint.deferral import (
    DEV_CORPORA,
    MAGE_ENDPOINT,
    PROBES,
    RADAR_ENDPOINT,
    build_reflow_variants,
)


CANONICAL_FIELDS = (
    "record_id", "corpus", "group_id", "text", "provenance_label",
    "partition", "source_order",
)
CORPUS_FILES = {
    **{corpus: f"{corpus}.csv" for corpus in DEV_CORPORA},
    "asap_aes": "asap_aes.csv",
    "pmc": "pmc.csv",
}
_BLOG_RECORD = re.compile(r"^(?P<author>.+):(?P<source>[0-9]+):(?P<chunk>[0-9]+)$")


def _tag(element: ET.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _clean_topic(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _active_corpora(corpora: Sequence[str] | None = None) -> tuple[str, ...]:
    active = tuple(str(corpus).strip() for corpus in (corpora or DEV_CORPORA))
    if not active or len(set(active)) != len(active):
        raise ValueError("Development corpora must be a non-empty unique sequence")
    unsupported = set(active) - set(CORPUS_FILES)
    if unsupported:
        raise ValueError(f"Unsupported input corpora: {sorted(unsupported)}")
    return active


def _read_csv_rows(data_root: Path, corpora: Sequence[str] | None = None) -> dict[str, dict[str, dict[str, str]]]:
    """Read the four normalized human corpora keyed by record ID."""
    result: dict[str, dict[str, dict[str, str]]] = {}
    for corpus in _active_corpora(corpora):
        filename = CORPUS_FILES[corpus]
        path = data_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing normalized corpus: {path}")
        rows: dict[str, dict[str, str]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                record_id = str(row.get("record_id", "")).strip()
                text = str(row.get("text", ""))
                if not record_id or not text.strip():
                    continue
                if record_id in rows:
                    raise ValueError(f"Duplicate record_id in {path.name}: {record_id}")
                row["record_id"] = record_id
                row["text"] = text
                rows[record_id] = row
        if not rows:
            raise ValueError(f"Normalized corpus is empty: {path}")
        result[corpus] = rows
    return result


def _resolve_db(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = (path / "fprint.sqlite3", path / "state" / "fprint.sqlite3")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No grouped study database under {path}")


def _grouped_rows(
    source_db: Path,
    normalized: Mapping[str, Mapping[str, Mapping[str, str]]],
    corpora: Sequence[str],
    partition: str | None = None,
) -> list[dict[str, str]]:
    """Join the grouped DB to normalized rows without inventing records.

    The normalized files define the human universe; the grouped database only
    supplies the already-frozen group and partition assignment.  This also
    prevents an accidental generated row in a future database from becoming a
    human candidate.
    """
    db_path = _resolve_db(source_db)
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" for _ in corpora)
        partition_clause = " AND partition_name=?" if partition else ""
        parameters = tuple(corpora) + ((str(partition),) if partition else ())
        rows = connection.execute(
            "SELECT record_id,corpus,group_id,partition_name,text FROM records "
            f"WHERE corpus IN ({placeholders}){partition_clause} ORDER BY corpus,group_id,record_id",
            parameters,
        )
        output: list[dict[str, str]] = []
        for record_id, corpus, group_id, partition, db_text in rows:
            record_key = str(record_id)
            source_key = record_key
            source = normalized.get(str(corpus), {}).get(source_key)
            if source is None:
                prefix = f"{corpus}:"
                if record_key.startswith(prefix):
                    source_key = record_key[len(prefix):]
                    source = normalized.get(str(corpus), {}).get(source_key)
            if source is None:
                continue
            text = str(db_text) if db_text is not None and str(db_text) else str(source["text"])
            output.append({
                **source,
                "record_id": str(record_id), "corpus": str(corpus),
                "source_record_id": source_key,
                "group_id": str(group_id), "text": text,
                "provenance_label": "human", "partition": str(partition or ""),
                "source_order": str(source.get("source_order", len(output))),
            })
        if not output:
            raise ValueError("Grouped database has no matching records in the four normalized corpora")
        return output
    finally:
        connection.close()


def canonical_human_rows(
    data_root: Path | str,
    source_db: Path | str | None = None,
    corpora: Sequence[str] | None = None,
    partition: str | None = None,
) -> list[dict[str, str]]:
    """Return one canonical human row per existing normalized/study record."""
    data_root = Path(data_root).expanduser().resolve()
    active = _active_corpora(corpora)
    normalized = _read_csv_rows(data_root, active)
    if source_db is not None:
        rows = _grouped_rows(Path(source_db).expanduser().resolve(), normalized, active, partition)
    else:
        rows = []
        for corpus in active:
            for index, source in enumerate(normalized[corpus].values()):
                metadata = dict(source)
                group_id = metadata.get("group_id") or grouping_key(corpus, metadata)
                rows.append({
                    **metadata,
                    "record_id": str(metadata["record_id"]), "corpus": corpus,
                    "group_id": group_id, "text": str(metadata["text"]),
                    "provenance_label": "human", "partition": str(metadata.get("partition", "")),
                    "source_order": str(metadata.get("source_order", index)),
                })
    rows.sort(key=lambda row: (row["corpus"], row["group_id"], row["record_id"]))
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Canonical human candidates contain duplicate record IDs")
    return rows


def _requested_by_metadata(rows: Sequence[Mapping[str, str]], field: str) -> dict[str, str]:
    return {str(row["record_id"]): str(row.get(field, "")).strip() for row in rows if str(row.get(field, "")).strip()}


def _blog_topics(rows: Sequence[Mapping[str, str]], raw_root: Path) -> dict[str, dict[str, object]]:
    """Map blog IDs to source-row topics using author and source order.

    The normalizer's record identity is ``author:source_order:chunk``.  When
    the raw Parquet includes a topic/title/category field it is used; otherwise
    the non-text source identity is used as a safe topic.  The latter is still
    metadata and never the passage body.
    """
    targets: dict[tuple[str, int], list[str]] = {}
    for row in rows:
        match = _BLOG_RECORD.fullmatch(str(row.get("source_record_id", row["record_id"])))
        if not match:
            continue
        targets.setdefault((match.group("author"), int(match.group("source"))), []).append(str(row["record_id"]))
    paths = sorted((raw_root / "blog_authorship").glob("*.parquet"))
    if not paths:
        # This path is useful for a normalized-only smoke test, while the real
        # run has the source files and therefore uses the mapping below.
        return {
            record_id: {
                "topic": f"blog author {author}; source row {source_order}",
                "topic_source": "blog_record_identity",
                "used_passage_text": False,
            }
            for (author, source_order), record_ids in targets.items()
            for record_id in record_ids
        }
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Blog topic recovery requires the existing pyarrow data dependency") from exc
    result: dict[str, dict[str, object]] = {}
    source_order = 0
    for path in paths:
        parquet_file = parquet.ParquetFile(path)
        names = set(parquet_file.schema_arrow.names)
        author_column = "author_id" if "author_id" in names else "id" if "id" in names else None
        if author_column is None:
            raise ValueError(f"Blog Parquet lacks author_id/id: {path}")
        topic_column = next((name for name in ("topic", "title", "category", "blog_id") if name in names), None)
        columns = [author_column] + ([topic_column] if topic_column else [])
        for batch in parquet_file.iter_batches(columns=columns, batch_size=4096):
            for entry in batch.to_pylist():
                author = str(entry.get(author_column) or "").strip()
                key = (author, source_order)
                if key in targets:
                    topic = _clean_topic(entry.get(topic_column)) if topic_column else ""
                    if not topic:
                        topic = f"blog author {author}; source row {source_order}"
                    for record_id in targets[key]:
                        result[record_id] = {
                            "topic": topic, "topic_source": f"raw_parquet:{topic_column or 'author_source_order'}",
                            "used_passage_text": False,
                        }
                source_order += 1
    missing = [record_id for ids in targets.values() for record_id in ids if record_id not in result]
    if missing:
        raise ValueError(f"Raw Blog Parquet did not resolve {len(missing)} normalized records")
    return result


def _article_title(article: ET.Element) -> str:
    for child in article.iter():
        if _tag(child) == "article-title":
            return _clean_topic("".join(child.itertext()))
    return ""


def _pmc_topics(rows: Sequence[Mapping[str, str]], raw_root: Path) -> dict[str, dict[str, object]]:
    wanted: dict[str, list[str]] = {}
    for row in rows:
        article_id = str(row.get("article_id", "")).upper().strip()
        if article_id:
            wanted.setdefault(article_id, []).append(str(row["record_id"]))
    paths = sorted((raw_root / "pmc").glob("*.xml"))
    if not paths:
        raise FileNotFoundError(f"No PMC XML batches under {raw_root / 'pmc'}")
    titles: dict[str, str] = {}
    for path in paths:
        for _, article in ET.iterparse(path, events=("end",)):
            if _tag(article) != "article":
                continue
            pmcid = ""
            for child in article.iter():
                if _tag(child) == "article-id" and str(child.attrib.get("pub-id-type", "")).casefold() in {"pmc", "pmcid", "pmcaid"}:
                    pmcid = _clean_topic(child.text).upper()
                    break
            if pmcid and not pmcid.startswith("PMC"):
                pmcid = "PMC" + pmcid
            if pmcid in wanted:
                titles[pmcid] = _article_title(article)
            article.clear()
    result = {}
    for article_id, record_ids in wanted.items():
        title = titles.get(article_id, "")
        if not title:
            raise ValueError(f"PMC XML did not resolve an article title for {article_id}")
        for record_id in record_ids:
            result[record_id] = {"topic": title, "topic_source": "pmc_article_title", "used_passage_text": False}
    return result


def _stack_topics(rows: Sequence[Mapping[str, str]], raw_root: Path) -> dict[str, dict[str, object]]:
    paths = sorted((raw_root / "stackexchange_20221005").glob("Posts.xml"))
    if not paths:
        raise FileNotFoundError(f"No Stack Exchange Posts.xml under {raw_root / 'stackexchange_20221005'}")
    wanted: dict[str, list[str]] = {}
    for row in rows:
        post_id = str(row.get("post_id", "")).strip()
        if post_id:
            wanted.setdefault(post_id, []).append(str(row["record_id"]))
    question_ids: set[str] = set()
    answer_parent: dict[str, str] = {}
    for path in paths:
        for _, element in ET.iterparse(path, events=("end",)):
            if _tag(element) != "row":
                continue
            values = element.attrib
            post_id = str(values.get("Id", "")).strip()
            if post_id in wanted and values.get("PostTypeId") == "1":
                question_ids.add(post_id)
            elif post_id in wanted and values.get("PostTypeId") == "2":
                parent = str(values.get("ParentId", "")).strip()
                if parent:
                    answer_parent[post_id] = parent
                    question_ids.add(parent)
            element.clear()
    titles: dict[str, str] = {}
    for path in paths:
        for _, element in ET.iterparse(path, events=("end",)):
            if _tag(element) != "row":
                continue
            values = element.attrib
            post_id = str(values.get("Id", "")).strip()
            if post_id in question_ids and values.get("PostTypeId") == "1":
                title = _clean_topic(values.get("Title"))
                if title:
                    titles[post_id] = title
            element.clear()
    result: dict[str, dict[str, object]] = {}
    for post_id, record_ids in wanted.items():
        question_id = answer_parent.get(post_id, post_id)
        title = titles.get(question_id, "")
        if not title:
            raise ValueError(f"Stack Exchange XML did not resolve a question title for {post_id}")
        for record_id in record_ids:
            result[record_id] = {"topic": title, "topic_source": "stack_question_title", "question_id": question_id, "used_passage_text": False}
    return result


def _asap_topics(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, object]]:
    """Use prompt metadata, never the essay body, as the generation topic."""
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        prompt_text = _clean_topic(row.get("prompt_text") or row.get("prompt_name"))
        prompt_id = _clean_topic(row.get("prompt_id") or row.get("essay_set") or row.get("prompt_name"))
        if not prompt_text or not prompt_id:
            raise ValueError(f"ASAP row lacks prompt metadata: {row['record_id']}")
        result[str(row["record_id"])] = {
            "topic": prompt_text, "prompt_id": prompt_id,
            "prompt_text": prompt_text, "topic_source": "asap_prompt_metadata",
            "used_passage_text": False,
        }
    return result


def source_topics(
    rows: Sequence[Mapping[str, str]], data_root: Path | str,
    corpora: Sequence[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Recover a record-keyed source topic map without text-derived topics."""
    raw_root = Path(data_root).expanduser().resolve() / "raw"
    active = _active_corpora(corpora)
    by_corpus: dict[str, list[Mapping[str, str]]] = {corpus: [] for corpus in active}
    for row in rows:
        by_corpus.setdefault(str(row["corpus"]), []).append(row)
    result = {}
    if "blog_authorship" in active:
        result.update(_blog_topics(by_corpus["blog_authorship"], raw_root))
    if "pmc" in active:
        result.update(_pmc_topics(by_corpus["pmc"], raw_root))
    if "asap_aes" in active:
        result.update(_asap_topics(by_corpus["asap_aes"]))
    if "stack_exchange" in active:
        result.update(_stack_topics(by_corpus["stack_exchange"], raw_root))
    if "wikitext_103" in active:
        for row in by_corpus["wikitext_103"]:
            title = _clean_topic(row.get("title"))
            if not title:
                raise ValueError(f"WikiText normalized row lacks title: {row['record_id']}")
            result[str(row["record_id"])] = {"topic": title, "topic_source": "wikitext_title_csv", "used_passage_text": False}
    if set(result) != {str(row["record_id"]) for row in rows}:
        raise ValueError("Source topic map is incomplete")
    return result


def tokenize_human_rows(
    rows: Sequence[Mapping[str, str]],
    radar_adapter: Any,
    mage_adapter: Any,
) -> dict[str, dict[str, dict[str, int]]]:
    """Return the exact panel shape consumed by ``prepare_pilot_manifest``."""
    output: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        base_text = " ".join(str(row["text"]).split())
        try:
            variants = build_reflow_variants(base_text, probes=PROBES)
        except ValueError:
            continue
        texts = {"original": base_text, **{variant.variant_id: variant.text for variant in variants}}
        output[str(row["record_id"])] = {
            RADAR_ENDPOINT: {variant: int(radar_adapter.token_count(text)) for variant, text in texts.items()},
            MAGE_ENDPOINT: {variant: int(mage_adapter.token_count(text)) for variant, text in texts.items()},
        }
    return output


class TokenCounter:
    def __init__(self, endpoint: str, preprocessor: Any = None):
        from fprint.detectors import SPECS
        from transformers import AutoTokenizer

        self.spec = SPECS[endpoint]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.model_id, revision=self.spec.tokenizer_revision,
            local_files_only=True,
        )
        self.preprocessor = preprocessor

    def token_count(self, text: str) -> int:
        effective = self.preprocessor(text) if self.preprocessor else text
        return len(self.tokenizer(effective, add_special_tokens=True, truncation=False)["input_ids"])


def build_pinned_token_counters(mage_repo: Path | str) -> tuple[TokenCounter, TokenCounter]:
    from fprint.detectors import _mage_preprocessor

    if not mage_repo:
        raise ValueError("Pinned MAGE preprocessing repository is required")
    return TokenCounter(RADAR_ENDPOINT), TokenCounter(MAGE_ENDPOINT, _mage_preprocessor(str(mage_repo)))


def tokenize_text_panel(
    text: str, radar: Any, mage: Any,
) -> dict[str, dict[str, int]]:
    base_text = " ".join(str(text).split())
    variants = build_reflow_variants(base_text, probes=PROBES)
    texts = {"original": base_text, **{variant.variant_id: variant.text for variant in variants}}
    return {
        RADAR_ENDPOINT: {variant: int(radar.token_count(value)) for variant, value in texts.items()},
        MAGE_ENDPOINT: {variant: int(mage.token_count(value)) for variant, value in texts.items()},
    }


def write_inputs(
    data_root: Path | str,
    output_dir: Path | str,
    source_db: Path | str | None = None,
    corpora: Sequence[str] | None = None,
    partition: str | None = None,
    *,
    tokenize: bool = False,
    device: int = -1,
    mage_repo: str | None = None,
) -> dict[str, object]:
    active = _active_corpora(corpora)
    rows = canonical_human_rows(data_root, source_db, active, partition)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "human_candidates.csv"
    with canonical_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CANONICAL_FIELDS} for row in rows)
    topics = source_topics(rows, data_root, active)
    (output_dir / "source_topics.json").write_text(json.dumps(topics, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    counts: dict[str, object] = {}
    provenance: dict[str, object] = {"tokenization_requested": bool(tokenize), "endpoints": [RADAR_ENDPOINT, MAGE_ENDPOINT], "development_corpora": list(active)}
    if tokenize:
        radar, mage = build_pinned_token_counters(mage_repo or "")
        counts = tokenize_human_rows(rows, radar, mage)
        provenance.update({"radar_model_revision": radar.spec.revision, "radar_tokenizer_revision": radar.spec.tokenizer_revision, "mage_model_revision": mage.spec.revision, "mage_tokenizer_revision": mage.spec.tokenizer_revision, "mage_preprocessing": "pinned deployment.preprocess"})
    (output_dir / "human_token_counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "human_token_counts.provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "rows": len(rows), "topics": len(topics), "tokenized": len(counts),
        "source_partition": partition, "output_dir": str(output_dir),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-db", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON config containing development_corpora")
    parser.add_argument("--corpora", nargs="+", help="Override the config development_corpora list")
    parser.add_argument("--partition", help="Use only this frozen source-database partition")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--mage-repo", type=str)
    args = parser.parse_args(argv)
    configured = None
    if args.config:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        configured = payload.get("development_corpora")
    corpora = args.corpora or configured
    partition = args.partition
    if args.config and partition is None:
        partition = payload.get("source_partition")
    print(json.dumps(write_inputs(
        args.data_root, args.output_dir, args.source_db, corpora=corpora,
        partition=partition, tokenize=args.tokenize, device=args.device,
        mage_repo=args.mage_repo,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
