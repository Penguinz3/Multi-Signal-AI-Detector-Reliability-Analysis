"""Verified, resumable scoring for the selective-deferral pilot.

The registry is built from the locked manifest and the exact text artifacts
that will be sent to a detector.  The journal keeps failures for diagnosis,
but only successful finite scores are exported to the canonical score table.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from fprint.deferral import (
    ENDPOINT_ROLES,
    LOGRANK_ENDPOINT,
    MAGE_ENDPOINT,
    PROBES,
    RADAR_ENDPOINT,
    DeferralPaths,
    build_reflow_variants,
    reflow_variant,
    verify_conditional_worklist,
    verify_generation_lock,
    verify_lock,
    verify_pilot_lock,
)


CANONICAL_FIELDS = (
    "record_id", "variant_id", "endpoint", "detector_revision", "text_sha256",
    "provenance_label", "canonical_ai_score", "native_score", "input_token_count",
    "truncated", "failure",
)
_HUMAN = {"human", "real", "authored", "human_written", "0", "false"}


def _base_text(text: object) -> str:
    return " ".join(str(text).split())


def _text_sha256(text: object) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _attempt_seed(locked_seed: int, request_id: str, attempt: int) -> int:
    payload = f"{int(locked_seed)}:{request_id}:{int(attempt)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647


@dataclass(frozen=True)
class RegistryRecord:
    record_id: str
    corpus: str
    variant_id: str
    text: str
    text_sha256: str
    provenance_label: str
    endpoint: str | None = None


@dataclass(frozen=True)
class ScoreWorkItem:
    record_id: str
    variant_id: str
    endpoint: str
    detector_revision: str
    text: str
    text_sha256: str
    provenance_label: str
    stage: str


class VerifiedRegistry:
    def __init__(self, records: Mapping[tuple[str, str], RegistryRecord], manifest: Mapping[str, object]):
        self.records = dict(records)
        self.manifest = dict(manifest)

    def get(self, record_id: str, variant_id: str) -> RegistryRecord:
        try:
            return self.records[(str(record_id), str(variant_id))]
        except KeyError as error:
            raise ValueError(f"Registry has no locked text: {(record_id, variant_id)}") from error


def _read_rows(path: Path | str) -> tuple[dict[str, str], ...]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _candidate_map(path: Path | str) -> dict[str, dict[str, str]]:
    rows = _read_rows(path)
    if not rows or not {"record_id", "corpus", "text"} <= set(rows[0]):
        raise ValueError("Human candidate CSV requires record_id, corpus, and text")
    result = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in result:
            raise ValueError(f"Duplicate or empty human candidate ID: {record_id}")
        if not str(row.get("text", "")).strip():
            raise ValueError(f"Empty human candidate text: {record_id}")
        result[record_id] = row
    return result


def _add_locked_record(
    records: dict[tuple[str, str], RegistryRecord],
    *,
    record_id: str,
    corpus: str,
    text: str,
    label: str,
    expected: Mapping[str, str],
    probes: Iterable[str] = (),
    width: int = 80,
    block_size: int = 2,
) -> None:
    base = _base_text(text)
    base_hash = _text_sha256(base)
    if base_hash != str(expected.get("original", "")):
        raise ValueError(f"Locked original text hash mismatch: {record_id}")
    records[(record_id, "original")] = RegistryRecord(record_id, corpus, "original", base, base_hash, label)
    for probe in probes:
        if probe not in PROBES:
            raise ValueError(f"Unknown locked probe: {probe}")
        variant = reflow_variant(base, probe, width=width, block_size=block_size)
        digest = _text_sha256(variant)
        if digest != str(expected.get(probe, "")):
            raise ValueError(f"Locked {probe} text hash mismatch: {record_id}")
        records[(record_id, probe)] = RegistryRecord(record_id, corpus, probe, variant, digest, label)
    if "original_repeat" in expected:
        if str(expected["original_repeat"]) != base_hash:
            raise ValueError(f"Original-repeat hash mismatch: {record_id}")
        records[(record_id, "original_repeat")] = RegistryRecord(record_id, corpus, "original_repeat", base, base_hash, label)


def _verify_accepted_generation(
    paths: DeferralPaths,
    accepted_rows: Sequence[Mapping[str, object]],
    panels: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    """Verify runner CSV rows against both generation and generated-panel locks.

    This intentionally does not call ``import_generation_outputs``: the local
    runner's exact importer CSV is text/provenance-only, while token panels are
    already represented by the existing generated-panel lock.
    """
    generation = verify_generation_lock(paths)["payload"]
    requests = {str(row["request_id"]): row for row in generation.get("requests", ())}
    panels_by_request = {str(row["request_id"]): row for row in panels}
    if len(accepted_rows) != len(requests) or set(panels_by_request) != set(requests):
        raise ValueError("Accepted generation outputs and generated-panel lock are incomplete")
    output_by_request: dict[str, Mapping[str, object]] = {}
    for row in accepted_rows:
        request_id = str(row.get("request_id", ""))
        request = requests.get(request_id)
        if request is None or request_id in output_by_request:
            raise ValueError(f"Unknown or duplicate accepted generation request: {request_id}")
        for field in ("generator_family", "generator_revision", "retry", "target_length"):
            if str(row.get(field, "")) != str(request.get(field, "")):
                raise ValueError(f"Accepted generation provenance mismatch: {request_id}/{field}")
        attempt = int(row.get("attempt", 0) or 0)
        if attempt < 0 or attempt > int(request.get("retry", 0)):
            raise ValueError(f"Accepted generation attempt mismatch: {request_id}")
        if str(row.get("seed", "")) not in {"", str(_attempt_seed(int(request["seed"]), request_id, attempt))}:
            raise ValueError(f"Accepted generation seed mismatch: {request_id}")
        if row.get("decoding") not in (None, ""):
            actual = row["decoding"]
            if isinstance(actual, str):
                actual = json.loads(actual)
            locked = request.get("decoding", {})
            if isinstance(locked, str):
                locked = json.loads(locked)
            if json.dumps(actual, sort_keys=True) != json.dumps(locked, sort_keys=True):
                raise ValueError(f"Accepted generation decoding mismatch: {request_id}")
        text = str(row.get("text", row.get("generated_text", "")))
        if not text.strip():
            raise ValueError(f"Accepted generation output is empty: {request_id}")
        base = _base_text(text)
        word_count = len(base.split())
        if not int(request["min_word_count"]) <= word_count <= int(request["max_word_count"]):
            raise ValueError(f"Accepted generation output violates locked length: {request_id}")
        panel = panels_by_request[request_id]
        if _text_sha256(base) != str(panel["base_text_sha256"]):
            raise ValueError(f"Accepted generation base hash mismatch: {request_id}")
        variants = {variant.variant_id: variant for variant in build_reflow_variants(text, probes=PROBES)}
        for probe in PROBES:
            if variants[probe].text_sha256 != str(panel["variants"][probe]["text_sha256"]):
                raise ValueError(f"Accepted generation {probe} hash mismatch: {request_id}")
        output_by_request[request_id] = row
    return output_by_request


def build_verified_registry(
    paths: DeferralPaths,
    human_candidates_csv: Path | str,
    accepted_generation_csv: Path | str,
) -> VerifiedRegistry:
    """Verify every human and generated text against all immutable locks."""
    manifest = verify_pilot_lock(paths)["payload"]
    candidates = _candidate_map(human_candidates_csv)
    records: dict[tuple[str, str], RegistryRecord] = {}
    transform = manifest.get("transform", {})
    transform_width = int(transform.get("width", 80))
    transform_block_size = int(transform.get("sentence_block_size", 2))
    for row in manifest.get("calibration", ()):
        record_id = str(row["record_id"])
        source = candidates.get(record_id)
        if source is None:
            raise ValueError(f"Human candidate is missing calibration record: {record_id}")
        if str(source.get("corpus", "")) != str(row["corpus"]):
            raise ValueError(f"Human candidate corpus mismatch: {record_id}")
        if str(source.get("group_id", "")) and str(source.get("group_id")) != str(row.get("group_id", source.get("group_id"))):
            raise ValueError(f"Human candidate group mismatch: {record_id}")
        _add_locked_record(
            records, record_id=record_id, corpus=str(row["corpus"]), text=source["text"],
            label="human", expected={"original": str(row["text_sha256"])},
            width=transform_width, block_size=transform_block_size,
        )
    for row in manifest.get("pilot", ()):
        record_id = str(row["record_id"])
        source = candidates.get(record_id)
        if source is None:
            raise ValueError(f"Human candidate is missing pilot record: {record_id}")
        if str(source.get("corpus", "")) != str(row["corpus"]):
            raise ValueError(f"Human candidate corpus mismatch: {record_id}")
        if str(source.get("group_id", "")) and str(source.get("group_id")) != str(row.get("group_id", source.get("group_id"))):
            raise ValueError(f"Human candidate group mismatch: {record_id}")
        expected = {"original": str(row["text_sha256"]), "original_repeat": str(row["text_sha256"])}
        expected.update({str(variant["variant_id"]): str(variant["text_sha256"]) for variant in row.get("variants", ())})
        _add_locked_record(
            records, record_id=record_id, corpus=str(row["corpus"]), text=source["text"],
            label="human", expected=expected, probes=tuple(key for key in PROBES if key in expected),
            width=transform_width, block_size=transform_block_size,
        )

    if not paths.panel_lock.exists():
        raise RuntimeError("Accepted generation output requires an existing generated-panel lock")
    verify_generation_lock(paths)
    panel_envelope = verify_lock(paths.panel_lock)
    if panel_envelope["payload"].get("generation_lock_sha256") != _file_sha256(paths.generation_lock):
        raise RuntimeError("Generated-panel lock is bound to a different generation lock")
    accepted_rows = _read_rows(accepted_generation_csv)
    panels = tuple(panel_envelope["payload"].get("panels", ()))
    output_by_request = _verify_accepted_generation(paths, accepted_rows, panels)
    for panel in panels:
        request_id = str(panel["request_id"])
        output = output_by_request.get(request_id)
        if output is None:
            raise ValueError(f"Accepted output missing request: {request_id}")
        record_id = str(panel["ai_record_id"])
        expected = {"original": str(panel["base_text_sha256"]), "original_repeat": str(panel["base_text_sha256"])}
        expected.update({probe: str(panel["variants"][probe]["text_sha256"]) for probe in PROBES})
        _add_locked_record(
            records, record_id=record_id, corpus=str(panel["corpus"]), text=str(output["text"]),
            label="ai", expected=expected, probes=PROBES,
        )
    return VerifiedRegistry(records, manifest)


def stage_work_items(
    paths: DeferralPaths,
    registry: VerifiedRegistry,
    stage: str,
    endpoint: str,
) -> tuple[ScoreWorkItem, ...]:
    """Build only stage-authorized queries for one endpoint/model process."""
    stage = str(stage).casefold()
    if endpoint not in registry.manifest.get("endpoint_revisions", {}):
        raise ValueError(f"Endpoint is not in the locked panel: {endpoint}")
    if stage == "calibration":
        if endpoint != RADAR_ENDPOINT:
            raise ValueError("Calibration is RADAR-only")
        rows = registry.manifest.get("calibration", ())
        return tuple(
            ScoreWorkItem(str(row["record_id"]), "original", endpoint, str(registry.manifest["endpoint_revisions"][endpoint]), registry.get(str(row["record_id"]), "original").text, str(row["text_sha256"]), "human", stage)
            for row in rows
        )
    if stage == "originals":
        if endpoint != RADAR_ENDPOINT:
            raise ValueError("Originals stage is RADAR-only; endpoint-only diagnostics are conditional")
        rows = list(registry.manifest.get("pilot", ()))
        rows.extend({"record_id": key[0], "corpus": value.corpus} for key, value in registry.records.items() if key[1] == "original" and value.provenance_label == "ai")
        seen: set[str] = set()
        output = []
        for row in rows:
            record_id = str(row["record_id"])
            if record_id in seen:
                continue
            seen.add(record_id)
            record = registry.get(record_id, "original")
            output.append(ScoreWorkItem(record_id, "original", endpoint, str(registry.manifest["endpoint_revisions"][endpoint]), record.text, record.text_sha256, record.provenance_label, stage))
        return tuple(output)
    if stage != "conditional":
        raise ValueError(f"Unknown scoring stage: {stage}")
    worklist = verify_conditional_worklist(paths)
    revision = str(registry.manifest["endpoint_revisions"][endpoint])
    output = []
    for row in worklist:
        if str(row["endpoint"]) != endpoint:
            continue
        record = registry.get(str(row["record_id"]), str(row["variant_id"]))
        if record.text_sha256 != str(row["text_sha256"]):
            raise ValueError(f"Conditional worklist hash mismatch: {(record.record_id, record.variant_id)}")
        output.append(ScoreWorkItem(record.record_id, record.variant_id, endpoint, revision, record.text, record.text_sha256, record.provenance_label, stage))
    return tuple(output)


class ScoreJournal:
    """Small resumable SQLite request journal shared by all endpoint runs."""
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS score_requests(
                record_id TEXT NOT NULL, variant_id TEXT NOT NULL, endpoint TEXT NOT NULL,
                detector_revision TEXT NOT NULL, text_sha256 TEXT NOT NULL,
                provenance_label TEXT NOT NULL, request_json TEXT NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                canonical_ai_score REAL, native_score REAL, input_token_count INTEGER,
                truncated INTEGER NOT NULL DEFAULT 0, failure TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL, PRIMARY KEY(record_id,variant_id,endpoint)
            )
        """)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def ensure(self, item: ScoreWorkItem) -> tuple[str, int]:
        key = (item.record_id, item.variant_id, item.endpoint)
        # Stage is intentionally omitted: the same RADAR original can appear
        # first in ``originals`` and later in the locked conditional worklist.
        request_json = json.dumps({"detector_revision": item.detector_revision, "text_sha256": item.text_sha256, "provenance_label": item.provenance_label}, sort_keys=True)
        row = self.connection.execute("SELECT status,attempts,request_json FROM score_requests WHERE record_id=? AND variant_id=? AND endpoint=?", key).fetchone()
        if row is not None and str(row[2]) != request_json:
            raise RuntimeError(f"Scoring journal request changed: {key}")
        if row is None:
            self.connection.execute("INSERT INTO score_requests(record_id,variant_id,endpoint,detector_revision,text_sha256,provenance_label,request_json,status,updated_at) VALUES(?,?,?,?,?,?,?,?,?)", (*key, item.detector_revision, item.text_sha256, item.provenance_label, request_json, "pending", time.time()))
            self.connection.commit()
            return "pending", 0
        return str(row[0]), int(row[1])

    def begin(self, item: ScoreWorkItem) -> int:
        _, attempts = self.ensure(item)
        attempts += 1
        self.connection.execute("UPDATE score_requests SET status='running',attempts=?,failure='',updated_at=? WHERE record_id=? AND variant_id=? AND endpoint=?", (attempts, time.time(), item.record_id, item.variant_id, item.endpoint))
        self.connection.commit()
        return attempts

    def success(self, item: ScoreWorkItem, *, canonical: float, native: float | None, input_tokens: int | None, truncated: bool) -> None:
        self.connection.execute("UPDATE score_requests SET status='success',canonical_ai_score=?,native_score=?,input_token_count=?,truncated=?,failure='',updated_at=? WHERE record_id=? AND variant_id=? AND endpoint=?", (canonical, native, input_tokens, int(bool(truncated)), time.time(), item.record_id, item.variant_id, item.endpoint))
        self.connection.commit()

    def failure(self, item: ScoreWorkItem, error: str) -> None:
        self.connection.execute("UPDATE score_requests SET status='failure',failure=?,updated_at=? WHERE record_id=? AND variant_id=? AND endpoint=?", (str(error), time.time(), item.record_id, item.variant_id, item.endpoint))
        self.connection.commit()

    def export(self, output: Path | str) -> tuple[dict[str, object], ...]:
        rows = self.connection.execute("SELECT record_id,variant_id,endpoint,detector_revision,text_sha256,provenance_label,canonical_ai_score,native_score,input_token_count,truncated FROM score_requests WHERE status='success' AND canonical_ai_score IS NOT NULL AND truncated=0 ORDER BY record_id,variant_id,endpoint").fetchall()
        normalized = tuple({**dict(zip(CANONICAL_FIELDS[:10], row)), "failure": ""} for row in rows)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
            writer.writeheader()
            writer.writerows(normalized)
            handle.flush()
        temporary.replace(output)
        return normalized


def run_stage(
    paths: DeferralPaths,
    human_candidates_csv: Path | str,
    accepted_generation_csv: Path | str,
    *,
    stage: str,
    endpoint: str,
    journal_path: Path | str,
    canonical_output: Path | str,
    adapter_factory: Callable[[str], Any],
    max_attempts: int = 3,
) -> dict[str, int]:
    registry = build_verified_registry(paths, human_candidates_csv, accepted_generation_csv)
    items = stage_work_items(paths, registry, stage, endpoint)
    adapter = adapter_factory(endpoint)
    adapter_spec = getattr(adapter, "spec", None)
    if adapter_spec is not None and str(getattr(adapter_spec, "config_id", endpoint)) != endpoint:
        raise ValueError(f"Adapter identity mismatch: expected {endpoint}")
    if adapter_spec is not None and str(getattr(adapter_spec, "revision", "")) != str(registry.manifest["endpoint_revisions"][endpoint]):
        raise ValueError(f"Adapter revision mismatch: expected {registry.manifest['endpoint_revisions'][endpoint]}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    journal = ScoreJournal(journal_path)
    completed = skipped = failures = 0
    try:
        for item in items:
            status, attempts = journal.ensure(item)
            if status == "success":
                skipped += 1
                continue
            if attempts >= max_attempts:
                failures += 1
                continue
            journal.begin(item)
            try:
                result = adapter.score(item.text)
                if isinstance(result, Mapping):
                    canonical = result.get("canonical_ai_score")
                    native = result.get("native_score")
                    tokens = result.get("input_token_count")
                    truncated = bool(result.get("truncated", False))
                    failure = result.get("failure")
                else:
                    canonical = getattr(result, "canonical_ai_score", None)
                    native = getattr(result, "native_score", None)
                    tokens = getattr(result, "input_token_count", None)
                    truncated = bool(getattr(result, "truncated", False))
                    failure = getattr(result, "failure", None)
                if failure or canonical is None or not math.isfinite(float(canonical)) or truncated:
                    raise ValueError(str(failure or "missing/non-finite/truncated score"))
                journal.success(item, canonical=float(canonical), native=None if native is None else float(native), input_tokens=None if tokens is None else int(tokens), truncated=False)
                completed += 1
            except Exception as error:
                journal.failure(item, f"{type(error).__name__}: {error}")
                failures += 1
        journal.export(canonical_output)
    finally:
        journal.close()
    if failures:
        raise RuntimeError(f"{failures} scoring request(s) failed or exhausted their retry limit")
    return {"requested": len(items), "completed": completed, "skipped": skipped, "failures": failures}


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    from fprint.detectors import build_adapter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--human-candidates", type=Path, required=True)
    parser.add_argument("--accepted-generation", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibration", "originals", "conditional"), required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--mage-repo", type=str)
    args = parser.parse_args(argv)
    paths = DeferralPaths.from_root(args.study_root)
    summary = run_stage(
        paths, args.human_candidates, args.accepted_generation,
        stage=args.stage, endpoint=args.endpoint, journal_path=args.journal,
        canonical_output=args.output,
        adapter_factory=lambda endpoint: build_adapter(endpoint, args.device, args.mage_repo),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["CANONICAL_FIELDS", "RegistryRecord", "ScoreJournal", "ScoreWorkItem", "VerifiedRegistry", "build_verified_registry", "run_stage", "stage_work_items"]
