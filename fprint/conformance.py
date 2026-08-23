from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import sqlite3
import statistics
import subprocess
import unicodedata
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .core import canonical_json, exact_sign_flip, lock_forecasts, make_probe_triplet, slope, verify_lock
from .detectors import SPECS, build_adapter


AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_triplets(
    triplet_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    corpus TEXT NOT NULL,
    group_id TEXT NOT NULL,
    probe TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    original_text TEXT NOT NULL,
    low_text TEXT NOT NULL,
    high_text TEXT NOT NULL,
    low_intensity REAL NOT NULL,
    high_intensity REAL NOT NULL,
    text_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_scores(
    triplet_id TEXT NOT NULL,
    intensity TEXT NOT NULL,
    audited_endpoint TEXT NOT NULL,
    fault_id TEXT NOT NULL,
    effective_endpoint TEXT NOT NULL,
    native_score REAL,
    canonical_ai_score REAL,
    input_token_count INTEGER,
    effective_token_count INTEGER,
    max_tokens INTEGER,
    truncated INTEGER NOT NULL DEFAULT 0,
    runtime_ms REAL,
    failure TEXT,
    adapter_json TEXT NOT NULL,
    PRIMARY KEY(triplet_id,intensity,audited_endpoint,fault_id)
);
CREATE TABLE IF NOT EXISTS audit_score_cache(
    effective_endpoint TEXT NOT NULL,
    scoring_mode TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(effective_endpoint,scoring_mode,input_sha256)
);
"""

PRIMARY_FAMILIES = ("input_handling", "output_policy", "core_computation")
@dataclass(frozen=True)
class FaultSpec:
    fault_id: str
    family: str
    severity: str
    stage: str
    mode: str
    applicable_endpoints: tuple[str, ...] = ()
    effective_implementation: str | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "FaultSpec":
        required = ("fault_id", "family", "severity", "stage", "mode")
        if any(not str(row.get(key, "")).strip() for key in required):
            raise ValueError(f"Malformed fault specification: {row}")
        return cls(
            *(str(row[key]) for key in required),
            tuple(str(value) for value in row.get("applicable_endpoints", ())),
            str(row.get("effective_implementation") or f"fprint.conformance:{row['stage']}:{row['mode']}:v1"),
            dict(row.get("parameters", {})),
        )

    def applies_to(self, endpoint: str) -> bool:
        return not self.applicable_endpoints or endpoint in self.applicable_endpoints


@dataclass(frozen=True)
class AuditPaths:
    root: Path
    database: Path
    lock: Path
    results: Path


def audit_paths(root: Path) -> AuditPaths:
    root = Path(root).resolve()
    return AuditPaths(root, root / "state" / "fault_audit.sqlite3", root / "locks" / "fault_audit.json", root / "results")


def load_fault_config(path: Path) -> tuple[dict, dict[str, FaultSpec]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    faults = [FaultSpec.from_mapping(row) for row in payload.get("faults", ())]
    by_id = {fault.fault_id: fault for fault in faults}
    if len(by_id) != len(faults) or "unchanged" not in by_id:
        raise ValueError("Fault IDs must be unique and include unchanged")
    endpoints = tuple(payload.get("primary_endpoints", ()))
    external = {str(value) for value in payload.get("external_endpoints", ())}
    if not endpoints or any(endpoint not in SPECS and endpoint not in external for endpoint in endpoints):
        raise ValueError("Primary endpoints must be local detector configurations or declared external endpoints")
    if any(fault.family not in (*PRIMARY_FAMILIES, "unchanged", "unknown") for fault in faults):
        raise ValueError("Unknown fault family")
    if any(endpoint not in SPECS and endpoint not in external for fault in faults for endpoint in fault.applicable_endpoints):
        raise ValueError("Fault applies to an unknown endpoint")
    return payload, by_id


def transform_input(mode: str, text: str) -> str:
    if mode == "identity":
        return text
    if mode == "newline_flatten":
        return re.sub(r"[ \t]*\r?\n+[ \t]*", " ", text).strip()
    if mode == "whitespace_collapse":
        return " ".join(text.split())
    if mode == "nfkc_whitespace":
        return " ".join(unicodedata.normalize("NFKC", text).split())
    raise ValueError(f"Unsupported input fault mode: {mode}")


def remap_percentile(value: float, fault: FaultSpec) -> float:
    value = min(1 - 1e-9, max(1e-9, float(value)))
    if fault.mode not in {"logit_bias", "temperature", "combined", "combined_core"}:
        return value
    logit = math.log(value / (1 - value))
    if fault.mode == "temperature":
        temperature = float(fault.parameters["temperature"])
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        logit /= temperature
    else:
        logit += float(fault.parameters.get("bias", 0.0))
    return 1 / (1 + math.exp(-logit))


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(AUDIT_SCHEMA)
    return connection


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _verify_audit_database(paths: AuditPaths, manifest: Mapping[str, object]) -> None:
    connection = _readonly(paths.database)
    try:
        metadata = connection.execute(
            "SELECT value FROM metadata WHERE key='manifest_sha256'"
        ).fetchone()
        if metadata is None or str(metadata[0]) != _digest(manifest):
            raise RuntimeError("Fault-audit database is not bound to the verified manifest")
        locked_ids = [
            *[str(value) for value in manifest.get("triplet_ids", ())],
            *[str(value) for value in manifest.get("confirmation_candidate_ids", ())],
        ]
        if locked_ids:
            rows = {
                str(row[0]): tuple(row)
                for row in connection.execute(
                    """SELECT triplet_id,record_id,corpus,group_id,probe,source_kind,
                              original_text,low_text,high_text,low_intensity,high_intensity,text_sha256
                         FROM audit_triplets"""
                )
            }
            if set(rows) != set(locked_ids) or _digest([list(rows[value]) for value in locked_ids]) != manifest.get("triplet_rows_sha256"):
                raise RuntimeError("Fault-audit triplet table disagrees with its manifest digest")
        if manifest.get("frozen_score_rows_sha256"):
            frozen = [
                tuple(row) for row in connection.execute(
                    """SELECT s.triplet_id,s.intensity,s.audited_endpoint,s.native_score,
                              s.canonical_ai_score,s.truncated,s.failure
                         FROM audit_scores s JOIN audit_triplets t USING(triplet_id)
                        WHERE s.fault_id='unchanged' AND t.source_kind='discovery'
                        ORDER BY 1,2,3"""
                )
            ]
            if _digest(frozen) != manifest["frozen_score_rows_sha256"]:
                raise RuntimeError("Imported frozen scores disagree with their manifest digest")
    finally:
        connection.close()


def _triplet_digest(row: Sequence[object]) -> str:
    return _digest([str(value) for value in row])


def _source_triplets(
    source_root: Path,
    probes: Sequence[str],
    corpora: Sequence[str],
    anchors_per_cell: int = 50,
) -> list[tuple]:
    profile_path = source_root / "folds" / "bawe" / "artifacts" / "zero" / "profiles.json"
    selected = set(json.loads(profile_path.read_text(encoding="utf-8"))["selected_triplet_ids"])
    connection = _readonly(source_root / "folds" / "bawe" / "fprint.sqlite3")
    try:
        rows = connection.execute(
            """SELECT p.triplet_id,p.record_id,p.corpus,r.group_id,p.probe,
                      p.original_text,p.low_text,p.high_text,p.low_intensity,p.high_intensity
                 FROM probe_triplets p JOIN records r USING(record_id)
                WHERE p.probe IN ({}) AND p.corpus IN ({})
                ORDER BY p.corpus,p.probe,p.triplet_id""".format(
                    ",".join("?" for _ in probes), ",".join("?" for _ in corpora),
                ),
            (*probes, *corpora),
        ).fetchall()
    finally:
        connection.close()
    result = []
    for row in rows:
        if str(row["triplet_id"]) not in selected:
            continue
        values = tuple(row)
        result.append((*values[:5], "discovery", *values[5:], _triplet_digest(values)))
    counts = {
        (corpus, probe): sum(row[2] == corpus and row[4] == probe for row in result)
        for corpus in corpora for probe in probes
    }
    sparse = {key: count for key, count in counts.items() if count < 5}
    if sparse:
        raise RuntimeError(f"Discovery panel has probe/corpus cells with fewer than five triplets: {sparse}")
    if any(count > anchors_per_cell for count in counts.values()):
        raise RuntimeError("Discovery panel exceeds its frozen per-cell cap")
    return result


def _confirmation_candidates(
    source_root: Path,
    corpora: Sequence[str],
    seed: int,
    limit: int,
) -> list[tuple]:
    connection = _readonly(source_root / "folds" / "bawe" / "fprint.sqlite3")
    try:
        used_groups = {
            str(row[0]) for row in connection.execute(
                "SELECT DISTINCT r.group_id FROM probe_triplets p JOIN records r USING(record_id)"
            )
        }
        rows = connection.execute(
            """SELECT record_id,corpus,group_id,text FROM records
                WHERE partition_name='anchor_candidates'
                ORDER BY corpus,group_id,record_id"""
        ).fetchall()
    finally:
        connection.close()
    by_corpus: dict[str, dict[str, tuple]] = {corpus: {} for corpus in corpora}
    for row in rows:
        corpus, group_id = str(row["corpus"]), str(row["group_id"])
        if corpus not in by_corpus or group_id in used_groups or group_id in by_corpus[corpus]:
            continue
        triplet = make_probe_triplet("paragraph_resegmentation", str(row["text"]), str(row["record_id"]))
        if triplet is None:
            continue
        triplet_id = hashlib.sha256(f"fault-confirmation:{row['record_id']}".encode()).hexdigest()
        values = (
            triplet_id, str(row["record_id"]), corpus, group_id,
            "paragraph_resegmentation", "confirmation_candidate",
            triplet.original, triplet.low, triplet.high,
            triplet.low_intensity, triplet.high_intensity,
        )
        by_corpus[corpus][group_id] = (*values, _triplet_digest(values))
    result = []
    for corpus in corpora:
        ordered = sorted(
            by_corpus[corpus].values(),
            key=lambda row: hashlib.sha256(f"{seed}:{row[0]}".encode()).hexdigest(),
        )[:limit]
        if len(ordered) < 50:
            raise RuntimeError(f"{corpus} has only {len(ordered)} unused paragraph-eligible groups")
        result.extend(ordered)
    return result


def _import_frozen_scores(source_root: Path, audit: sqlite3.Connection, triplet_ids: Sequence[str]) -> int:
    source = _readonly(source_root / "folds" / "bawe" / "fprint.sqlite3")
    try:
        rows = source.execute(
            """SELECT p.triplet_id,s.variant_id,s.detector_config,s.native_score,
                      s.canonical_ai_score,s.input_token_count,s.effective_token_count,
                      s.max_tokens,s.truncated,s.runtime_ms,s.failure,s.adapter_json
                 FROM probe_triplets p JOIN scores s USING(record_id)
                WHERE p.triplet_id IN ({}) AND s.variant_id<>'original'""".format(
                    ",".join("?" for _ in triplet_ids)
                ),
            tuple(triplet_ids),
        ).fetchall()
    finally:
        source.close()
    payload = []
    for row in rows:
        intensity = str(row["variant_id"]).rsplit(":", 1)[-1]
        if intensity not in {"original", "low", "high"}:
            continue
        endpoint = str(row["detector_config"])
        payload.append((
            row["triplet_id"], intensity, endpoint, "unchanged", endpoint,
            row["native_score"], row["canonical_ai_score"], row["input_token_count"],
            row["effective_token_count"], row["max_tokens"], row["truncated"],
            row["runtime_ms"], row["failure"], row["adapter_json"],
        ))
    with audit:
        audit.executemany("INSERT OR REPLACE INTO audit_scores VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", payload)
    return len(payload)


def prepare_fault_audit(source_root: Path, audit_root: Path, config_path: Path, evaluation_path: Path) -> dict:
    source_root, paths = Path(source_root).resolve(), audit_paths(audit_root)
    config, faults = load_fault_config(config_path)
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    primary_corpora = tuple(str(value) for value in evaluation["primary_corpora"])
    all_corpora = (*primary_corpora, *(
        corpus for corpus in ("bawe",) if corpus not in primary_corpora
    ))
    triplets = _source_triplets(
        source_root, config["probes"], primary_corpora,
        int(config.get("anchors_per_probe_per_corpus", 50)),
    )
    candidates = _confirmation_candidates(
        source_root, all_corpora, int(config["seed"]),
        int(config["confirmation_candidates_per_corpus"]),
    )
    manifest = {
        "schema_version": 1,
        "construct": "black_box_behavioral_conformance_and_coarse_fault_localization",
        "source_root": str(source_root),
        "source_database": str(source_root / "folds" / "bawe" / "fprint.sqlite3"),
        "source_database_size": (source_root / "folds" / "bawe" / "fprint.sqlite3").stat().st_size,
        "reference_database": str(source_root / "state" / "fprint.sqlite3"),
        "reference_database_size": (source_root / "state" / "fprint.sqlite3").stat().st_size,
        "frozen_threshold_artifact_sha256": _file_digest(source_root / "state" / "frozen_thresholds.json"),
        "primary_corpora": list(primary_corpora),
        "confirmation_corpora": list(all_corpora),
        "config": config,
        "faults": [asdict(fault) for fault in faults.values()],
        "triplet_ids": [row[0] for row in triplets],
        "discovery_cell_counts": {
            f"{corpus}:{probe}": sum(row[2] == corpus and row[4] == probe for row in triplets)
            for corpus in primary_corpora for probe in config["probes"]
        },
        "confirmation_candidate_ids": [row[0] for row in candidates],
        "triplet_rows_sha256": _digest([list(row) for row in (*triplets, *candidates)]),
        "detector_revisions": {
            endpoint: {
                "model": SPECS[endpoint].model_id,
                "model_revision": SPECS[endpoint].revision,
                "tokenizer_revision": SPECS[endpoint].tokenizer_revision,
                "implementation_revision": SPECS[endpoint].implementation_revision,
                "preprocessing_revision": SPECS[endpoint].preprocessing_revision,
            }
            for endpoint in SPECS
        },
        "code_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "code_sha256": {
            path.name: _file_digest(path)
            for path in (
                Path(__file__), Path(__file__).with_name("core.py"),
                Path(__file__).with_name("detectors.py"), Path(config_path),
            )
        },
        "analysis_contract": {
            "channels": ["raw", "fingerprint", "combined"],
            "normalization": "within_run_rank_geometry_plus_frozen_reference_output_percentiles",
            "outer_split": "leave_one_corpus_out",
            "diagnosis_split": "leave_one_corpus_and_fault_variant_out",
            "unknown_policy": "distance_and_margin_abstention",
            "claims_excluded": ["deployment_fpr", "exact_proprietary_internal_change"],
        },
    }
    if paths.lock.exists():
        existing = verify_lock(paths.lock)["payload"]
        immutable_keys = (
            "source_root", "reference_database", "frozen_threshold_artifact_sha256",
            "primary_corpora", "confirmation_corpora", "config",
            "faults", "triplet_ids", "confirmation_candidate_ids", "triplet_rows_sha256",
        )
        if any(_digest(existing.get(key)) != _digest(manifest.get(key)) for key in immutable_keys):
            raise RuntimeError("Existing fault-audit lock disagrees with the requested manifest")
        return existing
    if paths.database.exists():
        raise RuntimeError("Unbound fault-audit database exists without a manifest lock")
    audit = _connect(paths.database)
    try:
        with audit:
            audit.executemany("INSERT INTO audit_triplets VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (*triplets, *candidates))
        imported = _import_frozen_scores(source_root, audit, [row[0] for row in triplets])
        manifest["imported_frozen_score_rows"] = imported
        manifest["frozen_score_rows_sha256"] = _digest([
            tuple(row) for row in audit.execute(
                """SELECT triplet_id,intensity,audited_endpoint,native_score,canonical_ai_score,
                          truncated,failure FROM audit_scores ORDER BY 1,2,3"""
            )
        ])
        with audit:
            audit.execute("INSERT INTO metadata VALUES('manifest_sha256',?)", (_digest(manifest),))
    finally:
        audit.close()
    lock_forecasts(paths.lock, manifest)
    return manifest


def _score_payload(adapter: object, text: str, mode: str) -> dict:
    if mode == "mean_logprob":
        started = __import__("time").perf_counter()
        sequence = adapter.scorer.sequence(text)  # type: ignore[attr-defined]
        values = sequence["log_probs"]
        native = float(sum(float(value) for value in values) / len(values))
        return {
            "native_score": native, "canonical_ai_score": native,
            "input_token_count": int(sequence["token_count"]),
            "effective_token_count": int(sequence["token_count"]),
            "max_tokens": adapter.spec.max_tokens, "truncated": False,
            "runtime_ms": (__import__("time").perf_counter() - started) * 1000,
            "failure": None, "cache_hash": sequence.get("cache_hash"),
            "statistic": "mean_token_log_probability",
        }
    return asdict(adapter.score(text))  # type: ignore[attr-defined]


def score_fault_audit(
    audit_root: Path,
    endpoint: str,
    fault_id: str,
    *,
    device: int = 0,
    mage_repo: str | None = None,
    source_kind: str = "all",
) -> dict[str, int]:
    paths = audit_paths(audit_root)
    manifest = verify_lock(paths.lock)["payload"]
    _verify_audit_database(paths, manifest)
    _config, faults = load_fault_config(Path(paths.root) / "fault_audit_config.json") if (paths.root / "fault_audit_config.json").is_file() else (
        manifest["config"], {row["fault_id"]: FaultSpec.from_mapping(row) for row in manifest["faults"]}
    )
    if endpoint not in SPECS or fault_id not in faults:
        raise ValueError("Unknown endpoint or fault")
    fault = faults[fault_id]
    if not fault.applies_to(endpoint):
        raise ValueError(f"{fault_id} does not apply to {endpoint}")
    if fault.stage in {"post_score", "decision", "derived"} or fault.mode == "endpoint_replacement":
        return {"completed": 0, "skipped": 0, "rejected_triplets": 0, "derived": 1}
    effective = str(fault.parameters.get("replacement", endpoint))
    adapter_endpoint = "logrank__qwen2_5_0_5b_fp32" if fault.mode == "mean_logprob" else effective
    adapter = build_adapter(adapter_endpoint, device, mage_repo)
    audit = _connect(paths.database)
    filters, parameters = "", ()
    if source_kind != "all":
        if source_kind not in {"discovery", "confirmation_candidate"}:
            raise ValueError("source_kind must be all, discovery, or confirmation_candidate")
        filters, parameters = " WHERE source_kind=?", (source_kind,)
    rows = audit.execute(
        "SELECT * FROM audit_triplets" + filters + " ORDER BY corpus,probe,triplet_id",
        parameters,
    ).fetchall()
    completed = skipped = rejected = 0
    try:
        for row in rows:
            existing = audit.execute(
                """SELECT COUNT(*) FROM audit_scores WHERE triplet_id=?
                   AND audited_endpoint=? AND fault_id=?""",
                (row["triplet_id"], endpoint, fault_id),
            ).fetchone()[0]
            if existing == 3:
                skipped += 3
                continue
            texts = {
                level: transform_input(fault.mode, str(row[f"{level}_text"]))
                for level in ("original", "low", "high")
            }
            capacity = min(SPECS[adapter_endpoint].max_tokens - 32, 460)
            counts = {level: adapter.token_count(text) for level, text in texts.items()}
            if max(counts.values()) > capacity:
                rejected += 1
                for level in texts:
                    payload = {
                        "native_score": None, "canonical_ai_score": None,
                        "input_token_count": counts[level], "effective_token_count": 0,
                        "max_tokens": SPECS[adapter_endpoint].max_tokens,
                        "truncated": True, "runtime_ms": 0.0,
                        "failure": "full_triplet_rejected_capacity",
                    }
                    _insert_score(audit, row["triplet_id"], level, endpoint, fault_id, effective, payload)
                continue
            for level, text in texts.items():
                if audit.execute(
                    "SELECT 1 FROM audit_scores WHERE triplet_id=? AND intensity=? AND audited_endpoint=? AND fault_id=?",
                    (row["triplet_id"], level, endpoint, fault_id),
                ).fetchone():
                    skipped += 1
                    continue
                if fault.mode != "mean_logprob" and text == str(row[f"{level}_text"]):
                    frozen = audit.execute(
                        """SELECT adapter_json FROM audit_scores
                            WHERE triplet_id=? AND intensity=? AND audited_endpoint=? AND fault_id='unchanged'""",
                        (row["triplet_id"], level, endpoint),
                    ).fetchone()
                    if frozen:
                        payload = json.loads(frozen[0])
                        payload["reused_from_frozen_unchanged"] = True
                        _insert_score(audit, row["triplet_id"], level, endpoint, fault_id, effective, payload)
                        completed += 1
                        continue
                input_sha256 = hashlib.sha256(text.encode()).hexdigest()
                scoring_mode = "mean_logprob" if fault.mode == "mean_logprob" else "adapter_score"
                cached = audit.execute(
                    """SELECT payload_json FROM audit_score_cache
                        WHERE effective_endpoint=? AND scoring_mode=? AND input_sha256=?""",
                    (effective, scoring_mode, input_sha256),
                ).fetchone()
                if cached:
                    payload = json.loads(cached[0])
                    payload["reused_from_input_cache"] = True
                else:
                    try:
                        payload = _score_payload(adapter, text, fault.mode)
                    except Exception as error:
                        payload = {
                            "native_score": None, "canonical_ai_score": None,
                            "input_token_count": counts[level], "effective_token_count": 0,
                            "max_tokens": SPECS[adapter_endpoint].max_tokens,
                            "truncated": False, "runtime_ms": 0.0,
                            "failure": f"{type(error).__name__}: {error}",
                        }
                    if not payload.get("failure"):
                        with audit:
                            audit.execute(
                                "INSERT OR IGNORE INTO audit_score_cache VALUES(?,?,?,?)",
                                (effective, scoring_mode, input_sha256, json.dumps(payload, sort_keys=True, default=str)),
                            )
                _insert_score(audit, row["triplet_id"], level, endpoint, fault_id, effective, payload)
                completed += 1
    finally:
        audit.close()
    return {"completed": completed, "skipped": skipped, "rejected_triplets": rejected, "derived": 0}


def _insert_score(
    connection: sqlite3.Connection,
    triplet_id: str,
    intensity: str,
    endpoint: str,
    fault_id: str,
    effective_endpoint: str,
    payload: Mapping[str, object],
) -> None:
    with connection:
        connection.execute(
            "INSERT OR REPLACE INTO audit_scores VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                triplet_id, intensity, endpoint, fault_id, effective_endpoint,
                payload.get("native_score"), payload.get("canonical_ai_score"),
                payload.get("input_token_count"), payload.get("effective_token_count"),
                payload.get("max_tokens"), int(bool(payload.get("truncated"))),
                payload.get("runtime_ms"), payload.get("failure"),
                json.dumps(payload, sort_keys=True, default=str),
            ),
        )


def import_score_table(audit_root: Path, table: Path) -> int:
    paths = audit_paths(audit_root)
    manifest = verify_lock(paths.lock)["payload"]
    _verify_audit_database(paths, manifest)
    faults = {row["fault_id"] for row in manifest["faults"]}
    config = manifest["config"]
    allowed_endpoints = {
        str(value) for key in ("primary_endpoints", "control_endpoints", "external_endpoints")
        for value in config.get(key, ())
    }
    allowed_endpoints.update(
        str(row.get("parameters", {}).get("replacement"))
        for row in manifest["faults"] if row.get("parameters", {}).get("replacement")
    )
    required = {
        "triplet_id", "intensity", "audited_endpoint", "fault_id",
        "effective_endpoint", "native_score", "canonical_ai_score",
    }
    connection = _connect(paths.database)
    imported = 0
    try:
        known_triplets = {str(row[0]) for row in connection.execute("SELECT triplet_id FROM audit_triplets")}
        with Path(table).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not required <= set(reader.fieldnames or ()):
                raise ValueError(f"Score table lacks fields: {sorted(required - set(reader.fieldnames or ())) }")
            for row in reader:
                if row["triplet_id"] not in known_triplets or row["fault_id"] not in faults:
                    raise ValueError("Score table references an unlocked triplet or fault")
                if (
                    row["audited_endpoint"] not in allowed_endpoints
                    or row["effective_endpoint"] not in allowed_endpoints
                    or row["intensity"] not in {"original", "low", "high"}
                ):
                    raise ValueError("Invalid endpoint or intensity in score table")
                payload = {
                    key: _optional_number(row.get(key))
                    for key in (
                        "native_score", "canonical_ai_score", "input_token_count",
                        "effective_token_count", "max_tokens", "runtime_ms",
                    )
                }
                payload.update({
                    "truncated": str(row.get("truncated", "0")).casefold() in {"1", "true", "yes"},
                    "failure": row.get("failure") or None,
                    "external_import": True,
                })
                _insert_score(
                    connection, row["triplet_id"], row["intensity"],
                    row["audited_endpoint"], row["fault_id"], row["effective_endpoint"], payload,
                )
                imported += 1
    finally:
        connection.close()
    return imported


def _optional_number(value: object) -> float | None:
    return None if value is None or str(value).strip() == "" else float(value)


def fault_audit_readiness(audit_root: Path) -> dict:
    paths = audit_paths(audit_root)
    manifest = verify_lock(paths.lock)["payload"]
    _verify_audit_database(paths, manifest)
    config = manifest["config"]
    faults = {row["fault_id"]: FaultSpec.from_mapping(row) for row in manifest["faults"]}
    required: set[tuple[str, str]] = set()
    endpoints = {str(value) for value in config["primary_endpoints"]}
    endpoints.update(
        str(fault.parameters["replacement"])
        for fault in faults.values() if fault.mode == "endpoint_replacement"
    )
    required.update((endpoint, "unchanged") for endpoint in endpoints)
    required.update(
        (endpoint, fault.fault_id)
        for fault in faults.values() if fault.stage == "pre_inference"
        for endpoint in config["primary_endpoints"] if fault.applies_to(str(endpoint))
    )
    required.update(
        (endpoint, fault.fault_id)
        for fault in faults.values() if fault.mode == "mean_logprob"
        for endpoint in config["primary_endpoints"] if fault.applies_to(str(endpoint))
    )
    connection = _connect(paths.database)
    try:
        triplet_counts = {
            str(row[0]): int(row[1])
            for row in connection.execute("SELECT source_kind,COUNT(*) FROM audit_triplets GROUP BY source_kind")
        }
        missing = []
        for source_kind, triplet_count in sorted(triplet_counts.items()):
            expected = triplet_count * 3
            for endpoint, fault_id in sorted(required):
                observed = connection.execute(
                    """SELECT COUNT(*) FROM audit_scores s JOIN audit_triplets t USING(triplet_id)
                        WHERE t.source_kind=? AND s.audited_endpoint=? AND s.fault_id=?""",
                    (source_kind, endpoint, fault_id),
                ).fetchone()[0]
                if observed != expected:
                    missing.append({
                        "source_kind": source_kind, "endpoint": endpoint, "fault_id": fault_id,
                        "expected_rows": expected, "observed_rows": observed,
                        "missing_rows": expected - observed,
                    })
    finally:
        connection.close()
    return {"ready": not missing, "required_endpoint_fault_pairs": len(required), "missing": missing}


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a quantile of no values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _reference_distributions(manifest: Mapping[str, object]) -> dict[str, list[float]]:
    source = _readonly(Path(str(manifest["reference_database"])))
    try:
        rows = source.execute(
            """SELECT s.detector_config,s.canonical_ai_score
                 FROM scores s JOIN records r USING(record_id)
                WHERE r.partition_name='threshold_reference'
                  AND s.canonical_ai_score IS NOT NULL AND s.failure IS NULL AND s.truncated=0"""
        ).fetchall()
    finally:
        source.close()
    result: dict[str, list[float]] = {}
    for row in rows:
        result.setdefault(str(row["detector_config"]), []).append(float(row["canonical_ai_score"]))
    for values in result.values():
        values.sort()
    return result


def _score_signal(score: float, endpoint: str, references: Mapping[str, Sequence[float]]) -> float:
    reference = references.get(endpoint)
    if reference:
        return bisect_right(reference, score) / len(reference)
    # Mean-log-probability has no separate frozen reference distribution. Its
    # bounded transform is intentionally simple and is standardized inside folds.
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, score))))


def _load_audit_state(paths: AuditPaths) -> tuple[list[dict], dict[tuple[str, str, str, str], dict]]:
    connection = _connect(paths.database)
    try:
        triplets = [dict(row) for row in connection.execute(
            "SELECT * FROM audit_triplets ORDER BY corpus,probe,triplet_id"
        )]
        score_rows = connection.execute(
            "SELECT * FROM audit_scores ORDER BY triplet_id,intensity,audited_endpoint,fault_id"
        ).fetchall()
    finally:
        connection.close()
    scores = {
        (str(row["triplet_id"]), str(row["intensity"]), str(row["audited_endpoint"]), str(row["fault_id"])): dict(row)
        for row in score_rows
    }
    return triplets, scores


def _core_fault_for(endpoint: str, faults: Mapping[str, FaultSpec]) -> FaultSpec | None:
    return next((
        fault for fault in faults.values()
        if fault.family == "core_computation" and fault.applies_to(endpoint)
    ), None)


def _resolve_score(
    triplet_id: str,
    intensity: str,
    endpoint: str,
    fault: FaultSpec,
    faults: Mapping[str, FaultSpec],
    scores: Mapping[tuple[str, str, str, str], Mapping[str, object]],
    references: Mapping[str, Sequence[float]],
) -> tuple[float, float] | None:
    source_fault, effective = fault, endpoint
    remap_fault: FaultSpec | None = None
    if fault.family == "output_policy":
        source_fault, remap_fault = faults["unchanged"], fault
    elif fault.mode == "combined":
        source_fault = faults[str(fault.parameters["source_fault"])]
        remap_fault = fault
    elif fault.mode == "combined_core":
        source_fault = _core_fault_for(endpoint, faults) or faults["unchanged"]
        remap_fault = fault
    if source_fault.mode == "endpoint_replacement":
        effective = str(source_fault.parameters["replacement"])
    explicit = scores.get((triplet_id, intensity, endpoint, source_fault.fault_id))
    if explicit is None and source_fault.mode == "endpoint_replacement":
        explicit = scores.get((triplet_id, intensity, effective, "unchanged"))
    if explicit is None:
        return None
    if explicit.get("failure") or bool(explicit.get("truncated")) or explicit.get("canonical_ai_score") is None:
        return None
    native = float(explicit["canonical_ai_score"])
    signal = _score_signal(native, effective, references)
    if remap_fault is not None and remap_fault.mode != "threshold_policy":
        signal = remap_percentile(signal, remap_fault)
    return signal, native


def _rank_percentiles(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    return [(bisect_right(ordered, value) - 0.5) / len(ordered) for value in values]


def _draw_triplets(
    triplets: Sequence[Mapping[str, object]],
    corpus: str,
    probes: Sequence[str],
    budget: int,
    draw: int,
    seed: int,
    cap_at_available: bool = False,
) -> list[Mapping[str, object]]:
    selected: list[Mapping[str, object]] = []
    for probe in probes:
        candidates = [row for row in triplets if row["corpus"] == corpus and row["probe"] == probe]
        candidates.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{draw}:{corpus}:{probe}:{row['group_id']}:{row['triplet_id']}".encode()
        ).hexdigest())
        groups: set[str] = set()
        for row in candidates:
            group = str(row["group_id"])
            if group in groups:
                continue
            groups.add(group)
            selected.append(row)
            if len(groups) == budget:
                break
        if len(groups) < budget and not cap_at_available:
            return []
    return selected


def _feature_vector(
    selected: Sequence[Mapping[str, object]],
    endpoint: str,
    fault: FaultSpec,
    faults: Mapping[str, FaultSpec],
    scores: Mapping[tuple[str, str, str, str], Mapping[str, object]],
    references: Mapping[str, Sequence[float]],
    probes: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]] | None:
    values: list[float] = []
    native_values: list[float] = []
    metadata: list[tuple[str, str, float, float]] = []
    for row in selected:
        intensities = {"original": 0.0, "low": float(row["low_intensity"]), "high": float(row["high_intensity"])}
        resolved = []
        for level in ("original", "low", "high"):
            score = _resolve_score(str(row["triplet_id"]), level, endpoint, fault, faults, scores, references)
            if score is None:
                return None
            signal, native = score
            values.append(signal)
            native_values.append(native)
            metadata.append((str(row["probe"]), level, intensities[level], native))
            resolved.append(signal)
    ranks = _rank_percentiles(values)
    originals = [values[index] for index, (_, level, _, _) in enumerate(metadata) if level == "original"]
    deployed_cutoff = .99 if fault.mode == "threshold_policy" else .95
    features = {
        "raw_mean": _mean(originals),
        "raw_sd": statistics.pstdev(originals) if len(originals) > 1 else 0.0,
        "raw_q10": _quantile(originals, .10),
        "raw_q50": _quantile(originals, .50),
        "raw_q90": _quantile(originals, .90),
        "raw_flag05": _mean([float(value >= deployed_cutoff) for value in originals]),
        "raw_flag01": _mean([float(value >= .99) for value in originals]),
    }
    by_probe: dict[str, list[tuple[str, float, float]]] = {probe: [] for probe in probes}
    for index, (probe, level, intensity, _) in enumerate(metadata):
        by_probe[probe].append((level, intensity, ranks[index]))
    contributions: dict[str, float] = {}
    for probe in probes:
        rows = by_probe[probe]
        triplet_count = len(rows) // 3
        lows, highs, slopes = [], [], []
        for index in range(triplet_count):
            chunk = rows[index * 3:(index + 1) * 3]
            level_scores = {level: value for level, _, value in chunk}
            lows.append(level_scores["low"] - level_scores["original"])
            highs.append(level_scores["high"] - level_scores["original"])
            slopes.append(slope([item[1] for item in chunk], [item[2] for item in chunk]))
        for suffix, aggregate in (("low_shift", lows), ("high_shift", highs), ("slope", slopes)):
            name = f"probe__{probe}__{suffix}"
            features[name] = _mean(aggregate)
            contributions[name] = features[name]
    diagnostics = {
        "native_original_mean": _mean([
            native_values[index] for index, (_, level, _, _) in enumerate(metadata) if level == "original"
        ]),
    }
    return features, diagnostics


def _make_observations(
    triplets: Sequence[Mapping[str, object]],
    scores: Mapping[tuple[str, str, str, str], Mapping[str, object]],
    references: Mapping[str, Sequence[float]],
    faults: Mapping[str, FaultSpec],
    config: Mapping[str, object],
    source_kind: str,
) -> list[dict]:
    probes = list(config["probes"]) if source_kind == "discovery" else ["paragraph_resegmentation"]
    corpora = sorted({str(row["corpus"]) for row in triplets if row["source_kind"] == source_kind})
    budgets = list(config["query_budgets"]) if source_kind == "discovery" else [int(config["confirmation_groups_per_corpus"])]
    draws = int(config["draws"]) if source_kind == "discovery" else 1
    endpoints = [str(value) for value in config["primary_endpoints"]]
    pool = [row for row in triplets if row["source_kind"] == source_kind]
    observations = []
    for corpus in corpora:
        for budget in budgets:
            for draw in range(draws):
                selected = _draw_triplets(
                    pool, corpus, probes, int(budget), draw, int(config["seed"]),
                    cap_at_available=source_kind == "discovery",
                )
                if not selected:
                    continue
                for endpoint in endpoints:
                    for fault in faults.values():
                        if not fault.applies_to(endpoint):
                            continue
                        result = _feature_vector(selected, endpoint, fault, faults, scores, references, probes)
                        if result is None:
                            continue
                        features, diagnostics = result
                        observations.append({
                            "source_kind": source_kind, "corpus": corpus, "budget": budget,
                            "draw": draw, "endpoint": endpoint, "fault_id": fault.fault_id,
                            "family": fault.family, "features": features, **diagnostics,
                            "group_ids": [str(row["group_id"]) for row in selected],
                            "probe_group_counts": {
                                probe: len({str(row["group_id"]) for row in selected if row["probe"] == probe})
                                for probe in probes
                            },
                        })
    return observations


def _feature_names(observations: Sequence[Mapping[str, object]], channel: str) -> list[str]:
    names = sorted({name for row in observations for name in row["features"]})  # type: ignore[index]
    if channel == "raw":
        return [name for name in names if name.startswith("raw_")]
    if channel == "fingerprint":
        return [name for name in names if name.startswith("probe__")]
    return names


def _fit_standardizer(rows: Sequence[Mapping[str, object]], names: Sequence[str]) -> tuple[list[float], list[float]]:
    columns = [[float(row["features"][name]) for row in rows] for name in names]  # type: ignore[index]
    means = [_mean(column) for column in columns]
    scales = [statistics.pstdev(column) if len(column) > 1 else 1.0 for column in columns]
    return means, [scale if scale > 1e-9 else 1.0 for scale in scales]


def _vector(row: Mapping[str, object], names: Sequence[str], means: Sequence[float], scales: Sequence[float]) -> list[float]:
    return [
        (float(row["features"][name]) - mean) / scale  # type: ignore[index]
        for name, mean, scale in zip(names, means, scales)
    ]


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    return [_mean([row[index] for row in vectors]) for index in range(len(vectors[0]))]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(_mean([(a - b) ** 2 for a, b in zip(left, right)]))


def _auc(labels: Sequence[int], values: Sequence[float]) -> float | None:
    pairs = sorted(zip(values, labels))
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None
    positive_rank_sum, index = 0.0, 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)


def _macro_f1(rows: Sequence[Mapping[str, object]]) -> float:
    scores = []
    for family in PRIMARY_FAMILIES:
        true_positive = sum(row["family"] == family and row["prediction"] == family for row in rows)
        false_positive = sum(row["family"] != family and row["prediction"] == family for row in rows)
        false_negative = sum(row["family"] == family and row["prediction"] != family for row in rows)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return _mean(scores)


def _evaluate_channel(
    observations: Sequence[Mapping[str, object]],
    channel: str,
) -> list[dict]:
    results = []
    corpora = sorted({str(row["corpus"]) for row in observations})
    budgets = sorted({int(row["budget"]) for row in observations})
    for budget in budgets:
        at_budget = [row for row in observations if int(row["budget"]) == budget]
        for held_out in corpora:
            train = [row for row in at_budget if row["corpus"] != held_out]
            test = [row for row in at_budget if row["corpus"] == held_out]
            if not test:
                continue
            names = _feature_names(at_budget, channel)
            reference_models = {}
            for endpoint in sorted({str(row["endpoint"]) for row in test}):
                unchanged = [
                    row for row in train
                    if row["family"] == "unchanged" and row["endpoint"] == endpoint
                ]
                if not unchanged:
                    continue
                means, scales = _fit_standardizer(unchanged, names)
                unchanged_vectors = [_vector(row, names, means, scales) for row in unchanged]
                reference = _centroid(unchanged_vectors)
                reference_models[endpoint] = {
                    "means": means, "scales": scales, "reference": reference,
                    "alarm_threshold": _quantile([
                        _distance(vector, reference) for vector in unchanged_vectors
                    ], .95),
                    "native_reference": _mean([
                        float(row["native_original_mean"]) for row in unchanged
                    ]),
                }
            known_train = [
                row for row in train
                if row["family"] in PRIMARY_FAMILIES and row["endpoint"] in reference_models
            ]
            train_vectors = {
                id(row): _vector(
                    row, names,
                    reference_models[str(row["endpoint"])]["means"],
                    reference_models[str(row["endpoint"])]["scales"],
                )
                for row in known_train
            }
            diagnosis_models: dict[str, tuple[dict[str, list[float]], float, float]] = {}

            def diagnosis_model(excluded_fault: str) -> tuple[dict[str, list[float]], float, float]:
                cached = diagnosis_models.get(excluded_fault)
                if cached is not None:
                    return cached
                centroids = {}
                for family in PRIMARY_FAMILIES:
                    family_rows = [
                        row for row in known_train
                        if row["family"] == family and row["fault_id"] != excluded_fault
                    ]
                    if family_rows:
                        centroids[family] = _centroid([train_vectors[id(row)] for row in family_rows])
                calibration_distances, calibration_margins = [], []
                if len(centroids) == len(PRIMARY_FAMILIES):
                    for row in known_train:
                        if row["fault_id"] == excluded_fault:
                            continue
                        distances = sorted(
                            (_distance(train_vectors[id(row)], center), family)
                            for family, center in centroids.items()
                        )
                        if distances[0][1] == row["family"]:
                            calibration_distances.append(distances[0][0])
                            calibration_margins.append(
                                (distances[1][0] - distances[0][0]) / max(distances[1][0], 1e-9)
                            )
                model = (
                    centroids,
                    _quantile(calibration_distances, .95) if calibration_distances else float("inf"),
                    _quantile(calibration_margins, .05) if calibration_margins else 0.0,
                )
                diagnosis_models[excluded_fault] = model
                return model

            for candidate in test:
                endpoint_model = reference_models.get(str(candidate["endpoint"]))
                if endpoint_model is None:
                    continue
                reference = endpoint_model["reference"]
                alarm_threshold = float(endpoint_model["alarm_threshold"])
                test_vector = _vector(
                    candidate, names, endpoint_model["means"], endpoint_model["scales"]
                )
                alarm_distance = _distance(test_vector, reference)
                centroids, max_family_distance, min_margin = diagnosis_model(str(candidate["fault_id"]))
                ordered = sorted(((_distance(test_vector, center), family) for family, center in centroids.items()))
                predicted_family, nearest, margin = "unknown", float("inf"), 0.0
                if len(ordered) == len(PRIMARY_FAMILIES):
                    nearest, nearest_family = ordered[0]
                    margin = (ordered[1][0] - nearest) / max(ordered[1][0], 1e-9)
                    if math.isfinite(max_family_distance):
                        if nearest <= max_family_distance and margin >= min_margin:
                            predicted_family = nearest_family
                changed = alarm_distance > alarm_threshold
                if not changed:
                    status = "unchanged"
                    predicted_family = "unknown"
                elif predicted_family == "unknown":
                    status = "inconclusive"
                else:
                    status = "changed"
                contributions = sorted(
                    ({"feature": name, "standardized_squared_change": (value - ref) ** 2}
                     for name, value, ref in zip(names, test_vector, reference)),
                    key=lambda row: row["standardized_squared_change"], reverse=True,
                )
                results.append({
                    "source_kind": candidate["source_kind"], "channel": channel,
                    "corpus": held_out, "budget": budget, "draw": candidate["draw"],
                    "endpoint": candidate["endpoint"], "fault_id": candidate["fault_id"],
                    "family": candidate["family"], "status": status,
                    "prediction": predicted_family, "likely_fault_family": predicted_family,
                    "change_alarm": changed, "alarm_distance": alarm_distance,
                    "alarm_threshold": alarm_threshold, "nearest_family_distance": nearest,
                    "maximum_accepted_family_distance": max_family_distance,
                    "nearest_second_margin": margin, "minimum_accepted_margin": min_margin,
                    "raw_score_change": float(candidate["native_original_mean"]) - float(endpoint_model["native_reference"]),
                    "per_probe_contributions": [row for row in contributions if row["feature"].startswith("probe__")],
                    "revalidation_required": status != "unchanged",
                    "group_ids": candidate["group_ids"],
                    "probe_group_counts": candidate.get("probe_group_counts", {}),
                })
    return results


def _summarize_predictions(predictions: Sequence[Mapping[str, object]]) -> dict:
    known = [row for row in predictions if row["family"] in PRIMARY_FAMILIES]
    unchanged = [row for row in predictions if row["family"] == "unchanged"]
    unknown = [row for row in predictions if row["family"] == "unknown"]
    family_sensitivity = {
        family: _mean([float(row["alarm_distance"] > row["alarm_threshold"]) for row in known if row["family"] == family])
        for family in PRIMARY_FAMILIES
    }
    family_auc = {}
    for family in PRIMARY_FAMILIES:
        rows = [row for row in predictions if row["family"] in {"unchanged", family}]
        family_auc[family] = _auc(
            [int(row["family"] == family) for row in rows],
            [float(row["alarm_distance"]) for row in rows],
        )
    available_auc = [value for value in family_auc.values() if value is not None]
    return {
        "macro_auroc": _mean(available_auc) if available_auc else None,
        "macro_sensitivity": _mean(list(family_sensitivity.values())),
        "unchanged_false_alarm_rate": _mean([float(row["alarm_distance"] > row["alarm_threshold"]) for row in unchanged]),
        "diagnosis_macro_f1": _macro_f1(known),
        "unknown_rejection_rate": _mean([float(row["prediction"] == "unknown") for row in unknown]),
        "sensitivity_by_family": family_sensitivity,
        "auroc_by_family": family_auc,
        "n_predictions": len(predictions),
    }


def _hierarchical_bootstrap(
    predictions: Sequence[Mapping[str, object]],
    seed: int,
    repeats: int = 1000,
) -> dict:
    cells: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in predictions:
        cells.setdefault((str(row["corpus"]), str(row["endpoint"])), []).append(row)
    corpora = sorted({key[0] for key in cells})
    endpoints = {corpus: sorted(key[1] for key in cells if key[0] == corpus) for corpus in corpora}
    if not corpora:
        return {"method": "corpus_endpoint_group_draw_hierarchical_bootstrap", "repeats": 0}
    rng = random.Random(seed)
    samples = {"macro_sensitivity": [], "unchanged_false_alarm_rate": [], "macro_auroc": []}
    for _ in range(repeats):
        cell_summaries = []
        for _ in corpora:
            corpus = rng.choice(corpora)
            endpoint_values = endpoints[corpus]
            for _ in endpoint_values:
                endpoint = rng.choice(endpoint_values)
                rows = cells[(corpus, endpoint)]
                cell_summaries.append(_summarize_predictions([rng.choice(rows) for _ in rows]))
        for metric in samples:
            values = [float(summary[metric]) for summary in cell_summaries if summary[metric] is not None]
            if values:
                samples[metric].append(_mean(values))
    return {
        "method": "corpus_endpoint_group_draw_hierarchical_bootstrap",
        "repeats": repeats,
        "intervals": {
            metric: {"median": _quantile(values, .5), "low_95": _quantile(values, .025), "high_95": _quantile(values, .975)}
            for metric, values in samples.items() if values
        },
    }


def _endpoint_family_summary(predictions: Sequence[Mapping[str, object]], family: str) -> dict:
    changed = [row for row in predictions if row["family"] == family]
    unchanged = [row for row in predictions if row["family"] == "unchanged"]
    rows = [*unchanged, *changed]
    return {
        "auroc": _auc(
            [int(row["family"] == family) for row in rows],
            [float(row["alarm_distance"]) for row in rows],
        ),
        "sensitivity": _mean([float(row["alarm_distance"] > row["alarm_threshold"]) for row in changed]),
        "unchanged_false_alarm_rate": _mean([
            float(row["alarm_distance"] > row["alarm_threshold"]) for row in unchanged
        ]),
        "diagnosis_accuracy_or_abstention": _mean([
            float(row["prediction"] == family) for row in changed
        ]),
        "n_changed": len(changed),
        "n_unchanged": len(unchanged),
    }


def _gate_report(predictions: Sequence[Mapping[str, object]], config: Mapping[str, object]) -> dict:
    gates = config["success_gates"]
    budget = int(gates["query_budget"])
    primary = [row for row in predictions if row["source_kind"] == "discovery" and int(row["budget"]) == budget]
    summaries = {
        channel: _summarize_predictions([row for row in primary if row["channel"] == channel])
        for channel in ("raw", "fingerprint", "combined")
    }
    corpus_improvements = {}
    for corpus in sorted({str(row["corpus"]) for row in primary}):
        aucs = {}
        for channel in ("raw", "combined"):
            rows = [row for row in primary if row["corpus"] == corpus and row["channel"] == channel and row["family"] != "unknown"]
            aucs[channel] = _summarize_predictions(rows)["macro_auroc"]
        if aucs["combined"] is None or aucs["raw"] is None:
            continue
        corpus_improvements[corpus] = aucs["combined"] - aucs["raw"]
    improvements = list(corpus_improvements.values())
    permutation_p = exact_sign_flip(improvements) if improvements else 1.0
    combined, raw = summaries["combined"], summaries["raw"]
    detection_pass = (
        combined["macro_sensitivity"] >= float(gates["minimum_sensitivity"])
        and combined["unchanged_false_alarm_rate"] <= float(gates["maximum_false_alarm_rate"])
        and combined["macro_auroc"] is not None and raw["macro_auroc"] is not None
        and combined["macro_auroc"] > raw["macro_auroc"]
        and sum(value > 0 for value in improvements) >= int(gates["minimum_corpus_wins"])
        and permutation_p < float(gates["maximum_permutation_p"])
    )
    diagnosis_pass = (
        combined["diagnosis_macro_f1"] >= float(gates["minimum_diagnosis_macro_f1"])
        and combined["diagnosis_macro_f1"] - raw["diagnosis_macro_f1"] >= float(gates["minimum_diagnosis_gain"])
    )
    unknown_pass = combined["unknown_rejection_rate"] >= float(gates["minimum_unknown_rejection"])
    return {
        "query_budget": budget, "channels": summaries,
        "combined_minus_raw_auc_by_corpus": corpus_improvements,
        "positive_corpora": sum(value > 0 for value in improvements),
        "exact_one_sided_sign_flip_p": permutation_p,
        "detection_gate_passed": detection_pass,
        "diagnosis_gate_passed": diagnosis_pass,
        "unknown_gate_passed": unknown_pass,
        "permitted_primary_claim": (
            "coarse_fault_detection_and_diagnosis" if detection_pass and diagnosis_pass and unknown_pass
            else "behavioral_change_detection_and_localization" if detection_pass
            else "negative_result"
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in fields})


def evaluate_fault_audit(audit_root: Path, output_dir: Path | None = None) -> dict:
    paths = audit_paths(audit_root)
    manifest = verify_lock(paths.lock)["payload"]
    readiness = fault_audit_readiness(paths.root)
    if not readiness["ready"]:
        raise RuntimeError(
            "Fault audit is not fully scored; missing endpoint/fault rows: "
            + json.dumps(readiness["missing"][:12], sort_keys=True)
        )
    config = manifest["config"]
    faults = {row["fault_id"]: FaultSpec.from_mapping(row) for row in manifest["faults"]}
    triplets, scores = _load_audit_state(paths)
    references = _reference_distributions(manifest)
    discovery = _make_observations(triplets, scores, references, faults, config, "discovery")
    confirmation = _make_observations(triplets, scores, references, faults, config, "confirmation_candidate")
    if not discovery:
        raise RuntimeError("No complete discovery observations; run score-fault-audit for required inference faults")
    predictions = []
    for source_rows in (discovery, confirmation):
        for channel in ("raw", "fingerprint", "combined"):
            predictions.extend(_evaluate_channel(source_rows, channel))
    report = {
        "schema_version": 1,
        "construct": manifest["construct"],
        "manifest_lock_sha256": _file_digest(paths.lock),
        "discovery_observations": len(discovery),
        "confirmation_observations": len(confirmation),
        "readiness": readiness,
        "success_gates": _gate_report(predictions, config),
        "discovery_metrics": {
            f"{channel}:{budget}": _summarize_predictions([
                row for row in predictions
                if row["source_kind"] == "discovery" and row["channel"] == channel and int(row["budget"]) == budget
            ])
            for channel in ("raw", "fingerprint", "combined")
            for budget in config["query_budgets"]
        },
        "confirmation_metrics": {
            channel: _summarize_predictions([
                row for row in predictions if row["source_kind"] == "confirmation_candidate" and row["channel"] == channel
            ])
            for channel in ("raw", "fingerprint", "combined")
        },
        "primary_stratified_metrics": {
            f"endpoint={endpoint}:family={family}": _endpoint_family_summary([
                row for row in predictions
                if row["source_kind"] == "discovery" and row["channel"] == "combined"
                and int(row["budget"]) == int(config["success_gates"]["query_budget"])
                and row["endpoint"] == endpoint and row["family"] in {"unchanged", family}
            ], family)
            for endpoint in config["primary_endpoints"] for family in PRIMARY_FAMILIES
        },
        "primary_uncertainty": _hierarchical_bootstrap([
            row for row in predictions
            if row["source_kind"] == "discovery" and row["channel"] == "combined"
            and int(row["budget"]) == int(config["success_gates"]["query_budget"])
        ], int(config["seed"]), int(config["bootstrap_replicates"])),
        "claim_boundary": (
            "A fault is an observable departure from reference behavior. This report does not "
            "estimate deployment false-positive rates or identify exact proprietary internals."
        ),
    }
    destination = Path(output_dir).resolve() if output_dir else paths.results
    destination.mkdir(parents=True, exist_ok=True)
    fields = (
        "source_kind", "channel", "corpus", "budget", "draw", "endpoint", "fault_id",
        "family", "status", "prediction", "likely_fault_family", "change_alarm",
        "alarm_distance", "alarm_threshold",
        "nearest_family_distance", "maximum_accepted_family_distance",
        "nearest_second_margin", "minimum_accepted_margin", "raw_score_change",
        "per_probe_contributions", "revalidation_required", "group_ids", "probe_group_counts",
    )
    _write_csv(destination / "fault_audit_predictions.csv", predictions, fields)
    report["artifacts"] = {
        "evaluation": str(destination / "fault_audit_evaluation.json"),
        "predictions": str(destination / "fault_audit_predictions.csv"),
    }
    (destination / "fault_audit_evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return report
