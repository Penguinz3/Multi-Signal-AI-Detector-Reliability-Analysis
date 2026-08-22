from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from .core import TARGET_CORPORA, canonical_json, jeffreys_posterior, lock_forecasts, verify_lock
from .workflow import (
    _database_artifact,
    assert_all_target_locks,
    fold_paths,
    lock_privileged_forecasts,
)


PRIVILEGED_SIZES = (25, 50, 100, 250)


def privileged_plan_path(root: Path, target_corpus: str) -> Path:
    return fold_paths(root, target_corpus).root / "artifacts" / "privileged" / "plan.json"


def privileged_comparator_path(root: Path, target_corpus: str) -> Path:
    return fold_paths(root, target_corpus).root / "artifacts" / "privileged" / "comparator.json"


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_commit() -> str:
    repository = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _zero_context(root: Path, target_corpus: str) -> tuple[object, Mapping[str, object]]:
    paths = fold_paths(root, target_corpus)
    envelope = verify_lock(paths.zero_lock)
    payload = envelope.get("payload")
    manifest = payload.get("manifest") if isinstance(payload, Mapping) else None
    if not isinstance(manifest, Mapping) or manifest.get("target_corpus") != target_corpus:
        raise RuntimeError("Invalid zero-score lock")
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    if state.get("zero_lock_sha256") != envelope["sha256"]:
        raise RuntimeError("Fold state and zero-score lock disagree")
    return envelope, manifest


def _nested_ids(manifest: Mapping[str, object], target_corpus: str) -> tuple[dict[str, list[str]], dict]:
    artifacts = manifest.get("id_artifacts")
    if not isinstance(artifacts, Mapping) or len(artifacts) != 1:
        raise RuntimeError("Zero-score manifest must bind exactly one ID artifact")
    artifact_name, expected_hash = next(iter(artifacts.items()))
    artifact_path = Path(str(artifact_name))
    if not artifact_path.is_file() or _file_sha256(artifact_path) != expected_hash:
        raise RuntimeError("Zero-score ID artifact hash mismatch")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    draws = artifact.get("draw_ids")
    data_ids = artifact.get("data_ids")
    if not isinstance(draws, Mapping) or not isinstance(data_ids, Mapping):
        raise RuntimeError("Malformed zero-score ID artifact")
    try:
        n50 = list(draws["draw:0:n:50"])
        n100 = list(draws["draw:0:n:100"])
        n250 = list(draws["draw:0:n:250"])
        signature_ids = set(data_ids[f"signature:{target_corpus}"])
    except (KeyError, TypeError) as error:
        raise RuntimeError("Zero-score artifact lacks the canonical draw-0 target IDs") from error
    if len(n50) != 50 or len(n100) != 100 or len(n250) != 250:
        raise RuntimeError("Canonical draw-0 signature sizes are invalid")
    if len(set(n250)) != 250 or n50 != n100[:50] or n100 != n250[:100]:
        raise RuntimeError("Canonical draw-0 target IDs are not unique and nested")
    if not set(n250) <= signature_ids:
        raise RuntimeError("Canonical draw-0 IDs escape the target signature partition")
    sizes = {str(size): n250[:size] for size in PRIVILEGED_SIZES}
    return sizes, {"path": str(artifact_path.resolve()), "sha256": str(expected_hash)}


def _validate_plan_records(database: Path, target_corpus: str, ids: Sequence[str]) -> None:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"""SELECT record_id,group_id FROM records
                 WHERE corpus=? AND partition_name='signature'
                   AND record_id IN ({placeholders})""",
            (target_corpus, *ids),
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != len(ids) or len({row[0] for row in rows}) != len(ids):
        raise RuntimeError("Privileged plan IDs do not exactly match the target signature")
    if len({row[1] for row in rows}) != len(ids):
        raise RuntimeError("Privileged plan must contain one record per author/source group")


def _expected_plan(root: Path, target_corpus: str) -> dict:
    paths = fold_paths(root, target_corpus)
    envelope, manifest = _zero_context(root, target_corpus)
    sizes, artifact = _nested_ids(manifest, target_corpus)
    _validate_plan_records(paths.database, target_corpus, sizes["250"])
    return {
        "schema_version": 1,
        "target_corpus": target_corpus,
        "draw": 0,
        "sizes": sizes,
        "nested_ids_sha256": _digest(sizes),
        "zero_lock_sha256": envelope["sha256"],
        "zero_manifest_sha256": _digest(manifest),
        "id_artifact": artifact,
        "admitted_detectors": list(manifest["panel_revisions"]),
        "panel_sha256": manifest["panel_sha256"],
        "thresholds_sha256": manifest["thresholds_sha256"],
        "code_commit": manifest["code_commit"],
        "privileged_code_commit": _code_commit(),
    }


def build_privileged_plan(root: Path, target_corpus: str) -> Path:
    assert_all_target_locks(root, TARGET_CORPORA)
    paths = fold_paths(root, target_corpus)
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    if state.get("phase") != "zero_locked":
        raise RuntimeError(f"Privileged planning requires zero_locked state, found {state.get('phase')}")
    connection = sqlite3.connect(f"file:{paths.database.resolve()}?mode=ro", uri=True)
    try:
        scored = connection.execute(
            """SELECT COUNT(*) FROM scores s JOIN records r USING(record_id)
               WHERE r.corpus=? AND r.partition_name IN ('signature','test')""",
            (target_corpus,),
        ).fetchone()[0]
    finally:
        connection.close()
    if scored:
        raise RuntimeError("Target scores exist before privileged-plan publication")
    expected = _expected_plan(root, target_corpus)
    path = privileged_plan_path(root, target_corpus)
    if path.exists():
        if verify_lock(path).get("payload") != expected:
            raise RuntimeError("Existing privileged plan disagrees with the canonical zero-score IDs")
    else:
        lock_forecasts(path, expected)
    return path


def verify_privileged_plan(root: Path, target_corpus: str) -> dict:
    path = privileged_plan_path(root, target_corpus)
    envelope = verify_lock(path)
    expected = _expected_plan(root, target_corpus)
    if envelope.get("payload") != expected:
        raise RuntimeError("Privileged plan no longer matches the zero-score lock")
    return expected


def build_privileged_comparator(root: Path, target_corpus: str) -> Path:
    assert_all_target_locks(root, TARGET_CORPORA)
    paths = fold_paths(root, target_corpus)
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    if state.get("phase") != "signature_scored":
        raise RuntimeError(f"Privileged comparator requires signature_scored state, found {state.get('phase')}")
    plan = verify_privileged_plan(root, target_corpus)
    zero_envelope, zero_manifest = _zero_context(root, target_corpus)
    detectors = tuple(zero_manifest["panel_revisions"])
    ids = plan["sizes"]["250"]
    placeholders = ",".join("?" for _ in ids)
    connection = sqlite3.connect(f"file:{paths.database.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            f"""SELECT s.record_id,s.detector_config,s.canonical_ai_score
                 FROM scores s JOIN records r USING(record_id)
                 WHERE r.corpus=? AND r.partition_name='signature'
                   AND s.variant_id='original' AND s.record_id IN ({placeholders})
                   AND s.detector_config IN ({','.join('?' for _ in detectors)})
                   AND s.failure IS NULL AND s.canonical_ai_score IS NOT NULL
                   AND s.truncated=0""",
            (target_corpus, *ids, *detectors),
        ).fetchall()
    finally:
        connection.close()
    scores = {(str(record_id), str(detector)): float(score) for record_id, detector, score in rows}
    expected = {(record_id, detector) for record_id in ids for detector in detectors}
    if set(scores) != expected:
        missing = len(expected - set(scores))
        raise RuntimeError(f"Privileged signature panel is incomplete or invalid: missing={missing}")
    entries = []
    thresholds = zero_manifest["thresholds"]["detectors"]
    for detector in detectors:
        for fpr in (.05, .01):
            cutoff = float(thresholds[detector]["thresholds"][f"{fpr:.2f}"])
            for size in PRIVILEGED_SIZES:
                selected = plan["sizes"][str(size)]
                flagged = sum(scores[(record_id, detector)] >= cutoff for record_id in selected)
                mean, low, high = jeffreys_posterior(flagged, size)
                entries.append({
                    "detector_config": detector,
                    "operating_fpr": fpr,
                    "signature_size": size,
                    "flagged": flagged,
                    "total": size,
                    "raw_fpr": flagged / size,
                    "posterior_mean": mean,
                    "posterior_low_95": low,
                    "posterior_high_95": high,
                })
    comparator = {
        "schema_version": 1,
        "target_corpus": target_corpus,
        "analysis_role": "descriptive_privileged_comparator_not_a_primary_success_gate",
        "zero_lock_sha256": zero_envelope["sha256"],
        "privileged_plan_sha256": verify_lock(privileged_plan_path(root, target_corpus))["sha256"],
        "nested_ids_sha256": plan["nested_ids_sha256"],
        "panel_sha256": zero_manifest["panel_sha256"],
        "thresholds_sha256": zero_manifest["thresholds_sha256"],
        "score_rows_sha256": _digest(sorted((record_id, detector, score) for (record_id, detector), score in scores.items())),
        "comparators": entries,
    }
    output = privileged_comparator_path(root, target_corpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != comparator:
            raise RuntimeError("Existing privileged comparator disagrees with scored target data")
    else:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(comparator, handle, sort_keys=True, indent=2)
    manifest = dict(zero_manifest)
    manifest.update({
        "database": _database_artifact(paths.database),
        "privileged_builder_schema_version": 1,
        "zero_lock_sha256": zero_envelope["sha256"],
        "zero_manifest_sha256": _digest(zero_manifest),
        "privileged_code_commit": plan["privileged_code_commit"],
        "privileged_plan_artifacts": {
            str(privileged_plan_path(root, target_corpus).resolve()):
                _file_sha256(privileged_plan_path(root, target_corpus)),
        },
        "privileged_comparator_artifacts": {str(output.resolve()): _file_sha256(output)},
    })
    lock_privileged_forecasts(paths, manifest, comparator)
    return output
