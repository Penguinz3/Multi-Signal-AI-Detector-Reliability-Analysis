from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .core import canonical_json, lock_forecasts, threshold, verify_lock


PRIMARY_FPRS = (0.05, 0.01)
PHASES = (
    "prelock",
    "zero_locked",
    "signature_scored",
    "privileged_locked",
    "test_scored",
)
HEX64 = re.compile(r"[0-9a-f]{64}")
GIT_REVISION = re.compile(r"[0-9a-f]{7,64}")
SAFE_TARGET = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class FoldPaths:
    target_corpus: str
    root: Path
    database: Path
    state: Path
    zero_lock: Path
    privileged_lock: Path


def fold_paths(root: Path, target_corpus: str) -> FoldPaths:
    if not SAFE_TARGET.fullmatch(target_corpus) or ".." in target_corpus:
        raise ValueError(f"Unsafe target corpus name: {target_corpus!r}")
    fold = Path(root).resolve() / "folds" / target_corpus
    return FoldPaths(
        target_corpus,
        fold,
        fold / "fprint.sqlite3",
        fold / "state.json",
        fold / "locks" / "zero_score.json",
        fold / "locks" / "privileged.json",
    )


def initialize_fold(root: Path, target_corpus: str) -> FoldPaths:
    paths = fold_paths(root, target_corpus)
    paths.root.mkdir(parents=True, exist_ok=True)
    if paths.state.exists():
        state = _read_state(paths)
        if state["target_corpus"] != target_corpus:
            raise RuntimeError(f"Fold state belongs to {state['target_corpus']!r}")
    else:
        _write_json_exclusive(
            paths.state,
            {"target_corpus": target_corpus, "phase": "prelock"},
        )
    return paths


def assert_prelock_corpora(
    target_corpus: str,
    allowed_corpora: Sequence[str],
    scored_corpora: Sequence[str],
) -> None:
    allowed = set(allowed_corpora)
    if target_corpus in allowed:
        raise ValueError("The held-out target cannot be an allowed pre-lock corpus")
    unexpected = set(scored_corpora) - allowed
    if unexpected:
        raise RuntimeError(f"Pre-lock scores exist for forbidden corpora: {sorted(unexpected)}")


def assert_prelock_database(
    database: Path,
    target_corpus: str,
    allowed_corpora: Sequence[str],
) -> None:
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT r.corpus
            FROM scores AS s
            JOIN records AS r USING(record_id)
            """
        )
        scored_corpora = [str(row[0]) for row in rows]
    finally:
        connection.close()
    assert_prelock_corpora(target_corpus, allowed_corpora, scored_corpora)


def build_threshold_artifact(
    retained_raid: Sequence[tuple[str, str]],
    scores: Mapping[str, Mapping[str, float]],
    *,
    minimum_retained: int = 10_000,
    fprs: Sequence[float] = PRIMARY_FPRS,
) -> dict:
    if len(retained_raid) < minimum_retained:
        raise ValueError(f"Need at least {minimum_retained} retained RAID-human records")
    record_ids = [record_id for record_id, _ in retained_raid]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Retained RAID record IDs must be unique")
    ordered = sorted((str(record_id), str(text_hash)) for record_id, text_hash in retained_raid)
    if any(not HEX64.fullmatch(text_hash) for _, text_hash in ordered):
        raise ValueError("Retained RAID records require canonical-text SHA-256 hashes")
    expected_ids = set(record_ids)
    detector_rows = {}
    for detector, by_record in sorted(scores.items()):
        if set(by_record) != expected_ids:
            raise ValueError(f"{detector} scores do not match retained RAID IDs")
        values = [float(by_record[record_id]) for record_id, _ in ordered]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{detector} has non-finite RAID scores")
        detector_rows[detector] = {
            "score_sha256": _digest(sorted((record_id, float(by_record[record_id])) for record_id in expected_ids)),
            "thresholds": {_fpr_key(fpr): threshold(values, fpr) for fpr in fprs},
        }
    if not detector_rows:
        raise ValueError("Threshold calibration requires detector scores")
    return {
        "schema_version": 1,
        "retained_raid_count": len(ordered),
        "retained_raid_sha256": _digest(ordered),
        "fprs": [_fpr_key(fpr) for fpr in fprs],
        "detectors": detector_rows,
    }


def build_forecast_manifest(
    *,
    paths: FoldPaths,
    data_ids: Mapping[str, Sequence[str]],
    draw_ids: Mapping[str, Sequence[str]],
    panel_revisions: Mapping[str, Mapping[str, str]],
    thresholds: Mapping[str, object],
    selected_c: Mapping[str, float],
    feature_artifacts: Mapping[str, str],
    profile_artifacts: Mapping[str, str],
    code_commit: str,
) -> dict:
    _validate_threshold_artifact(thresholds, panel_revisions)
    _validate_artifact_hashes("feature", feature_artifacts)
    _validate_artifact_hashes("profile", profile_artifacts)
    if not GIT_REVISION.fullmatch(code_commit):
        raise ValueError("code_commit must be a hexadecimal Git revision")
    if not selected_c or not all(math.isfinite(float(value)) and float(value) > 0 for value in selected_c.values()):
        raise ValueError("selected_c must contain positive finite values")
    for detector, revisions in panel_revisions.items():
        if not revisions.get("model_revision") or not revisions.get("tokenizer_revision"):
            raise ValueError(f"{detector} lacks pinned model/tokenizer revisions")

    normalized_data = {name: list(ids) for name, ids in sorted(data_ids.items())}
    flat_ids = [record_id for ids in normalized_data.values() for record_id in ids]
    flat_id_set = set(flat_ids)
    if not flat_ids or len(flat_ids) != len(flat_id_set):
        raise ValueError("Data IDs must be unique across manifest partitions")
    normalized_draws = {name: list(ids) for name, ids in sorted(draw_ids.items())}
    if not normalized_draws:
        raise ValueError("Forecast manifest requires draw IDs")
    unknown = {
        record_id
        for ids in normalized_draws.values()
        for record_id in ids
        if record_id not in flat_id_set
    }
    if unknown:
        raise ValueError(f"Draw IDs are absent from the data manifest: {sorted(unknown)[:3]}")

    return {
        "schema_version": 1,
        "target_corpus": paths.target_corpus,
        "database": _database_artifact(paths.database),
        "data_ids_sha256": _digest(normalized_data),
        "data_id_count": len(flat_ids),
        "draw_ids_sha256": _digest(normalized_draws),
        "draw_count": len(normalized_draws),
        "panel_revisions": panel_revisions,
        "panel_sha256": _digest(panel_revisions),
        "thresholds": thresholds,
        "thresholds_sha256": _digest(thresholds),
        "selected_c": {name: float(value) for name, value in sorted(selected_c.items())},
        "feature_artifacts": dict(sorted(feature_artifacts.items())),
        "profile_artifacts": dict(sorted(profile_artifacts.items())),
        "code_commit": code_commit,
    }


def lock_zero_score_forecasts(
    paths: FoldPaths,
    manifest: Mapping[str, object],
    forecasts: object,
    allowed_prelock_corpora: Sequence[str],
) -> str:
    state = _read_state(paths)
    if state["phase"] != "prelock":
        raise RuntimeError(f"Zero-score lock requires prelock state, found {state['phase']}")
    assert_prelock_database(paths.database, paths.target_corpus, allowed_prelock_corpora)
    _validate_manifest_for_fold(manifest, paths, check_database=True)
    digest = lock_forecasts(paths.zero_lock, {"manifest": manifest, "forecasts": forecasts})
    _replace_state(paths, {"target_corpus": paths.target_corpus, "phase": "zero_locked", "zero_lock_sha256": digest})
    return digest


def lock_privileged_forecasts(
    paths: FoldPaths,
    manifest: Mapping[str, object],
    forecasts: object,
) -> str:
    state = _read_state(paths)
    if state["phase"] != "signature_scored":
        raise RuntimeError(f"Privileged lock requires signature_scored state, found {state['phase']}")
    _validate_manifest_for_fold(manifest, paths, check_database=True)
    digest = lock_forecasts(paths.privileged_lock, {"manifest": manifest, "forecasts": forecasts})
    _replace_state(
        paths,
        {
            "target_corpus": paths.target_corpus,
            "phase": "privileged_locked",
            "zero_lock_sha256": state["zero_lock_sha256"],
            "privileged_lock_sha256": digest,
        },
    )
    return digest


def assert_all_target_locks(root: Path, targets: Sequence[str], *, privileged: bool = False) -> None:
    missing = []
    for target in targets:
        paths = fold_paths(root, target)
        lock_path = paths.privileged_lock if privileged else paths.zero_lock
        if not lock_path.is_file():
            missing.append(target)
            continue
        envelope = verify_lock(lock_path)
        payload = envelope["payload"]
        manifest = payload.get("manifest") if isinstance(payload, Mapping) else None
        if not isinstance(manifest, Mapping) or manifest.get("target_corpus") != target:
            raise RuntimeError(f"Lock does not belong to target {target}: {lock_path}")
        _validate_manifest_for_fold(manifest, paths, check_database=False)
        state = _read_state(paths)
        digest_key = "privileged_lock_sha256" if privileged else "zero_lock_sha256"
        if state.get(digest_key) != envelope["sha256"]:
            raise RuntimeError(f"Fold state and lock disagree for target {target}")
    if missing:
        kind = "privileged" if privileged else "zero-score"
        raise RuntimeError(f"Missing {kind} locks for targets: {sorted(missing)}")


def assert_target_score_allowed(
    root: Path,
    targets: Sequence[str],
    target_corpus: str,
    partition: str,
) -> None:
    if target_corpus not in targets:
        raise ValueError(f"Unknown target corpus: {target_corpus}")
    paths = fold_paths(root, target_corpus)
    state = _read_state(paths)
    if partition == "privileged_signature":
        assert_all_target_locks(root, targets)
        if state["phase"] != "zero_locked":
            raise RuntimeError(f"Signature scoring requires zero_locked state, found {state['phase']}")
        return
    if partition == "test":
        assert_all_target_locks(root, targets, privileged=True)
        if state["phase"] != "privileged_locked":
            raise RuntimeError(f"Test scoring requires privileged_locked state, found {state['phase']}")
        return
    raise ValueError(f"Unknown target partition: {partition}")


def mark_signature_scored(root: Path, targets: Sequence[str], target_corpus: str) -> None:
    assert_target_score_allowed(root, targets, target_corpus, "privileged_signature")
    paths = fold_paths(root, target_corpus)
    state = _read_state(paths)
    _replace_state(
        paths,
        {
            "target_corpus": target_corpus,
            "phase": "signature_scored",
            "zero_lock_sha256": state["zero_lock_sha256"],
        },
    )


def mark_test_scored(root: Path, targets: Sequence[str], target_corpus: str) -> None:
    assert_target_score_allowed(root, targets, target_corpus, "test")
    paths = fold_paths(root, target_corpus)
    state = _read_state(paths)
    _replace_state(
        paths,
        {
            "target_corpus": target_corpus,
            "phase": "test_scored",
            "zero_lock_sha256": state["zero_lock_sha256"],
            "privileged_lock_sha256": state["privileged_lock_sha256"],
        },
    )


def _fpr_key(value: float) -> str:
    if value not in PRIMARY_FPRS:
        raise ValueError(f"Only preregistered FPRs are supported: {PRIMARY_FPRS}")
    return f"{value:.2f}"


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _database_artifact(database: Path) -> dict:
    if not database.is_file():
        raise FileNotFoundError(database)
    files = {database.name: _file_sha256(database)}
    wal = Path(f"{database}-wal")
    if wal.is_file():
        files[wal.name] = _file_sha256(wal)
    return {
        "path": str(database.resolve()),
        "files": files,
        "sha256": _digest(files),
    }


def _validate_threshold_artifact(
    artifact: Mapping[str, object],
    panel_revisions: Mapping[str, Mapping[str, str]],
) -> None:
    if int(artifact.get("retained_raid_count", 0)) < 10_000:
        raise ValueError("Threshold artifact requires at least 10,000 retained RAID-human records")
    if set(artifact.get("fprs", ())) != {_fpr_key(value) for value in PRIMARY_FPRS}:
        raise ValueError("Threshold artifact must contain the 5% and 1% operating points")
    if not HEX64.fullmatch(str(artifact.get("retained_raid_sha256", ""))):
        raise ValueError("Threshold artifact lacks a retained RAID hash")
    detectors = artifact.get("detectors")
    if not isinstance(detectors, Mapping) or set(detectors) != set(panel_revisions):
        raise ValueError("Threshold detectors must exactly match the admitted panel")
    for detector, row in detectors.items():
        values = row.get("thresholds") if isinstance(row, Mapping) else None
        if not isinstance(values, Mapping) or set(values) != set(artifact["fprs"]):
            raise ValueError(f"Incomplete thresholds for {detector}")
        if not HEX64.fullmatch(str(row.get("score_sha256", ""))):
            raise ValueError(f"Threshold artifact lacks a score hash for {detector}")
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"Threshold artifact has a non-finite value for {detector}")


def _validate_artifact_hashes(name: str, artifacts: Mapping[str, str]) -> None:
    if not artifacts or any(not HEX64.fullmatch(str(value)) for value in artifacts.values()):
        raise ValueError(f"{name} artifacts require SHA-256 hashes")


def _validate_manifest_for_fold(
    manifest: Mapping[str, object],
    paths: FoldPaths,
    *,
    check_database: bool,
) -> None:
    if manifest.get("target_corpus") != paths.target_corpus:
        raise ValueError("Forecast manifest target does not match fold")
    database = manifest.get("database")
    if not isinstance(database, Mapping) or database.get("path") != str(paths.database.resolve()):
        raise ValueError("Forecast manifest database does not match fold")
    if check_database and database.get("sha256") != _database_artifact(paths.database)["sha256"]:
        raise RuntimeError("Fold database changed after forecast manifest creation")
    for key in ("data_ids_sha256", "draw_ids_sha256", "panel_sha256", "thresholds_sha256"):
        if not HEX64.fullmatch(str(manifest.get(key, ""))):
            raise ValueError(f"Forecast manifest is missing {key}")
    panel = manifest.get("panel_revisions")
    thresholds = manifest.get("thresholds")
    if not isinstance(panel, Mapping) or not isinstance(thresholds, Mapping):
        raise ValueError("Forecast manifest lacks panel or threshold provenance")
    for detector, revisions in panel.items():
        if not isinstance(revisions, Mapping):
            raise ValueError(f"Invalid revision provenance for {detector}")
        if not revisions.get("model_revision") or not revisions.get("tokenizer_revision"):
            raise ValueError(f"{detector} lacks pinned model/tokenizer revisions")
    _validate_threshold_artifact(thresholds, panel)
    if manifest["panel_sha256"] != _digest(panel):
        raise ValueError("Panel revision hash mismatch")
    if manifest["thresholds_sha256"] != _digest(thresholds):
        raise ValueError("Threshold artifact hash mismatch")
    feature_artifacts = manifest.get("feature_artifacts")
    profile_artifacts = manifest.get("profile_artifacts")
    if not isinstance(feature_artifacts, Mapping) or not isinstance(profile_artifacts, Mapping):
        raise ValueError("Forecast manifest lacks feature/profile artifacts")
    _validate_artifact_hashes("feature", feature_artifacts)
    _validate_artifact_hashes("profile", profile_artifacts)
    selected_c = manifest.get("selected_c")
    if not isinstance(selected_c, Mapping) or not selected_c:
        raise ValueError("Forecast manifest lacks selected C values")
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in selected_c.values()):
        raise ValueError("Forecast manifest has invalid selected C values")
    if not GIT_REVISION.fullmatch(str(manifest.get("code_commit", ""))):
        raise ValueError("Forecast manifest lacks a pinned code commit")


def _read_state(paths: FoldPaths) -> dict:
    if not paths.state.is_file():
        raise RuntimeError(f"Fold is not initialized: {paths.target_corpus}")
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    if state.get("phase") not in PHASES:
        raise RuntimeError(f"Invalid fold phase: {state.get('phase')!r}")
    return state


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)


def _replace_state(paths: FoldPaths, state: Mapping[str, object]) -> None:
    temporary = paths.state.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
    temporary.replace(paths.state)
