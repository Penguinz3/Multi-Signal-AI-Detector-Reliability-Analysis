"""Locked, resumable scorer for the prospective operational validation panel.

This module deliberately has no dependency on the analyzer.  It only turns one
``endpoint + opaque condition + run label`` into a locked score table.  The
caller must create a scoring-protocol lock before any detector is loaded.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .conformance import _score_payload
from .core import lock_forecasts, verify_lock
from .detectors import (
    SPECS,
    CausalTokenScorer,
    SequenceClassifierAdapter,
    StatisticalAdapter,
    _mage_preprocessor,
)
from .validation import PRIMARY_ENDPOINTS


SCHEMA_VERSION = 1
RUN_LABELS = ("reference-a", "reference-b", "current")
LEVELS = ("original", "low", "high")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
INTEGRITY_AMENDMENT = "scoring_integrity_amendment_v2.lock.json"
PRIOR_INTEGRITY_AMENDMENT = "scoring_integrity_amendment.lock.json"
EXECUTION_PATCH = "execution_integrity_patch.lock.json"


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scores(
    endpoint TEXT NOT NULL,
    condition_code TEXT NOT NULL,
    run_label TEXT NOT NULL,
    triplet_id TEXT NOT NULL,
    probe TEXT NOT NULL,
    intensity TEXT NOT NULL,
    effective_endpoint TEXT NOT NULL,
    native_score REAL,
    canonical_ai_score REAL,
    input_token_count INTEGER,
    effective_token_count INTEGER,
    max_tokens INTEGER,
    truncated INTEGER NOT NULL DEFAULT 0,
    runtime_ms REAL,
    failure TEXT,
    precision TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(endpoint,condition_code,run_label,triplet_id,intensity)
);
CREATE TABLE IF NOT EXISTS audit_score_cache(
    effective_endpoint TEXT NOT NULL,
    scoring_mode TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(effective_endpoint,scoring_mode,input_sha256)
);
CREATE TABLE IF NOT EXISTS audit_token_sequences(
    observer_key TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    ranks_int32_zlib BLOB NOT NULL,
    log_probs_float32_zlib BLOB NOT NULL,
    cache_hash TEXT NOT NULL,
    PRIMARY KEY(observer_key,input_sha256)
);
"""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_files() -> tuple[Path, ...]:
    return (
        Path(__file__), Path(__file__).with_name("validation_evaluate.py"),
        Path(__file__).with_name("validation.py"), Path(__file__).with_name("operational.py"),
        Path(__file__).with_name("detectors.py"), Path(__file__).with_name("core.py"),
    )


def _verify_code_hashes(payload: Mapping[str, object], label: str) -> None:
    expected = payload.get("code_sha256")
    if not isinstance(expected, Mapping):
        raise RuntimeError(f"{label} lacks frozen code hashes")
    actual = {path.name: _file_sha256(path) for path in _code_files()}
    if {str(key): str(value) for key, value in expected.items()} != actual:
        raise RuntimeError(f"{label} code hashes no longer match the scoring implementation")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _safe(value: str) -> str:
    if not RUN_ID.fullmatch(value):
        raise ValueError(f"Invalid run component: {value!r}")
    return value


def _lock(path: Path) -> tuple[dict, str]:
    envelope = verify_lock(Path(path))
    return envelope["payload"], envelope["sha256"]


def _linked(payload: Mapping[str, object], names: Sequence[str], expected: str | Sequence[str], label: str) -> None:
    values = [payload[name] for name in names if name in payload]
    accepted = {str(expected)} if isinstance(expected, str) else {str(value) for value in expected}
    if values and any(str(value) not in accepted for value in values):
        raise RuntimeError(f"{label} is not bound to the requested validation lock")


def _required_link(payload: Mapping[str, object], names: Sequence[str], expected: str, label: str) -> None:
    values = [payload[name] for name in names if name in payload]
    if not values or any(str(value) != expected for value in values):
        raise RuntimeError(f"{label} must bind to the requested validation lock")


def _panel_csv_hash(
    root: Path, panel: Mapping[str, object], panel_sha: str,
    manifest_sha: str, protocol_sha: str,
) -> tuple[str, str | None]:
    """Resolve an inline panel hash or a pre-score amendment for legacy locks."""
    expected = panel.get("panel_csv_sha256", panel.get("panel_sha256"))
    if expected:
        return str(expected), None
    amendment_path = root / INTEGRITY_AMENDMENT
    amendment, amendment_sha = _lock(amendment_path)
    if amendment.get("construct") != "prospective_scoring_integrity_amendment":
        raise RuntimeError("Unsupported scoring integrity amendment")
    _required_link(amendment, ("manifest_sha256",), manifest_sha, "Integrity amendment")
    _required_link(amendment, ("panel_lock_sha256",), panel_sha, "Integrity amendment")
    _required_link(amendment, ("scoring_protocol_sha256",), protocol_sha, "Integrity amendment")
    expected = amendment.get("panel_csv_sha256")
    if not expected:
        raise RuntimeError("Integrity amendment lacks the panel CSV hash")
    return str(expected), amendment_sha


def _verify_execution_code(
    root: Path, amendment_sha: str | None, manifest_sha: str,
    panel_sha: str, protocol_sha: str,
) -> str | None:
    if amendment_sha is None:
        return None
    patch_path = root / EXECUTION_PATCH
    if not patch_path.exists():
        amendment = verify_lock(root / INTEGRITY_AMENDMENT)["payload"]
        _verify_code_hashes(amendment, "Integrity amendment")
        return None
    patch, patch_sha = _lock(patch_path)
    if patch.get("construct") != "prospective_score_preserving_execution_patch":
        raise RuntimeError("Unsupported execution integrity patch")
    for field, expected in (
        ("manifest_sha256", manifest_sha), ("panel_lock_sha256", panel_sha),
        ("scoring_protocol_sha256", protocol_sha),
        ("parent_integrity_amendment_sha256", amendment_sha),
    ):
        if patch.get(field) != expected:
            raise RuntimeError("Execution integrity patch belongs to another validation state")
    if patch.get("score_math_unchanged") is not True:
        raise RuntimeError("Execution patch is not declared score preserving")
    _verify_code_hashes(patch, "Execution integrity patch")
    return patch_sha


def _load_context(validation_root: Path, protocol_lock: Path) -> tuple[Path, dict, str, list[dict], dict, str, dict, str, dict, str]:
    root = Path(validation_root).resolve()
    manifest_path, panel_path, truth_path = (
        root / "manifest.lock.json", root / "panel.lock.json", root / "condition_truth.private.lock.json"
    )
    manifest, manifest_sha = _lock(manifest_path)
    manifest_file_sha = _file_sha256(manifest_path)
    panel, panel_sha = _lock(panel_path)
    truth, truth_sha = _lock(truth_path)
    protocol, protocol_sha = _lock(protocol_lock)
    if manifest.get("construct") != "prospective_operational_black_box_validation":
        raise ValueError("Unsupported prospective validation manifest")
    _linked(panel, ("parent_manifest_sha256", "manifest_sha256", "manifest_lock_sha256"), (manifest_sha, manifest_file_sha), "Panel lock")
    if truth.get("parent_manifest_payload_sha256") not in (None, _digest(manifest)):
        raise RuntimeError("Truth lock belongs to another manifest")
    _required_link(protocol, ("manifest_sha256", "manifest_lock_sha256"), manifest_sha, "Scoring protocol")
    _required_link(protocol, ("panel_sha256", "panel_lock_sha256"), panel_sha, "Scoring protocol")
    _required_link(protocol, ("truth_sha256", "truth_lock_sha256"), truth_sha, "Scoring protocol")
    if protocol.get("schema_version") != SCHEMA_VERSION or protocol.get("construct") != "prospective_validation_scoring_protocol":
        raise ValueError("Unsupported scoring protocol lock")
    if str(protocol.get("baseline_precision", "fp32")).casefold() != "fp32":
        raise ValueError("Prospective reference scoring must use the frozen FP32 baseline")
    panel_csv = root / "panel.csv"
    expected_panel_hash, amendment_sha = _panel_csv_hash(
        root, panel, panel_sha, manifest_sha, protocol_sha,
    )
    if not expected_panel_hash or _file_sha256(panel_csv) != expected_panel_hash:
        raise RuntimeError("Panel CSV disagrees with the panel lock")
    if amendment_sha:
        protocol = dict(protocol)
        protocol["integrity_amendment_sha256"] = amendment_sha
        protocol["execution_patch_sha256"] = _verify_execution_code(
            root, amendment_sha, manifest_sha, panel_sha, protocol_sha,
        )
    with panel_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"triplet_id", "probe", "original_text", "low_text", "high_text", "low_intensity", "high_intensity"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise RuntimeError("Panel CSV has an unsupported shape")
    ids = [str(row["triplet_id"]) for row in rows]
    locked_ids = {str(value) for value in panel.get("triplet_ids", ())}
    if len(ids) != len(set(ids)) or (locked_ids and set(ids) != locked_ids):
        raise RuntimeError("Panel CSV does not match the panel lock")
    if not locked_ids:
        raise RuntimeError("Panel lock lacks frozen triplet IDs")
    return root, manifest, manifest_sha, rows, panel, panel_sha, truth, truth_sha, protocol, protocol_sha


def lock_scoring_protocol(
    validation_root: Path,
    payload: Mapping[str, object] | None = None,
    *,
    protocol_path: Path | None = None,
    frozen_empirical_cdf: Mapping[str, object] | None = None,
    precision: str = "fp32",
) -> Path:
    """Create the pre-inference protocol lock used by :func:`score_validation_run`.

    The lock may contain CDF score arrays, or a path to a separately locked
    normalization artifact.  It never contains condition truth.
    """
    root = Path(validation_root).resolve()
    manifest, manifest_sha = _lock(root / "manifest.lock.json")
    panel, panel_sha = _lock(root / "panel.lock.json")
    truth, truth_sha = _lock(root / "condition_truth.private.lock.json")
    if manifest.get("construct") != "prospective_operational_black_box_validation":
        raise ValueError("Unsupported prospective validation manifest")
    _linked(panel, ("parent_manifest_sha256", "manifest_sha256", "manifest_lock_sha256"), (manifest_sha, _file_sha256(root / "manifest.lock.json")), "Panel lock")
    if truth.get("parent_manifest_payload_sha256") not in (None, _digest(manifest)):
        raise RuntimeError("Truth lock belongs to another manifest")
    if precision.casefold() != "fp32":
        raise ValueError("Prospective baseline must use FP32; BF16 is the locked precision-change condition")
    protocol = dict(payload or {})
    protocol.update({
        "schema_version": SCHEMA_VERSION,
        "construct": "prospective_validation_scoring_protocol",
        "manifest_sha256": manifest_sha,
        "panel_sha256": panel_sha,
        "truth_sha256": truth_sha,
        "baseline_precision": "fp32",
        "precision_change": "bf16",
        "run_labels": list(protocol.get("run_labels", RUN_LABELS)),
    })
    if frozen_empirical_cdf is not None:
        protocol["frozen_empirical_cdf"] = dict(frozen_empirical_cdf)
    target = Path(protocol_path or (root / "scoring_protocol.lock.json")).resolve()
    if target.exists():
        raise FileExistsError(f"Scoring protocol lock already exists: {target}")
    lock_forecasts(target, protocol)
    return target


def lock_scoring_protocol_from_database(validation_root: Path, reference_database: Path) -> Path:
    """Bind code and frozen human-reference score distributions before inference."""
    reference_database = Path(reference_database).resolve()
    connection = sqlite3.connect(f"file:{reference_database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    endpoints = (*PRIMARY_ENDPOINTS, "lastde__qwen2_5_0_5b_fp32")
    try:
        cdf = {
            endpoint: [float(row[0]) for row in connection.execute(
                """SELECT s.native_score FROM scores s JOIN records r USING(record_id)
                    WHERE r.partition_name='threshold_reference' AND s.detector_config=?
                      AND s.native_score IS NOT NULL AND s.failure IS NULL AND s.truncated=0
                    ORDER BY s.native_score""",
                (endpoint,),
            )]
            for endpoint in endpoints
        }
    finally:
        connection.close()
    if any(not values for values in cdf.values()):
        raise RuntimeError("Every effective endpoint requires a frozen human-reference score distribution")
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    return lock_scoring_protocol(
        validation_root,
        {
            "code_commit": commit,
            "code_sha256": {path.name: _file_sha256(path) for path in _code_files()},
            "reference_database": str(reference_database),
            "reference_database_sha256": _file_sha256(reference_database),
            "normalization": "frozen_human_reference_empirical_cdf",
        },
        frozen_empirical_cdf=cdf,
    )


def lock_scoring_integrity_amendment(validation_root: Path) -> Path:
    """Bind a legacy ID-only panel lock before any prospective scores exist."""
    root = Path(validation_root).resolve()
    target = root / INTEGRITY_AMENDMENT
    if target.exists():
        verify_lock(target)
        return target
    if any((root / "runs").glob("*.lock.json")):
        raise RuntimeError("Integrity amendment must be locked before every score run")
    database = root / "validation_scores.sqlite3"
    if database.exists():
        connection = sqlite3.connect(database)
        try:
            count = connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        except sqlite3.Error:
            count = 1
        finally:
            connection.close()
        if count:
            raise RuntimeError("Integrity amendment must be locked before any score rows")
    manifest, manifest_sha = _lock(root / "manifest.lock.json")
    panel, panel_sha = _lock(root / "panel.lock.json")
    protocol, protocol_sha = _lock(root / "scoring_protocol.lock.json")
    prior_path = root / PRIOR_INTEGRITY_AMENDMENT
    prior_sha = verify_lock(prior_path)["sha256"] if prior_path.exists() else None
    panel_path = root / "panel.csv"
    with panel_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    locked_ids = [str(value) for value in panel.get("triplet_ids", ())]
    if [str(row.get("triplet_id", "")) for row in rows] != locked_ids:
        raise RuntimeError("Legacy panel table does not match its locked row order")
    if any(
        row.get("triplet_sha256") and _digest([
            row["original_text"], row["low_text"], row["high_text"],
            float(row["low_intensity"]), float(row["high_intensity"]),
        ]) != row["triplet_sha256"]
        for row in rows
    ):
        raise RuntimeError("Legacy panel triplet content hash mismatch")
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    lock_forecasts(target, {
        "schema_version": SCHEMA_VERSION,
        "construct": "prospective_scoring_integrity_amendment",
        "reason": "legacy_panel_lock_bound_ids_and_counts_but_not_panel_csv_bytes",
        "created_before_scores": True,
        "parent_integrity_amendment_sha256": prior_sha,
        "manifest_sha256": manifest_sha,
        "panel_lock_sha256": panel_sha,
        "scoring_protocol_sha256": protocol_sha,
        "panel_csv_sha256": _file_sha256(panel_path),
        "rows": len(rows),
        "triplet_ids_sha256": _digest(locked_ids),
        "code_commit": commit,
        "code_sha256": {path.name: _file_sha256(path) for path in _code_files()},
    })
    return target


def lock_execution_integrity_patch(validation_root: Path) -> Path:
    """Bind score-preserving guard fixes after collection began but before unblinding."""
    root = Path(validation_root).resolve()
    target = root / EXECUTION_PATCH
    if target.exists():
        verify_lock(target)
        return target
    if (root / "results").exists():
        raise RuntimeError("Execution patch must be locked before prospective evaluation")
    manifest, manifest_sha = _lock(root / "manifest.lock.json")
    panel, panel_sha = _lock(root / "panel.lock.json")
    protocol, protocol_sha = _lock(root / "scoring_protocol.lock.json")
    amendment, amendment_sha = _lock(root / INTEGRITY_AMENDMENT)
    if amendment.get("panel_csv_sha256") != _file_sha256(root / "panel.csv"):
        raise RuntimeError("Panel bytes changed after the integrity amendment")
    run_locks = {}
    for path in sorted((root / "runs").glob("*.lock.json")):
        payload = verify_lock(path)["payload"]
        if payload.get("construct") != "prospective_validation_score_run" or payload.get("completion") != "complete":
            raise RuntimeError("Only complete score runs may precede the execution patch")
        run_locks[path.name] = _file_sha256(path)
    database = root / "validation_scores.sqlite3"
    rows: list[tuple] = []
    if database.exists():
        connection = sqlite3.connect(database)
        try:
            rows = connection.execute(
                """SELECT endpoint,condition_code,run_label,triplet_id,intensity,
                          native_score,canonical_ai_score,truncated,failure
                   FROM scores ORDER BY endpoint,condition_code,run_label,triplet_id,intensity"""
            ).fetchall()
        finally:
            connection.close()
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    lock_forecasts(target, {
        "schema_version": SCHEMA_VERSION,
        "construct": "prospective_score_preserving_execution_patch",
        "created_before_unblinding": True,
        "score_math_unchanged": True,
        "changes": [
            "route_offline_calibration_without_text_transformation",
            "verify_frozen_code_before_inference",
            "validate_existing_completion_locks",
            "validate_score_run_provenance_before_evaluation",
        ],
        "manifest_sha256": manifest_sha,
        "panel_lock_sha256": panel_sha,
        "scoring_protocol_sha256": protocol_sha,
        "parent_integrity_amendment_sha256": amendment_sha,
        "completed_run_lock_files_sha256": run_locks,
        "pre_patch_score_rows": len(rows),
        "pre_patch_score_rows_sha256": _digest(rows),
        "code_commit": commit,
        "code_sha256": {path.name: _file_sha256(path) for path in _code_files()},
    })
    return target


def _condition(
    manifest: Mapping[str, object], truth: Mapping[str, object], endpoint: str, condition_code: str,
) -> dict[str, object]:
    if endpoint not in {str(value) for value in manifest.get("endpoints", ())}:
        raise ValueError(f"Endpoint is not in the locked prospective panel: {endpoint}")
    public = {
        str(row.get("condition_code")): str(row.get("endpoint"))
        for row in manifest.get("opaque_conditions", ()) if isinstance(row, Mapping)
    }
    if public.get(condition_code) != endpoint:
        raise ValueError("Opaque condition is not declared for this endpoint")
    conditions = [row for row in truth.get("conditions", ()) if str(row.get("condition_code")) == condition_code]
    if len(conditions) != 1:
        raise ValueError("Opaque condition is absent or duplicated in the truth lock")
    row = dict(conditions[0])
    if str(row.get("endpoint")) != endpoint or not str(row.get("mode", "")).strip():
        raise ValueError("Condition truth does not match the requested endpoint")
    return row


def _extract_cdf(value: object, endpoint: str) -> object | None:
    if isinstance(value, Mapping):
        if endpoint in value:
            return _extract_cdf(value[endpoint], endpoint)
        for key in ("scores", "reference_scores", "values", "cdf_scores"):
            if key in value:
                found = _extract_cdf(value[key], endpoint)
                if found is not None:
                    return found
        return None
    return value


def _cdf_scores(protocol: Mapping[str, object], endpoint: str, protocol_path: Path) -> list[float]:
    sources: list[object] = []
    for key in ("frozen_empirical_cdf", "frozen_cdf", "normalization", "cdf_scores"):
        value = protocol.get(key)
        if isinstance(value, Mapping):
            sources.append(value.get(endpoint))
            sources.append(value.get("scores"))
            sources.append(value.get("reference_scores"))
            sources.append(value.get("values"))
        elif key == "cdf_scores":
            sources.append(value)
    source: object | None = next((item for item in sources if item is not None), None)
    source = _extract_cdf(source, endpoint)
    if source is None:
        path_value = protocol.get("frozen_empirical_cdf_path") or protocol.get("normalization_path")
        if path_value:
            source_path = Path(str(path_value))
            if not source_path.is_absolute():
                source_path = protocol_path.parent / source_path
            source = json.loads(source_path.read_text(encoding="utf-8"))
            if isinstance(source, Mapping) and isinstance(source.get("payload"), Mapping):
                source = source["payload"]
            source = _extract_cdf(source, endpoint)
    source = _extract_cdf(source, endpoint)
    try:
        result = [float(value) for value in source]  # type: ignore[union-attr]
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"No frozen empirical-CDF reference exists for {endpoint}") from error
    if not result or not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"Frozen empirical-CDF reference is empty or non-finite for {endpoint}")
    return sorted(result)


def _empirical_cdf(reference: Sequence[float], value: float) -> float:
    return bisect.bisect_right(reference, float(value)) / len(reference)


def _transform_input(mode: str, text: str) -> str:
    if mode in {
        "identity", "bf16", "unchanged", "threshold_only", "endpoint_replacement",
        "mean_logprob", "lastde", "logit_bias", "score_recalibration",
        "monotone_recalibration", "temperature", "temperature_remap",
    }:
        return text
    if mode == "newline_flatten":
        return re.sub(r"[ \t]*\r?\n+[ \t]*", " ", text).strip()
    if mode == "whitespace_collapse":
        return " ".join(text.split())
    if mode in {"nfkc_whitespace", "nfkc_plus_whitespace"}:
        return " ".join(unicodedata.normalize("NFKC", text).split())
    raise ValueError(f"Unsupported input fault mode: {mode}")


def _remap(value: float, mode: str, parameters: Mapping[str, object]) -> float:
    value = min(1 - 1e-9, max(1e-9, float(value)))
    logit = math.log(value / (1 - value))
    if mode in {"logit_bias", "score_recalibration", "monotone_recalibration"}:
        logit += float(parameters.get("bias", parameters.get("logit_bias", 0.0)))
    elif mode in {"temperature", "temperature_remap"}:
        temperature = float(parameters.get("temperature", 1.0))
        if temperature <= 0:
            raise ValueError("Temperature remapping requires a positive temperature")
        logit /= temperature
    else:
        raise ValueError(f"Unsupported offline calibration mode: {mode}")
    return 1 / (1 + math.exp(-logit))


def _open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def _condition_code_for_unchanged(truth: Mapping[str, object], endpoint: str) -> str:
    rows = [
        row for row in truth.get("conditions", ())
        if str(row.get("endpoint")) == endpoint
        and str(row.get("family", "")).casefold() == "unchanged"
    ]
    if len(rows) != 1:
        rows = [
            row for row in truth.get("conditions", ())
            if str(row.get("endpoint")) == endpoint and str(row.get("mode", "")).casefold() == "unchanged"
        ]
    if len(rows) != 1:
        raise RuntimeError("Offline calibration requires one locked unchanged condition for the endpoint")
    return str(rows[0]["condition_code"])


def _build_adapter(
    endpoint: str, *, device: int, mage_repo: Path | None,
    disable_mage_preprocess: bool, precision: str,
) -> object:
    if endpoint not in SPECS:
        raise ValueError(f"Unknown local detector configuration: {endpoint}")
    import torch

    spec = replace(SPECS[endpoint], precision=precision)
    if spec.method_family.startswith("supervised"):
        preprocessor = None
        if spec.config_id == "mage_longformer__paper":
            if not mage_repo:
                raise RuntimeError("MAGE scoring requires its pinned repository checkout")
            preprocessor = _mage_preprocessor(str(mage_repo))
        adapter = SequenceClassifierAdapter(spec, device, preprocessor)
        if disable_mage_preprocess and spec.config_id == "mage_longformer__paper":
            adapter.preprocessor = None
        if precision == "bf16":
            adapter.model.to(dtype=torch.bfloat16)
        adapter.model.eval()
        return adapter
    scorer = CausalTokenScorer(spec)
    if precision == "bf16":
        scorer.model.to(dtype=torch.bfloat16)
    scorer.model.eval()
    return StatisticalAdapter(spec, scorer)


def _score(adapter: object, text: str, mode: str, connection: sqlite3.Connection) -> dict[str, object]:
    if hasattr(adapter, "scorer"):
        return dict(_score_payload(adapter, text, "mean_logprob" if mode == "mean_logprob" else mode, connection))
    return asdict(adapter.score(text))  # type: ignore[attr-defined]


def _adapter_mode(condition: Mapping[str, object]) -> tuple[str, str, bool]:
    mode = str(condition.get("mode", "identity"))
    parameters = condition.get("parameters", {})
    parameters = parameters if isinstance(parameters, Mapping) else {}
    replacement = str(parameters.get("replacement", ""))
    if mode in {"endpoint_replacement", "replace_endpoint", "logrank_to_lastde"}:
        if not replacement:
            raise ValueError("Endpoint replacement condition lacks replacement endpoint")
        return replacement, "adapter_score", False
    if mode == "mean_logprob":
        return "logrank__qwen2_5_0_5b_fp32", "mean_logprob", False
    if mode == "preprocessor_disabled":
        return str(condition["endpoint"]), "adapter_score", True
    return str(condition["endpoint"]), "adapter_score", False


def _run_stem(endpoint: str, condition_code: str, run_label: str) -> str:
    return f"{_safe(endpoint)}__{_safe(condition_code)}__{_safe(run_label)}"


def _write_score_csv(path: Path, rows: Sequence[sqlite3.Row]) -> None:
    fields = (
        "triplet_id", "probe", "intensity", "effective_endpoint", "native_score",
        "canonical_ai_score", "truncated", "failure", "input_token_count",
        "effective_token_count", "precision",
    )
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in fields})
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _existing_lock(
    root: Path, stem: str, endpoint: str, condition_code: str, run_label: str,
    expected_ids: set[str], manifest_sha: str, panel_sha: str,
    protocol_sha: str, amendment_sha: str | None,
) -> Path | None:
    path = root / "runs" / f"{stem}.lock.json"
    if not path.exists():
        return None
    payload, _ = _lock(path)
    if (payload.get("endpoint"), payload.get("condition_code"), payload.get("run_label")) != (endpoint, condition_code, run_label):
        raise RuntimeError("Existing run lock has a conflicting identity")
    expected_rows = len(expected_ids) * len(LEVELS)
    scores = payload.get("scores")
    expected_challenges = {f"{triplet_id}:{level}" for triplet_id in expected_ids for level in LEVELS}
    if (
        payload.get("construct") != "prospective_validation_score_run"
        or payload.get("completion") != "complete"
        or int(payload.get("expected_triplets", -1)) != len(expected_ids)
        or int(payload.get("score_rows", -1)) != expected_rows
        or int(payload.get("valid_rows", -1)) != expected_rows
        or int(payload.get("rejected_triplets", -1)) != 0
        or not isinstance(scores, Mapping)
        or set(map(str, scores)) != expected_challenges
        or payload.get("manifest_sha256") != manifest_sha
        or payload.get("panel_sha256") != panel_sha
        or payload.get("scoring_protocol_sha256") != protocol_sha
        or (amendment_sha and payload.get("integrity_amendment_sha256") != amendment_sha)
    ):
        raise RuntimeError("Existing run lock is incomplete or has invalid provenance")
    table_name = str(payload.get("score_table_path", ""))
    table_path = root / "runs" / table_name
    if Path(table_name).name != table_name or not table_path.exists() or _file_sha256(table_path) != payload.get("score_table_sha256"):
        raise RuntimeError("Existing run score table is missing or altered")
    return path


def score_validation_run(
    validation_root: Path,
    endpoint: str,
    condition_code: str,
    run_label: str,
    *,
    protocol_lock: Path | None = None,
    database: Path | None = None,
    device: int = 0,
    mage_repo: Path | None = None,
    precision: str | None = None,
) -> Path:
    """Score exactly one endpoint/opaque-condition/run-label, resuming SQLite rows.

    No model is loaded until every validation lock and the scoring-protocol lock
    has passed.  Offline calibration and threshold-only conditions use the
    locked unchanged rows and never run inference.
    """
    endpoint, condition_code, run_label = _safe(endpoint), _safe(condition_code), _safe(run_label)
    if run_label not in RUN_LABELS:
        raise ValueError(f"Run label must be one of {RUN_LABELS}")
    root = Path(validation_root).resolve()
    protocol_path = Path(protocol_lock or (root / "scoring_protocol.lock.json")).resolve()
    # Lock checks intentionally happen before opening a model or beginning inference.
    root, manifest, manifest_sha, panel_rows, panel, panel_sha, truth, truth_sha, protocol, protocol_sha = _load_context(root, protocol_path)
    condition = _condition(manifest, truth, endpoint, condition_code)
    if run_label != "current" and str(condition.get("family", "")).casefold() != "unchanged":
        raise ValueError("Reference runs must use the endpoint's unchanged condition")
    effective_precision = (
        "bf16" if str(condition.get("mode", "")).casefold() == "bf16"
        else str(protocol.get("baseline_precision", "fp32")).casefold()
    )
    if precision is not None and precision.casefold() != effective_precision:
        raise ValueError(f"This condition is locked to {effective_precision}")
    effective_endpoint, scoring_mode, disable_mage_preprocess = _adapter_mode(condition)
    cdf = _cdf_scores(protocol, effective_endpoint, protocol_path)
    stem = _run_stem(endpoint, condition_code, run_label)
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    existing = _existing_lock(
        root, stem, endpoint, condition_code, run_label, set(expected_triplets),
        manifest_sha, panel_sha, protocol_sha,
        str(protocol.get("integrity_amendment_sha256")) if protocol.get("integrity_amendment_sha256") else None,
    )
    if existing:
        return existing
    db_path = Path(database or (root / "validation_scores.sqlite3")).resolve()
    connection = _open_database(db_path)
    expected_triplets = {str(row["triplet_id"]): row for row in panel_rows}
    run_key = (endpoint, condition_code, run_label)
    condition_mode = str(condition.get("mode", "identity"))
    parameters = condition.get("parameters", {})
    parameters = parameters if isinstance(parameters, Mapping) else {}
    offline = condition_mode in {
        "logit_bias", "score_recalibration", "monotone_recalibration", "temperature",
        "temperature_remap", "threshold_only",
    }
    unchanged_code = None
    base_rows: dict[tuple[str, str], sqlite3.Row] = {}
    if offline:
        unchanged_code = _condition_code_for_unchanged(truth, endpoint)
        base_rows = {
            (str(row["triplet_id"]), str(row["intensity"])): row
            for row in connection.execute(
                """SELECT * FROM scores WHERE endpoint=? AND condition_code=? AND run_label=?""",
                (endpoint, unchanged_code, run_label),
            )
        }
        expected_base = {(triplet_id, level) for triplet_id in expected_triplets for level in LEVELS}
        if set(base_rows) != expected_base:
            connection.close()
            raise RuntimeError("Offline calibration requires a complete unchanged run for the same label")
    adapter: object | None = None
    if not offline:
        try:
            adapter = _build_adapter(
                effective_endpoint, device=device, mage_repo=mage_repo,
                disable_mage_preprocess=disable_mage_preprocess, precision=effective_precision,
            )
        except Exception:
            connection.close()
            raise
    try:
        for triplet_id, triplet in expected_triplets.items():
            existing_rows = connection.execute(
                """SELECT intensity,failure,truncated FROM scores
                   WHERE endpoint=? AND condition_code=? AND run_label=? AND triplet_id=?""",
                (*run_key, triplet_id),
            ).fetchall()
            if len(existing_rows) == 3:
                complete = all(
                    not row["failure"] and not row["truncated"]
                    for row in existing_rows
                )
                rejected = all(
                    row["failure"] == "full_triplet_rejected_capacity" and bool(row["truncated"])
                    for row in existing_rows
                )
                if complete or rejected:
                    continue
            if existing_rows:
                with connection:
                    connection.execute(
                        "DELETE FROM scores WHERE endpoint=? AND condition_code=? AND run_label=? AND triplet_id=?",
                        (*run_key, triplet_id),
                    )
            texts = {
                level: _transform_input(condition_mode, str(triplet[f"{level}_text"]))
                for level in LEVELS
            }
            if offline:
                for level in LEVELS:
                    base = base_rows[(triplet_id, level)]
                    base_native = base["native_score"]
                    base_canonical = base["canonical_ai_score"]
                    if base_native is None or base_canonical is None:
                        raise RuntimeError("Unchanged base row lacks a score")
                    if condition_mode == "threshold_only":
                        native, canonical = float(base_native), float(base_canonical)
                    else:
                        native = float(base_native)
                        canonical = _remap(float(base_canonical), condition_mode, parameters)
                    payload = json.loads(base["payload_json"])
                    payload.update({
                        "offline_calibration": True,
                        "source_condition_code": unchanged_code,
                        "threshold_policy": parameters.get("policy", "unchanged"),
                    })
                    with connection:
                        connection.execute(
                            """INSERT INTO scores VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (
                                endpoint, condition_code, run_label, triplet_id, triplet["probe"], level,
                                effective_endpoint, native, canonical, base["input_token_count"],
                                base["effective_token_count"], base["max_tokens"], 0, 0.0, None,
                                effective_precision, json.dumps(payload, sort_keys=True, default=str),
                            ),
                        )
                continue
            assert adapter is not None
            counts = {level: int(adapter.token_count(texts[level])) for level in LEVELS}  # type: ignore[attr-defined]
            spec = SPECS[effective_endpoint]
            capacity = min(spec.max_tokens - 32, 460)
            if max(counts.values()) > capacity:
                payloads = {
                    level: {
                        "native_score": None, "canonical_ai_score": None,
                        "input_token_count": counts[level], "effective_token_count": 0,
                        "max_tokens": spec.max_tokens, "truncated": True,
                        "runtime_ms": 0.0, "failure": "full_triplet_rejected_capacity",
                    }
                    for level in LEVELS
                }
            else:
                payloads = {}
                for level in LEVELS:
                    started = time.perf_counter()
                    try:
                        payload = _score(adapter, texts[level], scoring_mode, connection)
                        native = payload.get("native_score")
                        if native is None or not math.isfinite(float(native)):
                            raise FloatingPointError("Detector returned a non-finite score")
                        payload["canonical_ai_score"] = _empirical_cdf(cdf, float(native))
                        payload["precision"] = effective_precision
                        payload["effective_endpoint"] = effective_endpoint
                    except Exception as error:
                        if "CUDA" in str(error) or "AcceleratorError" in type(error).__name__:
                            raise
                        payload = {
                            "native_score": None, "canonical_ai_score": None,
                            "input_token_count": counts[level], "effective_token_count": 0,
                            "max_tokens": spec.max_tokens, "truncated": False,
                            "runtime_ms": (time.perf_counter() - started) * 1000,
                            "failure": f"{type(error).__name__}: {error}",
                        }
                    payloads[level] = payload
            for level, payload in payloads.items():
                with connection:
                    connection.execute(
                        """INSERT OR REPLACE INTO scores VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            endpoint, condition_code, run_label, triplet_id, triplet["probe"], level,
                            effective_endpoint, payload.get("native_score"), payload.get("canonical_ai_score"),
                            payload.get("input_token_count"), payload.get("effective_token_count"),
                            payload.get("max_tokens"), int(bool(payload.get("truncated"))),
                            payload.get("runtime_ms"), payload.get("failure"), effective_precision,
                            json.dumps(payload, sort_keys=True, default=str),
                        ),
                    )
    except Exception:
        connection.close()
        raise
    finally:
        adapter = None
    rows = connection.execute(
        """SELECT * FROM scores WHERE endpoint=? AND condition_code=? AND run_label=?
           ORDER BY triplet_id,CASE intensity WHEN 'original' THEN 0 WHEN 'low' THEN 1 ELSE 2 END""",
        run_key,
    ).fetchall()
    expected_rows = len(expected_triplets) * 3
    if len(rows) != expected_rows:
        connection.close()
        raise RuntimeError(f"Scoring did not complete: {len(rows)}/{expected_rows} rows")
    by_triplet = {}
    for row in rows:
        by_triplet.setdefault(row["triplet_id"], []).append(row)
    if any({item["intensity"] for item in values} != set(LEVELS) for values in by_triplet.values()):
        connection.close()
        raise RuntimeError("Run completion requires all three members of every triplet")
    csv_path = runs / f"{stem}.scores.csv"
    _write_score_csv(csv_path, rows)
    rejected = sum(
        1 for values in by_triplet.values()
        if any(item["failure"] or item["truncated"] for item in values)
    )
    valid_rows = sum(1 for row in rows if not row["failure"] and not row["truncated"] and row["canonical_ai_score"] is not None)
    if rejected or valid_rows != expected_rows:
        connection.close()
        raise RuntimeError(f"Run is not complete and valid: valid_rows={valid_rows}, rejected_triplets={rejected}")
    run_payload = {
        "schema_version": SCHEMA_VERSION,
        "construct": "prospective_validation_score_run",
        "manifest_sha256": manifest_sha,
        "panel_sha256": panel_sha,
        "truth_sha256": truth_sha,
        "scoring_protocol_sha256": protocol_sha,
        "integrity_amendment_sha256": protocol.get("integrity_amendment_sha256"),
        "execution_patch_sha256": protocol.get("execution_patch_sha256"),
        "endpoint": endpoint,
        "condition_code": condition_code,
        "run_label": run_label,
        "role": run_label.replace("-", "_"),
        "effective_endpoint": effective_endpoint,
        "precision": effective_precision,
        "expected_triplets": len(expected_triplets),
        "score_rows": len(rows),
        "valid_rows": valid_rows,
        "rejected_triplets": rejected,
        "score_table_sha256": _file_sha256(csv_path),
        "score_table_path": csv_path.name,
        "database": db_path.name,
        "completion": "complete",
        "metadata": {
            "version": str(protocol.get("version", f"{condition_code}:{run_label}")),
            "configuration": str(protocol.get("configuration", f"{effective_endpoint}:{effective_precision}")),
            "threshold_policy": str(parameters.get("policy", "unchanged")),
            "collected_at_utc": str(protocol.get("collected_at_utc", "locked_prospective_run")),
        },
        "scores": {
            f"{row['triplet_id']}:{row['intensity']}": float(row["canonical_ai_score"])
            for row in rows
            if row["canonical_ai_score"] is not None and not row["failure"] and not row["truncated"]
        },
    }
    lock_path = runs / f"{stem}.lock.json"
    lock_forecasts(lock_path, run_payload)
    connection.close()
    return lock_path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Score one locked prospective validation run")
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--condition-code", required=True)
    parser.add_argument("--run-label", choices=RUN_LABELS, required=True)
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--mage-repo", type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    print(score_validation_run(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
