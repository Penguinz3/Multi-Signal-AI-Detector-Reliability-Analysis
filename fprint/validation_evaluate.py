"""Prospective, score-blind evaluation for the operational validation panel.

The scoring stage writes one verified ``prospective_score_state`` lock below
the validation root.  Its payload contains ``runs`` with
``endpoint``, ``condition_code``, ``role`` (``reference_a``, ``reference_b``
or ``current``), and a complete score map.  For convenience, the evaluator
also accepts a list of per-condition records with ``reference_a``,
``reference_b`` and ``current`` maps.  This small adapter keeps collection
and analysis separate without adding a scoring dependency.

The private condition lock is intentionally not opened until the blinded
prediction table and its hash lock have been written.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .core import canonical_json, lock_forecasts, verify_lock
from .operational import LEVELS, PROBES, analyze_score_maps


SCHEMA_VERSION = 1
ROLES = {"reference_a", "reference_b", "current"}
PRIMARY_FAMILIES = {"input_handling", "output_policy", "core_computation"}
DEFAULT_RULE = {
    "alpha": .05,
    "absolute_tolerance": .01,
    "noise_multiplier": 3.0,
    "minimum_affected_fraction": .20,
    "maximum_reference_noise": .02,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _as_map(value: object, expected_ids: set[str]) -> dict[str, float]:
    """Normalize a score map and fail closed on missing, extra, or bad rows."""
    rows: object = value
    if isinstance(value, Mapping):
        for key in ("score_map", "scores", "canonical_scores", "score_table"):
            if key in value:
                rows = value[key]
                break
    result: dict[str, float] = {}
    if isinstance(rows, Mapping):
        iterator = rows.items()
        for challenge_id, raw in iterator:
            if isinstance(raw, Mapping):
                raw = raw.get("canonical_ai_score", raw.get("score"))
            result[str(challenge_id)] = float(raw)
    elif isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Score rows must be mappings")
            challenge_id = row.get("challenge_id")
            if not challenge_id and row.get("triplet_id") and (row.get("intensity") or row.get("level")):
                challenge_id = f"{row['triplet_id']}:{row.get('intensity', row.get('level'))}"
            raw = row.get("canonical_ai_score", row.get("score"))
            if not challenge_id or raw is None:
                raise ValueError("Score rows require challenge_id and canonical_ai_score")
            if str(challenge_id) in result:
                raise ValueError(f"Duplicate challenge ID in locked score state: {challenge_id}")
            result[str(challenge_id)] = float(raw)
    else:
        raise ValueError("Locked score state has no score map")
    if set(result) != expected_ids or len(result) != len(expected_ids):
        raise ValueError(
            f"Score map IDs do not match panel: missing={len(expected_ids - set(result))}, "
            f"extra={len(set(result) - expected_ids)}"
        )
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in result.values()):
        raise ValueError("Canonical scores must be finite values in [0,1]")
    return result


def _challenge(panel: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    rows = []
    for triplet in panel:
        triplet_id = str(triplet["triplet_id"])
        probe = str(triplet["probe"])
        rows.extend(
            {
                "challenge_id": f"{triplet_id}:{level}",
                "triplet_id": triplet_id,
                "probe": probe,
                "intensity": level,
                "intensity_value": str(
                    0.0 if level == "original" else triplet[f"{level}_intensity"]
                ),
            }
            for level in LEVELS
        )
    return rows


def _load_panel(root: Path, manifest: Mapping[str, object], manifest_digest: str) -> list[dict[str, str]]:
    panel_lock = verify_lock(root / "panel.lock.json")
    panel_payload = panel_lock["payload"]
    if panel_payload.get("parent_manifest_payload_sha256") not in (None, manifest_digest):
        raise RuntimeError("Panel lock belongs to a different prospective manifest")
    if panel_payload.get("parent_manifest_sha256") not in (None, _sha256(root / "manifest.lock.json")):
        raise RuntimeError("Panel lock belongs to a different prospective manifest")
    panel_path = root / "panel.csv"
    expected_hash = panel_payload.get("panel_csv_sha256", panel_payload.get("panel_sha256"))
    if not expected_hash:
        amendment = verify_lock(root / "scoring_integrity_amendment_v2.lock.json")["payload"]
        if amendment.get("construct") != "prospective_scoring_integrity_amendment":
            raise RuntimeError("Unsupported scoring integrity amendment")
        if amendment.get("manifest_sha256") != manifest_digest:
            raise RuntimeError("Integrity amendment belongs to another manifest")
        if amendment.get("panel_lock_sha256") != panel_lock["sha256"]:
            raise RuntimeError("Integrity amendment belongs to another panel lock")
        protocol = verify_lock(root / "scoring_protocol.lock.json")
        if amendment.get("scoring_protocol_sha256") != protocol["sha256"]:
            raise RuntimeError("Integrity amendment belongs to another scoring protocol")
        expected_hash = amendment.get("panel_csv_sha256")
    if not expected_hash or _sha256(panel_path) != expected_hash:
        raise RuntimeError("Panel CSV hash disagrees with its lock")
    with panel_path.open(encoding="utf-8-sig", newline="") as handle:
        panel = list(csv.DictReader(handle))
    required = {"triplet_id", "record_id", "corpus", "group_id", "probe", "low_intensity", "high_intensity"}
    if not panel or any(not required.issubset(row) for row in panel):
        raise RuntimeError("Prospective panel is missing required columns")
    expected_triplets = [str(value) for value in panel_payload.get("triplet_ids", [])]
    actual_triplets = [str(row["triplet_id"]) for row in panel]
    if actual_triplets != expected_triplets or len(panel) != int(panel_payload.get("rows", len(panel))):
        raise RuntimeError("Panel table disagrees with its lock")
    if len(actual_triplets) != len(set(actual_triplets)):
        raise RuntimeError("Panel contains duplicate triplet IDs")
    return panel


def _public_conditions(manifest: Mapping[str, object]) -> dict[str, str]:
    result = {}
    for row in manifest.get("opaque_conditions", []):
        if not isinstance(row, Mapping) or not row.get("condition_code") or not row.get("endpoint"):
            raise RuntimeError("Malformed public condition manifest")
        code, endpoint = str(row["condition_code"]), str(row["endpoint"])
        if code in result or endpoint not in set(map(str, manifest.get("endpoints", []))):
            raise RuntimeError("Unknown or duplicate public condition")
        result[code] = endpoint
    if not result:
        raise RuntimeError("Prospective manifest has no opaque conditions")
    return result


def _role(value: object) -> str:
    name = str(value or "").strip().lower().replace("-", "_")
    return {
        "reference": "reference_a",
        "ref": "reference_a",
        "ref_a": "reference_a",
        "reference_a": "reference_a",
        "ref_b": "reference_b",
        "reference_b": "reference_b",
        "current": "current",
        "target": "current",
        "variant": "current",
    }.get(name, name)


def _map_fields(item: Mapping[str, object], expected_ids: set[str]) -> dict[str, float] | None:
    for key in ("score_map", "scores", "canonical_scores", "score_table"):
        if key in item:
            return _as_map(item[key], expected_ids)
    if any(key in item for key in ("challenge_id", "triplet_id")):
        return _as_map([item], expected_ids)
    return None


def _collect_runs(
    payload: Mapping[str, object], expected_ids: set[str], inherited_endpoint: str = "", inherited_code: str = ""
) -> list[dict[str, object]]:
    """Accept the canonical state shape and the compact per-condition shape."""
    entries: list[dict[str, object]] = []
    endpoint = str(payload.get("endpoint", inherited_endpoint))
    code = str(payload.get("condition_code", inherited_code))
    runs = payload.get("runs")
    if isinstance(runs, Mapping):
        iterable = []
        for run_id, value in runs.items():
            item = dict(value) if isinstance(value, Mapping) else {"score_map": value}
            item.setdefault("run_id", run_id)
            iterable.append(item)
    elif isinstance(runs, Sequence) and not isinstance(runs, (str, bytes, bytearray)):
        iterable = list(runs)
    else:
        iterable = []
    for raw in iterable:
        if not isinstance(raw, Mapping):
            raise RuntimeError("Locked prospective run is not a mapping")
        item = dict(raw)
        item.setdefault("endpoint", endpoint)
        item.setdefault("condition_code", code)
        entries.extend(_collect_runs(item, expected_ids, str(item.get("endpoint", "")), str(item.get("condition_code", ""))) if "runs" in item else [])
        score_map = _map_fields(item, expected_ids)
        if score_map is not None:
            entries.append({
                "endpoint": str(item.get("endpoint", endpoint)),
                "condition_code": str(item.get("condition_code", code)),
                "role": _role(item.get("role")),
                "run_id": str(item.get("run_id", "")),
                "score_map": score_map,
                "metadata": item.get("metadata", {}),
            })
    conditions = payload.get("conditions")
    if isinstance(conditions, Mapping):
        condition_items = []
        for condition_code, value in conditions.items():
            item = dict(value) if isinstance(value, Mapping) else {}
            item.setdefault("condition_code", condition_code)
            condition_items.append(item)
    elif isinstance(conditions, Sequence) and not isinstance(conditions, (str, bytes, bytearray)):
        condition_items = list(conditions)
    else:
        condition_items = []
    for raw in condition_items:
        if not isinstance(raw, Mapping):
            raise RuntimeError("Locked condition is not a mapping")
        item = dict(raw)
        current_endpoint = str(item.get("endpoint", endpoint))
        current_code = str(item.get("condition_code", code))
        for role_name in ("reference_a", "reference_b", "current"):
            value = item.get(role_name)
            if value is None:
                continue
            if isinstance(value, Mapping) and any(key in value for key in ("score_map", "scores", "score_table")):
                metadata = value.get("metadata", {})
                value = value.get("score_map", value.get("scores", value.get("score_table")))
            else:
                metadata = {}
            entries.append({
                "endpoint": current_endpoint,
                "condition_code": current_code,
                "role": role_name,
                "run_id": f"{current_code}:{role_name}",
                "score_map": _as_map(value, expected_ids),
                "metadata": metadata,
            })
        refs = item.get("references")
        if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes, bytearray)):
            for index, value in enumerate(refs[:2]):
                entries.append({
                    "endpoint": current_endpoint,
                    "condition_code": current_code,
                    "role": "reference_a" if index == 0 else "reference_b",
                    "run_id": f"{current_code}:reference_{index + 1}",
                    "score_map": _as_map(value, expected_ids),
                    "metadata": {},
                })
    direct = _map_fields(payload, expected_ids)
    if direct is not None:
        entries.append({
            "endpoint": endpoint,
            "condition_code": code,
            "role": _role(payload.get("role")),
            "run_id": str(payload.get("run_id", "")),
            "score_map": direct,
            "metadata": payload.get("metadata", {}),
        })
    return entries


def _load_score_state(root: Path, manifest: Mapping[str, object], manifest_digest: str, expected_ids: set[str]) -> tuple[dict[str, dict[str, dict[str, object]]], list[str]]:
    public = _public_conditions(manifest)
    panel_sha = verify_lock(root / "panel.lock.json")["sha256"]
    protocol_sha = verify_lock(root / "scoring_protocol.lock.json")["sha256"]
    amendment_sha = verify_lock(root / "scoring_integrity_amendment_v2.lock.json")["sha256"]
    patch_envelope = verify_lock(root / "execution_integrity_patch.lock.json")
    patch = patch_envelope["payload"]
    if (
        patch.get("construct") != "prospective_score_preserving_execution_patch"
        or patch.get("manifest_sha256") != manifest_digest
        or patch.get("panel_lock_sha256") != panel_sha
        or patch.get("scoring_protocol_sha256") != protocol_sha
        or patch.get("parent_integrity_amendment_sha256") != amendment_sha
        or patch.get("score_math_unchanged") is not True
    ):
        raise RuntimeError("Execution integrity patch does not match the prospective state")
    code_dir = Path(__file__).resolve().parent
    code_files = (
        Path(__file__), code_dir / "validation_scoring.py", code_dir / "validation.py",
        code_dir / "operational.py", code_dir / "detectors.py", code_dir / "core.py",
    )
    expected_code = patch.get("code_sha256")
    actual_code = {path.name: _sha256(path) for path in code_files}
    if not isinstance(expected_code, Mapping) or dict(expected_code) != actual_code:
        raise RuntimeError("Evaluator code no longer matches the execution integrity patch")
    pre_patch_locks = patch.get("completed_run_lock_files_sha256", {})
    if not isinstance(pre_patch_locks, Mapping):
        raise RuntimeError("Execution patch lacks its pre-patch run inventory")
    candidates = [root / name for name in ("score_state.lock.json", "scoring_state.lock.json", "scores.lock.json")]
    candidates.extend(
        path for path in sorted(root.rglob("*.lock.json"))
        if path not in candidates and path.name not in {"manifest.lock.json", "panel.lock.json", "condition_truth.private.lock.json"}
        and "result" not in {part.casefold() for part in path.parts}
    )
    entries: list[dict[str, object]] = []
    verified_sources: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        envelope = verify_lock(path)
        payload = envelope.get("payload")
        if not isinstance(payload, Mapping):
            continue
        construct = str(payload.get("construct", ""))
        if path.name not in {"score_state.lock.json", "scoring_state.lock.json", "scores.lock.json"} and "score" not in construct and "run" not in construct:
            continue
        if construct == "prospective_validation_score_run":
            expected_rows = len(expected_ids)
            if (
                payload.get("completion") != "complete"
                or int(payload.get("score_rows", -1)) != expected_rows
                or int(payload.get("valid_rows", -1)) != expected_rows
                or int(payload.get("rejected_triplets", -1)) != 0
                or payload.get("manifest_sha256") != manifest_digest
                or payload.get("panel_sha256") != panel_sha
                or payload.get("scoring_protocol_sha256") != protocol_sha
                or payload.get("integrity_amendment_sha256") != amendment_sha
            ):
                raise RuntimeError(f"Incomplete or unbound prospective score run: {path}")
            table_name = str(payload.get("score_table_path", ""))
            table_path = path.parent / table_name
            if Path(table_name).name != table_name or not table_path.exists() or _sha256(table_path) != payload.get("score_table_sha256"):
                raise RuntimeError(f"Prospective score table is missing or altered: {path}")
            pre_patch = pre_patch_locks.get(path.name) == _sha256(path)
            post_patch = payload.get("execution_patch_sha256") == patch_envelope["sha256"]
            if not (pre_patch or post_patch):
                raise RuntimeError(f"Prospective run is not covered by the execution patch: {path}")
        elif path.name not in {"score_state.lock.json", "scoring_state.lock.json", "scores.lock.json"}:
            continue
        if payload.get("manifest_sha256") not in (None, manifest_digest) and payload.get("parent_manifest_sha256") not in (None, manifest_digest):
            raise RuntimeError(f"Score state belongs to another prospective manifest: {path}")
        entries.extend(_collect_runs(payload, expected_ids))
        verified_sources.append(str(path))
    if not entries:
        raise RuntimeError("No completed, hash-locked prospective scoring state was found")
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for entry in entries:
        endpoint, code, role = str(entry["endpoint"]), str(entry["condition_code"]), str(entry["role"])
        if code not in public or public[code] != endpoint or role not in ROLES:
            raise RuntimeError("Score state contains an unknown endpoint, condition, or role")
        if role in grouped[code]:
            raise RuntimeError(f"Duplicate locked prospective run: {code}:{role}")
        grouped[code][role] = entry
    for endpoint in set(public.values()):
        reference_codes = [
            code for code, declared_endpoint in public.items()
            if declared_endpoint == endpoint and {"reference_a", "reference_b"}.issubset(grouped.get(code, {}))
        ]
        if len(reference_codes) != 1:
            raise RuntimeError(f"Endpoint {endpoint} must have exactly one locked reference condition")
        reference = grouped[reference_codes[0]]
        for code, declared_endpoint in public.items():
            if declared_endpoint != endpoint:
                continue
            if "current" not in grouped.get(code, {}):
                raise RuntimeError(f"Condition {code} lacks a current run")
            grouped[code].setdefault("reference_a", reference["reference_a"])
            grouped[code].setdefault("reference_b", reference["reference_b"])
    return dict(grouped), verified_sources


def _draw_panel(panel: Sequence[Mapping[str, str]], corpus: str, budget: int, draw: int, seed: str) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    used_groups: set[str] = set()
    for probe in PROBES:
        rows = [row for row in panel if str(row["corpus"]) == corpus and str(row["probe"]) == probe]
        rows.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{corpus}:{probe}:{draw}:{row['group_id']}:{row['triplet_id']}".encode()
        ).hexdigest())
        for row in rows:
            group = str(row["group_id"])
            if group in used_groups:
                continue
            selected.append(dict(row))
            used_groups.add(group)
            if sum(item["probe"] == probe for item in selected) >= budget:
                break
    return selected


def _alarm_score(analysis: Mapping[str, object]) -> float:
    rule = analysis["decision_rule"]
    tolerance = float(rule["absolute_tolerance"])
    multiplier = float(rule["noise_multiplier"])
    values = []
    for probe, cells in analysis["probe_results"].items():
        for feature, cell in cells.items():
            values.append(float(cell["median_current_delta"]) / max(
                tolerance, multiplier * float(cell["median_reference_repeat_delta"]), 1e-12
            ))
    return max(values, default=0.0)


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    positives = [row for row in rows if bool(row["truth_changed"])]
    negatives = [row for row in rows if not bool(row["truth_changed"])]
    scores = [(float(row["alarm_score"]), bool(row["truth_changed"])) for row in rows]
    positive_scores = [score for score, truth in scores if truth]
    negative_scores = [score for score, truth in scores if not truth]
    if positive_scores and negative_scores:
        wins = sum((left > right) + .5 * (left == right) for left in positive_scores for right in negative_scores)
        auroc = wins / (len(positive_scores) * len(negative_scores))
    else:
        auroc = 0.0
    return {
        "cases": len(rows),
        "changed_cases": len(positives),
        "unchanged_cases": len(negatives),
        "auroc": auroc,
        "sensitivity": sum(row["status"] == "changed" for row in positives) / len(positives) if positives else 0.0,
        "false_alarm_rate": sum(row["status"] == "changed" for row in negatives) / len(negatives) if negatives else 0.0,
        "unchanged_false_alarm_rate": sum(row["status"] == "changed" for row in negatives) / len(negatives) if negatives else 0.0,
        "inconclusive_rate": sum(row["status"] == "inconclusive" for row in rows) / len(rows) if rows else 0.0,
    }


def _group_metrics(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, dict]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: _metrics(values) for name, values in sorted(groups.items())}


def _macro(metrics: Mapping[str, Mapping[str, object]], field: str) -> float:
    values = [float(item[field]) for item in metrics.values() if item.get("cases")]
    return sum(values) / len(values) if values else 0.0


def _canonical_family(family: str, mode: str) -> str:
    if mode == "threshold_only" or family in {"negative_control", "threshold_policy"}:
        return "threshold_policy"
    return {
        "precision": "core_computation",
        "calibration": "output_policy",
    }.get(family, family)


def _metadata_policy_changed(condition: Mapping[str, object]) -> bool:
    current = condition["current"].get("metadata", {})
    references = [condition[name].get("metadata", {}) for name in ("reference_a", "reference_b")]
    if not isinstance(current, Mapping) or not current.get("threshold_policy"):
        return False
    policies = [ref.get("threshold_policy") for ref in references if isinstance(ref, Mapping) and ref.get("threshold_policy")]
    return bool(policies) and any(str(current["threshold_policy"]) != str(policy) for policy in policies)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0]) if rows else ["condition_code"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_prospective_validation(validation_root: Path, output_dir: Path | None = None) -> Path:
    """Evaluate locked prospective runs and publish a blinded-then-unblinded report."""
    root = Path(validation_root).resolve()
    manifest_envelope = verify_lock(root / "manifest.lock.json")
    manifest = manifest_envelope["payload"]
    manifest_digest = manifest_envelope["sha256"]
    if manifest.get("construct") != "prospective_operational_black_box_validation":
        raise RuntimeError("Unsupported prospective validation manifest")
    panel = _load_panel(root, manifest, manifest_digest)
    challenge_all = _challenge(panel)
    expected_ids_all = {row["challenge_id"] for row in challenge_all}
    runs, sources = _load_score_state(root, manifest, manifest_digest, expected_ids_all)
    seed = str(manifest.get("seed", "fprint-prospective-operational-v1"))
    draws = int(manifest.get("draws", 20))
    budgets = tuple(int(value) for value in manifest.get("query_budgets", (10, 25, 50)))
    analysis_manifest = {
        "probes": list(PROBES),
        "decision_rule": dict(manifest.get("decision_rule", DEFAULT_RULE)),
    }
    blinded: list[dict[str, object]] = []
    for code, condition_runs in sorted(runs.items()):
        endpoint = str(condition_runs["current"]["endpoint"])
        for budget in budgets:
            for draw in range(draws):
                for corpus in sorted({str(row["corpus"]) for row in panel}):
                    selected = _draw_panel(panel, corpus, budget, draw, seed)
                    counts = {probe: sum(row["probe"] == probe for row in selected) for probe in PROBES}
                    if min(counts.values(), default=0) < 3:
                        continue
                    challenge = _challenge(selected)
                    expected_ids = {row["challenge_id"] for row in challenge}
                    maps = {
                        role: {key: value for key, value in condition_runs[role]["score_map"].items() if key in expected_ids}
                        for role in ROLES
                    }
                    analysis = analyze_score_maps(
                        analysis_manifest,
                        challenge,
                        [maps["reference_a"], maps["reference_b"]],
                        maps["current"],
                    )
                    changed_cells = [
                        f"{probe}:{feature}"
                        for probe, cells in analysis["probe_results"].items()
                        for feature, cell in cells.items() if cell["changed"]
                    ]
                    probe_contributions = {
                        probe: max(
                            float(cell["median_current_delta"]) / max(
                                float(analysis["decision_rule"]["absolute_tolerance"]),
                                float(analysis["decision_rule"]["noise_multiplier"]) * float(cell["median_reference_repeat_delta"]),
                                1e-12,
                            )
                            for cell in cells.values()
                        )
                        for probe, cells in analysis["probe_results"].items()
                    }
                    blinded.append({
                        "condition_code": code,
                        "endpoint": endpoint,
                        "corpus": corpus,
                        "budget": budget,
                        "draw": draw,
                        "status": analysis["status"],
                        "alarm_score": _alarm_score(analysis),
                        "changed_cells": ";".join(changed_cells),
                        "probe_contributions_json": json.dumps(probe_contributions, sort_keys=True),
                        "raw_score_change": statistics.mean(
                            abs(maps["current"][key] - (maps["reference_a"][key] + maps["reference_b"][key]) / 2)
                            for key in expected_ids
                        ),
                        "triplets_by_probe_json": json.dumps(counts, sort_keys=True),
                        "full_budget": min(counts.values()) >= budget,
                    })
    if not blinded:
        raise RuntimeError("No complete group-aware prospective draws were available")

    destination = Path(output_dir or root / "results" / "prospective_validation").resolve()
    if destination.exists():
        raise FileExistsError(f"Prospective evaluation output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        blinded_path = staging / "blinded_predictions.csv"
        _write_csv(blinded_path, blinded)
        blinded_lock = {
            "schema_version": SCHEMA_VERSION,
            "construct": "prospective_blinded_predictions",
            "manifest_sha256": manifest_digest,
            "panel_lock_sha256": _sha256(root / "panel.lock.json"),
            "score_state_locks": sources,
            "rows": len(blinded),
            "predictions_sha256": _sha256(blinded_path),
        }
        lock_forecasts(staging / "blinded_predictions.lock.json", blinded_lock)

        # Do not move this read above the blinded lock. This is the prospective blind.
        truth_envelope = verify_lock(root / "condition_truth.private.lock.json")
        truth_payload = truth_envelope["payload"]
        if truth_payload.get("parent_manifest_payload_sha256") not in (None, _digest(manifest)):
            raise RuntimeError("Private condition truth belongs to another manifest")
        truth_by_code = {
            str(row["condition_code"]): row
            for row in truth_payload.get("conditions", [])
            if isinstance(row, Mapping) and row.get("condition_code")
        }
        if set(truth_by_code) != set(runs):
            raise RuntimeError("Private truth conditions do not match the public scoring state")

        final_rows: list[dict[str, object]] = []
        for row in blinded:
            truth = truth_by_code[str(row["condition_code"])]
            declared_family = str(truth.get("family", "unknown"))
            mode = str(truth.get("mode", ""))
            family = _canonical_family(declared_family, mode)
            metadata_changed = _metadata_policy_changed(runs[str(row["condition_code"])])
            observable = family != "threshold_policy" or metadata_changed
            enriched = dict(row)
            enriched.update({
                "declared_family": declared_family,
                "family": family,
                "mode": mode,
                "truth_changed": family in PRIMARY_FAMILIES,
                "evaluation_class": "threshold_policy_metadata_only" if family == "threshold_policy" and metadata_changed else (
                    "threshold_policy_unobservable" if family == "threshold_policy" else (
                        "unchanged_control" if family == "unchanged" else "behavioral_change"
                    )
                ),
                "threshold_policy_metadata_changed": metadata_changed,
                "fingerprint_positive": bool(family in PRIMARY_FAMILIES and observable),
            })
            if family == "threshold_policy":
                enriched["status"] = "inconclusive"
            final_rows.append(enriched)

        primary_rows = [row for row in final_rows if row["evaluation_class"] in {"behavioral_change", "unchanged_control"}]
        gate = dict(manifest.get("success_gate", {}))
        primary_budget = int(gate.get("budget", 50))
        primary = [row for row in primary_rows if int(row["budget"]) == primary_budget]
        by_budget = {
            str(budget): _metrics([row for row in primary_rows if int(row["budget"]) == budget])
            for budget in budgets
        }
        by_endpoint = _group_metrics(primary, "endpoint")
        by_family = {
            family: _metrics([row for row in primary if row["family"] in {family, "unchanged"}])
            for family in sorted(PRIMARY_FAMILIES)
        }
        by_corpus = _group_metrics(primary, "corpus")
        endpoint_sensitivities = [float(value["sensitivity"]) for value in by_endpoint.values() if value["changed_cases"]]
        full_corpora = sorted({str(row["corpus"]) for row in primary if bool(row["full_budget"])})
        overall = _metrics(primary)
        overall["macro_auroc_by_endpoint"] = _macro(by_endpoint, "auroc")
        overall["macro_auroc_by_corpus"] = _macro(by_corpus, "auroc")
        overall["macro_sensitivity_by_endpoint"] = _macro(by_endpoint, "sensitivity")
        overall["macro_false_alarm_rate_by_endpoint"] = _macro(by_endpoint, "false_alarm_rate")
        gates = {
            "budget": primary_budget,
            "minimum_sensitivity": float(gate.get("minimum_sensitivity", .80)),
            "maximum_unchanged_false_alarm_rate": float(gate.get("maximum_unchanged_false_alarm_rate", .05)),
            "minimum_endpoint_sensitivity": float(gate.get("minimum_endpoint_sensitivity", .70)),
            "minimum_full_budget_corpora": int(gate.get("minimum_full_budget_corpora", 4)),
        }
        gates["passed"] = bool(
            overall["macro_sensitivity_by_endpoint"] >= gates["minimum_sensitivity"]
            and overall["macro_false_alarm_rate_by_endpoint"] <= gates["maximum_unchanged_false_alarm_rate"]
            and endpoint_sensitivities
            and min(endpoint_sensitivities) >= gates["minimum_endpoint_sensitivity"]
            and len(full_corpora) >= gates["minimum_full_budget_corpora"]
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "construct": "prospective_operational_validation_evaluation",
            "evidence_status": "prospective_blind_validation",
            "manifest_sha256": manifest_digest,
            "panel_lock_sha256": _sha256(root / "panel.lock.json"),
            "truth_lock_sha256": truth_envelope["sha256"],
            "blinded_predictions_lock_sha256": verify_lock(staging / "blinded_predictions.lock.json")["sha256"],
            "draws": draws,
            "budgets": list(budgets),
            "overall_by_budget": by_budget,
            "primary_budget": overall,
            "macro_auroc": overall["macro_auroc_by_endpoint"],
            "by_endpoint": by_endpoint,
            "by_family": by_family,
            "by_corpus": by_corpus,
            "full_budget_corpora": full_corpora,
            "threshold_policy_cases": [row for row in final_rows if row["family"] == "threshold_policy"],
            "success_gate": gates,
            "claim_boundary": "Detect observable behavioral departures and localize changed probe responses; do not infer authorship, deployment FPR, exact internal cause, or diagnose fault families.",
        }
        _write_csv(staging / "predictions.csv", final_rows)
        metrics_path = staging / "validation_metrics.json"
        metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lock_forecasts(staging / "prospective_evaluation.lock.json", {
            "schema_version": SCHEMA_VERSION,
            "construct": "prospective_operational_validation_evaluation_lock",
            "manifest_sha256": manifest_digest,
            "blinded_predictions_sha256": _sha256(blinded_path),
            "blinded_predictions_lock_sha256": verify_lock(staging / "blinded_predictions.lock.json")["sha256"],
            "predictions_sha256": _sha256(staging / "predictions.csv"),
            "metrics_sha256": _sha256(metrics_path),
            "truth_read_after_blind_lock": True,
        })
        staging.replace(destination)
    except Exception:
        for child in staging.iterdir():
            if child.is_file():
                child.unlink()
        staging.rmdir()
        raise
    return destination


__all__ = ["evaluate_prospective_validation"]
