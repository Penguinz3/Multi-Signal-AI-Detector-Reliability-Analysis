from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import tempfile
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Mapping, Sequence

from .core import lock_forecasts, make_probe_triplet, slope, verify_lock


SCHEMA_VERSION = 1
PROBES = ("punctuation_normalization", "sentence_splitting", "paragraph_resegmentation")
LEVELS = ("original", "low", "high")
FEATURES = ("original_score", "low_shift", "high_shift", "slope")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
RUN_METADATA_FIELDS = ("version", "configuration", "threshold_policy", "collected_at_utc")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot take a quantile of an empty collection")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _read_records(path: Path) -> list[tuple[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"record_id", "text"}.issubset(reader.fieldnames):
            raise ValueError("Records CSV requires record_id and text columns")
        records, seen = [], set()
        for row in reader:
            record_id, text = str(row.get("record_id", "")).strip(), str(row.get("text", "")).strip()
            if not record_id or not text:
                raise ValueError("Every record requires a non-empty record_id and text")
            if record_id in seen:
                raise ValueError(f"Duplicate record_id: {record_id}")
            seen.add(record_id)
            records.append((record_id, text))
    if not records:
        raise ValueError("Records CSV is empty")
    return records


def initialize_audit(
    records_path: Path,
    audit_root: Path,
    endpoint: str,
    *,
    seed: str = "fprint-operational-v1",
    minimum_triplets_per_probe: int = 10,
    minimum_sites: int = 4,
    alpha: float = .05,
    absolute_tolerance: float = .01,
    noise_multiplier: float = 3.0,
    minimum_affected_fraction: float = .20,
    maximum_reference_noise: float = .02,
) -> dict:
    if not endpoint.strip():
        raise ValueError("Endpoint identifier is required")
    if minimum_triplets_per_probe < 3 or minimum_sites < 1:
        raise ValueError("Require at least three triplets per probe and one eligible site")
    if not 0 < alpha < 1 or not 0 <= minimum_affected_fraction <= 1:
        raise ValueError("Invalid alarm rule")
    if min(absolute_tolerance, noise_multiplier, maximum_reference_noise) < 0:
        raise ValueError("Tolerances cannot be negative")
    audit_root = Path(audit_root).resolve()
    if audit_root.exists():
        raise FileExistsError(f"Audit root already exists: {audit_root}")
    records = _read_records(records_path)
    rows, counts = [], {probe: 0 for probe in PROBES}
    for record_id, text in records:
        for probe in PROBES:
            triplet = make_probe_triplet(probe, text, f"{seed}:{record_id}:{probe}", minimum_sites)
            if triplet is None:
                continue
            triplet_id = hashlib.sha256(
                f"{endpoint}\0{record_id}\0{probe}\0{_text_sha256(text)}".encode()
            ).hexdigest()[:24]
            counts[probe] += 1
            for level, value, intensity in (
                ("original", triplet.original, 0.0),
                ("low", triplet.low, triplet.low_intensity),
                ("high", triplet.high, triplet.high_intensity),
            ):
                rows.append({
                    "challenge_id": f"{triplet_id}:{level}",
                    "triplet_id": triplet_id,
                    "record_id": record_id,
                    "probe": probe,
                    "intensity": level,
                    "intensity_value": f"{intensity:.12g}",
                    "text_sha256": _text_sha256(value),
                    "text": value,
                })
    sparse = {probe: count for probe, count in counts.items() if count < minimum_triplets_per_probe}
    if sparse:
        raise ValueError(f"Insufficient eligible records for probes: {sparse}")
    audit_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{audit_root.name}-", dir=audit_root.parent))
    try:
        challenge = staging / "challenge.csv"
        with challenge.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        template = staging / "scores_template.csv"
        with template.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("challenge_id", "canonical_ai_score", "native_score", "truncated", "failure"))
            writer.writerows((row["challenge_id"], "", "", "", "") for row in rows)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "construct": "operational_behavioral_conformance",
            "endpoint": endpoint.strip(),
            "seed": seed,
            "probes": list(PROBES),
            "challenge_rows": len(rows),
            "triplets_by_probe": counts,
            "challenge_sha256": _file_sha256(challenge),
            "decision_rule": {
                "name": "paired_repeat_noise_v1",
                "alpha": alpha,
                "absolute_tolerance": absolute_tolerance,
                "noise_multiplier": noise_multiplier,
                "minimum_affected_fraction": minimum_affected_fraction,
                "maximum_reference_noise": maximum_reference_noise,
            },
        }
        lock_forecasts(staging / "manifest.lock.json", manifest)
        (staging / "runs").mkdir()
        staging.replace(audit_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"audit_root": str(audit_root), "endpoint": endpoint.strip(), "triplets_by_probe": counts}


def _load_audit(audit_root: Path) -> tuple[dict, str, list[dict[str, str]]]:
    root = Path(audit_root).resolve()
    envelope = verify_lock(root / "manifest.lock.json")
    manifest, digest = envelope["payload"], envelope["sha256"]
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("construct") != "operational_behavioral_conformance":
        raise ValueError("Unsupported operational audit manifest")
    challenge = root / "challenge.csv"
    if _file_sha256(challenge) != manifest.get("challenge_sha256"):
        raise RuntimeError("Challenge table disagrees with the locked manifest")
    with challenge.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"challenge_id", "triplet_id", "record_id", "probe", "intensity", "intensity_value", "text_sha256", "text"}
    if len(rows) != manifest.get("challenge_rows") or any(not required.issubset(row) for row in rows):
        raise RuntimeError("Challenge table shape disagrees with the locked manifest")
    ids = [row["challenge_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Challenge table contains duplicate IDs")
    if any(_text_sha256(row["text"]) != row["text_sha256"] for row in rows):
        raise RuntimeError("Challenge text hash mismatch")
    return manifest, digest, rows


def export_challenge(audit_root: Path, output_dir: Path) -> Path:
    _, _, _ = _load_audit(audit_root)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Challenge export already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        shutil.copyfile(Path(audit_root).resolve() / "challenge.csv", staging / "challenge.csv")
        shutil.copyfile(Path(audit_root).resolve() / "scores_template.csv", staging / "scores_template.csv")
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def import_run(
    audit_root: Path,
    run_id: str,
    role: str,
    scores_path: Path,
    metadata_path: Path | None,
) -> Path:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, numbers, dots, underscores, or hyphens")
    if role not in {"reference", "current"}:
        raise ValueError("Run role must be reference or current")
    manifest, manifest_digest, challenge = _load_audit(audit_root)
    if metadata_path is None:
        raise ValueError(f"Run metadata is required: {', '.join(RUN_METADATA_FIELDS)}")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8-sig"))
    if not isinstance(metadata, dict) or any(not str(metadata.get(field, "")).strip() for field in RUN_METADATA_FIELDS):
        raise ValueError(f"Run metadata requires non-empty fields: {', '.join(RUN_METADATA_FIELDS)}")
    metadata = {field: str(metadata[field]).strip() for field in RUN_METADATA_FIELDS}
    with Path(scores_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"challenge_id", "canonical_ai_score"}.issubset(reader.fieldnames):
            raise ValueError("Score table requires challenge_id and canonical_ai_score")
        scores = {}
        for row in reader:
            challenge_id = str(row.get("challenge_id", "")).strip()
            if challenge_id in scores:
                raise ValueError(f"Duplicate challenge_id in scores: {challenge_id}")
            if str(row.get("failure", "")).strip():
                raise ValueError(f"Failed detector query for {challenge_id}")
            truncated = str(row.get("truncated", "")).strip().casefold()
            if truncated not in {"", "0", "false", "no"}:
                if truncated in {"1", "true", "yes"}:
                    raise ValueError(f"Truncated detector query for {challenge_id}")
                raise ValueError(f"Invalid truncated value for {challenge_id}")
            try:
                score = float(row["canonical_ai_score"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid canonical score for {challenge_id}") from error
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Canonical score must be finite and in [0,1]: {challenge_id}")
            scores[challenge_id] = score
    expected = {row["challenge_id"] for row in challenge}
    if set(scores) != expected:
        raise ValueError(f"Score IDs do not match challenge: missing={len(expected - set(scores))}, extra={len(set(scores) - expected)}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "construct": "operational_detector_run",
        "manifest_sha256": manifest_digest,
        "endpoint": manifest["endpoint"],
        "run_id": run_id,
        "role": role,
        "source_score_sha256": _file_sha256(scores_path),
        "metadata": metadata,
        "scores": [{"challenge_id": row["challenge_id"], "canonical_ai_score": scores[row["challenge_id"]]} for row in challenge],
    }
    destination = Path(audit_root).resolve() / "runs" / f"{run_id}.lock.json"
    lock_forecasts(destination, payload)
    return destination


def _load_run(audit_root: Path, run_id: str, manifest_digest: str, expected_ids: set[str]) -> dict:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"Invalid run ID: {run_id}")
    payload = verify_lock(Path(audit_root).resolve() / "runs" / f"{run_id}.lock.json")["payload"]
    if payload.get("manifest_sha256") != manifest_digest:
        raise RuntimeError(f"Run {run_id} belongs to a different audit")
    scores = {str(row["challenge_id"]): float(row["canonical_ai_score"]) for row in payload.get("scores", [])}
    if set(scores) != expected_ids or len(scores) != len(payload.get("scores", [])):
        raise RuntimeError(f"Run {run_id} is incomplete or duplicated")
    payload["score_map"] = scores
    return payload


def _triplet_features(challenge: Sequence[Mapping[str, str]], scores: Mapping[str, float]) -> list[dict]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in challenge:
        grouped[row["triplet_id"]].append(row)
    result = []
    for triplet_id, rows in grouped.items():
        by_level = {row["intensity"]: row for row in rows}
        if set(by_level) != set(LEVELS):
            raise RuntimeError(f"Triplet {triplet_id} is incomplete")
        values = [scores[by_level[level]["challenge_id"]] for level in LEVELS]
        intensities = [float(by_level[level]["intensity_value"]) for level in LEVELS]
        result.append({
            "triplet_id": triplet_id,
            "probe": by_level["original"]["probe"],
            "features": {
                "original_score": values[0],
                "low_shift": values[1] - values[0],
                "high_shift": values[2] - values[0],
                "slope": slope(intensities, values),
            },
        })
    return result


def _binomial_upper_tail(successes: int, trials: int) -> float:
    return sum(math.comb(trials, count) for count in range(successes, trials + 1)) / (2 ** trials)


def _render_report(report: Mapping[str, object]) -> str:
    status = str(report["status"])
    rows = []
    for probe, cells in report["probe_results"].items():
        for feature, cell in cells.items():
            rows.append(
                "<tr>"
                f"<td>{escape(str(probe).replace('_', ' '))}</td>"
                f"<td>{escape(str(feature).replace('_', ' '))}</td>"
                f"<td>{float(cell['median_current_delta']):.4f}</td>"
                f"<td>{float(cell['affected_fraction']):.1%}</td>"
                f"<td>{'yes' if cell['changed'] else 'no'}</td>"
                "</tr>"
            )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FPRINT operational conformance report</title><style>
:root{{font-family:system-ui,sans-serif;line-height:1.5;color-scheme:light dark}}body{{max-width:72rem;margin:auto;padding:2rem}}
header,section{{border:1px solid #8886;border-radius:.75rem;padding:1.25rem;margin-bottom:1rem}}table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #8885;padding:.5rem;text-align:left}}.status{{font-size:1.2rem;font-weight:700;border-left:.45rem solid #b86e00;padding:.7rem}}
</style></head><body><header><p>FPRINT · operational beta</p><h1>Detector behavioral conformance</h1>
<p class="status">{escape(status.upper())}</p><p>Endpoint: <strong>{escape(str(report['endpoint']))}</strong></p></header>
<section><h2>Runs</h2><p>Reference versions: {escape(', '.join(item['version'] for item in report['reference_metadata']))}</p>
<p>Current version: {escape(report['current_metadata']['version'])}</p></section>
<section><h2>Decision</h2><p>Reference noise p95: {float(report['reference_noise_p95']):.4f}</p>
<p>Reference feature noise p95: {float(report['reference_feature_noise_p95']):.4f}</p>
<p>Changed cells: {int(report['changed_cells'])} of {int(report['tested_cells'])}</p>
<p>Revalidation required: {'yes' if report['revalidation_required'] else 'no'}</p></section>
<section><h2>Probe-level localization</h2><table><thead><tr><th>Probe</th><th>Behavior</th><th>Median change</th><th>Affected</th><th>Changed</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></section><section><h2>Boundary</h2><p>{escape(str(report['claim_boundary']))}</p></section></body></html>
"""


def compare_runs(
    audit_root: Path,
    reference_run_ids: Sequence[str],
    current_run_id: str,
    output_dir: Path,
) -> Path:
    if len(reference_run_ids) != 2 or len(set(reference_run_ids)) != 2:
        raise ValueError("Exactly two distinct reference runs are required")
    manifest, manifest_digest, challenge = _load_audit(audit_root)
    expected_ids = {row["challenge_id"] for row in challenge}
    references = [_load_run(audit_root, run_id, manifest_digest, expected_ids) for run_id in reference_run_ids]
    current = _load_run(audit_root, current_run_id, manifest_digest, expected_ids)
    if any(run.get("role") != "reference" for run in references) or current.get("role") != "current":
        raise ValueError("Run roles do not match the requested comparison")
    reference_noise = [
        abs(references[0]["score_map"][challenge_id] - references[1]["score_map"][challenge_id])
        for challenge_id in sorted(expected_ids)
    ]
    reference_noise_p95 = _quantile(reference_noise, .95)
    feature_sets = [_triplet_features(challenge, run["score_map"]) for run in (*references, current)]
    indexed = [
        {(row["triplet_id"], row["probe"]): row["features"] for row in rows}
        for rows in feature_sets
    ]
    rule = manifest["decision_rule"]
    cells: dict[str, dict[str, dict]] = {probe: {} for probe in manifest["probes"]}
    total_cells = len(manifest["probes"]) * len(FEATURES)
    adjusted_alpha = float(rule["alpha"]) / total_cells
    all_feature_repeat_deltas = []
    for probe in manifest["probes"]:
        keys = sorted(key for key in indexed[0] if key[1] == probe)
        for feature in FEATURES:
            repeat_deltas, current_deltas = [], []
            for key in keys:
                left, right, now = (float(index[ key ][feature]) for index in indexed)
                repeat_deltas.append(abs(left - right))
                current_deltas.append(abs(now - (left + right) / 2))
            all_feature_repeat_deltas.extend(repeat_deltas)
            tolerance = float(rule["absolute_tolerance"])
            multiplier = float(rule["noise_multiplier"])
            affected = [current_delta > max(tolerance, multiplier * repeat_delta) for current_delta, repeat_delta in zip(current_deltas, repeat_deltas)]
            wins = sum(current_delta > repeat_delta + tolerance for current_delta, repeat_delta in zip(current_deltas, repeat_deltas))
            p_value = _binomial_upper_tail(wins, len(keys))
            affected_fraction = sum(affected) / len(affected)
            median_current = statistics.median(current_deltas)
            median_repeat = statistics.median(repeat_deltas)
            changed = (
                p_value <= adjusted_alpha
                and affected_fraction >= float(rule["minimum_affected_fraction"])
                and median_current > max(tolerance, multiplier * median_repeat)
            )
            cells[probe][feature] = {
                "triplets": len(keys),
                "median_current_delta": median_current,
                "median_reference_repeat_delta": median_repeat,
                "affected_fraction": affected_fraction,
                "one_sided_binomial_p": p_value,
                "adjusted_alpha": adjusted_alpha,
                "changed": changed,
            }
    changed_cells = sum(bool(cell["changed"]) for probe in cells.values() for cell in probe.values())
    reference_feature_noise_p95 = _quantile(all_feature_repeat_deltas, .95)
    unstable = max(reference_noise_p95, reference_feature_noise_p95) > float(rule["maximum_reference_noise"])
    status = "inconclusive" if unstable else "changed" if changed_cells else "unchanged"
    report = {
        "schema_version": SCHEMA_VERSION,
        "construct": "operational_behavioral_conformance_report",
        "evidence_status": "engineering_beta_not_yet_validated_on_external_endpoints",
        "manifest_lock_sha256": manifest_digest,
        "endpoint": manifest["endpoint"],
        "reference_runs": list(reference_run_ids),
        "current_run": current_run_id,
        "reference_metadata": [run["metadata"] for run in references],
        "current_metadata": current["metadata"],
        "status": status,
        "revalidation_required": status != "unchanged",
        "reference_noise_p95": reference_noise_p95,
        "reference_feature_noise_p95": reference_feature_noise_p95,
        "changed_cells": changed_cells,
        "tested_cells": total_cells,
        "decision_rule": rule,
        "probe_results": cells,
        "claim_boundary": "This engineering-beta report identifies observable score and probe-response changes only. It does not determine authorship, estimate deployment accuracy, or identify an internal detector cause.",
    }
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Report output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (staging / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "index.html").write_text(_render_report(report), encoding="utf-8")
        files = []
        for path in sorted(staging.iterdir()):
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)})
        (staging / "release_manifest.json").write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "manifest_lock_sha256": manifest_digest, "files": files}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir
