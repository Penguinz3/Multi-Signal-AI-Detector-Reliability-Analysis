from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from html import escape
from pathlib import Path
from typing import Mapping


PRODUCT_SCHEMA_VERSION = 1
SUPPORTED_CONSTRUCTS = (
    "black_box_behavioral_conformance_and_coarse_fault_localization",
    "behavioral_conformance",
)
SCORE_TABLE_REQUIRED_FIELDS = (
    "triplet_id",
    "intensity",
    "audited_endpoint",
    "fault_id",
    "effective_endpoint",
    "native_score",
    "canonical_ai_score",
)
SCORE_TABLE_OPTIONAL_FIELDS = (
    "input_token_count",
    "effective_token_count",
    "max_tokens",
    "truncated",
    "runtime_ms",
    "failure",
)
REPORT_REQUIRED_FIELDS = (
    "schema_version",
    "construct",
    "manifest_lock_sha256",
    "success_gates",
    "discovery_metrics",
    "confirmation_metrics",
    "claim_boundary",
)


def _atomic_write(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def export_contracts(output_dir: Path) -> dict[str, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    score_contract = {
        "contract": "fprint.external_fault_score_table",
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "format": "CSV with a header row",
        "required_fields": list(SCORE_TABLE_REQUIRED_FIELDS),
        "optional_fields": list(SCORE_TABLE_OPTIONAL_FIELDS),
        "enums": {"intensity": ["original", "low", "high"]},
        "constraints": [
            "IDs, endpoints, and fault IDs must belong to the locked audit manifest.",
            "canonical_ai_score must increase with evidence for AI-generated text.",
            "Failed or truncated rows must retain their failure and truncation provenance.",
        ],
    }
    report_contract = {
        "contract": "fprint.fault_audit_evaluation",
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "format": "JSON",
        "required_top_level_fields": list(REPORT_REQUIRED_FIELDS),
        "supported_constructs": list(SUPPORTED_CONSTRUCTS),
        "allowed_primary_claims": [
            "coarse_fault_detection_and_diagnosis",
            "behavioral_change_detection_and_localization",
            "negative_result",
        ],
    }
    score_path = output_dir / "fault_score_table_contract_v1.json"
    report_path = output_dir / "fault_audit_report_contract_v1.json"
    template_path = output_dir / "fault_score_table_template_v1.csv"
    _atomic_write(score_path, json.dumps(score_contract, indent=2, sort_keys=True) + "\n")
    _atomic_write(report_path, json.dumps(report_contract, indent=2, sort_keys=True) + "\n")
    rows = [SCORE_TABLE_REQUIRED_FIELDS + SCORE_TABLE_OPTIONAL_FIELDS]
    temporary = template_path.with_name(template_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    temporary.replace(template_path)
    return {
        "score_contract": str(score_path),
        "report_contract": str(report_path),
        "score_template": str(template_path),
    }


def validate_evaluation_report(report: object) -> dict:
    if not isinstance(report, dict):
        raise ValueError("Fault-audit evaluation must be a JSON object")
    missing = sorted(set(REPORT_REQUIRED_FIELDS) - set(report))
    if missing:
        raise ValueError(f"Fault-audit evaluation lacks fields: {missing}")
    if report["schema_version"] != PRODUCT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported fault-audit schema version: {report['schema_version']}")
    if report["construct"] not in SUPPORTED_CONSTRUCTS:
        raise ValueError("Fault-audit evaluation has an unsupported construct")
    if not re.fullmatch(r"[0-9a-f]{64}", str(report["manifest_lock_sha256"])):
        raise ValueError("manifest_lock_sha256 must be a lowercase SHA-256 digest")
    gates = report["success_gates"]
    if not isinstance(gates, dict) or gates.get("permitted_primary_claim") not in {
        "coarse_fault_detection_and_diagnosis",
        "behavioral_change_detection_and_localization",
        "negative_result",
    }:
        raise ValueError("Fault-audit evaluation has an invalid permitted primary claim")
    combined = gates.get("channels", {}).get("combined")
    if not isinstance(combined, dict):
        raise ValueError("Fault-audit evaluation lacks primary combined-channel metrics")
    for name in ("macro_auroc", "macro_sensitivity", "unchanged_false_alarm_rate"):
        value = combined.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"Primary metric {name} must be finite")
        if not 0 <= value <= 1:
            raise ValueError(f"Primary metric {name} must be between zero and one")
    if not isinstance(report["claim_boundary"], str) or not report["claim_boundary"].strip():
        raise ValueError("Fault-audit evaluation requires a claim boundary")
    if not isinstance(report["discovery_metrics"], dict) or not isinstance(report["confirmation_metrics"], dict):
        raise ValueError("Fault-audit metric collections must be JSON objects")
    return report


def load_evaluation_report(path: Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as handle:
        return validate_evaluation_report(json.load(handle))


def _percent(value: object) -> str:
    return f"{100 * float(value):.1f}%"


def _metric_rows(metrics: Mapping[str, object]) -> str:
    labels = (
        ("Macro AUROC", "macro_auroc", False),
        ("Sensitivity", "macro_sensitivity", True),
        ("Unchanged false-alarm rate", "unchanged_false_alarm_rate", True),
    )
    return "".join(
        f"<tr><th scope=\"row\">{escape(label)}</th><td>{_percent(metrics[key]) if percent else f'{float(metrics[key]):.3f}'}</td></tr>"
        for label, key, percent in labels
    )


def render_evaluation_html(report: Mapping[str, object]) -> str:
    report = validate_evaluation_report(dict(report))
    gates = report["success_gates"]
    primary = gates["channels"]["combined"]
    confirmation = report.get("confirmation_metrics", {}).get("combined", {})
    detection_passed = bool(gates.get("detection_gate_passed"))
    diagnosis_passed = bool(gates.get("diagnosis_gate_passed"))
    outcome = "Validated for behavioral change detection" if detection_passed else "Primary validation gate not met"
    outcome_class = "pass" if detection_passed else "fail"
    diagnosis = (
        "Coarse fault-family diagnosis passed its validation gate."
        if diagnosis_passed
        else "Coarse fault-family diagnosis did not pass validation and must not be used operationally."
    )
    confirmation_table = (
        f"<table><caption>Prospective confirmation</caption><tbody>{_metric_rows(confirmation)}</tbody></table>"
        if all(key in confirmation for key in ("macro_auroc", "macro_sensitivity", "unchanged_false_alarm_rate"))
        else "<p>Prospective confirmation metrics were not supplied.</p>"
    )
    family_rows = "".join(
        f"<tr><th scope=\"row\">{escape(str(name).replace('_', ' ').title())}</th><td>{float(value):.3f}</td></tr>"
        for name, value in sorted(primary.get("auroc_by_family", {}).items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FPRINT behavioral conformance report</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; line-height: 1.5; }}
body {{ margin: 0 auto; max-width: 72rem; padding: 2rem; }}
header, section {{ border: 1px solid #8886; border-radius: .75rem; margin: 0 0 1rem; padding: 1.25rem; }}
h1, h2 {{ line-height: 1.15; }}
.status {{ border-left: .45rem solid; font-size: 1.1rem; font-weight: 700; padding: .8rem 1rem; }}
.pass {{ border-color: #16803c; }} .fail {{ border-color: #b42318; }}
.warning {{ background: #b86e0018; border-left: .35rem solid #b86e00; padding: .8rem 1rem; }}
.grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr)); }}
table {{ border-collapse: collapse; width: 100%; }} caption {{ font-weight: 700; text-align: left; margin-bottom: .5rem; }}
th, td {{ border-bottom: 1px solid #8885; padding: .5rem; text-align: left; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<header>
<p>FPRINT · system-level behavioral assurance</p>
<h1>Black-box AI-text detector conformance report</h1>
<p class="status {outcome_class}">{escape(outcome)}</p>
<p>Permitted claim: <strong>{escape(str(gates['permitted_primary_claim']).replace('_', ' '))}</strong></p>
</header>
<main>
<section>
<h2>Validation evidence</h2>
<div class="grid">
<table><caption>Held-out-corpus evaluation</caption><tbody>{_metric_rows(primary)}</tbody></table>
{confirmation_table}
</div>
</section>
<section>
<h2>Behavior by declared change family</h2>
<table><caption>Primary AUROC</caption><tbody>{family_rows}</tbody></table>
</section>
<section>
<h2>Operational boundary</h2>
<p class="warning">{escape(diagnosis)}</p>
<p>{escape(str(report['claim_boundary']))}</p>
<p>FPRINT is not an authorship detector and must not be used to adjudicate whether an individual person used AI.</p>
</section>
<section>
<h2>Integrity</h2>
<p>Evaluation schema: v{PRODUCT_SCHEMA_VERSION}</p>
<p>Locked manifest: <code>{escape(str(report['manifest_lock_sha256']))}</code></p>
</section>
</main>
</body>
</html>
"""


def write_evaluation_html(evaluation_path: Path, output_path: Path) -> Path:
    report = load_evaluation_report(evaluation_path)
    output_path = Path(output_path).resolve()
    _atomic_write(output_path, render_evaluation_html(report))
    return output_path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_evaluation_summary(report: Mapping[str, object]) -> dict:
    validated = validate_evaluation_report(dict(report))
    return {field: validated[field] for field in REPORT_REQUIRED_FIELDS}


def build_release_bundle(evaluation_path: Path, output_dir: Path) -> Path:
    """Publish a privacy-safe release atomically; refuse to mix with existing files."""
    evaluation_path = Path(evaluation_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Release output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    report = load_evaluation_report(evaluation_path)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        summary = public_evaluation_summary(report)
        _atomic_write(staging / "evaluation_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _atomic_write(staging / "index.html", render_evaluation_html(summary))
        export_contracts(staging / "contracts")
        files = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            files.append({
                "path": path.relative_to(staging).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            })
        manifest = {
            "product": "FPRINT behavioral conformance report",
            "schema_version": PRODUCT_SCHEMA_VERSION,
            "manifest_lock_sha256": report["manifest_lock_sha256"],
            "source_evaluation_sha256": _file_sha256(evaluation_path),
            "permitted_primary_claim": report["success_gates"]["permitted_primary_claim"],
            "files": files,
        }
        _atomic_write(staging / "release_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir
