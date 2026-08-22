from __future__ import annotations

import csv
import io
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from .core import (
    EXTERNAL_VALIDATION_CORPORA,
    FORECAST_MODELS,
    STUDY_CORPORA,
    TARGET_CORPORA,
    canonical_json,
    validate_forecast_payload,
    verify_lock,
)
from .detectors import SPECS
from .evaluation import ForecastEvaluationRow, SuccessGateResult, evaluate_success_gate
from .workflow import PRIMARY_FPRS, assert_all_target_locks, fold_paths


def run_final_evaluation(
    root: Path,
    output_dir: Path | None = None,
    *,
    primary_corpora: Sequence[str] = STUDY_CORPORA,
    external_corpora: Sequence[str] = EXTERNAL_VALIDATION_CORPORA,
    detectors: Sequence[str] = tuple(SPECS),
) -> dict[str, object]:
    """Evaluate locked zero-score forecasts after every target test is scored."""
    root = Path(root).resolve()
    primary_corpora = tuple(primary_corpora)
    external_corpora = tuple(external_corpora)
    targets = primary_corpora + external_corpora
    detectors = tuple(detectors)
    if targets != TARGET_CORPORA or set(detectors) != set(SPECS):
        raise ValueError("Final evaluation requires the frozen nine-corpus, five-detector panel")

    assert_all_target_locks(root, targets)
    assert_all_target_locks(root, targets, privileged=True)

    rows: list[ForecastEvaluationRow] = []
    observed: dict[str, dict[str, dict[str, float]]] = {}
    frozen_thresholds: Mapping[str, object] | None = None
    for corpus in targets:
        paths = fold_paths(root, corpus)
        state = json.loads(paths.state.read_text(encoding="utf-8"))
        if state.get("phase") != "test_scored":
            raise RuntimeError(f"Final evaluation requires test_scored state for {corpus}")

        payload = verify_lock(paths.zero_lock)["payload"]
        manifest = payload.get("manifest") if isinstance(payload, Mapping) else None
        forecasts = payload.get("forecasts") if isinstance(payload, Mapping) else None
        if not isinstance(manifest, Mapping) or not isinstance(forecasts, list):
            raise RuntimeError(f"Invalid zero-score lock payload for {corpus}")
        thresholds = manifest.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise RuntimeError(f"Zero-score lock lacks frozen thresholds for {corpus}")
        if frozen_thresholds is None:
            frozen_thresholds = thresholds
        elif canonical_json(thresholds) != canonical_json(frozen_thresholds):
            raise RuntimeError("Frozen thresholds differ across target folds")

        validate_forecast_payload(
            {"admitted_detectors": list(detectors), "forecasts": forecasts},
            corpora=(corpus,), detectors=detectors,
        )
        corpus_observed = _observed_fprs(paths.database, corpus, detectors, thresholds)
        observed[corpus] = {
            f"{fpr:.2f}": corpus_observed[fpr] for fpr in PRIMARY_FPRS
        }
        for entry in forecasts:
            detector = str(entry["detector_config"])
            operating_fpr = float(entry["operating_fpr"])
            rows.append(ForecastEvaluationRow(
                corpus=corpus,
                detector=detector,
                dependency_group=SPECS[detector].dependency_group,
                signature_size=int(entry["signature_size"]),
                draw=int(entry["draw"]),
                model=str(entry["model"]),
                prediction=float(entry["prediction"]),
                observed_fpr=corpus_observed[operating_fpr][detector],
                operating_fpr=operating_fpr,
            ))

    primary_rows = [row for row in rows if row.corpus in primary_corpora]
    gates = {
        f"{fpr:.2f}": evaluate_success_gate(
            primary_rows,
            required_corpora=primary_corpora,
            operating_fpr=fpr,
        )
        for fpr in PRIMARY_FPRS
    }
    external = {
        corpus: {
            "observed_fpr": observed[corpus],
            "backend_macro_mae": _descriptive_mae(
                [row for row in rows if row.corpus == corpus]
            ),
            "gate_status": "descriptive_external_validation_only",
        }
        for corpus in external_corpora
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "primary_corpora": list(primary_corpora),
        "external_validation": external,
        "observed_fpr": {corpus: observed[corpus] for corpus in primary_corpora},
        "success_gates": {key: _gate_payload(value) for key, value in gates.items()},
    }
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        _atomic_text(
            output_dir / "final_evaluation.json",
            json.dumps(report, sort_keys=True, indent=2),
        )
        _atomic_text(output_dir / "forecast_evaluation_rows.csv", _rows_csv(rows, primary_corpora))
    return report


def _observed_fprs(
    database: Path,
    corpus: str,
    detectors: Sequence[str],
    threshold_artifact: Mapping[str, object],
) -> dict[float, dict[str, float]]:
    detector_thresholds = threshold_artifact.get("detectors")
    if not isinstance(detector_thresholds, Mapping) or set(detector_thresholds) != set(detectors):
        raise RuntimeError("Frozen threshold detector panel is incomplete")
    thresholds: dict[float, dict[str, float]] = {fpr: {} for fpr in PRIMARY_FPRS}
    for detector in detectors:
        row = detector_thresholds[detector]
        values = row.get("thresholds") if isinstance(row, Mapping) else None
        if not isinstance(values, Mapping):
            raise RuntimeError(f"Frozen thresholds are missing for {detector}")
        for fpr in PRIMARY_FPRS:
            value = float(values[f"{fpr:.2f}"])
            if not math.isfinite(value):
                raise RuntimeError(f"Frozen threshold is non-finite for {detector}")
            thresholds[fpr][detector] = value

    connection = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        record_ids = {
            str(row[0]) for row in connection.execute(
                "SELECT record_id FROM records WHERE corpus=? AND partition_name='test'",
                (corpus,),
            )
        }
        if not record_ids:
            raise RuntimeError(f"No test records for {corpus}")
        placeholders = ",".join("?" for _ in detectors)
        score_rows = connection.execute(
            f"""SELECT s.record_id,s.detector_config,s.detector_group,
                       s.canonical_ai_score,s.truncated,s.failure
                FROM scores s JOIN records r USING(record_id)
                WHERE r.corpus=? AND r.partition_name='test'
                  AND s.variant_id='original'
                  AND s.detector_config IN ({placeholders})""",
            (corpus, *detectors),
        ).fetchall()
    finally:
        connection.close()

    by_cell = {(str(row["record_id"]), str(row["detector_config"])): row for row in score_rows}
    expected = {(record_id, detector) for record_id in record_ids for detector in detectors}
    if set(by_cell) != expected:
        raise RuntimeError(
            f"Incomplete test score panel for {corpus}: missing={len(expected - set(by_cell))}"
        )
    scores: dict[str, list[float]] = defaultdict(list)
    for (_, detector), row in by_cell.items():
        score = row["canonical_ai_score"]
        if (
            row["detector_group"] != SPECS[detector].dependency_group
            or row["failure"] is not None
            or bool(row["truncated"])
            or score is None
            or not math.isfinite(float(score))
        ):
            raise RuntimeError(f"Invalid or truncated test score for {corpus}/{detector}")
        scores[detector].append(float(score))
    return {
        fpr: {
            detector: mean(score >= thresholds[fpr][detector] for score in scores[detector])
            for detector in detectors
        }
        for fpr in PRIMARY_FPRS
    }


def _descriptive_mae(rows: Sequence[ForecastEvaluationRow]) -> dict[str, float]:
    grouped: dict[tuple[float, int, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.operating_fpr, row.signature_size, row.draw, row.model, row.dependency_group)].append(
            abs(row.prediction - row.observed_fpr)
        )
    output: dict[str, float] = {}
    for fpr in PRIMARY_FPRS:
        for size in (50, 100, 250):
            for model in FORECAST_MODELS:
                per_draw = []
                for draw in range(20):
                    group_losses = [
                        mean(grouped[(fpr, size, draw, model, group)])
                        for group in {spec.dependency_group for spec in SPECS.values()}
                    ]
                    per_draw.append(mean(group_losses))
                output[f"{fpr:.2f}:{size}:{model}"] = mean(per_draw)
    return output


def _gate_payload(result: SuccessGateResult) -> dict[str, object]:
    return {
        "passed": result.passed,
        "corpus_losses": {
            f"{corpus}:{size}:{model}": value
            for (corpus, size, model), value in result.corpus_losses.items()
        },
        "overall_mae": {
            f"{size}:{model}": value for (size, model), value in result.overall_mae.items()
        },
        "wins_over_detector_id": dict(result.wins_over_detector_id),
        "corpus_improvements": dict(result.corpus_improvements),
        "sign_flip_p": result.sign_flip_p,
        "simpler_baselines_beaten": {
            f"{size}:{model}": value
            for (size, model), value in result.simpler_baselines_beaten.items()
        },
        "failures": list(result.failures),
    }


def _rows_csv(rows: Sequence[ForecastEvaluationRow], primary_corpora: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    fields = (
        "validation_scope", "corpus", "detector", "dependency_group", "signature_size",
        "draw", "model", "prediction", "observed_fpr", "operating_fpr",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "validation_scope": "primary" if row.corpus in primary_corpora else "external",
            **row.__dict__,
        })
    return output.getvalue()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False,
        ) as handle:
            handle.write(content)
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()
