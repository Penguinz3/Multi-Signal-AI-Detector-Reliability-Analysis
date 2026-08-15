from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
import tempfile
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from .core import (
    FORECAST_MODELS, PROBES, STUDY_CORPORA, TextRecord, canonical_json,
    repeated_signature_samples, slope, validate_forecast_payload,
)
from .detectors import SPECS
from .features import FEATURE_NAMES, target_features
from .modeling import Observation, RecomputedFold, fit_forecaster, tune_c_nested
from .workflow import (
    assert_prelock_database, build_forecast_manifest, fold_paths,
    lock_zero_score_forecasts,
)

OPERATING_FPRS = (.05, .01)


@dataclass(frozen=True)
class ProbeRow:
    triplet_id: str
    corpus: str
    probe: str
    group_id: str
    slopes: Mapping[str, float]


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)


def _clean_commit() -> str:
    repository = Path(__file__).resolve().parents[1]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "fprint", "fprint_config.json"],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Forecast building requires committed FPRINT code and configuration")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _threshold_data(
    master: sqlite3.Connection,
    artifact_path: Path,
    detectors: Sequence[str],
) -> tuple[dict, dict[str, list[float]], dict[float, dict[str, float]]]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if set(artifact.get("detectors", ())) != set(detectors):
        raise RuntimeError("Frozen threshold panel differs from the admitted panel")
    cdfs: dict[str, list[float]] = {}
    thresholds = {fpr: {} for fpr in OPERATING_FPRS}
    retained = [
        (str(row["record_id"]), str(row["text_hash"]))
        for row in master.execute(
            "SELECT record_id,text_hash FROM records WHERE partition_name='threshold_reference' ORDER BY record_id"
        )
    ]
    retained_digest = hashlib.sha256(canonical_json(sorted(retained))).hexdigest()
    if len(retained) != int(artifact["retained_raid_count"]) or retained_digest != artifact["retained_raid_sha256"]:
        raise RuntimeError("Frozen retained-RAID identity hash mismatch")
    for detector in detectors:
        rows = master.execute(
            """SELECT s.record_id,s.canonical_ai_score
               FROM scores s JOIN records r USING(record_id)
               WHERE r.partition_name='threshold_reference'
                 AND s.variant_id='original' AND s.detector_config=?
                 AND s.failure IS NULL AND s.truncated=0
                 AND s.canonical_ai_score IS NOT NULL
               ORDER BY s.record_id""",
            (detector,),
        ).fetchall()
        if len(rows) != int(artifact["retained_raid_count"]):
            raise RuntimeError(f"Incomplete frozen RAID scores for {detector}")
        score_pairs = [(str(row["record_id"]), float(row["canonical_ai_score"])) for row in rows]
        digest = hashlib.sha256(canonical_json(sorted(score_pairs))).hexdigest()
        if digest != artifact["detectors"][detector]["score_sha256"]:
            raise RuntimeError(f"Frozen RAID score hash mismatch for {detector}")
        cdfs[detector] = sorted(score for _, score in score_pairs)
        for fpr in OPERATING_FPRS:
            thresholds[fpr][detector] = float(
                artifact["detectors"][detector]["thresholds"][f"{fpr:.2f}"]
            )
    return artifact, cdfs, thresholds


def _feature_rows(
    fold: sqlite3.Connection,
    source_corpora: Sequence[str],
    target_corpus: str,
) -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[str, str]], dict[str, list[str]]]:
    placeholders = ",".join("?" for _ in source_corpora)
    rows = fold.execute(
        f"""SELECT record_id,corpus,group_id,partition_name,text FROM records
             WHERE (partition_name='source_model' AND corpus IN ({placeholders}))
                OR (partition_name='signature' AND corpus=?)
             ORDER BY corpus,partition_name,record_id""",
        (*source_corpora, target_corpus),
    ).fetchall()
    features: dict[str, tuple[float, ...]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    data_ids: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        values = target_features(str(row["text"]))
        record_id = str(row["record_id"])
        features[record_id] = tuple(float(values[name]) for name in FEATURE_NAMES)
        metadata[record_id] = (str(row["corpus"]), str(row["group_id"]))
        data_ids[f"{row['partition_name']}:{row['corpus']}"].append(record_id)
    if not data_ids.get(f"signature:{target_corpus}"):
        raise RuntimeError(f"No unscored signature records for {target_corpus}")
    return features, metadata, dict(data_ids)


def _source_outcomes(
    fold: sqlite3.Connection,
    source_corpora: Sequence[str],
    detectors: Sequence[str],
    features: Mapping[str, tuple[float, ...]],
    thresholds: Mapping[str, float],
) -> tuple[Observation, ...]:
    placeholders = ",".join("?" for _ in source_corpora)
    records = fold.execute(
        f"SELECT record_id,corpus FROM records WHERE partition_name='source_model' AND corpus IN ({placeholders})",
        tuple(source_corpora),
    ).fetchall()
    scores = {
        (str(row["record_id"]), str(row["detector_config"])): float(row["canonical_ai_score"])
        for row in fold.execute(
            f"""SELECT s.record_id,s.detector_config,s.canonical_ai_score
                FROM scores s JOIN records r USING(record_id)
                WHERE r.partition_name='source_model' AND r.corpus IN ({placeholders})
                  AND s.variant_id='original' AND s.detector_config IN ({','.join('?' for _ in detectors)})
                  AND s.failure IS NULL AND s.truncated=0
                  AND s.canonical_ai_score IS NOT NULL""",
            (*source_corpora, *detectors),
        )
    }
    expected = len(records) * len(detectors)
    if len(scores) != expected:
        raise RuntimeError(f"Source-model panel is incomplete: {len(scores)}/{expected}")
    return tuple(
        Observation(
            str(record["corpus"]), detector, SPECS[detector].dependency_group,
            float(scores[(str(record["record_id"]), detector)] >= thresholds[detector]),
            0.0, (), features[str(record["record_id"])],
        )
        for record in records for detector in detectors
    )


def _probe_rows(
    fold: sqlite3.Connection,
    detectors: Sequence[str],
    cdfs: Mapping[str, Sequence[float]],
) -> tuple[ProbeRow, ...]:
    triplets = fold.execute(
        """SELECT p.triplet_id,p.record_id,p.corpus,p.probe,p.low_intensity,
                  p.high_intensity,r.group_id
           FROM probe_triplets p JOIN records r USING(record_id)
           WHERE p.corpus IN ({}) ORDER BY p.corpus,p.probe,p.triplet_id""".format(
               ",".join("?" for _ in STUDY_CORPORA)
           ),
        STUDY_CORPORA,
    ).fetchall()
    checks = {
        (str(row["triplet_id"]), str(row["detector_config"])): bool(row["fits"])
        for row in fold.execute(
            "SELECT triplet_id,detector_config,fits FROM probe_token_checks WHERE detector_config IN ({})".format(
                ",".join("?" for _ in detectors)
            ),
            tuple(detectors),
        )
    }
    score_rows = fold.execute(
        """SELECT p.triplet_id,s.detector_config,s.variant_id,s.canonical_ai_score,
                  s.failure,s.truncated
           FROM probe_triplets p JOIN scores s USING(record_id)
           WHERE s.detector_config IN ({}) AND s.variant_id<>'original'""".format(
               ",".join("?" for _ in detectors)
           ),
        tuple(detectors),
    ).fetchall()
    scores = {
        (str(row["triplet_id"]), str(row["detector_config"]), str(row["variant_id"])): row
        for row in score_rows
    }
    eligible: list[ProbeRow] = []
    for triplet in triplets:
        triplet_id, probe = str(triplet["triplet_id"]), str(triplet["probe"])
        if any(not checks.get((triplet_id, detector), False) for detector in detectors):
            continue
        by_detector = {}
        for detector in detectors:
            rows = [scores.get((triplet_id, detector, f"{probe}:{level}")) for level in ("original", "low", "high")]
            if any(row is None or row["failure"] is not None or row["truncated"] or row["canonical_ai_score"] is None for row in rows):
                break
            values = [
                bisect_right(cdfs[detector], float(row["canonical_ai_score"])) / len(cdfs[detector])
                for row in rows
            ]
            by_detector[detector] = slope(
                (0.0, float(triplet["low_intensity"]), float(triplet["high_intensity"])),
                values,
            )
        if len(by_detector) == len(detectors):
            eligible.append(ProbeRow(
                triplet_id, str(triplet["corpus"]), probe,
                str(triplet["group_id"]), by_detector,
            ))
    selected, seen, counts = [], set(), defaultdict(int)
    for row in eligible:
        group_key = (row.corpus, row.probe, row.group_id)
        cell = (row.corpus, row.probe)
        if group_key in seen or counts[cell] >= 50:
            continue
        seen.add(group_key)
        counts[cell] += 1
        selected.append(row)
    return tuple(selected)


def _quantities(
    fold: sqlite3.Connection,
    allowed: frozenset[str],
    detectors: Sequence[str],
    thresholds: Mapping[str, float],
    probes: Sequence[ProbeRow],
) -> tuple[dict[str, float], dict[str, tuple[float, ...]], dict]:
    corpus_fprs: dict[tuple[str, str], float] = {}
    for corpus in sorted(allowed):
        for detector in detectors:
            row = fold.execute(
                """SELECT COUNT(*) total,COUNT(s.record_id) scored,
                          SUM(s.canonical_ai_score>=?) flagged,
                          SUM(s.failure IS NOT NULL OR s.truncated=1 OR s.canonical_ai_score IS NULL) invalid
                   FROM records r LEFT JOIN scores s
                     ON s.record_id=r.record_id AND s.variant_id='original' AND s.detector_config=?
                   WHERE r.partition_name='source_summary' AND r.corpus=?""",
                (thresholds[detector], detector, corpus),
            ).fetchone()
            if not row["total"] or row["scored"] != row["total"] or row["invalid"]:
                raise RuntimeError(f"Incomplete source-summary scores for {corpus}/{detector}")
            corpus_fprs[(corpus, detector)] = float(row["flagged"]) / int(row["total"])
    source_fpr = {
        detector: mean(corpus_fprs[(corpus, detector)] for corpus in allowed)
        for detector in detectors
    }
    profiles: dict[str, tuple[float, ...]] = {}
    detail: dict[str, dict] = {}
    for detector in detectors:
        values, detector_detail = [], {}
        for probe in PROBES:
            corpus_values = {
                corpus: mean(row.slopes[detector] for row in probes if row.corpus == corpus and row.probe == probe)
                for corpus in allowed
                if any(row.corpus == corpus and row.probe == probe for row in probes)
            }
            if not corpus_values:
                raise RuntimeError(f"No panel-valid anchors for {detector}/{probe}/{sorted(allowed)}")
            values.append(mean(corpus_values.values()))
            detector_detail[probe] = corpus_values
        profiles[detector] = tuple(values)
        detail[detector] = detector_detail
    return source_fpr, profiles, {
        "derived_from": sorted(allowed),
        "source_fpr": source_fpr,
        "profile_corpus_slopes": detail,
    }


def _apply_quantities(
    rows: Sequence[Observation],
    source_fpr: Mapping[str, float],
    profiles: Mapping[str, tuple[float, ...]],
) -> tuple[Observation, ...]:
    return tuple(replace(row, source_fpr=source_fpr[row.detector], profile=profiles[row.detector]) for row in rows)


def build_zero_forecasts(
    root: Path,
    target_corpus: str,
    threshold_artifact: Path,
    output_dir: Path | None = None,
    detectors: Sequence[str] = tuple(SPECS),
) -> dict[str, Path]:
    detectors = tuple(dict.fromkeys(detectors))
    groups = {SPECS[detector].dependency_group for detector in detectors}
    if len(detectors) < 4 or groups != {"openai_roberta", "radar", "mage", "qwen25_shared"}:
        raise ValueError("Forecasting requires all four admitted backend groups")
    code_commit = _clean_commit()
    paths = fold_paths(root, target_corpus)
    source_corpora = tuple(corpus for corpus in STUDY_CORPORA if corpus != target_corpus)
    assert_prelock_database(paths.database, target_corpus, source_corpora)
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    if state.get("phase") != "prelock" or paths.zero_lock.exists() or paths.privileged_lock.exists():
        raise RuntimeError("Zero forecasts require an untouched prelock fold")
    before_hash = _sha256(paths.database)
    fold, master = _connect(paths.database), _connect(root / "state" / "fprint.sqlite3")
    try:
        target_scores = fold.execute(
            """SELECT COUNT(*) FROM scores s JOIN records r USING(record_id)
               WHERE r.corpus=? AND r.partition_name IN ('signature','test')""",
            (target_corpus,),
        ).fetchone()[0]
        if target_scores:
            raise RuntimeError("Target scores exist; zero-score forecasting is forbidden")
        artifact, cdfs, thresholds = _threshold_data(master, threshold_artifact, detectors)
        feature_map, metadata, data_ids = _feature_rows(fold, source_corpora, target_corpus)
        signature_records = [
            TextRecord(record_id, target_corpus, "", metadata[record_id][1])
            for record_id in data_ids[f"signature:{target_corpus}"]
        ]
        draws = repeated_signature_samples(signature_records)
        draw_ids = {f"draw:{draw}:n:{size}": list(ids) for (draw, size), ids in draws.items()}
        probes = _probe_rows(fold, detectors, cdfs)
        selected_c, forecasts, quantity_artifact = {}, [], {
            "schema_version": 1,
            "probe_order": list(PROBES),
            "selected_triplet_ids": [row.triplet_id for row in probes],
            "operating_points": {},
        }
        target_ids = data_ids[f"signature:{target_corpus}"]
        for operating_fpr in OPERATING_FPRS:
            raw = _source_outcomes(
                fold, source_corpora, detectors, feature_map,
                thresholds[operating_fpr],
            )
            cache: dict[frozenset[str], tuple[dict, dict, dict]] = {}

            def quantities(allowed: frozenset[str]):
                if allowed not in cache:
                    cache[allowed] = _quantities(
                        fold, allowed, detectors, thresholds[operating_fpr], probes,
                    )
                return cache[allowed]

            def recompute(train, valid, allowed):
                source_fpr, profiles, _ = quantities(allowed)
                return RecomputedFold(
                    _apply_quantities(train, source_fpr, profiles),
                    _apply_quantities(valid, source_fpr, profiles),
                    allowed, allowed,
                )

            outer_allowed = frozenset(source_corpora)
            source_fpr, profiles, _ = quantities(outer_allowed)
            outer = _apply_quantities(raw, source_fpr, profiles)
            target_rows = tuple(
                Observation(
                    target_corpus, detector, SPECS[detector].dependency_group,
                    0.0, source_fpr[detector], profiles[detector], feature_map[record_id],
                )
                for record_id in target_ids for detector in detectors
            )
            index = {
                (record_id, detector): position
                for position, (record_id, detector) in enumerate(
                    (record_id, detector)
                    for record_id in target_ids for detector in detectors
                )
            }
            for model in FORECAST_MODELS:
                C = tune_c_nested(raw, model, recompute)
                selected_c[f"{operating_fpr:.2f}:{model}"] = C
                predictions = fit_forecaster(
                    outer, target_rows, model, C,
                    source_fpr_derived_from=outer_allowed,
                    profile_derived_from=outer_allowed,
                )
                for (draw, size), ids in draws.items():
                    for detector in detectors:
                        forecasts.append({
                            "target_corpus": target_corpus,
                            "detector_config": detector,
                            "operating_fpr": operating_fpr,
                            "signature_size": size,
                            "draw": draw,
                            "model": model,
                            "prediction": mean(predictions[index[(record_id, detector)]] for record_id in ids),
                        })
            quantity_artifact["operating_points"][f"{operating_fpr:.2f}"] = {
                "outer": cache[outer_allowed][2],
                "inner": {
                    ",".join(sorted(allowed)): detail
                    for allowed, (_, _, detail) in cache.items()
                    if allowed != outer_allowed
                },
            }
    finally:
        fold.close()
        master.close()

    payload = {"admitted_detectors": list(detectors), "forecasts": forecasts}
    validate_forecast_payload(payload, corpora=(target_corpus,), detectors=detectors)
    if _sha256(paths.database) != before_hash or json.loads(paths.state.read_text(encoding="utf-8")) != state:
        raise RuntimeError("Fold database or state changed while forecasts were built")
    output_dir = (output_dir or paths.root / "artifacts" / "zero").resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    final_paths = {
        name: output_dir / filename
        for name, filename in (
            ("features", "features.json"), ("profiles", "profiles.json"),
            ("ids", "ids.json"), ("forecasts", "forecasts.json"),
            ("manifest", "manifest.json"),
        )
    }
    panel = {
        detector: {
            "model_revision": SPECS[detector].revision,
            "tokenizer_revision": SPECS[detector].tokenizer_revision,
            "implementation_revision": SPECS[detector].implementation_revision or "none",
        }
        for detector in detectors
    }
    with tempfile.TemporaryDirectory(prefix=".zero-", dir=output_dir.parent) as temporary:
        temporary = Path(temporary)
        temporary_paths = {name: temporary / path.name for name, path in final_paths.items()}
        _write_exclusive(temporary_paths["features"], {
            "schema_version": 1,
            "feature_names": list(FEATURE_NAMES),
            "values": {record_id: list(values) for record_id, values in sorted(feature_map.items())},
        })
        _write_exclusive(temporary_paths["profiles"], quantity_artifact)
        _write_exclusive(temporary_paths["ids"], {
            "schema_version": 1, "data_ids": data_ids, "draw_ids": draw_ids,
        })
        _write_exclusive(temporary_paths["forecasts"], payload)
        manifest = build_forecast_manifest(
            paths=paths, data_ids=data_ids, draw_ids=draw_ids,
            panel_revisions=panel, thresholds=artifact, selected_c=selected_c,
            feature_artifacts={str(final_paths["features"]): _sha256(temporary_paths["features"])},
            profile_artifacts={str(final_paths["profiles"]): _sha256(temporary_paths["profiles"])},
            id_artifacts={str(final_paths["ids"]): _sha256(temporary_paths["ids"])},
            forecast_artifacts={str(final_paths["forecasts"]): _sha256(temporary_paths["forecasts"])},
            code_commit=code_commit,
        )
        _write_exclusive(temporary_paths["manifest"], manifest)
        if _sha256(paths.database) != before_hash or json.loads(paths.state.read_text(encoding="utf-8")) != state:
            raise RuntimeError("Fold database or state changed before forecast publication")
        temporary.replace(output_dir)
    lock_zero_score_forecasts(paths, manifest, forecasts, source_corpora)
    return {**final_paths, "zero_lock": paths.zero_lock}
