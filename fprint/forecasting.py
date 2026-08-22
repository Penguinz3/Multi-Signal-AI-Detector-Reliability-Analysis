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
    repeated_signature_samples, slope, threshold, validate_forecast_payload,
)
from .detectors import SPECS
from .features import FEATURE_NAMES, target_features
from .modeling import Observation, RecomputedFold, fit_forecaster, tune_c_nested
from .uncertainty import (
    cluster_bootstrap_indices, forecast_identity_hashes, split_half_pairs,
    summarize_replicates, validate_replicate_completeness,
)
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
    intensities: tuple[float, float, float]
    scores: Mapping[str, tuple[float, float, float]]


@dataclass(frozen=True)
class SourceExample:
    record_id: str
    corpus: str
    group_id: str
    features: tuple[float, ...] | None
    scores: Mapping[str, float]


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
) -> tuple[dict, dict[str, list[float]], dict[float, dict[str, float]], tuple[Mapping[str, float], ...]]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if set(artifact.get("detectors", ())) != set(detectors):
        raise RuntimeError("Frozen threshold panel differs from the admitted panel")
    cdfs: dict[str, list[float]] = {}
    scores_by_detector: dict[str, dict[str, float]] = {}
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
        scores_by_detector[detector] = dict(score_pairs)
        for fpr in OPERATING_FPRS:
            thresholds[fpr][detector] = float(
                artifact["detectors"][detector]["thresholds"][f"{fpr:.2f}"]
            )
    matrix = tuple(
        {detector: scores_by_detector[detector][record_id] for detector in detectors}
        for record_id, _ in retained
    )
    return artifact, cdfs, thresholds, matrix


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
        by_detector, raw_scores = {}, {}
        intensities = (0.0, float(triplet["low_intensity"]), float(triplet["high_intensity"]))
        for detector in detectors:
            rows = [scores.get((triplet_id, detector, f"{probe}:{level}")) for level in ("original", "low", "high")]
            if any(row is None or row["failure"] is not None or row["truncated"] or row["canonical_ai_score"] is None for row in rows):
                break
            raw_scores[detector] = tuple(float(row["canonical_ai_score"]) for row in rows)
            values = [bisect_right(cdfs[detector], value) / len(cdfs[detector]) for value in raw_scores[detector]]
            by_detector[detector] = slope(intensities, values)
        if len(by_detector) == len(detectors):
            eligible.append(ProbeRow(
                triplet_id, str(triplet["corpus"]), probe,
                str(triplet["group_id"]), by_detector, intensities, raw_scores,
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


def _source_examples(
    fold: sqlite3.Connection,
    partition: str,
    source_corpora: Sequence[str],
    detectors: Sequence[str],
    features: Mapping[str, tuple[float, ...]],
) -> tuple[SourceExample, ...]:
    corpus_placeholders = ",".join("?" for _ in source_corpora)
    detector_placeholders = ",".join("?" for _ in detectors)
    records = fold.execute(
        f"""SELECT record_id,corpus,group_id FROM records
             WHERE partition_name=? AND corpus IN ({corpus_placeholders})
             ORDER BY corpus,record_id""",
        (partition, *source_corpora),
    ).fetchall()
    score_map: dict[str, dict[str, float]] = defaultdict(dict)
    for row in fold.execute(
        f"""SELECT s.record_id,s.detector_config,s.canonical_ai_score
             FROM scores s JOIN records r USING(record_id)
             WHERE r.partition_name=? AND r.corpus IN ({corpus_placeholders})
               AND s.variant_id='original' AND s.detector_config IN ({detector_placeholders})
               AND s.failure IS NULL AND s.truncated=0 AND s.canonical_ai_score IS NOT NULL""",
        (partition, *source_corpora, *detectors),
    ):
        score_map[str(row["record_id"])][str(row["detector_config"])] = float(row["canonical_ai_score"])
    if len(records) * len(detectors) != sum(len(values) for values in score_map.values()):
        raise RuntimeError(f"Incomplete {partition} score panel")
    return tuple(
        SourceExample(
            str(row["record_id"]), str(row["corpus"]), str(row["group_id"]),
            features.get(str(row["record_id"])), score_map[str(row["record_id"])],
        )
        for row in records
    )


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(": ".join(map(str, parts)).encode()).digest()[:8], "big")


def _bootstrap_schedules(
    examples: Sequence[SourceExample],
    source_corpora: Sequence[str],
    label: str,
    replicates: int,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    result = {}
    for corpus in source_corpora:
        local = [row for row in examples if row.corpus == corpus]
        if not local:
            raise RuntimeError(f"No {label} examples for {corpus}")
        result[corpus] = cluster_bootstrap_indices(
            [row.group_id for row in local], seed=_seed(20260729, label, corpus),
            replicates=replicates,
        )
    return result


def _profile_from_sample(
    probes_by_cell: Mapping[tuple[str, str], Sequence[ProbeRow]],
    schedules: Mapping[tuple[str, str], Sequence[Sequence[int]]],
    replicate: int,
    source_corpora: Sequence[str],
    detectors: Sequence[str],
    cdfs: Mapping[str, Sequence[float]],
) -> dict[str, tuple[float, ...]]:
    profiles = {}
    for detector in detectors:
        values = []
        for probe in PROBES:
            corpus_slopes = []
            for corpus in source_corpora:
                rows = probes_by_cell.get((corpus, probe), ())
                if not rows:
                    continue
                slopes = []
                for index in schedules[(corpus, probe)][replicate]:
                    row = rows[index]
                    normalized = [
                        bisect_right(cdfs[detector], value) / len(cdfs[detector])
                        for value in row.scores[detector]
                    ]
                    slopes.append(slope(row.intensities, normalized))
                corpus_slopes.append(mean(slopes))
            if not corpus_slopes:
                raise RuntimeError(f"No bootstrap probes for {detector}/{probe}")
            values.append(mean(corpus_slopes))
        profiles[detector] = tuple(values)
    return profiles


def _split_half_diagnostics(
    probes_by_cell: Mapping[tuple[str, str], Sequence[ProbeRow]],
    source_corpora: Sequence[str],
    detectors: Sequence[str],
) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for detector in detectors:
        profile_halves: list[tuple[list[float], list[float]]] = [([], []) for _ in range(100)]
        for probe in PROBES:
            paired: list[tuple[list[int], Sequence[ProbeRow]]] = []
            excluded = []
            for corpus in source_corpora:
                rows = probes_by_cell.get((corpus, probe), ())
                group_count = len({row.group_id for row in rows})
                if group_count >= 2:
                    pairs = split_half_pairs(
                        [row.group_id for row in rows],
                        seed=_seed(20260729, "split-half", corpus, probe), pairs=100,
                    )
                    paired.append((pairs, rows))
                else:
                    excluded.append({
                        "corpus": corpus,
                        "panel_valid_triplets": len(rows),
                        "anchor_groups": group_count,
                        "reason": "fewer_than_two_panel_valid_anchor_groups",
                    })
            if not paired:
                result[detector][probe] = {
                    "status": "unavailable",
                    "reason": "fewer_than_two_panel_valid_anchor_groups",
                    "excluded_corpora": excluded,
                }
                continue
            left_values, right_values = [], []
            for pair_index in range(100):
                left_values.append(mean(mean(rows[index].slopes[detector] for index in pairs[pair_index][0]) for pairs, rows in paired))
                right_values.append(mean(mean(rows[index].slopes[detector] for index in pairs[pair_index][1]) for pairs, rows in paired))
                profile_halves[pair_index][0].append(left_values[-1])
                profile_halves[pair_index][1].append(right_values[-1])
            result[detector][probe] = {
                "status": "available",
                "pairs": 100,
                "included_corpora": len(paired),
                "excluded_corpora": excluded,
                "sign_agreement": mean(float(left * right >= 0) for left, right in zip(left_values, right_values)),
                "mean_absolute_difference": mean(abs(left - right) for left, right in zip(left_values, right_values)),
            }
        cosines = []
        for left, right in profile_halves:
            denominator = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
            if left and denominator:
                cosines.append(sum(a * b for a, b in zip(left, right)) / denominator)
        result[detector]["overall_profile_stability"] = {
            "status": "available" if cosines else "unavailable",
            "statistic": "mean_split_half_cosine",
            "available_probes": len(profile_halves[0][0]),
            "valid_pairs": len(cosines),
            "mean_cosine": mean(cosines) if cosines else None,
        }
    return dict(result)


def _joint_bootstrap(
    fold: sqlite3.Connection,
    forecasts: list[dict],
    selected_c: Mapping[str, float],
    draws: Mapping[tuple[int, int], Sequence[str]],
    target_corpus: str,
    target_ids: Sequence[str],
    feature_map: Mapping[str, tuple[float, ...]],
    source_corpora: Sequence[str],
    detectors: Sequence[str],
    source_model: Sequence[SourceExample],
    source_summary: Sequence[SourceExample],
    probes: Sequence[ProbeRow],
    raid_scores: Sequence[Mapping[str, float]],
    replicates: int = 100,
) -> dict:
    if replicates != 100:
        raise ValueError("The frozen pre-lock protocol requires exactly 100 bootstrap replicates")
    model_by_corpus = {corpus: [row for row in source_model if row.corpus == corpus] for corpus in source_corpora}
    summary_by_corpus = {corpus: [row for row in source_summary if row.corpus == corpus] for corpus in source_corpora}
    model_schedules = _bootstrap_schedules(source_model, source_corpora, "source-model", replicates)
    summary_schedules = _bootstrap_schedules(source_summary, source_corpora, "source-summary", replicates)
    raid_schedule = cluster_bootstrap_indices(
        tuple(range(len(raid_scores))), seed=_seed(20260729, "raid-threshold"),
        replicates=replicates,
    )
    probes_by_cell = {
        (corpus, probe): tuple(row for row in probes if row.corpus == corpus and row.probe == probe)
        for corpus in source_corpora for probe in PROBES
        if any(row.corpus == corpus and row.probe == probe for row in probes)
    }
    probe_schedules = {
        cell: cluster_bootstrap_indices(
            [row.group_id for row in rows], seed=_seed(20260729, "profile", *cell),
            replicates=replicates,
        )
        for cell, rows in probes_by_cell.items()
    }
    main_entries = {}
    for entry in forecasts:
        ids = draws[(int(entry["draw"]), int(entry["signature_size"]))]
        forecast_id, signature_hash = forecast_identity_hashes(entry, ids)
        entry.update({
            "forecast_id": forecast_id,
            "signature_ids_sha256": signature_hash,
            "fit_ref": f"fit:{float(entry['operating_fpr']):.2f}:{entry['model']}",
        })
        if entry["model"] == "main":
            entry.update({
                "uncertainty_status": "joint_cluster_bootstrap_v1",
                "uncertainty_ref": f"conditional:{forecast_id}",
            })
            main_entries[(
                float(entry["operating_fpr"]), str(entry["detector_config"]),
                int(entry["signature_size"]), int(entry["draw"]),
            )] = entry
        else:
            entry.update({
                "uncertainty_status": "point_only_preregistered_secondary",
                "uncertainty_ref": None,
            })

    replicate_predictions = {entry["forecast_id"]: [] for entry in main_entries.values()}
    component_thresholds = defaultdict(list)
    component_source_fpr = defaultdict(list)
    component_profiles = defaultdict(list)
    target_index = {
        (record_id, detector): position
        for position, (record_id, detector) in enumerate(
            (record_id, detector) for record_id in target_ids for detector in detectors
        )
    }
    completed_replicates = []
    for replicate in range(replicates):
        sampled_raid = [raid_scores[index] for index in raid_schedule[replicate]]
        cdfs = {
            detector: sorted(row[detector] for row in sampled_raid)
            for detector in detectors
        }
        thresholds = {
            operating_fpr: {
                detector: threshold(cdfs[detector], operating_fpr)
                for detector in detectors
            }
            for operating_fpr in OPERATING_FPRS
        }
        profiles = _profile_from_sample(
            probes_by_cell, probe_schedules, replicate,
            source_corpora, detectors, cdfs,
        )
        for detector in detectors:
            for probe_index, probe in enumerate(PROBES):
                component_profiles[(detector, probe)].append(profiles[detector][probe_index])
        for operating_fpr in OPERATING_FPRS:
            source_fpr = {}
            for detector in detectors:
                corpus_values = []
                for corpus in source_corpora:
                    examples = summary_by_corpus[corpus]
                    indices = summary_schedules[corpus][replicate]
                    corpus_values.append(mean(
                        float(examples[index].scores[detector] >= thresholds[operating_fpr][detector])
                        for index in indices
                    ))
                source_fpr[detector] = mean(corpus_values)
                component_thresholds[(operating_fpr, detector)].append(thresholds[operating_fpr][detector])
                component_source_fpr[(operating_fpr, detector)].append(source_fpr[detector])
            observations = []
            for corpus in source_corpora:
                examples = model_by_corpus[corpus]
                for index in model_schedules[corpus][replicate]:
                    example = examples[index]
                    if example.features is None:
                        raise RuntimeError("Source-model bootstrap lacks text features")
                    for detector in detectors:
                        observations.append(Observation(
                            corpus, detector, SPECS[detector].dependency_group,
                            float(example.scores[detector] >= thresholds[operating_fpr][detector]),
                            source_fpr[detector], profiles[detector], example.features,
                        ))
            target_rows = tuple(
                Observation(
                    target_corpus, detector, SPECS[detector].dependency_group,
                    0.0, source_fpr[detector], profiles[detector], feature_map[record_id],
                )
                for record_id in target_ids for detector in detectors
            )
            predictions = fit_forecaster(
                observations, target_rows, "main", selected_c[f"{operating_fpr:.2f}:main"],
                source_fpr_derived_from=frozenset(source_corpora),
                profile_derived_from=frozenset(source_corpora),
            )
            for (draw, size), ids in draws.items():
                for detector in detectors:
                    entry = main_entries[(operating_fpr, detector, size, draw)]
                    replicate_predictions[entry["forecast_id"]].append(mean(
                        predictions[target_index[(record_id, detector)]] for record_id in ids
                    ))
        completed_replicates.append(replicate)

    validate_replicate_completeness(completed_replicates)
    if any(len(values) != 100 for values in replicate_predictions.values()):
        raise RuntimeError("A conditional forecast is missing one or more bootstrap replicates")

    conditional = {
        forecast_id: {
            **summarize_replicates(values),
            "raw_predictions": values,
        }
        for forecast_id, values in replicate_predictions.items()
    }
    marginal = {}
    for operating_fpr in OPERATING_FPRS:
        for detector in detectors:
            for size in (50, 100, 250):
                point_values = [
                    main_entries[(operating_fpr, detector, size, draw)]["prediction"]
                    for draw in range(20)
                ]
                joint_values = [
                    value
                    for draw in range(20)
                    for value in replicate_predictions[
                        main_entries[(operating_fpr, detector, size, draw)]["forecast_id"]
                    ]
                ]
                marginal[f"{operating_fpr:.2f}:{detector}:{size}"] = {
                    "point_mean_across_20_draws": mean(point_values),
                    "between_draw_sd": summarize_replicates(point_values)["sd"],
                    "joint": summarize_replicates(joint_values),
                    "signature_draws": 20,
                    "bootstrap_replicates": replicates,
                    "valid_replicates": replicates,
                }
    return {
        "schema_version": 1,
        "method": "joint_cluster_bootstrap_v1",
        "master_seed": 20260729,
        "replicates": replicates,
        "interval_level": .90,
        "selected_c_treatment": "frozen_after_original_nested_cv",
        "resampling_units": {
            "threshold_reference": "retained_RAID_record",
            "source_model": "corpus_specific_group_id",
            "source_summary": "corpus_specific_group_id",
            "response_profile": "complete_triplet_group_id_within_corpus_probe",
            "target_signature": "20_preregistered_group_unique_nested_draws",
        },
        "failure_rule": "all_100_replicates_required",
        "conditional": conditional,
        "marginal": marginal,
        "component_diagnostics": {
            "thresholds": {f"{fpr:.2f}:{detector}": summarize_replicates(values) for (fpr, detector), values in component_thresholds.items()},
            "source_fpr": {f"{fpr:.2f}:{detector}": summarize_replicates(values) for (fpr, detector), values in component_source_fpr.items()},
            "profiles": {f"{detector}:{probe}": summarize_replicates(values) for (detector, probe), values in component_profiles.items()},
            "split_half": _split_half_diagnostics(probes_by_cell, source_corpora, detectors),
        },
    }


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
        artifact, cdfs, thresholds, raid_scores = _threshold_data(master, threshold_artifact, detectors)
        feature_map, metadata, data_ids = _feature_rows(fold, source_corpora, target_corpus)
        signature_records = [
            TextRecord(record_id, target_corpus, "", metadata[record_id][1])
            for record_id in data_ids[f"signature:{target_corpus}"]
        ]
        draws = repeated_signature_samples(signature_records)
        draw_ids = {f"draw:{draw}:n:{size}": list(ids) for (draw, size), ids in draws.items()}
        probes = _probe_rows(fold, detectors, cdfs)
        source_model_examples = _source_examples(
            fold, "source_model", source_corpora, detectors, feature_map,
        )
        source_summary_examples = _source_examples(
            fold, "source_summary", source_corpora, detectors, feature_map,
        )
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
        uncertainty_artifact = _joint_bootstrap(
            fold, forecasts, selected_c, draws, target_corpus, target_ids,
            feature_map, source_corpora, detectors, source_model_examples,
            source_summary_examples, probes, raid_scores,
        )
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
            ("uncertainty", "uncertainty.json"), ("manifest", "manifest.json"),
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
        _write_exclusive(temporary_paths["uncertainty"], uncertainty_artifact)
        manifest = build_forecast_manifest(
            paths=paths, data_ids=data_ids, draw_ids=draw_ids,
            panel_revisions=panel, thresholds=artifact, selected_c=selected_c,
            feature_artifacts={str(final_paths["features"]): _sha256(temporary_paths["features"])},
            profile_artifacts={str(final_paths["profiles"]): _sha256(temporary_paths["profiles"])},
            id_artifacts={str(final_paths["ids"]): _sha256(temporary_paths["ids"])},
            forecast_artifacts={str(final_paths["forecasts"]): _sha256(temporary_paths["forecasts"])},
            uncertainty_artifacts={str(final_paths["uncertainty"]): _sha256(temporary_paths["uncertainty"])},
            code_commit=code_commit,
        )
        _write_exclusive(temporary_paths["manifest"], manifest)
        if _sha256(paths.database) != before_hash or json.loads(paths.state.read_text(encoding="utf-8")) != state:
            raise RuntimeError("Fold database or state changed before forecast publication")
        temporary.replace(output_dir)
    lock_zero_score_forecasts(paths, manifest, forecasts, source_corpora)
    return {**final_paths, "zero_lock": paths.zero_lock}
