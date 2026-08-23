from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

from .core import exact_sign_flip, slope


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else None


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left)) * math.sqrt(
        sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _correlation(_ranks(left), _ranks(right))


def _fixed_effect_spearman(strata: Sequence[tuple[Sequence[float], Sequence[float]]]) -> float | None:
    left, right = [], []
    for left_values, right_values in strata:
        left_ranks, right_ranks = _ranks(left_values), _ranks(right_values)
        left_mean, right_mean = mean(left_ranks), mean(right_ranks)
        left.extend(value - left_mean for value in left_ranks)
        right.extend(value - right_mean for value in right_ranks)
    return _correlation(left, right)


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _identify(
    vectors: Mapping[tuple[str, str], Sequence[float]],
    detectors: Sequence[str],
    corpora: Sequence[str],
) -> tuple[float, float, dict[str, dict[str, str]]]:
    cosine_correct = distance_correct = 0
    predictions = {}
    for corpus in corpora:
        predictions[corpus] = {}
        for detector in detectors:
            centroids = {
                candidate: tuple(
                    mean(vectors[(candidate, source)][index] for source in corpora if source != corpus)
                    for index in range(len(vectors[(detector, corpus)]))
                )
                for candidate in detectors
            }
            vector = vectors[(detector, corpus)]
            cosine_scores = {
                candidate: _cosine(vector, centroid)
                for candidate, centroid in centroids.items()
            }
            predicted = max(
                cosine_scores,
                key=lambda candidate: (
                    cosine_scores[candidate]
                    if cosine_scores[candidate] is not None else -math.inf
                ),
            )
            predictions[corpus][detector] = predicted
            cosine_correct += predicted == detector
            distance_correct += min(
                centroids, key=lambda candidate: _distance(vector, centroids[candidate])
            ) == detector
    total = len(detectors) * len(corpora)
    return cosine_correct / total, distance_correct / total, predictions


def _identification_inference(
    vectors: Mapping[tuple[str, str], Sequence[float]],
    detectors: Sequence[str],
    corpora: Sequence[str],
    observed_accuracy: float,
    predictions: Mapping[str, Mapping[str, str]],
    *,
    replicates: int = 10000,
) -> dict[str, object]:
    rng = random.Random(20260729)
    null = []
    for _ in range(replicates):
        permuted = {}
        for corpus in corpora:
            labels = list(detectors)
            rng.shuffle(labels)
            for label, source in zip(detectors, labels):
                permuted[(label, corpus)] = vectors[(source, corpus)]
        null.append(_identify(permuted, detectors, corpora)[0])
    corpus_accuracies = [
        mean(predicted == detector for detector, predicted in predictions[corpus].items())
        for corpus in corpora
    ]
    bootstraps = [
        mean(rng.choice(corpus_accuracies) for _ in corpora)
        for _ in range(replicates)
    ]
    return {
        "label_permutation_replicates": replicates,
        "label_permutation_p_greater_equal": (
            1 + sum(value >= observed_accuracy for value in null)
        ) / (replicates + 1),
        "corpus_cluster_bootstrap_90_interval": [
            _percentile(bootstraps, .05), _percentile(bootstraps, .95)
        ],
        "per_corpus_accuracy": dict(zip(corpora, corpus_accuracies)),
    }


def _distance_risk_inference(
    vectors: Mapping[tuple[str, str], Sequence[float]],
    risk: Mapping[str, Mapping[str, float]],
    detectors: Sequence[str],
    corpora: Sequence[str],
    *,
    replicates: int = 10000,
) -> dict[str, object]:
    pairs = tuple(combinations(corpora, 2))
    profile = {
        detector: [_distance(vectors[(detector, left)], vectors[(detector, right)]) for left, right in pairs]
        for detector in detectors
    }
    risk_distance = {
        detector: [abs(float(risk[left][detector]) - float(risk[right][detector])) for left, right in pairs]
        for detector in detectors
    }
    observed = _fixed_effect_spearman([
        (profile[detector], risk_distance[detector]) for detector in detectors
    ])
    rng = random.Random(20260729)
    null = []
    for _ in range(replicates):
        permuted = {}
        for detector in detectors:
            values = [float(risk[corpus][detector]) for corpus in corpora]
            rng.shuffle(values)
            permuted[detector] = dict(zip(corpora, values))
        null.append(_fixed_effect_spearman([
            (
                profile[detector],
                [abs(permuted[detector][left] - permuted[detector][right]) for left, right in pairs],
            )
            for detector in detectors
        ]))
    return {
        "detector_fixed_effect_spearman": observed,
        "corpus_label_permutation_replicates": replicates,
        "corpus_label_permutation_p_two_sided": (
            1 + sum(abs(value) >= abs(observed) for value in null if value is not None)
        ) / (replicates + 1),
    }


def _metamorphic_audit(
    root: Path,
    detectors: Sequence[str],
    probe_order: Sequence[str],
) -> dict[str, object]:
    artifact = root / "folds" / "bawe" / "artifacts" / "zero" / "profiles.json"
    selected = set(json.loads(artifact.read_text(encoding="utf-8"))["selected_triplet_ids"])
    thresholds = json.loads(
        (root / "state" / "frozen_thresholds.json").read_text(encoding="utf-8")
    )["detectors"]
    database = root / "folds" / "bawe" / "fprint.sqlite3"
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT p.triplet_id,p.corpus,p.probe,r.group_id,
                  p.low_intensity,p.high_intensity,s.detector_config,
                  s.variant_id,s.canonical_ai_score,s.failure,s.truncated
             FROM probe_triplets p JOIN records r USING(record_id)
             JOIN scores s USING(record_id)
            WHERE s.variant_id<>'original'"""
    ).fetchall()
    connection.close()
    grouped = defaultdict(dict)
    metadata = {}
    for row in rows:
        triplet_id = str(row["triplet_id"])
        detector = str(row["detector_config"])
        if triplet_id not in selected or detector not in detectors:
            continue
        level = str(row["variant_id"]).rsplit(":", 1)[-1]
        if row["failure"] is not None or row["truncated"] or row["canonical_ai_score"] is None:
            raise RuntimeError(f"Invalid selected probe score: {triplet_id}/{detector}/{level}")
        grouped[(triplet_id, detector)][level] = float(row["canonical_ai_score"])
        metadata[triplet_id] = (
            str(row["corpus"]), str(row["probe"]), str(row["group_id"]),
            float(row["low_intensity"]), float(row["high_intensity"]),
        )
    if set(metadata) != selected or any(set(values) != {"original", "low", "high"} for values in grouped.values()):
        raise RuntimeError("Selected probe panel is incomplete in the final database")

    paired = []
    raw_slopes = defaultdict(list)
    for (triplet_id, detector), values in grouped.items():
        corpus, probe, group_id, low_intensity, high_intensity = metadata[triplet_id]
        raw_slopes[(detector, corpus, probe)].append(slope(
            (0.0, low_intensity, high_intensity),
            (values["original"], values["low"], values["high"]),
        ))
        for operating_point in ("0.05", "0.01"):
            threshold = float(thresholds[detector]["thresholds"][operating_point])
            original_flag = values["original"] >= threshold
            for level in ("low", "high"):
                variant_flag = values[level] >= threshold
                paired.append({
                    "corpus": corpus, "group_id": group_id, "probe": probe,
                    "detector": detector, "operating_point": operating_point,
                    "intensity": level, "score_shift": values[level] - values["original"],
                    "original_flag": original_flag, "variant_flag": variant_flag,
                })

    rng = random.Random(20260729)
    audit_cells = []
    for operating_point in ("0.05", "0.01"):
        for detector in detectors:
            for probe in probe_order:
                for intensity in ("low", "high"):
                    selected_rows = [
                        row for row in paired
                        if row["operating_point"] == operating_point
                        and row["detector"] == detector and row["probe"] == probe
                        and row["intensity"] == intensity
                    ]
                    by_corpus = defaultdict(list)
                    for row in selected_rows:
                        by_corpus[row["corpus"]].append(row)
                    corpora = tuple(sorted(by_corpus))
                    bootstrap = defaultdict(list)
                    for _ in range(1000):
                        sample = []
                        for corpus in (rng.choice(corpora) for _ in corpora):
                            source = by_corpus[corpus]
                            sample.extend(rng.choice(source) for _ in source)
                        metrics = {
                            "any_flip_rate": mean(row["original_flag"] != row["variant_flag"] for row in sample),
                            "human_to_ai_flip_rate": mean(not row["original_flag"] and row["variant_flag"] for row in sample),
                            "ai_to_human_flip_rate": mean(row["original_flag"] and not row["variant_flag"] for row in sample),
                            "mean_score_shift": mean(row["score_shift"] for row in sample),
                        }
                        for name, value in metrics.items():
                            bootstrap[name].append(value)
                    point = {
                        "any_flip_rate": mean(row["original_flag"] != row["variant_flag"] for row in selected_rows),
                        "human_to_ai_flip_rate": mean(not row["original_flag"] and row["variant_flag"] for row in selected_rows),
                        "ai_to_human_flip_rate": mean(row["original_flag"] and not row["variant_flag"] for row in selected_rows),
                        "mean_score_shift": mean(row["score_shift"] for row in selected_rows),
                    }
                    corpus_net = [
                        mean(
                            (not row["original_flag"] and row["variant_flag"])
                            - (row["original_flag"] and not row["variant_flag"])
                            for row in by_corpus[corpus]
                        )
                        for corpus in corpora
                    ]
                    directional_p = min(
                        1.0,
                        2 * min(
                            exact_sign_flip(corpus_net),
                            exact_sign_flip([-value for value in corpus_net]),
                        ),
                    )
                    audit_cells.append({
                        "operating_point": operating_point, "detector": detector,
                        "probe": probe, "intensity": intensity,
                        "n": len(selected_rows), "corpora": len(corpora),
                        "directional_corpus_sign_flip_p": directional_p,
                        **{
                            name: {
                                "estimate": value,
                                "bootstrap_90_interval": [
                                    _percentile(bootstrap[name], .05),
                                    _percentile(bootstrap[name], .95),
                                ],
                            }
                            for name, value in point.items()
                        },
                    })
    for operating_point in ("0.05", "0.01"):
        family = [row for row in audit_cells if row["operating_point"] == operating_point]
        adjusted = 0.0
        for rank, row in enumerate(
            sorted(family, key=lambda item: item["directional_corpus_sign_flip_p"])
        ):
            adjusted = max(
                adjusted,
                (len(family) - rank) * row["directional_corpus_sign_flip_p"],
            )
            row["holm_adjusted_directional_p"] = min(1.0, adjusted)
            row["holm_reject_0.05"] = adjusted <= .05
    raw_means = {key: mean(values) for key, values in raw_slopes.items()}
    raw_corpora = tuple(sorted({value[0] for value in metadata.values()}))
    raw_complete = tuple(
        probe for probe in probe_order
        if all((detector, corpus, probe) in raw_means for detector in detectors for corpus in raw_corpora)
    )
    raw_vectors = {
        (detector, corpus): tuple(raw_means[(detector, corpus, probe)] for probe in raw_complete)
        for detector in detectors for corpus in raw_corpora
    }
    raw_cosine, raw_distance, _ = _identify(raw_vectors, detectors, raw_corpora)
    return {
        "selected_triplets": len(selected),
        "hierarchical_bootstrap_replicates": 1000,
        "cells": audit_cells,
        "raw_score_sensitivity": {
            "panel_complete_probes": list(raw_complete),
            "leave_one_corpus_out_cosine_identification": raw_cosine,
            "leave_one_corpus_out_euclidean_identification": raw_distance,
            "cross_corpus_cosine_by_detector": {
                detector: _summary([
                    similarity
                    for left, right in combinations(raw_corpora, 2)
                    if (similarity := _cosine(
                        raw_vectors[(detector, left)], raw_vectors[(detector, right)]
                    )) is not None
                ])
                for detector in detectors
            },
            "slopes": {
                f"{detector}:{corpus}:{probe}": value
                for (detector, corpus, probe), value in raw_means.items()
            },
        },
    }


def _load_profile_cells(root: Path, corpora: Sequence[str]) -> tuple[
    dict[tuple[str, str, str], float], tuple[str, ...], tuple[str, ...]
]:
    cells: dict[tuple[str, str, str], float] = {}
    detectors, probe_order = set(), None
    for target in corpora:
        path = root / "folds" / target / "artifacts" / "zero" / "profiles.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        current_order = tuple(payload["probe_order"])
        if probe_order is not None and current_order != probe_order:
            raise RuntimeError("Probe order changes across frozen folds")
        probe_order = current_order
        outer = payload["operating_points"]["0.05"]["outer"]["profile_corpus_slopes"]
        for detector, probes in outer.items():
            detectors.add(detector)
            for probe, slopes in probes.items():
                for corpus, value in slopes.items():
                    if corpus not in corpora:
                        continue
                    key = (detector, corpus, probe)
                    value = float(value)
                    if key in cells and not math.isclose(cells[key], value, abs_tol=1e-12):
                        raise RuntimeError(f"Frozen profile value changes across folds: {key}")
                    cells[key] = value
    if probe_order is None or len(detectors) < 2:
        raise RuntimeError("Incomplete frozen profile panel")
    return cells, tuple(sorted(detectors)), probe_order


def _split_half(root: Path, detectors: Sequence[str]) -> dict[str, object]:
    path = root / "folds" / "bawe" / "artifacts" / "zero" / "uncertainty.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = payload["component_diagnostics"]["split_half"]
    result = {}
    for detector in detectors:
        rows = diagnostics[detector]
        overall = rows["overall_profile_stability"]
        result[detector] = {
            "overall_mean_split_half_cosine": overall["mean_cosine"],
            "available_probes": overall["available_probes"],
            "valid_pairs": overall["valid_pairs"],
            "probes": {
                probe: {
                    key: detail.get(key)
                    for key in (
                        "status", "included_corpora", "pairs",
                        "sign_agreement", "mean_absolute_difference",
                    )
                }
                for probe, detail in rows.items()
                if probe != "overall_profile_stability"
            },
        }
    return result


def analyze_fingerprint_geometry(root: Path, evaluation_path: Path) -> tuple[dict, list[dict]]:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    corpora = tuple(evaluation["primary_corpora"])
    cells, detectors, probe_order = _load_profile_cells(root, corpora)
    universal_probes = tuple(
        probe for probe in probe_order
        if all((detector, corpus, probe) in cells for detector in detectors for corpus in corpora)
    )
    if not universal_probes:
        raise RuntimeError("No probe is available for every detector-corpus cell")

    vectors = {
        (detector, corpus): tuple(cells[(detector, corpus, probe)] for probe in universal_probes)
        for detector in detectors for corpus in corpora
    }
    within = {
        detector: [
            similarity
            for left, right in combinations(corpora, 2)
            if (similarity := _cosine(vectors[(detector, left)], vectors[(detector, right)])) is not None
        ]
        for detector in detectors
    }
    between = [
        similarity
        for corpus in corpora
        for left, right in combinations(detectors, 2)
        if (similarity := _cosine(vectors[(left, corpus)], vectors[(right, corpus)])) is not None
    ]

    cosine_accuracy, distance_accuracy, predictions = _identify(vectors, detectors, corpora)
    leave_one_probe_out = {}
    for omitted in universal_probes:
        retained = tuple(probe for probe in universal_probes if probe != omitted)
        sensitivity_vectors = {
            key: tuple(value[index] for index, probe in enumerate(universal_probes) if probe in retained)
            for key, value in vectors.items()
        }
        cosine, euclidean, _ = _identify(sensitivity_vectors, detectors, corpora)
        leave_one_probe_out[omitted] = {
            "retained_probes": list(retained),
            "cosine_accuracy": cosine,
            "euclidean_accuracy": euclidean,
        }

    dependency_groups = {
        detector: (
            "qwen25_shared" if "qwen2_5" in detector
            else "openai_roberta" if detector.startswith("openai_")
            else "radar" if detector.startswith("radar_")
            else "mage"
        )
        for detector in detectors
    }
    groups = tuple(sorted(set(dependency_groups.values())))
    grouped_vectors = {
        (group, corpus): tuple(
            mean(
                vectors[(detector, corpus)][index]
                for detector in detectors if dependency_groups[detector] == group
            )
            for index in range(len(universal_probes))
        )
        for group in groups for corpus in corpora
    }
    group_cosine, group_euclidean, _ = _identify(grouped_vectors, groups, corpora)

    observed = evaluation["observed_fpr"]
    cell_rows = []
    for detector in detectors:
        for corpus in corpora:
            vector = vectors[(detector, corpus)]
            cell_rows.append({
                "detector": detector,
                "corpus": corpus,
                "profile_norm": math.sqrt(sum(value * value for value in vector)),
                "observed_fpr_0.05": float(observed[corpus]["0.05"][detector]),
                "observed_fpr_0.01": float(observed[corpus]["0.01"][detector]),
                **{f"slope__{probe}": value for probe, value in zip(universal_probes, vector)},
            })

    distance_risk = {}
    for operating_point in ("0.05", "0.01"):
        profile_distances, risk_distances = [], []
        for detector in detectors:
            for left, right in combinations(corpora, 2):
                profile_distances.append(_distance(vectors[(detector, left)], vectors[(detector, right)]))
                risk_distances.append(abs(
                    float(observed[left][operating_point][detector])
                    - float(observed[right][operating_point][detector])
                ))
        per_detector = {
            detector: _spearman(
                [
                    _distance(vectors[(detector, left)], vectors[(detector, right)])
                    for left, right in combinations(corpora, 2)
                ],
                [
                    abs(
                        float(observed[left][operating_point][detector])
                        - float(observed[right][operating_point][detector])
                    )
                    for left, right in combinations(corpora, 2)
                ],
            )
            for detector in detectors
        }
        distance_risk[operating_point] = {
            "pairs": len(profile_distances),
            "naive_pooled_spearman_profile_distance_vs_fpr_distance": _spearman(
                profile_distances, risk_distances
            ),
            "by_detector": per_detector,
            "mean_by_detector_spearman": mean(value for value in per_detector.values() if value is not None),
            **_distance_risk_inference(
                vectors,
                {corpus: observed[corpus][operating_point] for corpus in corpora},
                detectors,
                corpora,
            ),
        }

    gate = evaluation["success_gates"]["0.05"]
    local_to_global = {
        "passed_preregistered_gate": gate["passed"],
        "sign_flip_p": gate["sign_flip_p"],
        "wins_over_detector_id": gate["wins_over_detector_id"],
        "mae": {
            size: {
                "main": gate["overall_mae"][f"{size}:main"],
                "detector_id_x_text": gate["overall_mae"][f"{size}:detector_id_x_text"],
                "main_minus_detector_id_x_text": (
                    gate["overall_mae"][f"{size}:main"]
                    - gate["overall_mae"][f"{size}:detector_id_x_text"]
                ),
            }
            for size in (100, 250)
        },
    }

    report = {
        "schema_version": 1,
        "research_construct": "local_response_geometry_to_global_false_positive_tail_risk",
        "corpora": list(corpora),
        "detectors": list(detectors),
        "probe_order": list(probe_order),
        "panel_complete_probes": list(universal_probes),
        "probe_coverage": {
            probe: sum((detector, corpus, probe) in cells for detector in detectors for corpus in corpora)
            for probe in probe_order
        },
        "split_half_reliability": _split_half(root, detectors),
        "cross_corpus_cosine": {
            "by_detector": {detector: _summary(values) for detector, values in within.items()},
            "pooled_within_detector": _summary([value for values in within.values() for value in values]),
            "pooled_between_detector_same_corpus": _summary(between),
        },
        "leave_one_corpus_out_detector_identification": {
            "cells": len(detectors) * len(corpora),
            "chance_accuracy": 1 / len(detectors),
            "cosine_accuracy": cosine_accuracy,
            "euclidean_accuracy": distance_accuracy,
            "predictions": predictions,
            **_identification_inference(
                vectors, detectors, corpora, cosine_accuracy, predictions
            ),
            "leave_one_probe_out": leave_one_probe_out,
            "shared_backend_collapsed": {
                "groups": list(groups),
                "chance_accuracy": 1 / len(groups),
                "cosine_accuracy": group_cosine,
                "euclidean_accuracy": group_euclidean,
            },
        },
        "profile_distance_vs_fpr_distance": distance_risk,
        "local_to_global_forecast_test": local_to_global,
        "metamorphic_threshold_audit": _metamorphic_audit(
            root, detectors, probe_order
        ),
    }
    return report, cell_rows


def write_fingerprint_geometry(root: Path, evaluation_path: Path, output_dir: Path) -> dict:
    report, rows = analyze_fingerprint_geometry(root, evaluation_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fingerprint_geometry.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "fingerprint_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    crossing_rows = []
    for row in report["metamorphic_threshold_audit"]["cells"]:
        flat = {
            key: value for key, value in row.items()
            if not isinstance(value, dict)
        }
        for metric in (
            "any_flip_rate", "human_to_ai_flip_rate",
            "ai_to_human_flip_rate", "mean_score_shift",
        ):
            detail = row[metric]
            flat[f"{metric}__estimate"] = detail["estimate"]
            flat[f"{metric}__lower90"], flat[f"{metric}__upper90"] = detail[
                "bootstrap_90_interval"
            ]
        crossing_rows.append(flat)
    with (output_dir / "metamorphic_threshold_crossings.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(crossing_rows[0]))
        writer.writeheader()
        writer.writerows(crossing_rows)
    identification_rows = [
        {"corpus": corpus, "detector": detector, "predicted_detector": predicted}
        for corpus, predictions in report[
            "leave_one_corpus_out_detector_identification"
        ]["predictions"].items()
        for detector, predicted in predictions.items()
    ]
    with (output_dir / "fingerprint_identification.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(identification_rows[0]))
        writer.writeheader()
        writer.writerows(identification_rows)
    return report
