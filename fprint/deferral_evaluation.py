"""Evaluation helpers for the forecast-locked human-FP deferral pilot.

This module deliberately has no dependency on the completed forecasting or
fault-audit artifacts.  A row is one original detector-positive passage with
its cached RADAR metamorphic scores and the human/AI outcome used only for
evaluation.  All fitted preprocessing is rebuilt inside each corpus-held-out
fold.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FINGERPRINT_FEATURE_NAMES = (
    "radar_margin",
    "radar_wrap_80_minus_original",
    "radar_sentence_blocks_2_minus_original",
    "radar_sentence_per_paragraph_minus_original",
)

REQUIRED_COLUMNS = (
    "record_id", "pair_id", "corpus", "generator_family", "label", "text",
    "radar_threshold", "radar_original", "radar_wrap_80",
    "radar_sentence_blocks_2", "radar_sentence_per_paragraph",
    "mage_original", "logrank_original",
)

_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$", re.UNICODE)
_SURFACE_NAMES = (
    "surface_log1p_chars", "surface_log1p_words", "surface_log1p_sentence_units",
    "surface_mean_word_length", "surface_whitespace_fraction",
    "surface_punctuation_fraction", "surface_digit_fraction",
    "surface_uppercase_fraction",
)


def _finite(value: Any, name: str, *, missing_ok: bool = True) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        if missing_ok:
            return float("nan")
        raise ValueError(f"{name} is required")
    result = float(value)
    if not math.isfinite(result):
        if missing_ok:
            return float("nan")
        raise ValueError(f"{name} must be finite")
    return result


def _normalise_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("At least one evaluation row is required")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"row {index} must be a mapping")
        row = dict(raw)
        missing = [name for name in REQUIRED_COLUMNS if name not in row]
        if missing:
            raise ValueError(f"row {index} is missing columns: {', '.join(missing)}")
        for key in ("record_id", "pair_id", "corpus", "generator_family"):
            row[key] = str(row[key])
            if not row[key]:
                raise ValueError(f"row {index} has an empty {key}")
        if row["record_id"] in seen:
            raise ValueError(f"duplicate record_id: {row['record_id']}")
        seen.add(row["record_id"])
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("label must be 1 (human false accusation) or 0 (AI true positive)")
        row["label"] = label
        row["text"] = str(row["text"])
        for key in (
            "radar_threshold", "radar_original", "radar_wrap_80",
            "radar_sentence_blocks_2", "radar_sentence_per_paragraph",
            "mage_original", "logrank_original", "radar_original_repeat",
        ):
            if key in row:
                row[key] = _finite(row[key], key)
        result.append(row)
    return result


def fingerprint_features(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the fixed signed RADAR fingerprint features in preregistered order."""
    rows = _normalise_rows(rows)
    matrix = np.asarray([
        [
            row["radar_original"] - row["radar_threshold"],
            row["radar_wrap_80"] - row["radar_original"],
            row["radar_sentence_blocks_2"] - row["radar_original"],
            row["radar_sentence_per_paragraph"] - row["radar_original"],
        ]
        for row in rows
    ], dtype=float)
    return matrix, FINGERPRINT_FEATURE_NAMES


def _surface_features(text: str) -> list[float]:
    words = _WORD_RE.findall(text)
    count = len(words)
    chars = len(text)
    sentences = [part for part in _SENTENCE_RE.findall(text) if part.strip()]
    punctuation = sum(character in ".,;:!?()[]{}\"'" for character in text)
    digits = sum(character.isdigit() for character in text)
    whitespace = sum(character.isspace() for character in text)
    alphabetic = sum(character.isalpha() for character in text)
    uppercase = sum(character.isupper() for character in text)
    return [
        float(np.log1p(chars)), float(np.log1p(count)), float(np.log1p(len(sentences))),
        sum(map(len, words)) / count if count else 0.0,
        whitespace / chars if chars else 0.0,
        punctuation / chars if chars else 0.0,
        digits / chars if chars else 0.0,
        uppercase / alphabetic if alphabetic else 0.0,
    ]


def _impute_scale(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-impute and standardize using train rows only."""
    train = np.asarray(train, dtype=float)
    values = np.asarray(values, dtype=float)
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(train), train, medians)
    values = np.where(np.isfinite(values), values, medians)
    means = np.mean(train, axis=0)
    scales = np.std(train, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    return (train - means) / scales, (values - means) / scales, medians


def _detector_comparator_numeric(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    original = np.asarray([
        [
            row["radar_original"] - row["radar_threshold"],
            row["mage_original"], row["logrank_original"],
        ]
        for row in rows
    ], dtype=float)
    radar_views = np.asarray([
        [
            row["radar_original"], row["radar_wrap_80"],
            row["radar_sentence_blocks_2"], row["radar_sentence_per_paragraph"],
        ]
        for row in rows
    ], dtype=float)
    return np.column_stack((original, np.nanmean(radar_views, axis=1)))


def generic_tta_features(
    train_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, tuple[str, ...]]:
    """Build the fixed generic+TTA comparator without signed probe deltas.

    The character vectorizer is fitted on ``train_rows`` only.  The three
    original detector scores are standardized on that same fold and their
    absolute pairwise differences are appended.  Individual RADAR deltas are
    intentionally absent; only the mean of the four RADAR views is retained.
    """
    train_rows = _normalise_rows(train_rows)
    rows = _normalise_rows(rows)
    train_numeric = _detector_comparator_numeric(train_rows)
    values_numeric = _detector_comparator_numeric(rows)
    train_scores, values_scores, _ = _impute_scale(train_numeric[:, :3], values_numeric[:, :3])
    train_pairwise = np.column_stack((
        np.abs(train_scores[:, 0] - train_scores[:, 1]),
        np.abs(train_scores[:, 0] - train_scores[:, 2]),
        np.abs(train_scores[:, 1] - train_scores[:, 2]),
    ))
    values_pairwise = np.column_stack((
        np.abs(values_scores[:, 0] - values_scores[:, 1]),
        np.abs(values_scores[:, 0] - values_scores[:, 2]),
        np.abs(values_scores[:, 1] - values_scores[:, 2]),
    ))
    tta_train = train_numeric[:, 3:4]
    tta_values = values_numeric[:, 3:4]
    tta_median = np.nanmedian(tta_train, axis=0)
    tta_median = np.where(np.isfinite(tta_median), tta_median, 0.0)
    tta_train = np.where(np.isfinite(tta_train), tta_train, tta_median)
    tta_values = np.where(np.isfinite(tta_values), tta_values, tta_median)
    surface_train = np.asarray([_surface_features(row["text"]) for row in train_rows], dtype=float)
    surface_values = np.asarray([_surface_features(row["text"]) for row in rows], dtype=float)
    surface_train, surface_values, _ = _impute_scale(surface_train, surface_values)

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=2, max_features=50_000,
        sublinear_tf=True, dtype=np.float64,
    )
    try:
        char_train = vectorizer.fit_transform([row["text"] for row in train_rows])
        char_values = vectorizer.transform([row["text"] for row in rows])
        char_names = tuple(f"char_tfidf:{gram}" for gram in vectorizer.get_feature_names_out())
    except ValueError:
        # Empty text is valid at this boundary; the fixed numeric part still works.
        char_train = np.zeros((len(train_rows), 0), dtype=float)
        char_values = np.zeros((len(rows), 0), dtype=float)
        char_names = ()
    numeric_names = (
        tuple(_SURFACE_NAMES)
        + ("radar_margin_z", "mage_original_z", "logrank_original_z")
        + ("abs_radar_mage_z", "abs_radar_logrank_z", "abs_mage_logrank_z")
        + ("radar_four_view_mean",)
    )
    train_matrix = np.column_stack((surface_train, train_scores, train_pairwise, tta_train))
    values_matrix = np.column_stack((surface_values, values_scores, values_pairwise, tta_values))
    from scipy import sparse

    return (
        sparse.hstack((sparse.csr_matrix(train_matrix), char_train), format="csr"),
        sparse.hstack((sparse.csr_matrix(values_matrix), char_values), format="csr"),
        numeric_names + char_names,
    )


def _fit_probabilities(train_x: Any, train_y: np.ndarray, values_x: Any, *, seed: int = 20260824) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if len(set(train_y.tolist())) < 2:
        return np.full(len(values_x), float(train_y[0]) if len(train_y) else 0.0)
    estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False) if hasattr(train_x, "tocsr") else StandardScaler()),
        ("logistic", LogisticRegression(
            C=1.0, class_weight="balanced", solver="liblinear", max_iter=2_000,
            random_state=seed,
        )),
    ])
    estimator.fit(train_x, train_y)
    probabilities = estimator.predict_proba(values_x)
    classes = list(estimator.named_steps["logistic"].classes_)
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def loco_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str = "fingerprint",
    seed: int = 20260824,
) -> list[dict[str, Any]]:
    """Return leave-one-corpus-out human-FP probabilities.

    ``model`` is either ``fingerprint`` or ``generic_tta``.  No C tuning is
    performed: the locked model is balanced L2 logistic regression with C=1.
    """
    rows = _normalise_rows(rows)
    if model not in {"fingerprint", "generic_tta"}:
        raise ValueError("model must be 'fingerprint' or 'generic_tta'")
    corpora = sorted({row["corpus"] for row in rows})
    output: list[dict[str, Any]] = []
    for held_out in corpora:
        train_rows = [row for row in rows if row["corpus"] != held_out]
        test_rows = [row for row in rows if row["corpus"] == held_out]
        if not train_rows or not test_rows:
            raise ValueError(f"Cannot fit held-out corpus fold: {held_out}")
        if model == "fingerprint":
            train_x, _ = fingerprint_features(train_rows)
            test_x, _ = fingerprint_features(test_rows)
            train_x, test_x, _ = _impute_scale(train_x, test_x)
        else:
            train_x, test_x, _ = generic_tta_features(train_rows, test_rows)
        probabilities = _fit_probabilities(
            train_x,
            np.asarray([row["label"] for row in train_rows], dtype=int),
            test_x,
            seed=seed,
        )
        for row, probability in zip(test_rows, probabilities):
            output.append({
                **row,
                "prediction": float(probability),
                "model": model,
                "held_out_corpus": held_out,
            })
    return sorted(output, key=lambda row: (row["corpus"], row["record_id"]))


def _balanced_cell_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    for row in rows:
        counts[(str(row["corpus"]), str(row["generator_family"]), int(row["label"]))] += 1
    return np.asarray([
        1.0 / counts[(str(row["corpus"]), str(row["generator_family"]), int(row["label"]))]
        for row in rows
    ], dtype=float)


def _ranking_metric(rows: Sequence[Mapping[str, Any]], scores: Sequence[float], ai_deferral: float = .10) -> dict[str, Any]:
    if len(rows) != len(scores) or not rows:
        raise ValueError("rows and scores must be non-empty and equally sized")
    if not 0.0 <= ai_deferral <= 1.0:
        raise ValueError("ai_deferral must be in [0, 1]")
    weights = _balanced_cell_weights(rows)
    human_total = float(sum(weights[index] for index, row in enumerate(rows) if row["label"] == 1))
    ai_total = float(sum(weights[index] for index, row in enumerate(rows) if row["label"] == 0))
    if not human_total or not ai_total:
        return {
            "human_fp_removal": 0.0, "ai_tp_retention": 1.0, "ai_deferral": 0.0,
            "weighting": "equal_corpus_generator_cells_within_label",
        }
    order = sorted(
        range(len(rows)),
        key=lambda i: (-float(scores[i]), str(rows[i]["pair_id"]), str(rows[i]["record_id"])),
    )
    x = [0.0]
    y = [0.0]
    human_deferred = ai_deferred = 0.0
    for index in order:
        human_deferred += weights[index] * int(rows[index]["label"] == 1)
        ai_deferred += weights[index] * int(rows[index]["label"] == 0)
        x.append(ai_deferred / ai_total)
        y.append(human_deferred / human_total)
    removal = float(np.interp(ai_deferral, np.asarray(x), np.asarray(y)))
    return {
        "human_fp_removal": removal,
        "ai_tp_retention": 1.0 - ai_deferral,
        "ai_deferral": ai_deferral,
        "weighting": "equal_corpus_generator_cells_within_label",
    }


def review_budget_curve(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    budgets: Sequence[float] = (.05, .10, .20),
) -> list[dict[str, float]]:
    if len(rows) != len(scores) or not rows:
        raise ValueError("rows and scores must be non-empty and equally sized")
    human_total = sum(row["label"] == 1 for row in rows)
    ai_total = sum(row["label"] == 0 for row in rows)
    order = sorted(range(len(rows)), key=lambda i: (-float(scores[i]), str(rows[i]["pair_id"]), str(rows[i]["record_id"])))
    output = []
    for budget in budgets:
        if not 0.0 <= float(budget) <= 1.0:
            raise ValueError("review budgets must be in [0, 1]")
        count = min(len(rows), int(math.ceil(float(budget) * len(rows))))
        deferred = [rows[index] for index in order[:count]]
        human_removed = sum(row["label"] == 1 for row in deferred)
        ai_removed = sum(row["label"] == 0 for row in deferred)
        retained = len(rows) - count
        retained_human = human_total - human_removed
        output.append({
            "review_budget": float(budget),
            "defer_fraction": count / len(rows),
            "coverage": retained / len(rows),
            "human_fp_removal": human_removed / human_total if human_total else 0.0,
            "ai_tp_retention": (ai_total - ai_removed) / ai_total if ai_total else 1.0,
            "selective_risk": retained_human / retained if retained else 0.0,
        })
    return output


def risk_coverage_rows(rows: Sequence[Mapping[str, Any]], scores: Sequence[float], *, steps: int = 20) -> list[dict[str, float]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    budgets = np.linspace(0.0, 1.0, steps + 1)
    return review_budget_curve(rows, scores, budgets=budgets)


def paired_group_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    fingerprint_scores: Sequence[float],
    comparator_scores: Sequence[float],
    *,
    replicates: int = 10_000,
    seed: int = 20260824,
    ai_deferral: float = .10,
) -> dict[str, Any]:
    """Stratified corpus/generator group bootstrap of paired metric contrast."""
    rows = _normalise_rows(rows)
    if len(rows) != len(fingerprint_scores) or len(rows) != len(comparator_scores):
        raise ValueError("rows and score arrays must have equal length")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    groups: dict[str, list[int]] = defaultdict(list)
    pair_metadata: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows):
        pair = row["pair_id"]
        metadata = (row["corpus"], row["generator_family"])
        existing = pair_metadata.setdefault(pair, metadata)
        if existing != metadata:
            raise ValueError("pair_id must belong to one corpus/generator stratum")
        groups[pair].append(index)
    strata: dict[tuple[str, str], list[list[int]]] = defaultdict(list)
    for pair, members in sorted(groups.items()):
        corpus, generator = pair_metadata[pair]
        strata[(corpus, generator)].append(members)
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=float)
    fp = np.asarray(fingerprint_scores, dtype=float)
    comp = np.asarray(comparator_scores, dtype=float)
    for replicate in range(replicates):
        indices: list[int] = []
        for key in sorted(strata):
            members = strata[key]
            for selected in rng.integers(0, len(members), size=len(members)):
                indices.extend(members[int(selected)])
        sample = [rows[index] for index in indices]
        differences[replicate] = (
            _ranking_metric(sample, fp[indices], ai_deferral)["human_fp_removal"]
            - _ranking_metric(sample, comp[indices], ai_deferral)["human_fp_removal"]
        )
    return {
        "seed": int(seed),
        "replicates": int(replicates),
        "differences": differences.tolist(),
        "mean": float(np.mean(differences)),
        "lower_80": float(np.quantile(differences, .20, method="linear")),
        "upper_80": float(np.quantile(differences, .80, method="linear")),
        "strata": {f"{corpus}:{generator}": len(groups_) for (corpus, generator), groups_ in sorted(strata.items())},
    }


def sentinel_ratio(rows: Sequence[Mapping[str, Any]]) -> float | None:
    rows = _normalise_rows(rows)
    sentinels = [
        row for row in rows
        if "radar_original_repeat" in row and math.isfinite(row["radar_original_repeat"])
    ]
    if not sentinels:
        return None
    repeat_error = float(np.median([
        abs(row["radar_original"] - row["radar_original_repeat"])
        for row in sentinels
    ]))
    probe_scale = float(np.median([
        np.median([
            abs(row["radar_wrap_80"] - row["radar_original"]),
            abs(row["radar_sentence_blocks_2"] - row["radar_original"]),
            abs(row["radar_sentence_per_paragraph"] - row["radar_original"]),
        ])
        for row in rows
    ]))
    # A zero response scale cannot normalize a repeat error.  Treat it as a
    # failed sentinel rather than allowing a NaN to pass the gate.
    if not math.isfinite(probe_scale) or probe_scale <= 0:
        return float("inf")
    return repeat_error / probe_scale


def _summary_gate(summary: Mapping[str, Any], names: Sequence[str]) -> bool | None:
    for name in names:
        if name in summary:
            value = summary[name]
            if isinstance(value, Mapping):
                value = value.get("passed", value.get("status") == "pass")
            return bool(value)
    return None


def evaluate_gates(
    rows: Sequence[Mapping[str, Any]],
    *,
    pooled_incremental: float,
    bootstrap_lower_80: float | None,
    per_corpus_incremental: Mapping[str, float],
    validation_summary: Mapping[str, Any] | None = None,
    sentinel: float | None = None,
) -> dict[str, Any]:
    rows = _normalise_rows(rows)
    validation_summary = validation_summary or {}
    human_total = sum(row["label"] == 1 for row in rows)
    ai_total = sum(row["label"] == 0 for row in rows)
    corpus_human = {
        corpus: sum(row["label"] == 1 for row in rows if row["corpus"] == corpus)
        for corpus in {row["corpus"] for row in rows}
    }
    manual = _summary_gate(validation_summary, ("manual_gate", "manual_validation_passed", "manual_passed", "manual"))
    automated = _summary_gate(validation_summary, ("automated_gate", "automated_validation_passed", "automated_passed", "automated"))
    mage = _summary_gate(validation_summary, ("mage_gate", "mage_invariance_passed", "mage_passed", "mage"))
    checks = {
        "human_fp_floor": human_total >= 200,
        "ai_tp_floor": ai_total >= 400,
        "corpus_human_fp_floor": sum(value >= 30 for value in corpus_human.values()) >= 3,
        "incremental_floor": pooled_incremental >= .075,
        "bootstrap_lower_positive": bootstrap_lower_80 is not None and bootstrap_lower_80 > 0,
        "positive_corpora_floor": sum(value > 0 for value in per_corpus_incremental.values()) >= 3,
        "manual_validation": manual is True,
        "automated_validation": automated is True,
        "mage_invariance": mage is True,
        "sentinel_ratio": sentinel is not None and sentinel <= .20,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "event_counts": {"human_false_accusations": human_total, "ai_true_positives": ai_total, "corpora_with_30_human_fp": sum(value >= 30 for value in corpus_human.values())},
        "validation_summary_status": {"manual": manual, "automated": automated, "mage": mage},
        "sentinel_ratio": sentinel,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def _metric_bundle(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> dict[str, Any]:
    return {
        "ai_deferral_10": _ranking_metric(rows, scores, .10),
        "review_budget": review_budget_curve(rows, scores),
        "risk_coverage": risk_coverage_rows(rows, scores),
    }


def _delta_correlations(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    features, names = fingerprint_features(rows)
    deltas = features[:, 1:]
    output: dict[str, float | None] = {}
    for left in range(deltas.shape[1]):
        for right in range(left + 1, deltas.shape[1]):
            key = f"{names[left + 1]}__{names[right + 1]}"
            if np.std(deltas[:, left]) == 0 or np.std(deltas[:, right]) == 0:
                output[key] = None
            else:
                output[key] = float(np.corrcoef(deltas[:, left], deltas[:, right])[0, 1])
    return output


def evaluate_pilot(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_summary: Mapping[str, Any] | None = None,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260824,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the locked LOCO pilot and return a JSON-serializable report."""
    rows = _normalise_rows(rows)
    if len({row["corpus"] for row in rows}) != 4:
        raise ValueError("The pilot requires exactly four development corpora")
    fingerprint = loco_predictions(rows, model="fingerprint", seed=seed)
    comparator = loco_predictions(rows, model="generic_tta", seed=seed)
    by_id = {row["record_id"]: row for row in fingerprint}
    comparator_by_id = {row["record_id"]: row for row in comparator}
    if set(by_id) != set(comparator_by_id):
        raise RuntimeError("Fingerprint and comparator predictions have different record IDs")
    ordered = sorted(rows, key=lambda row: (row["corpus"], row["record_id"]))
    fp_scores = np.asarray([by_id[row["record_id"]]["prediction"] for row in ordered], dtype=float)
    comp_scores = np.asarray([comparator_by_id[row["record_id"]]["prediction"] for row in ordered], dtype=float)
    pooled_fp = _ranking_metric(ordered, fp_scores)
    pooled_comp = _ranking_metric(ordered, comp_scores)
    pooled_incremental = float(pooled_fp["human_fp_removal"] - pooled_comp["human_fp_removal"])
    margin_scores = np.asarray([
        -(row["radar_original"] - row["radar_threshold"]) for row in ordered
    ], dtype=float)
    tta_scores = np.asarray([
        -np.mean([
            row["radar_original"], row["radar_wrap_80"],
            row["radar_sentence_blocks_2"], row["radar_sentence_per_paragraph"],
        ])
        for row in ordered
    ], dtype=float)
    per_corpus: dict[str, Any] = {}
    per_corpus_incremental: dict[str, float] = {}
    for corpus in sorted({row["corpus"] for row in ordered}):
        subset = [row for row in ordered if row["corpus"] == corpus]
        indices = [index for index, row in enumerate(ordered) if row["corpus"] == corpus]
        fp_metric = _ranking_metric(subset, fp_scores[indices])
        comp_metric = _ranking_metric(subset, comp_scores[indices])
        per_corpus_incremental[corpus] = float(fp_metric["human_fp_removal"] - comp_metric["human_fp_removal"])
        per_corpus[corpus] = {
            "fingerprint": {"ai_deferral_10": fp_metric},
            "generic_tta": {"ai_deferral_10": comp_metric},
            "incremental_human_fp_removal": per_corpus_incremental[corpus],
        }
    bootstrap = paired_group_bootstrap(
        ordered, fp_scores, comp_scores, replicates=bootstrap_replicates, seed=seed,
    )
    sentinel = sentinel_ratio(ordered)
    gates = evaluate_gates(
        ordered,
        pooled_incremental=pooled_incremental,
        bootstrap_lower_80=bootstrap["lower_80"],
        per_corpus_incremental=per_corpus_incremental,
        validation_summary=validation_summary,
        sentinel=sentinel,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "loco_human_false_accusation_deferral_v1",
        "feature_order": list(FINGERPRINT_FEATURE_NAMES),
        "primary_model": "balanced_l2_logistic_C1_fingerprint",
        "comparator": "generic_tta_delta_free",
        "n_rows": len(ordered),
        "pooled": {
            "fingerprint": _metric_bundle(ordered, fp_scores),
            "generic_tta": _metric_bundle(ordered, comp_scores),
            "margin_only_diagnostic": _metric_bundle(ordered, margin_scores),
            "tta_mean_only_diagnostic": _metric_bundle(ordered, tta_scores),
            "incremental_human_fp_removal_at_90pct_ai_retention": pooled_incremental,
        },
        "per_corpus": per_corpus,
        "pairwise_delta_correlations": _delta_correlations(ordered),
        "bootstrap": bootstrap,
        "sentinel_ratio": sentinel,
        "gates": gates,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    return report


# Small aliases make the module convenient for CLI/adapters without adding a
# second implementation or changing the existing forecasting API.
fingerprint_feature_matrix = fingerprint_features
comparator_feature_matrix = generic_tta_features
run_loco = loco_predictions
run_pilot = evaluate_pilot
