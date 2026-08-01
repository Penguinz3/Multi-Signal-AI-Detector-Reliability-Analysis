from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

C_GRID = (.01, .1, 1.0, 10.0)
SUPPORTED_MODELS = frozenset({
    "source_fpr",
    "text_only",
    "profile_only",
    "detector_id_text",
    "detector_id_x_text",
    "source_fpr_id_text",
    "profile_text",
    "main",
})


@dataclass(frozen=True)
class Observation:
    corpus: str
    detector: str
    dependency_group: str
    outcome: float
    source_fpr: float
    profile: tuple[float, ...]
    features: tuple[float, ...]


@dataclass(frozen=True)
class RecomputedFold:
    train: tuple[Observation, ...]
    valid: tuple[Observation, ...]
    source_fpr_derived_from: frozenset[str]
    profile_derived_from: frozenset[str]


FoldRecomputer = Callable[
    [Sequence[Observation], Sequence[Observation], frozenset[str]],
    RecomputedFold,
]


def _detector_group_map(observations: Sequence[Observation]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in observations:
        existing = mapping.setdefault(row.detector, row.dependency_group)
        if existing != row.dependency_group:
            raise ValueError(
                f"Detector {row.detector!r} appears in both {existing!r} "
                f"and {row.dependency_group!r}"
            )
    return mapping


def _membership_counts(observations: Sequence[Observation]) -> Counter[tuple[str, str, str]]:
    return Counter(
        (row.corpus, row.detector, row.dependency_group)
        for row in observations
    )


def _validate_binary_outcomes(observations: Sequence[Observation], label: str) -> np.ndarray:
    outcomes = np.asarray([row.outcome for row in observations], dtype=float)
    if not len(outcomes) or not np.all(np.isfinite(outcomes)):
        raise ValueError(f"{label} outcomes must be non-empty and finite")
    values = set(outcomes.tolist())
    if not values <= {0.0, 1.0}:
        raise ValueError(f"{label} outcomes must be binary 0/1; found {sorted(values)}")
    if values != {0.0, 1.0}:
        raise ValueError(f"{label} requires both outcome classes; found {sorted(values)}")
    return outcomes


def design_row(
    observation: Observation,
    model: str,
    detector_levels: Sequence[str],
) -> list[float]:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model}")
    if len(set(detector_levels)) != len(detector_levels):
        raise ValueError("detector_levels must be unique")
    if observation.detector not in detector_levels:
        raise ValueError(f"Unknown detector level: {observation.detector}")

    detector = [
        float(observation.detector == level)
        for level in detector_levels[:-1]
    ]
    profile, features = list(observation.profile), list(observation.features)
    if model == "source_fpr":
        return [observation.source_fpr]
    if model == "text_only":
        return features
    if model == "profile_only":
        return profile
    if model == "detector_id_text":
        return detector + features
    if model == "detector_id_x_text":
        return detector + features + [
            indicator * feature
            for indicator in detector
            for feature in features
        ]
    if model == "source_fpr_id_text":
        return [observation.source_fpr] + detector + features
    if model == "profile_text":
        return profile + features
    return (
        [observation.source_fpr]
        + profile
        + features
        + [value * feature for value in profile for feature in features]
    )


def _pipeline(C: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=2_000,
                random_state=0,
            ),
        ),
    ])


def _probabilities(estimator, matrix: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(matrix), dtype=float)
    classes = list(estimator.named_steps["logistic"].classes_)
    if 1.0 not in classes:
        raise RuntimeError("Fitted logistic model has no positive class")
    return probabilities[:, classes.index(1.0)]


def grouped_brier(
    observations: Sequence[Observation],
    predictions: Sequence[float],
) -> float:
    """Macro-average corpus -> dependency group -> detector -> passage."""
    if len(observations) != len(predictions) or not observations:
        raise ValueError("Observations and predictions must be non-empty and equally sized")
    predicted = np.asarray(predictions, dtype=float)
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Predictions must be finite")
    _detector_group_map(observations)

    by_detector: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for observation, prediction in zip(observations, predicted):
        by_detector[
            (
                observation.corpus,
                observation.dependency_group,
                observation.detector,
            )
        ].append((float(prediction) - observation.outcome) ** 2)

    by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (corpus, group, _), losses in by_detector.items():
        by_group[(corpus, group)].append(float(np.mean(losses)))

    by_corpus: dict[str, list[float]] = defaultdict(list)
    for (corpus, _), detector_losses in by_group.items():
        by_corpus[corpus].append(float(np.mean(detector_losses)))

    return float(np.mean([
        np.mean(group_losses)
        for group_losses in by_corpus.values()
    ]))


def _validate_recomputed_fold(
    fold: RecomputedFold,
    raw_train: Sequence[Observation],
    raw_valid: Sequence[Observation],
    allowed: frozenset[str],
    held_out: str,
    detector_groups: dict[str, str],
) -> None:
    if fold.source_fpr_derived_from != allowed:
        raise RuntimeError(
            f"Source-FPR provenance must equal the inner training corpora for {held_out}"
        )
    if fold.profile_derived_from != allowed:
        raise RuntimeError(
            f"Profile provenance must equal the inner training corpora for {held_out}"
        )
    if {row.corpus for row in fold.train} != set(allowed):
        raise RuntimeError(f"Recomputed training membership is invalid for {held_out}")
    if {row.corpus for row in fold.valid} != {held_out}:
        raise RuntimeError(f"Recomputed validation membership is invalid for {held_out}")
    if _membership_counts(fold.train) != _membership_counts(raw_train):
        raise RuntimeError(f"Recomputation changed training membership for {held_out}")
    if _membership_counts(fold.valid) != _membership_counts(raw_valid):
        raise RuntimeError(f"Recomputation changed validation membership for {held_out}")

    expected_detectors = set(detector_groups)
    if {row.detector for row in fold.train} != expected_detectors:
        raise RuntimeError(f"Inner training detector coverage is incomplete for {held_out}")
    if {row.detector for row in fold.valid} != expected_detectors:
        raise RuntimeError(f"Inner validation detector coverage is incomplete for {held_out}")
    if _detector_group_map((*fold.train, *fold.valid)) != detector_groups:
        raise RuntimeError(f"Detector dependency-group mapping changed for {held_out}")


def tune_c_nested(
    observations: Sequence[Observation],
    model: str,
    recompute: FoldRecomputer,
) -> float:
    """Inner leave-one-corpus-out with strict source-quantity provenance."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model}")
    corpora = sorted({row.corpus for row in observations})
    if len(corpora) < 3:
        raise ValueError("Nested tuning requires at least three source corpora")
    detector_groups = _detector_group_map(observations)
    detector_levels = sorted(detector_groups)
    expected_detectors = set(detector_levels)
    for corpus in corpora:
        if {row.detector for row in observations if row.corpus == corpus} != expected_detectors:
            raise ValueError(f"Corpus {corpus} lacks one or more admitted detectors")

    losses: dict[float, list[float]] = defaultdict(list)
    for held_out in corpora:
        raw_train = tuple(row for row in observations if row.corpus != held_out)
        raw_valid = tuple(row for row in observations if row.corpus == held_out)
        allowed = frozenset(row.corpus for row in raw_train)
        fold = recompute(raw_train, raw_valid, allowed)
        if not isinstance(fold, RecomputedFold):
            raise TypeError("recompute must return RecomputedFold with provenance")
        _validate_recomputed_fold(
            fold,
            raw_train,
            raw_valid,
            allowed,
            held_out,
            detector_groups,
        )

        x_train = np.asarray([
            design_row(row, model, detector_levels)
            for row in fold.train
        ])
        y_train = _validate_binary_outcomes(fold.train, "Inner training")
        x_valid = np.asarray([
            design_row(row, model, detector_levels)
            for row in fold.valid
        ])
        for C in C_GRID:
            estimator = _pipeline(C).fit(x_train, y_train)
            losses[C].append(
                grouped_brier(fold.valid, _probabilities(estimator, x_valid))
            )
    return min(C_GRID, key=lambda value: (float(np.mean(losses[value])), value))


def fit_forecaster(
    observations: Sequence[Observation],
    targets: Sequence[Observation],
    model: str,
    C: float,
    *,
    source_fpr_derived_from: frozenset[str],
    profile_derived_from: frozenset[str],
) -> list[float]:
    """Fit the outer forecaster after all quantities are rebuilt from source corpora."""
    source_corpora = frozenset(row.corpus for row in observations)
    target_corpora = {row.corpus for row in targets}
    if not observations or not targets:
        raise ValueError("Forecaster requires source observations and target rows")
    if target_corpora & set(source_corpora):
        raise RuntimeError("Target corpus appears in outer training observations")
    if source_fpr_derived_from != source_corpora:
        raise RuntimeError("Outer source-FPR provenance must equal source corpora")
    if profile_derived_from != source_corpora:
        raise RuntimeError("Outer profile provenance must equal source corpora")

    detector_groups = _detector_group_map(observations)
    detector_levels = sorted(detector_groups)
    if _detector_group_map(targets) != detector_groups:
        raise RuntimeError("Target detector coverage or dependency grouping differs from source")

    x = np.asarray([
        design_row(row, model, detector_levels)
        for row in observations
    ])
    y = _validate_binary_outcomes(observations, "Outer training")
    target_x = np.asarray([
        design_row(row, model, detector_levels)
        for row in targets
    ])
    return list(_probabilities(_pipeline(C).fit(x, y), target_x))
