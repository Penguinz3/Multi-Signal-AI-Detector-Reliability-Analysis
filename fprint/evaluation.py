from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .core import exact_sign_flip

PRIMARY_SIGNATURE_SIZES = (100, 250)
REQUIRED_DEPENDENCY_GROUPS = frozenset({
    "openai_roberta",
    "radar",
    "mage",
    "qwen25_shared",
})
SIMPLE_BASELINES = (
    "source_fpr",
    "text_only",
    "profile_only",
    "profile_text",
    "detector_id_text",
)


@dataclass(frozen=True)
class ForecastEvaluationRow:
    corpus: str
    detector: str
    dependency_group: str
    signature_size: int
    draw: int
    model: str
    prediction: float
    observed_fpr: float


@dataclass(frozen=True)
class SuccessGateResult:
    passed: bool
    corpus_losses: Mapping[tuple[str, int, str], float]
    overall_mae: Mapping[tuple[int, str], float]
    wins_over_detector_id: Mapping[int, int]
    corpus_improvements: Mapping[str, float]
    sign_flip_p: float
    simpler_baselines_beaten: Mapping[tuple[int, str], bool]
    failures: tuple[str, ...]


def _finite_probability(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return result


def evaluate_success_gate(
    rows: Sequence[ForecastEvaluationRow],
    *,
    required_corpora: Sequence[str] | None = None,
    required_dependency_groups: frozenset[str] = REQUIRED_DEPENDENCY_GROUPS,
    signature_draws: int = 20,
    main_model: str = "main",
    detector_id_model: str = "detector_id_x_text",
    simple_baselines: Sequence[str] = SIMPLE_BASELINES,
    alpha: float = .05,
) -> SuccessGateResult:
    """Evaluate the preregistered eight-corpus, four-backend success gate."""
    if len(required_dependency_groups) != 4:
        raise ValueError("Success gate requires exactly four dependency groups")
    if signature_draws <= 0:
        raise ValueError("signature_draws must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not rows:
        raise ValueError("No forecast-evaluation rows")

    inferred_corpora = {row.corpus for row in rows}
    if required_corpora is None:
        corpora = tuple(sorted(inferred_corpora))
    else:
        corpora = tuple(required_corpora)
        if len(set(corpora)) != len(corpora):
            raise ValueError("required_corpora must be unique")
        if inferred_corpora != set(corpora):
            raise ValueError("Forecast rows do not match required_corpora")
    if len(corpora) != 8:
        raise ValueError("Success gate requires exactly eight held-out corpora")

    expected_draws = set(range(signature_draws))
    required_models = {
        main_model,
        detector_id_model,
        *simple_baselines,
    }
    detector_groups: dict[str, str] = {}
    observed_fprs: dict[tuple[str, str], float] = {}
    by_cell: dict[
        tuple[str, int, int, str],
        dict[str, ForecastEvaluationRow],
    ] = defaultdict(dict)

    for row in rows:
        if row.corpus not in corpora:
            raise ValueError(f"Unexpected corpus: {row.corpus}")
        if row.draw not in expected_draws:
            raise ValueError(f"Unexpected signature draw: {row.draw}")
        prediction = _finite_probability(row.prediction, "prediction")
        observed = _finite_probability(row.observed_fpr, "observed_fpr")

        existing_group = detector_groups.setdefault(
            row.detector,
            row.dependency_group,
        )
        if existing_group != row.dependency_group:
            raise ValueError(f"Detector {row.detector} changes dependency group")
        fpr_key = (row.corpus, row.detector)
        existing_fpr = observed_fprs.setdefault(fpr_key, observed)
        if not math.isclose(existing_fpr, observed, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Observed FPR changes across forecasts for {fpr_key}")

        if (
            row.signature_size in PRIMARY_SIGNATURE_SIZES
            and row.model in required_models
        ):
            cell = (
                row.corpus,
                row.signature_size,
                row.draw,
                row.model,
            )
            if row.detector in by_cell[cell]:
                raise ValueError(f"Duplicate forecast row for {cell + (row.detector,)}")
            by_cell[cell][row.detector] = ForecastEvaluationRow(
                row.corpus,
                row.detector,
                row.dependency_group,
                row.signature_size,
                row.draw,
                row.model,
                prediction,
                observed,
            )

    actual_groups = frozenset(detector_groups.values())
    if actual_groups != required_dependency_groups:
        raise ValueError(
            "Forecast rows must span exactly the four required dependency groups"
        )
    expected_detectors = set(detector_groups)

    cell_losses: dict[tuple[str, int, int, str], float] = {}
    for corpus in corpora:
        for size in PRIMARY_SIGNATURE_SIZES:
            for draw in range(signature_draws):
                for model in required_models:
                    cell_key = (corpus, size, draw, model)
                    detector_rows = by_cell.get(cell_key, {})
                    if set(detector_rows) != expected_detectors:
                        raise ValueError(f"Incomplete detector panel for {cell_key}")
                    losses_by_group: dict[str, list[float]] = defaultdict(list)
                    for row in detector_rows.values():
                        losses_by_group[row.dependency_group].append(
                            abs(row.prediction - row.observed_fpr)
                        )
                    if frozenset(losses_by_group) != required_dependency_groups:
                        raise ValueError(f"Incomplete dependency groups for {cell_key}")
                    cell_losses[cell_key] = sum(
                        sum(config_losses) / len(config_losses)
                        for config_losses in losses_by_group.values()
                    ) / len(required_dependency_groups)

    corpus_losses: dict[tuple[str, int, str], float] = {}
    for corpus in corpora:
        for size in PRIMARY_SIGNATURE_SIZES:
            for model in required_models:
                corpus_losses[(corpus, size, model)] = sum(
                    cell_losses[(corpus, size, draw, model)]
                    for draw in range(signature_draws)
                ) / signature_draws

    overall_mae: dict[tuple[int, str], float] = {}
    for size in PRIMARY_SIGNATURE_SIZES:
        for model in required_models:
            overall_mae[(size, model)] = sum(
                corpus_losses[(corpus, size, model)]
                for corpus in corpora
            ) / len(corpora)

    wins = {
        size: sum(
            corpus_losses[(corpus, size, main_model)]
            < corpus_losses[(corpus, size, detector_id_model)]
            for corpus in corpora
        )
        for size in PRIMARY_SIGNATURE_SIZES
    }
    improvements = {
        corpus: sum(
            corpus_losses[(corpus, size, detector_id_model)]
            - corpus_losses[(corpus, size, main_model)]
            for size in PRIMARY_SIGNATURE_SIZES
        ) / len(PRIMARY_SIGNATURE_SIZES)
        for corpus in corpora
    }
    if len(improvements) != 8 or not all(
        math.isfinite(value)
        for value in improvements.values()
    ):
        raise RuntimeError("Sign-flip input must be eight finite corpus improvements")
    sign_flip_p = exact_sign_flip([
        improvements[corpus]
        for corpus in corpora
    ])

    baseline_checks = {
        (size, baseline): (
            overall_mae[(size, main_model)]
            < overall_mae[(size, baseline)]
        )
        for size in PRIMARY_SIGNATURE_SIZES
        for baseline in simple_baselines
    }
    failures: list[str] = []
    for size in PRIMARY_SIGNATURE_SIZES:
        if wins[size] < 6:
            failures.append(
                f"main beats detector-ID interaction in only {wins[size]}/8 "
                f"corpora at n={size}"
            )
        if not (
            overall_mae[(size, main_model)]
            < overall_mae[(size, detector_id_model)]
        ):
            failures.append(
                f"main does not lower overall backend-macro MAE at n={size}"
            )
    if sign_flip_p > alpha:
        failures.append(
            f"exact sign-flip p={sign_flip_p:.6g} exceeds alpha={alpha}"
        )
    for (size, baseline), beaten in baseline_checks.items():
        if not beaten:
            failures.append(f"main does not beat {baseline} at n={size}")

    return SuccessGateResult(
        passed=not failures,
        corpus_losses=corpus_losses,
        overall_mae=overall_mae,
        wins_over_detector_id=wins,
        corpus_improvements=improvements,
        sign_flip_p=sign_flip_p,
        simpler_baselines_beaten=baseline_checks,
        failures=tuple(failures),
    )
