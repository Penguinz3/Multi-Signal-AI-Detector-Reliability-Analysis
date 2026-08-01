from __future__ import annotations

import unittest
from dataclasses import replace
import importlib.util

from fprint.evaluation import (
    REQUIRED_DEPENDENCY_GROUPS,
    SIMPLE_BASELINES,
    ForecastEvaluationRow,
    evaluate_success_gate,
)
from fprint.modeling import (
    C_GRID,
    Observation,
    RecomputedFold,
    design_row,
    fit_forecaster,
    grouped_brier,
    tune_c_nested,
)


def observation(
    corpus: str,
    detector: str,
    group: str,
    outcome: float,
    feature: float,
) -> Observation:
    return Observation(
        corpus=corpus,
        detector=detector,
        dependency_group=group,
        outcome=outcome,
        source_fpr=.05,
        profile=(.1, -.2),
        features=(feature, 1.0 - feature),
    )


class ModelingTests(unittest.TestCase):
    def test_detector_interactions_and_profile_only(self):
        row = observation("c", "d1", "g1", 0, .25)
        levels = ("d1", "d2", "d3")
        self.assertEqual(design_row(row, "profile_only", levels), [.1, -.2])
        self.assertEqual(
            design_row(row, "detector_id_x_text", levels),
            [1.0, 0.0, .25, .75, .25, .75, 0.0, 0.0],
        )
        with self.assertRaises(ValueError):
            design_row(row, "detector_id_x_text", ("d2", "d3"))

    def test_grouped_brier_balances_detectors_before_backends(self):
        rows = [observation("c", "d1", "g1", 0, 0)]
        predictions = [0.0]
        rows.extend(observation("c", "d2", "g1", 0, 0) for _ in range(9))
        predictions.extend([1.0] * 9)
        rows.append(observation("c", "d3", "g2", 0, 0))
        predictions.append(0.0)
        self.assertAlmostEqual(grouped_brier(rows, predictions), .25)
        with self.assertRaises(ValueError):
            grouped_brier(rows, predictions[:-1])

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed in smoke-test runtime")
    def test_nested_tuning_requires_exact_provenance_and_membership(self):
        rows = []
        for corpus_index, corpus in enumerate(("a", "b", "c")):
            for detector, group in (("d1", "g1"), ("d2", "g2")):
                rows.append(observation(corpus, detector, group, 0, .1 + corpus_index / 10))
                rows.append(observation(corpus, detector, group, 1, .8 - corpus_index / 10))

        def recompute(raw_train, raw_valid, allowed):
            detector_fpr = {
                detector: sum(row.outcome for row in raw_train if row.detector == detector)
                / sum(row.detector == detector for row in raw_train)
                for detector in ("d1", "d2")
            }
            rebuilt = lambda row: replace(
                row,
                source_fpr=detector_fpr[row.detector],
                profile=(detector_fpr[row.detector], -detector_fpr[row.detector]),
            )
            return RecomputedFold(
                tuple(map(rebuilt, raw_train)),
                tuple(map(rebuilt, raw_valid)),
                allowed,
                allowed,
            )

        chosen = tune_c_nested(rows, "detector_id_x_text", recompute)
        self.assertIn(chosen, C_GRID)

        def leaked(raw_train, raw_valid, allowed):
            held_out = frozenset(row.corpus for row in raw_valid)
            return RecomputedFold(
                tuple(raw_train),
                tuple(raw_valid),
                allowed | held_out,
                allowed,
            )

        with self.assertRaises(RuntimeError):
            tune_c_nested(rows, "main", leaked)

        def changed_membership(raw_train, raw_valid, allowed):
            return RecomputedFold(
                tuple(raw_train[:-1]),
                tuple(raw_valid),
                allowed,
                allowed,
            )

        with self.assertRaises(RuntimeError):
            tune_c_nested(rows, "main", changed_membership)

        source = tuple(row for row in rows if row.corpus != "c")
        targets = tuple(replace(row, corpus="target") for row in rows if row.corpus == "c")
        probabilities = fit_forecaster(
            source,
            targets,
            "main",
            chosen,
            source_fpr_derived_from=frozenset({"a", "b"}),
            profile_derived_from=frozenset({"a", "b"}),
        )
        self.assertEqual(len(probabilities), len(targets))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities))


class SuccessGateTests(unittest.TestCase):
    detectors = {
        "openai": "openai_roberta",
        "radar": "radar",
        "mage": "mage",
        "logrank": "qwen25_shared",
        "lastde": "qwen25_shared",
    }
    models = (
        "main",
        "detector_id_x_text",
        *SIMPLE_BASELINES,
    )

    def rows(self):
        result = []
        observed = .05
        predictions = {
            "main": .05,
            "detector_id_x_text": .07,
            "source_fpr": .09,
            "text_only": .09,
            "profile_only": .08,
            "profile_text": .08,
            "detector_id_text": .075,
        }
        for corpus_index in range(8):
            corpus = f"c{corpus_index}"
            for size in (100, 250):
                for draw in range(20):
                    for model in self.models:
                        for detector, group in self.detectors.items():
                            result.append(ForecastEvaluationRow(
                                corpus=corpus,
                                detector=detector,
                                dependency_group=group,
                                signature_size=size,
                                draw=draw,
                                model=model,
                                prediction=predictions[model],
                                observed_fpr=observed,
                            ))
        return result

    def test_complete_success_gate(self):
        report = evaluate_success_gate(self.rows())
        self.assertTrue(report.passed, report.failures)
        self.assertEqual(report.wins_over_detector_id, {100: 8, 250: 8})
        self.assertEqual(len(report.corpus_improvements), 8)
        self.assertEqual(report.sign_flip_p, 1 / 256)
        self.assertEqual(
            set(REQUIRED_DEPENDENCY_GROUPS),
            {"openai_roberta", "radar", "mage", "qwen25_shared"},
        )

    def test_gate_rejects_wrong_corpus_or_backend_count(self):
        seven_corpora = [row for row in self.rows() if row.corpus != "c7"]
        with self.assertRaises(ValueError):
            evaluate_success_gate(seven_corpora)

        three_groups = [
            row
            for row in self.rows()
            if row.dependency_group != "radar"
        ]
        with self.assertRaises(ValueError):
            evaluate_success_gate(three_groups)


if __name__ == "__main__":
    unittest.main()
