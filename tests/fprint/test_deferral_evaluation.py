from __future__ import annotations

import unittest

import numpy as np

from fprint.deferral_evaluation import (
    FINGERPRINT_FEATURE_NAMES,
    evaluate_gates,
    fingerprint_features,
    generic_tta_features,
    loco_predictions,
    paired_group_bootstrap,
    review_budget_curve,
    risk_coverage_rows,
    sentinel_ratio,
    evaluate_pilot,
)


def row(index: int, corpus: str = "a", label: int = 1, *, repeat: float | None = None) -> dict:
    original = .60 if label else .30
    result = {
        "record_id": f"{corpus}-{index}",
        "pair_id": f"{corpus}-pair-{index // 2}",
        "corpus": corpus,
        "generator_family": "g1" if (index // 2) % 2 else "g2",
        "label": label,
        "text": f"A short test passage number {index}. It is deterministic.",
        "radar_threshold": .50,
        "radar_original": original,
        "radar_wrap_80": original + (.20 if label else -.02),
        "radar_sentence_blocks_2": original + (.10 if label else -.01),
        "radar_sentence_per_paragraph": original + (.15 if label else -.01),
        "mage_original": .50 + index / 1000,
        "logrank_original": .40 + index / 1000,
    }
    if repeat is not None:
        result["radar_original_repeat"] = repeat
    return result


class FeatureTests(unittest.TestCase):
    def test_fingerprint_order_and_signed_deltas(self):
        matrix, names = fingerprint_features([row(0)])
        self.assertEqual(names, FINGERPRINT_FEATURE_NAMES)
        self.assertEqual(matrix.shape, (1, 4))
        self.assertAlmostEqual(matrix[0, 0], .10)
        self.assertAlmostEqual(matrix[0, 1], .20)

    def test_comparator_has_tta_mean_but_no_individual_deltas(self):
        train = [row(i, "a", i % 2) for i in range(6)]
        values = [row(6, "b", 1)]
        train_x, values_x, names = generic_tta_features(train, values)
        self.assertEqual(train_x.shape[0], 6)
        self.assertEqual(values_x.shape[0], 1)
        self.assertIn("radar_four_view_mean", names)
        self.assertFalse(any("minus_original" in name for name in names))
        self.assertFalse(any(name.startswith("radar_wrap") for name in names))


class MetricTests(unittest.TestCase):
    def test_interpolated_metric_and_review_budgets(self):
        rows = [row(i, label=int(i % 2 == 0)) for i in range(10)]
        scores = np.asarray([1.0 - i / 10 for i in range(10)])
        metric = __import__("fprint.deferral_evaluation", fromlist=["_ranking_metric"])._ranking_metric(rows, scores)
        self.assertGreaterEqual(metric["human_fp_removal"], 0.0)
        curve = review_budget_curve(rows, scores)
        self.assertEqual([point["review_budget"] for point in curve], [.05, .10, .20])
        self.assertEqual(len(risk_coverage_rows(rows, scores, steps=4)), 5)

    def test_bootstrap_is_deterministic_and_grouped(self):
        rows = [row(i, "a" if i < 4 else "b", i % 2) for i in range(8)]
        fp = np.asarray([r["label"] for r in rows], dtype=float)
        baseline = np.zeros(len(rows))
        first = paired_group_bootstrap(rows, fp, baseline, replicates=8, seed=7)
        second = paired_group_bootstrap(rows, fp, baseline, replicates=8, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["strata"], {"a:g1": 1, "a:g2": 1, "b:g1": 1, "b:g2": 1})

    def test_sentinel_ratio(self):
        rows = [row(i, repeat=.60) for i in range(3)]
        self.assertEqual(sentinel_ratio(rows), 0.0)
        for item in rows:
            item["radar_original_repeat"] = .59
        self.assertAlmostEqual(sentinel_ratio(rows), .01 / .15)
        rows[1].pop("radar_original_repeat")
        rows[2].pop("radar_original_repeat")
        self.assertAlmostEqual(sentinel_ratio(rows), .01 / .15)
        self.assertIsNone(sentinel_ratio([row(0)]))


class GateTests(unittest.TestCase):
    def test_gate_reports_every_missing_or_failed_requirement(self):
        result = evaluate_gates(
            [row(0)],
            pooled_incremental=.01,
            bootstrap_lower_80=-.01,
            per_corpus_incremental={"a": .01},
            validation_summary={"manual_gate": True, "automated_gate": True, "mage_gate": False},
            sentinel=1.0,
        )
        self.assertFalse(result["passed"])
        self.assertIn("human_fp_floor", result["failures"])
        self.assertIn("mage_invariance", result["failures"])
        self.assertIn("sentinel_ratio", result["failures"])


class LocoTests(unittest.TestCase):
    def test_loco_predictions_preserve_rows_and_are_folded_by_corpus(self):
        rows = [row(i, "a" if i < 4 else "b", i % 2) for i in range(8)]
        predictions = loco_predictions(rows)
        self.assertEqual({p["record_id"] for p in predictions}, {r["record_id"] for r in rows})
        self.assertEqual({p["held_out_corpus"] for p in predictions}, {"a", "b"})
        self.assertTrue(all(0.0 <= p["prediction"] <= 1.0 for p in predictions))

    def test_pilot_requires_exactly_four_corpora(self):
        with self.assertRaises(ValueError):
            evaluate_pilot([row(i, "a" if i < 4 else "b", i % 2) for i in range(8)], bootstrap_replicates=2)


if __name__ == "__main__":
    unittest.main()
