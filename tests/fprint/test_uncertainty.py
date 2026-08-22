from __future__ import annotations

import hashlib
import unittest

from fprint.core import canonical_json
from fprint.uncertainty import (
    cluster_bootstrap_indices,
    forecast_identity_hashes,
    split_half_pairs,
    summarize_replicates,
    validate_replicate_completeness,
)


class UncertaintyTests(unittest.TestCase):
    def test_cluster_bootstrap_is_seeded_and_preserves_whole_groups(self):
        group_ids = ("a", "a", "b", "c", "c", "c")
        first = cluster_bootstrap_indices(group_ids, seed=7, replicates=5)
        self.assertEqual(first, cluster_bootstrap_indices(group_ids, seed=7, replicates=5))
        self.assertNotEqual(first, cluster_bootstrap_indices(group_ids, seed=8, replicates=5))
        for indices in first:
            counts = {index: indices.count(index) for index in range(len(group_ids))}
            self.assertEqual(counts[0], counts[1])
            self.assertEqual(counts[3], counts[4])
            self.assertEqual(counts[4], counts[5])

    def test_summary_uses_90_percent_percentiles_and_sample_sd(self):
        summary = summarize_replicates((0.0, 1.0, 2.0, 3.0, 4.0))
        self.assertEqual(summary["n"], 5)
        self.assertAlmostEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["sd"], 2.5 ** .5)
        self.assertAlmostEqual(summary["lower"], .2)
        self.assertAlmostEqual(summary["upper"], 3.8)
        self.assertEqual(summary["confidence"], .90)

    def test_forecast_identity_excludes_prediction_and_sorts_signature_ids(self):
        forecast = {
            "target_corpus": "pmc", "detector_config": "detector",
            "operating_fpr": .05, "signature_size": 100,
            "draw": 2, "model": "main", "prediction": .12,
        }
        first = forecast_identity_hashes(forecast, ("record:b", "record:a"))
        changed_prediction = {**forecast, "prediction": .99}
        self.assertEqual(first, forecast_identity_hashes(changed_prediction, ("record:a", "record:b")))
        expected_signature = hashlib.sha256(canonical_json(["record:a", "record:b"])).hexdigest()
        self.assertEqual(first[1], expected_signature)

    def test_split_halves_are_seeded_disjoint_and_keep_groups_whole(self):
        group_ids = ("a", "a", "b", "c", "c", "d", "e")
        pairs = split_half_pairs(group_ids, seed=19, pairs=4)
        self.assertEqual(pairs, split_half_pairs(group_ids, seed=19, pairs=4))
        for left, right in pairs:
            self.assertFalse(set(left) & set(right))
            self.assertEqual(set(left) | set(right), set(range(len(group_ids))))
            for group in set(group_ids):
                members = {index for index, value in enumerate(group_ids) if value == group}
                self.assertTrue(members <= set(left) or members <= set(right))

    def test_replicate_completeness_is_strict_and_defaults_to_100(self):
        self.assertEqual(validate_replicate_completeness(reversed(range(100))), tuple(range(100)))
        with self.assertRaisesRegex(ValueError, "99/100"):
            validate_replicate_completeness(range(99))
        with self.assertRaisesRegex(ValueError, r"duplicates=\[0\]"):
            validate_replicate_completeness((*range(99), 0))


if __name__ == "__main__":
    unittest.main()
