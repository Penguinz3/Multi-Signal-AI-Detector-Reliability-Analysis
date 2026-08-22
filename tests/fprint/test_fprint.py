from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fprint.core import (
    ProbeTriplet, TextRecord, assign_grouped_partitions, backend_macro, canonical_text, deduplicate,
    exact_sign_flip, jeffreys_posterior, lock_forecasts, make_probe_triplet,
    repeated_signature_samples, threshold, triplet_fits, validate_forecast_payload,
    verify_lock,
)
from fprint.detectors import SPECS
from fprint.modeling import Observation, RecomputedFold, tune_c_nested


class FprintTests(unittest.TestCase):
    def test_primary_signature_partition_requires_250_groups_and_1000_records(self):
        records = []
        for group in range(1200):
            size = 20 if group < 300 else 1
            records.extend(
                TextRecord(f"r:{group}:{item}", "gutenberg", "text", f"author:{group}")
                for item in range(size)
            )
        assignments = assign_grouped_partitions(records, seed=7)
        signature = [record for record in records if assignments[record.record_id] == "signature"]
        test = [record for record in records if assignments[record.record_id] == "test"]
        self.assertGreaterEqual(len(signature), 1000)
        self.assertGreaterEqual(len({record.group_id for record in signature}), 250)
        self.assertGreaterEqual(len(test), 2000)
        self.assertFalse({record.group_id for record in signature} & {record.group_id for record in test})

    def test_global_dedup_drops_cross_corpus_and_keeps_deterministic_local_winner(self):
        rows = [
            TextRecord("a2", "pmc", "Unique local sentence repeated enough for shingles.", "g2", 2),
            TextRecord("a1", "pmc", "Unique local sentence repeated enough for shingles.", "g1", 1),
            TextRecord("b", "blog_authorship", "Shared exact text across both evaluation corpora.", "g3"),
            TextRecord("c", "pmc", "Shared exact text across both evaluation corpora.", "g4"),
        ]
        kept, audit = deduplicate(rows)
        self.assertEqual([row.record_id for row in kept], ["a1"])
        self.assertIn("drop_cross_corpus_component", {row["action"] for row in audit})

    def test_global_dedup_drops_raid_evaluation_collision(self):
        text = "The same passage appears in RAID and the evaluation collection with enough words."
        kept, audit = deduplicate([
            TextRecord("raid", "raid_threshold", text, "raid", 0, True),
            TextRecord("eval", "pmc", text, "article"),
        ])
        self.assertEqual(kept, [])
        self.assertEqual({row["action"] for row in audit}, {"drop_raid_collision"})

    def test_probe_intensity_and_triplet_token_gate(self):
        text = "I can't go, and I won't stay. You aren't ready, but you're trying. It isn't easy, yet it doesn't stop."
        triplet = make_probe_triplet("contraction_expansion", text, "seed", min_sites=4)
        self.assertIsNotNone(triplet)
        assert triplet
        self.assertGreater(triplet.high_intensity, triplet.low_intensity)
        self.assertTrue(triplet_fits(triplet, [("words", lambda value: len(value.split()), 100)]))
        self.assertFalse(triplet_fits(triplet, [("tiny", lambda value: len(value.split()), 2)]))
        self.assertFalse(triplet_fits(
            triplet,
            [("variant_only", lambda value: 101 if "cannot" in value else 10, 100)],
        ))

    def test_threshold_backend_macro_and_sign_flip(self):
        self.assertEqual(threshold(list(range(100)), .05), 94)
        self.assertAlmostEqual(backend_macro({
            "qwen25_shared": {"logrank": .1, "lastde": .3},
            "radar": {"radar": .6},
        }), .4)
        self.assertEqual(exact_sign_flip([1] * 8), 1 / 256)

    def test_forecast_lock_is_exclusive_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            digest = lock_forecasts(path, {"forecast": [.1, .2]})
            self.assertEqual(verify_lock(path)["sha256"], digest)
            with self.assertRaises(FileExistsError):
                lock_forecasts(path, {})
            envelope = json.loads(path.read_text())
            envelope["payload"]["forecast"][0] = .9
            path.write_text(json.dumps(envelope))
            with self.assertRaises(RuntimeError):
                verify_lock(path)

    def test_detector_orientations_are_not_generic(self):
        self.assertEqual(SPECS["openai_roberta_base__gpt2_legacy"].ai_label, 0)
        self.assertEqual(SPECS["mage_longformer__paper"].ai_label, 0)
        self.assertEqual(SPECS["radar_roberta_large__vicuna7b_training"].ai_label, 0)
        self.assertEqual(
            SPECS["logrank__qwen2_5_0_5b_fp32"].dependency_group,
            SPECS["lastde__qwen2_5_0_5b_fp32"].dependency_group,
        )

    def test_jeffreys_posterior(self):
        mean, low, high = jeffreys_posterior(1, 25)
        self.assertAlmostEqual(mean, 1.5 / 26)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)

    def test_repeated_signature_samples_are_nested_and_reproducible(self):
        pool = [f"r{i}" for i in range(300)]
        first = repeated_signature_samples(pool, draws=2)
        second = repeated_signature_samples(pool, draws=2)
        self.assertEqual(first, second)
        self.assertEqual(first[(0, 50)], first[(0, 100)][:50])
        self.assertNotEqual(first[(0, 250)], first[(1, 250)])

    def test_forecast_lock_requires_every_cell(self):
        forecasts = []
        for corpus in ("a", "b"):
            for detector in ("d",):
                for size in (50, 100):
                    for draw in range(2):
                        for model in ("main", "baseline"):
                            forecasts.append({
                                "target_corpus": corpus, "detector_config": detector,
                                "operating_fpr": .05,
                                "signature_size": size, "draw": draw, "model": model,
                                "prediction": .05,
                                "forecast_id": "a" * 64,
                                "signature_ids_sha256": "b" * 64,
                                "fit_ref": f"fit:.05:{model}",
                                "uncertainty_status": (
                                    "joint_cluster_bootstrap_v1" if model == "main"
                                    else "point_only_preregistered_secondary"
                                ),
                                "uncertainty_ref": "conditional:a" if model == "main" else None,
                            })
        payload = {"forecasts": forecasts}
        validate_forecast_payload(
            payload, ("a", "b"), ("d",), (50, 100), 2,
            ("main", "baseline"), (.05,),
        )
        payload["forecasts"].pop()
        with self.assertRaises(ValueError):
            validate_forecast_payload(
                payload, ("a", "b"), ("d",), (50, 100), 2,
                ("main", "baseline"), (.05,),
            )

    def test_nested_cv_rejects_source_quantity_leakage(self):
        rows = [
            Observation("a", "d", "g", .1, .1, (.1,), (.2,)),
            Observation("b", "d", "g", .2, .1, (.1,), (.2,)),
            Observation("c", "d", "g", .3, .1, (.1,), (.2,)),
        ]
        def leaking(train, valid, allowed):
            return RecomputedFold(
                tuple(train), tuple(valid),
                allowed | {valid[0].corpus}, allowed,
            )
        with self.assertRaises(RuntimeError):
            tune_c_nested(rows, "main", leaking)


if __name__ == "__main__":
    unittest.main()
