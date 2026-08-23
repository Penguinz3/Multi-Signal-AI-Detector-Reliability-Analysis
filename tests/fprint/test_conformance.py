import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fprint.conformance import (
    FaultSpec, _connect, _digest, _draw_triplets, _evaluate_channel, _feature_vector,
    _summarize_predictions, audit_paths, import_score_table, remap_percentile,
    score_fault_audit, transform_input,
)
from fprint.core import lock_forecasts, verify_lock


ENDPOINT = "radar_roberta_large__vicuna7b_training"


def fault(fault_id, family, stage, mode, parameters=None):
    return FaultSpec(fault_id, family, "test", stage, mode, (), None, parameters or {})


class _RejectingAdapter:
    def token_count(self, text):
        return 999


class ConformanceUnitTests(unittest.TestCase):
    def test_input_faults_are_deterministic(self):
        text = "Ａ  line\n\n next\tword"
        self.assertEqual(transform_input("newline_flatten", text), "Ａ  line next\tword")
        self.assertEqual(transform_input("whitespace_collapse", text), "Ａ line next word")
        self.assertEqual(transform_input("nfkc_whitespace", text), "A line next word")
        self.assertEqual(transform_input("nfkc_whitespace", text), transform_input("nfkc_whitespace", text))

    def test_probe_geometry_is_invariant_to_monotone_remapping(self):
        unchanged = fault("unchanged", "unchanged", "none", "identity")
        calibration = fault("cal", "output_policy", "post_score", "logit_bias", {"bias": .8})
        faults = {"unchanged": unchanged, "cal": calibration}
        selected, scores = [], {}
        for probe_index, probe in enumerate(("punctuation", "sentence", "paragraph")):
            triplet_id = f"t{probe_index}"
            selected.append({
                "triplet_id": triplet_id, "probe": probe,
                "low_intensity": .25, "high_intensity": 1.0,
            })
            for level, value in (("original", .2), ("low", .4), ("high", .8)):
                scores[(triplet_id, level, ENDPOINT, "unchanged")] = {
                    "canonical_ai_score": value + probe_index * .01,
                    "failure": None, "truncated": 0,
                }
        references = {ENDPOINT: [index / 100 for index in range(101)]}
        base = _feature_vector(selected, ENDPOINT, unchanged, faults, scores, references, [row["probe"] for row in selected])
        shifted = _feature_vector(selected, ENDPOINT, calibration, faults, scores, references, [row["probe"] for row in selected])
        self.assertIsNotNone(base)
        self.assertIsNotNone(shifted)
        for name, value in base[0].items():
            if name.startswith("probe__"):
                self.assertAlmostEqual(value, shifted[0][name])
        self.assertNotAlmostEqual(base[0]["raw_mean"], shifted[0]["raw_mean"])

    def test_group_aware_draw_never_reuses_a_group(self):
        triplets = []
        for probe in ("a", "b"):
            for index in range(6):
                triplets.append({
                    "corpus": "c", "probe": probe, "group_id": f"{probe}:{index // 2}",
                    "triplet_id": f"{probe}:{index}",
                })
        selected = _draw_triplets(triplets, "c", ("a", "b"), 3, 0, 9)
        for probe in ("a", "b"):
            groups = [row["group_id"] for row in selected if row["probe"] == probe]
            self.assertEqual(len(groups), len(set(groups)))
            self.assertEqual(len(groups), 3)

    def test_remapping_is_monotone(self):
        calibration = fault("cal", "output_policy", "post_score", "temperature", {"temperature": .8})
        values = [remap_percentile(value, calibration) for value in (.1, .3, .7, .9)]
        self.assertEqual(values, sorted(values))


class ConformanceStorageTests(unittest.TestCase):
    def _locked_root(self, root):
        paths = audit_paths(root)
        config = {"primary_endpoints": [ENDPOINT], "probes": ["p"], "seed": 1}
        faults = [
            {
                "fault_id": "unchanged", "family": "unchanged", "severity": "none",
                "stage": "none", "mode": "identity", "applicable_endpoints": [],
                "effective_implementation": None, "parameters": {},
            },
            {
                "fault_id": "input", "family": "input_handling", "severity": "test",
                "stage": "pre_inference", "mode": "whitespace_collapse",
                "applicable_endpoints": [], "effective_implementation": None, "parameters": {},
            },
        ]
        manifest = {"config": config, "faults": faults}
        lock_forecasts(paths.lock, manifest)
        connection = _connect(paths.database)
        connection.execute(
            "INSERT INTO audit_triplets VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            ("t", "r", "c", "g", "p", "discovery", "one", "two", "three", .25, 1.0, "hash"),
        )
        connection.execute("INSERT INTO metadata VALUES('manifest_sha256',?)", (_digest(manifest),))
        connection.commit()
        connection.close()
        return paths

    def test_full_triplet_rejection_is_atomic_and_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._locked_root(Path(temporary))
            with patch("fprint.conformance.build_adapter", return_value=_RejectingAdapter()):
                first = score_fault_audit(paths.root, ENDPOINT, "input")
                second = score_fault_audit(paths.root, ENDPOINT, "input")
            self.assertEqual(first["rejected_triplets"], 1)
            self.assertEqual(second["skipped"], 3)
            connection = _connect(paths.database)
            rows = connection.execute(
                "SELECT intensity,truncated,failure,audited_endpoint,fault_id FROM audit_scores ORDER BY intensity"
            ).fetchall()
            connection.close()
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["truncated"] == 1 for row in rows))
            self.assertTrue(all(row["failure"] == "full_triplet_rejected_capacity" for row in rows))
            self.assertTrue(all(row["audited_endpoint"] == ENDPOINT and row["fault_id"] == "input" for row in rows))

    def test_lock_tampering_and_unlocked_score_rows_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._locked_root(Path(temporary))
            table = paths.root / "scores.csv"
            with table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "triplet_id", "intensity", "audited_endpoint", "fault_id",
                    "effective_endpoint", "native_score", "canonical_ai_score",
                ))
                writer.writeheader()
                writer.writerow({
                    "triplet_id": "not-locked", "intensity": "original",
                    "audited_endpoint": ENDPOINT, "fault_id": "unchanged",
                    "effective_endpoint": ENDPOINT, "native_score": .1,
                    "canonical_ai_score": .1,
                })
            with self.assertRaises(ValueError):
                import_score_table(paths.root, table)
            envelope = json.loads(paths.lock.read_text(encoding="utf-8"))
            envelope["payload"]["config"]["seed"] = 2
            paths.lock.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_lock(paths.lock)


class ConformanceIntegrationTests(unittest.TestCase):
    def test_fault_families_separate_and_unknown_abstains(self):
        observations = []
        positions = {
            "unchanged": (0.0, 0.0),
            "input_handling": (.1, 3.0),
            "output_policy": (3.0, .1),
            "core_computation": (3.0, 3.0),
            "unknown": (20.0, 20.0),
        }
        variants = {
            "unchanged": ("same",),
            "input_handling": ("input_a", "input_b"),
            "output_policy": ("output_a", "output_b"),
            "core_computation": ("core_a", "core_b"),
            "unknown": ("multi",),
        }
        for corpus_index, corpus in enumerate(("a", "b", "c", "d")):
            for draw in range(20):
                noise = (draw - 9.5) * .002 + corpus_index * .001
                for family, fault_ids in variants.items():
                    for fault_id in fault_ids:
                        raw, geometry = positions[family]
                        observations.append({
                            "source_kind": "discovery", "corpus": corpus, "budget": 50,
                            "draw": draw, "endpoint": ENDPOINT, "fault_id": fault_id,
                            "family": family,
                            "features": {"raw_mean": raw + noise, "probe__p__slope": geometry + noise},
                            "native_original_mean": raw, "group_ids": [f"{corpus}:{draw}"],
                        })
        predictions = _evaluate_channel(observations, "combined")
        summary = _summarize_predictions(predictions)
        # The training calibration is fixed at 5%; a finite held-out corpus may vary.
        self.assertLessEqual(summary["unchanged_false_alarm_rate"], .10)
        self.assertGreaterEqual(summary["diagnosis_macro_f1"], .95)
        self.assertEqual(summary["unknown_rejection_rate"], 1.0)
        self.assertTrue(all(
            row["prediction"] == "unknown" and row["status"] == "inconclusive"
            for row in predictions if row["family"] == "unknown"
        ))
        # Every held-out prediction carries only groups from that corpus.
        self.assertTrue(all(all(group.startswith(f"{row['corpus']}:") for group in row["group_ids"]) for row in predictions))

        # Held-out values can change their own distances, but never their fold's
        # training-derived preprocessing or alarm threshold.
        shifted = []
        for row in observations:
            copied = {**row, "features": dict(row["features"])}
            if row["corpus"] == "a":
                copied["features"]["raw_mean"] += 100
            shifted.append(copied)
        shifted_predictions = _evaluate_channel(shifted, "combined")
        original_thresholds = {
            (row["draw"], row["fault_id"]): row["alarm_threshold"]
            for row in predictions if row["corpus"] == "a"
        }
        shifted_thresholds = {
            (row["draw"], row["fault_id"]): row["alarm_threshold"]
            for row in shifted_predictions if row["corpus"] == "a"
        }
        self.assertEqual(original_thresholds, shifted_thresholds)


if __name__ == "__main__":
    unittest.main()
