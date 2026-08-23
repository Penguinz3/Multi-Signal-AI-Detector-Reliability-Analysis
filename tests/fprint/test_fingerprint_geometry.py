import json
import tempfile
import unittest
from pathlib import Path

from fprint.fingerprint_geometry import write_fingerprint_geometry


class FingerprintGeometryTests(unittest.TestCase):
    def test_geometry_separates_stable_detector_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpora = ("a", "b", "c")
            detectors = ("left", "right")
            vectors = {
                "left": {"a": (1.0, .1), "b": (.9, .1), "c": (1.1, .1)},
                "right": {"a": (-1.0, -.1), "b": (-.9, -.1), "c": (-1.1, -.1)},
            }
            for target in corpora:
                slopes = {
                    detector: {
                        probe: {
                            corpus: vectors[detector][corpus][index]
                            for corpus in corpora if corpus != target
                        }
                        for index, probe in enumerate(("p1", "p2"))
                    }
                    for detector in detectors
                }
                path = root / "folds" / target / "artifacts" / "zero"
                path.mkdir(parents=True)
                (path / "profiles.json").write_text(json.dumps({
                    "probe_order": ["p1", "p2"],
                    "operating_points": {"0.05": {"outer": {
                        "profile_corpus_slopes": slopes,
                    }}},
                }), encoding="utf-8")

            bawe = root / "folds" / "bawe" / "artifacts" / "zero"
            bawe.mkdir(parents=True)
            (bawe / "uncertainty.json").write_text(json.dumps({
                "component_diagnostics": {"split_half": {
                    detector: {
                        "overall_profile_stability": {
                            "mean_cosine": .99, "available_probes": 2, "valid_pairs": 100,
                        },
                        **{
                            probe: {
                                "status": "available", "included_corpora": 3,
                                "pairs": 100, "sign_agreement": 1.0,
                                "mean_absolute_difference": .01,
                            }
                            for probe in ("p1", "p2")
                        },
                    }
                    for detector in detectors
                }},
            }), encoding="utf-8")

            observed = {
                corpus: {
                    fpr: {detector: .01 * (index + 1) for detector in detectors}
                    for fpr in ("0.05", "0.01")
                }
                for index, corpus in enumerate(corpora)
            }
            evaluation = root / "evaluation.json"
            evaluation.write_text(json.dumps({
                "primary_corpora": list(corpora),
                "observed_fpr": observed,
                "success_gates": {"0.05": {
                    "passed": False, "sign_flip_p": 1.0,
                    "wins_over_detector_id": {"100": 0, "250": 0},
                    "overall_mae": {
                        "100:main": .2, "100:detector_id_x_text": .1,
                        "250:main": .2, "250:detector_id_x_text": .1,
                    },
                }},
            }), encoding="utf-8")

            thresholds = root / "state"
            thresholds.mkdir()
            (thresholds / "frozen_thresholds.json").write_text(json.dumps({
                "detectors": {
                    detector: {"thresholds": {"0.05": .5, "0.01": .8}}
                    for detector in detectors
                }
            }), encoding="utf-8")
            import sqlite3
            database = sqlite3.connect(root / "folds" / "bawe" / "fprint.sqlite3")
            database.executescript("""
                CREATE TABLE probe_triplets (
                    triplet_id TEXT, record_id TEXT, corpus TEXT, probe TEXT,
                    low_intensity REAL, high_intensity REAL
                );
                CREATE TABLE records (record_id TEXT, group_id TEXT);
                CREATE TABLE scores (
                    record_id TEXT, variant_id TEXT, detector_config TEXT,
                    canonical_ai_score REAL, failure TEXT, truncated INTEGER
                );
            """)
            selected = []
            for corpus in corpora:
                for probe in ("p1", "p2"):
                    triplet = f"{corpus}:{probe}"
                    selected.append(triplet)
                    database.execute(
                        "INSERT INTO probe_triplets VALUES(?,?,?,?,?,?)",
                        (triplet, triplet, corpus, probe, .25, 1.0),
                    )
                    database.execute("INSERT INTO records VALUES(?,?)", (triplet, triplet))
                    for detector in detectors:
                        for level, score in (("original", .1), ("low", .2), ("high", .6)):
                            database.execute(
                                "INSERT INTO scores VALUES(?,?,?,?,?,?)",
                                (triplet, f"{probe}:{level}", detector, score, None, 0),
                            )
            database.commit()
            database.close()
            profiles = json.loads((bawe / "profiles.json").read_text(encoding="utf-8")) if (bawe / "profiles.json").exists() else {}
            profiles["selected_triplet_ids"] = selected
            (bawe / "profiles.json").write_text(json.dumps(profiles), encoding="utf-8")

            output = root / "output"
            report = write_fingerprint_geometry(root, evaluation, output)
            self.assertEqual(report["panel_complete_probes"], ["p1", "p2"])
            self.assertEqual(report["leave_one_corpus_out_detector_identification"]["cosine_accuracy"], 1.0)
            self.assertEqual(report["metamorphic_threshold_audit"]["selected_triplets"], 6)
            self.assertEqual(len((output / "fingerprint_cells.csv").read_text().splitlines()), 7)
            self.assertTrue((output / "metamorphic_threshold_crossings.csv").is_file())
            self.assertTrue((output / "fingerprint_identification.csv").is_file())


if __name__ == "__main__":
    unittest.main()
