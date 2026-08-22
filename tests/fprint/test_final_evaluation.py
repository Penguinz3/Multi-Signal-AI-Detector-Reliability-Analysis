import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fprint.core import (
    FORECAST_MODELS, STUDY_CORPORA, TARGET_CORPORA, StudyDB, TextRecord, lock_forecasts,
)
from fprint.detectors import SPECS
from fprint.final_evaluation import run_final_evaluation
from fprint.workflow import fold_paths


class FinalEvaluationTests(unittest.TestCase):
    def test_locked_evaluation_and_invalid_score_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            thresholds = {
                "detectors": {
                    detector: {"thresholds": {"0.05": .5, "0.01": .8}}
                    for detector in SPECS
                }
            }
            for corpus in TARGET_CORPORA:
                paths = fold_paths(root, corpus)
                paths.zero_lock.parent.mkdir(parents=True)
                paths.state.write_text(json.dumps({"phase": "test_scored"}), encoding="utf-8")
                db = StudyDB(paths.database)
                record = TextRecord(f"{corpus}:test", corpus, "human text", f"{corpus}:group")
                db.add_records([record], {record.record_id: "test"})
                for detector, spec in SPECS.items():
                    db.add_score(record.record_id, detector, spec.dependency_group, {
                        "canonical_ai_score": .1, "native_score": .1, "truncated": False,
                        "failure": None,
                    })
                db.close()
                forecasts = []
                for fpr in (.05, .01):
                    for size in (50, 100, 250):
                        for draw in range(20):
                            for model in FORECAST_MODELS:
                                for detector in SPECS:
                                    forecasts.append({
                                        "target_corpus": corpus,
                                        "detector_config": detector,
                                        "operating_fpr": fpr,
                                        "signature_size": size,
                                        "draw": draw,
                                        "model": model,
                                        "prediction": 0.0 if model == "main" else .2,
                                        "forecast_id": "a" * 64,
                                        "signature_ids_sha256": "b" * 64,
                                        "fit_ref": "fit",
                                        "uncertainty_status": (
                                            "joint_cluster_bootstrap_v1" if model == "main"
                                            else "point_only_preregistered_secondary"
                                        ),
                                        "uncertainty_ref": "ref" if model == "main" else None,
                                    })
                lock_forecasts(paths.zero_lock, {
                    "manifest": {"thresholds": thresholds}, "forecasts": forecasts,
                })

            output = root / "results"
            with patch("fprint.final_evaluation.assert_all_target_locks"):
                report = run_final_evaluation(root, output)
            self.assertTrue(report["success_gates"]["0.05"]["passed"])
            self.assertTrue(report["success_gates"]["0.01"]["passed"])
            self.assertEqual(
                report["external_validation"]["bawe"]["gate_status"],
                "descriptive_external_validation_only",
            )
            self.assertTrue((output / "final_evaluation.json").is_file())
            self.assertTrue((output / "forecast_evaluation_rows.csv").is_file())

            paths = fold_paths(root, STUDY_CORPORA[0])
            connection = StudyDB(paths.database)
            connection.connection.execute(
                "UPDATE scores SET truncated=1 WHERE detector_config=?",
                (next(iter(SPECS)),),
            )
            connection.connection.commit()
            connection.close()
            with patch("fprint.final_evaluation.assert_all_target_locks"):
                with self.assertRaisesRegex(RuntimeError, "Invalid or truncated"):
                    run_final_evaluation(root)


if __name__ == "__main__":
    unittest.main()
