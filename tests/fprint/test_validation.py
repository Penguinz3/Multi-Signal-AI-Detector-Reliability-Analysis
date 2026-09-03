import csv
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fprint.core import lock_forecasts, verify_lock
from fprint.validation import _condition_specs, _metrics, prepare_prospective_validation
from fprint.validation_evaluate import evaluate_prospective_validation
from fprint.validation_scoring import SCHEMA
import fprint.validation_evaluate as validation_evaluate_module


def _eligible_text(index):
    return (
        f"“Record {index} begins with a careful account of several ordinary events — all documented clearly, "
        "and every reviewer can inspect the original notes without special access. "
        "The second long clause contains enough words for safe sentence splitting, but it remains easy for a reader to follow. "
        "A third complete sentence supplies another paragraph boundary for the locked transformation. "
        "A fourth complete sentence makes the example robust during deterministic testing.”"
    )


class ProspectiveValidationTests(unittest.TestCase):
    def test_score_schema_accepts_locked_row_shape(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript(SCHEMA)
        row = (
            "endpoint", "condition", "current", "triplet", "probe", "original",
            "endpoint", 0.1, 0.2, 10, 10, 512, 0, 1.0, None, "fp32", "{}",
        )
        connection.execute(
            "INSERT INTO scores VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row,
        )
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 1)
        connection.close()

    def test_metrics_and_opaque_conditions(self):
        rows = [
            {"truth_changed": False, "alarm_score": 0.0, "status": "unchanged"},
            {"truth_changed": True, "alarm_score": 2.0, "status": "changed"},
        ]
        self.assertEqual(_metrics(rows)["auroc"], 1.0)
        conditions = _condition_specs("seed")
        self.assertEqual(len(conditions), 18)
        self.assertEqual(len({row["condition_code"] for row in conditions}), 18)

    def test_candidate_lock_is_group_disjoint_and_hides_condition_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            database = source_root / "folds" / "bawe" / "fprint.sqlite3"
            database.parent.mkdir(parents=True)
            connection = sqlite3.connect(database)
            connection.executescript("""
                CREATE TABLE records(record_id TEXT PRIMARY KEY, corpus TEXT, group_id TEXT, partition_name TEXT, text_hash TEXT, text TEXT);
                CREATE TABLE probe_triplets(triplet_id TEXT PRIMARY KEY, record_id TEXT);
            """)
            corpora = (
                "asap_aes", "blog_authorship", "cnn_dailymail", "govreport",
                "gutenberg", "pmc", "stack_exchange", "wikitext_103",
            )
            connection.executemany(
                "INSERT INTO records VALUES(?,?,?,?,?,?)",
                (
                    (f"{corpus}-{index}", corpus, f"{corpus}-g{index}", "anchor_candidates", "hash", _eligible_text(index))
                    for corpus in corpora for index in range(180)
                ),
            )
            connection.commit()
            connection.close()
            prior = root / "prior"
            (prior / "locks").mkdir(parents=True)
            (prior / "state").mkdir()
            lock_forecasts(prior / "locks" / "fault_audit.json", {"schema_version": 1})
            connection = sqlite3.connect(prior / "state" / "fault_audit.sqlite3")
            connection.execute("CREATE TABLE audit_triplets(group_id TEXT)")
            connection.commit()
            connection.close()
            output = prepare_prospective_validation(
                source_root, prior, root / "validation", candidates_per_cell=50,
            )
            with (output / "candidates.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8 * 3 * 50)
            self.assertEqual(len({row["group_id"] for row in rows}), len(rows))
            public = verify_lock(output / "manifest.lock.json")["payload"]
            self.assertFalse(any("family" in row for row in public["opaque_conditions"]))
            self.assertTrue((output / "condition_truth.private.lock.json").is_file())

    def test_evaluator_reuses_locked_endpoint_references_then_unblinds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            panel = []
            for probe in ("punctuation_normalization", "sentence_splitting", "paragraph_resegmentation"):
                for index in range(3):
                    panel.append({
                        "triplet_id": f"{probe}-{index}", "record_id": f"r-{probe}-{index}",
                        "corpus": "test_corpus", "group_id": f"g-{probe}-{index}", "probe": probe,
                        "low_intensity": ".25", "high_intensity": "1",
                    })
            panel_path = root / "panel.csv"
            with panel_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(panel[0]))
                writer.writeheader()
                writer.writerows(panel)
            manifest = {
                "schema_version": 1, "construct": "prospective_operational_black_box_validation",
                "seed": "test", "corpora": ["test_corpus"],
                "endpoints": ["endpoint"], "query_budgets": [10, 25, 50], "draws": 1,
                "opaque_conditions": [
                    {"condition_code": "base", "endpoint": "endpoint"},
                    {"condition_code": "changed", "endpoint": "endpoint"},
                ],
                "success_gate": {"budget": 50, "minimum_full_budget_corpora": 0},
            }
            manifest_digest = lock_forecasts(root / "manifest.lock.json", manifest)
            file_digest = hashlib.sha256((root / "manifest.lock.json").read_bytes()).hexdigest()
            panel_digest = lock_forecasts(root / "panel.lock.json", {
                "parent_manifest_sha256": file_digest, "rows": len(panel),
                "triplet_ids": [row["triplet_id"] for row in panel],
                "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
            })
            lock_forecasts(root / "condition_truth.private.lock.json", {
                "parent_manifest_payload_sha256": manifest_digest,
                "conditions": [
                    {"condition_code": "base", "endpoint": "endpoint", "family": "unchanged", "mode": "unchanged"},
                    {"condition_code": "changed", "endpoint": "endpoint", "family": "input_handling", "mode": "test"},
                ],
            })
            protocol_digest = lock_forecasts(root / "scoring_protocol.lock.json", {
                "construct": "prospective_validation_scoring_protocol",
            })
            amendment_digest = lock_forecasts(root / "scoring_integrity_amendment_v2.lock.json", {
                "construct": "prospective_scoring_integrity_amendment",
            })
            code_dir = Path(validation_evaluate_module.__file__).resolve().parent
            code_files = (
                code_dir / "validation_evaluate.py", code_dir / "validation_scoring.py",
                code_dir / "validation.py", code_dir / "operational.py",
                code_dir / "detectors.py", code_dir / "core.py",
            )
            lock_forecasts(root / "execution_integrity_patch.lock.json", {
                "construct": "prospective_score_preserving_execution_patch",
                "manifest_sha256": manifest_digest, "panel_lock_sha256": panel_digest,
                "scoring_protocol_sha256": protocol_digest,
                "parent_integrity_amendment_sha256": amendment_digest,
                "score_math_unchanged": True, "completed_run_lock_files_sha256": {},
                "code_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files
                },
            })
            ids = [f"{row['triplet_id']}:{level}" for row in panel for level in ("original", "low", "high")]
            base = {challenge_id: .2 + (index % 5) * .01 for index, challenge_id in enumerate(ids)}
            changed = dict(base)
            for challenge_id in ids:
                if challenge_id.startswith("paragraph_resegmentation") and not challenge_id.endswith(":original"):
                    changed[challenge_id] += .3
            lock_forecasts(root / "score_state.lock.json", {
                "construct": "prospective_score_state", "manifest_sha256": manifest_digest,
                "conditions": [
                    {"endpoint": "endpoint", "condition_code": "base",
                     "reference_a": base, "reference_b": base, "current": base},
                    {"endpoint": "endpoint", "condition_code": "changed", "current": changed},
                ],
            })
            output = evaluate_prospective_validation(root, root / "results")
            report = json.loads((output / "validation_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue((output / "blinded_predictions.lock.json").is_file())
            self.assertEqual(report["evidence_status"], "prospective_blind_validation")
            self.assertEqual(report["by_endpoint"]["endpoint"]["unchanged_false_alarm_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
