import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fprint.product import (
    SCORE_TABLE_OPTIONAL_FIELDS,
    SCORE_TABLE_REQUIRED_FIELDS,
    build_release_bundle,
    export_contracts,
    render_evaluation_html,
    validate_evaluation_report,
    write_evaluation_html,
)


def _report():
    metrics = {
        "macro_auroc": .974,
        "macro_sensitivity": .948,
        "unchanged_false_alarm_rate": 0.0,
        "auroc_by_family": {"input_handling": .979, "output_policy": .943},
    }
    return {
        "schema_version": 1,
        "construct": "behavioral_conformance",
        "manifest_lock_sha256": "a" * 64,
        "success_gates": {
            "permitted_primary_claim": "behavioral_change_detection_and_localization",
            "detection_gate_passed": True,
            "diagnosis_gate_passed": False,
            "channels": {"combined": metrics},
        },
        "discovery_metrics": {"combined:50": metrics},
        "confirmation_metrics": {"combined": metrics},
        "claim_boundary": "Observable departure only; <not internal cause>.",
    }


class ProductContractTests(unittest.TestCase):
    def test_contract_bundle_has_stable_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = export_contracts(Path(temporary))
            with Path(paths["score_template"]).open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(header, list(SCORE_TABLE_REQUIRED_FIELDS + SCORE_TABLE_OPTIONAL_FIELDS))
            score_contract = json.loads(Path(paths["score_contract"]).read_text(encoding="utf-8"))
            self.assertEqual(score_contract["schema_version"], 1)
            self.assertEqual(score_contract["required_fields"], list(SCORE_TABLE_REQUIRED_FIELDS))

    def test_html_is_deterministic_escaped_and_privacy_preserving(self):
        report = _report()
        first = render_evaluation_html(report)
        second = render_evaluation_html(report)
        self.assertEqual(first, second)
        self.assertIn("&lt;not internal cause&gt;", first)
        self.assertNotIn("original_text", first)
        self.assertIn("must not be used to adjudicate", first)

    def test_write_html_validates_before_atomic_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "evaluation.json", root / "report.html"
            source.write_text(json.dumps(_report()), encoding="utf-8")
            self.assertEqual(write_evaluation_html(source, output), output.resolve())
            self.assertTrue(output.read_text(encoding="utf-8").startswith("<!doctype html>"))
            self.assertFalse((root / "report.html.tmp").exists())

    def test_invalid_report_is_refused(self):
        report = _report()
        report["success_gates"]["channels"]["combined"]["macro_auroc"] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            validate_evaluation_report(report)

    def test_release_bundle_is_atomic_hashed_and_public_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "private-evaluation.json", root / "release"
            report = _report()
            report["artifacts"] = {"predictions": r"F:\private\predictions.csv"}
            source.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(build_release_bundle(source, output), output.resolve())
            manifest = json.loads((output / "release_manifest.json").read_text(encoding="utf-8"))
            for item in manifest["files"]:
                content = (output / item["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])
                self.assertEqual(len(content), item["bytes"])
            public = (output / "evaluation_summary.json").read_text(encoding="utf-8")
            self.assertNotIn("predictions.csv", public)
            self.assertNotIn("original_text", (output / "index.html").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                build_release_bundle(source, output)


if __name__ == "__main__":
    unittest.main()
