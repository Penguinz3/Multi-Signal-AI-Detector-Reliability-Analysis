import csv
import json
import tempfile
import unittest
from pathlib import Path

from fprint.operational import compare_runs, export_challenge, import_run, initialize_audit


def _source_text(index):
    return (
        f"“During trial {index}, the careful review team examined every available measurement in considerable detail — twice, "
        "and they recorded each unexpected deviation for the final committee report.” "
        "The second review used a separate checklist throughout the long afternoon session, but it still reached the same documented conclusion. "
        "A third analyst inspected the notes before the meeting began. "
        "The committee then archived the complete record for later verification. "
        "Finally, an independent reader confirmed that the summary matched the evidence."
    )


def _write_scores(path, challenge, offset=.0, paragraph_change=False, transform=None):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("challenge_id", "canonical_ai_score", "truncated", "failure"))
        for index, row in enumerate(challenge):
            score = .20 + (index % 7) * .01 + offset
            if paragraph_change and row["probe"] == "paragraph_resegmentation" and row["intensity"] != "original":
                score += .20
            if transform:
                score = transform(score)
            writer.writerow((row["challenge_id"], score, "false", ""))


def _write_metadata(path, version="v1"):
    path.write_text(json.dumps({
        "version": version,
        "configuration": "test-default",
        "threshold_policy": "unchanged",
        "collected_at_utc": "2026-09-02T12:00:00Z",
    }), encoding="utf-8")


class OperationalWorkflowTests(unittest.TestCase):
    def test_complete_black_box_lifecycle_detects_and_localizes_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.csv"
            with records.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("record_id", "text"))
                writer.writerows((f"r{index}", _source_text(index)) for index in range(10))
            audit = root / "audit"
            result = initialize_audit(records, audit, "opaque-detector-v1", minimum_sites=2)
            self.assertEqual(result["triplets_by_probe"], {
                "punctuation_normalization": 10,
                "sentence_splitting": 10,
                "paragraph_resegmentation": 10,
            })
            exported = export_challenge(audit, root / "export")
            self.assertTrue((exported / "scores_template.csv").is_file())
            self.assertTrue((exported / "manifest.lock.json").is_file())
            with (audit / "challenge.csv").open(encoding="utf-8", newline="") as handle:
                challenge = list(csv.DictReader(handle))
            run_specs = (
                ("ref-a", "reference", 0.0, False),
                ("ref-b", "reference", .001, False),
                ("ref-noisy", "reference", .05, False),
                ("current-same", "current", .0005, False),
                ("current-changed", "current", .0005, True),
            )
            for run_id, role, offset, changed in run_specs:
                scores = root / f"{run_id}.csv"
                metadata = root / f"{run_id}.json"
                _write_scores(scores, challenge, offset, changed)
                _write_metadata(metadata, "v2" if changed else "v1")
                import_run(audit, run_id, role, scores, metadata)
            same = compare_runs(audit, ("ref-a", "ref-b"), "current-same", root / "same-report")
            changed = compare_runs(audit, ("ref-a", "ref-b"), "current-changed", root / "changed-report")
            inconclusive = compare_runs(audit, ("ref-a", "ref-noisy"), "current-same", root / "noisy-report")
            same_report = json.loads((same / "report.json").read_text(encoding="utf-8"))
            changed_report = json.loads((changed / "report.json").read_text(encoding="utf-8"))
            inconclusive_report = json.loads((inconclusive / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(same_report["status"], "unchanged")
            self.assertEqual(changed_report["status"], "changed")
            self.assertEqual(inconclusive_report["status"], "inconclusive")
            self.assertTrue(changed_report["probe_results"]["paragraph_resegmentation"]["high_shift"]["changed"])
            self.assertNotIn("During trial", (changed / "index.html").read_text(encoding="utf-8"))

    def test_probe_geometry_is_invariant_to_monotone_score_remapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.csv"
            with records.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("record_id", "text"))
                writer.writerows((f"r{index}", _source_text(index)) for index in range(10))
            audit = root / "audit"
            initialize_audit(records, audit, "opaque-detector-v1", minimum_sites=2)
            with (audit / "challenge.csv").open(encoding="utf-8", newline="") as handle:
                challenge = list(csv.DictReader(handle))
            metadata = root / "metadata.json"
            _write_metadata(metadata)
            for run_id, role, transform in (
                ("ref-a", "reference", None),
                ("ref-b", "reference", None),
                ("remapped", "current", lambda value: value * value),
            ):
                scores = root / f"{run_id}.csv"
                _write_scores(scores, challenge, transform=transform)
                import_run(audit, run_id, role, scores, metadata)
            output = compare_runs(audit, ("ref-a", "ref-b"), "remapped", root / "report")
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(any(
                cells["original_score"]["changed"] for cells in report["probe_results"].values()
            ))
            self.assertFalse(any(
                cell["changed"]
                for cells in report["probe_results"].values()
                for feature, cell in cells.items()
                if feature != "original_score"
            ))

    def test_import_refuses_incomplete_scores_and_tampered_challenge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.csv"
            with records.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("record_id", "text"))
                writer.writerows((f"r{index}", _source_text(index)) for index in range(10))
            audit = root / "audit"
            initialize_audit(records, audit, "opaque-detector-v1", minimum_sites=2)
            incomplete = root / "incomplete.csv"
            metadata = root / "metadata.json"
            _write_metadata(metadata)
            incomplete.write_text(
                "challenge_id,canonical_ai_score,truncated,failure\nmissing,0.5,false,\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "do not match"):
                import_run(audit, "bad", "reference", incomplete, metadata)
            with (audit / "challenge.csv").open(encoding="utf-8", newline="") as handle:
                challenge = list(csv.DictReader(handle))
            truncated = root / "truncated.csv"
            with truncated.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("challenge_id", "canonical_ai_score", "truncated", "failure"))
                for index, row in enumerate(challenge):
                    writer.writerow((row["challenge_id"], .5, "true" if index == 0 else "false", ""))
            with self.assertRaisesRegex(ValueError, "Truncated detector query"):
                import_run(audit, "truncated", "reference", truncated, metadata)
            with self.assertRaisesRegex(ValueError, "metadata is required"):
                import_run(audit, "no-metadata", "reference", truncated, None)
            missing_provenance = root / "missing-provenance.csv"
            missing_provenance.write_text("challenge_id,canonical_ai_score\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires challenge_id"):
                import_run(audit, "missing-provenance", "reference", missing_provenance, metadata)
            with (audit / "challenge.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered")
            with self.assertRaisesRegex(RuntimeError, "locked manifest"):
                export_challenge(audit, root / "should-not-exist")


if __name__ == "__main__":
    unittest.main()
