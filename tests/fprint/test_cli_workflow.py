from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fprint.cli import _read_corpora, build_parser
from fprint.core import STUDY_CORPORA, TARGET_CORPORA, StudyDB, TextRecord


class CliWorkflowTests(unittest.TestCase):
    def test_pilot_requires_labeled_ai_reference(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "pilot", "--detector", "openai_roberta_base__gpt2_legacy",
            ])
        args = parser.parse_args([
            "pilot", "--detector", "openai_roberta_base__gpt2_legacy",
            "--ai-reference", "raid_ai_pilot.csv",
        ])
        self.assertEqual(args.ai_reference, Path("raid_ai_pilot.csv"))

    def test_source_cache_import_excludes_held_out_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                TextRecord("a:1", "a", "source a", "ga"),
                TextRecord("b:1", "b", "source b", "gb"),
            ]
            partitions = {"a:1": "source_model", "b:1": "source_model"}
            master = StudyDB(root / "master.sqlite3")
            master.add_records(records, partitions)
            for record in records:
                master.add_score(
                    record.record_id, "detector", "group",
                    {
                        "native_score": .2, "canonical_ai_score": .2,
                        "input_token_count": 2, "effective_token_count": 2,
                        "max_tokens": 512, "truncated": False,
                        "runtime_ms": 1, "failure": None,
                    },
                )
            master.close()

            fold = StudyDB(root / "fold.sqlite3")
            fold.add_records(records, partitions)
            fold.import_source_results(root / "master.sqlite3", "detector", "b")
            scored = {
                row[0] for row in fold.connection.execute("SELECT record_id FROM scores")
            }
            fold.close()
            self.assertEqual(scored, {"a:1"})

    def test_target_scoring_requires_explicit_fold_phase_and_panel(self):
        args = build_parser().parse_args([
            "score-target",
            "--target-corpus", "pmc",
            "--partition", "test",
            "--detector", "openai_roberta_base__gpt2_legacy",
            "--admitted-detectors", "openai_roberta_base__gpt2_legacy",
        ])
        self.assertEqual(args.target_corpus, "pmc")
        self.assertEqual(args.partition, "test")

    def test_bawe_is_target_only_and_never_a_primary_source(self):
        self.assertEqual(TARGET_CORPORA, STUDY_CORPORA + ("bawe",))
        args = build_parser().parse_args([
            "score-source", "--target-corpus", "bawe",
            "--detector", "openai_roberta_base__gpt2_legacy",
        ])
        self.assertEqual(args.target_corpus, "bawe")
        with tempfile.TemporaryDirectory() as directory:
            db = StudyDB(Path(directory) / "state.sqlite3")
            records = [
                TextRecord("pmc:1", "pmc", "primary source", "p"),
                TextRecord("bawe:1", "bawe", "external target", "b"),
            ]
            db.add_records(records, {record.record_id: "source_model" for record in records})
            selected = {
                record_id for record_id, _ in db.records(
                    ["source_model"], include_corpora=STUDY_CORPORA,
                )
            }
            db.close()
        self.assertEqual(selected, {"pmc:1"})

    def test_bawe_fails_closed_without_writer_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bawe.csv"
            path.write_text(
                "record_id,text,document_id,disciplinary_group,writer_stratum\n"
                "paper:00,student prose,paper,SS,SS\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _read_corpora([("bawe", path)])


if __name__ == "__main__":
    unittest.main()
