import csv
import json
import sqlite3
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from fprint.core import lock_forecasts
from fprint.deferral import (
    ENDPOINT_ROLES,
    DeferralPaths,
    build_conditional_worklist,
    import_generation_outputs,
    lock_human_token_panels,
    prepare_generation_requests,
    prepare_pilot_manifest,
)
from tools.run_deferral_scoring import build_verified_registry, run_stage
from tools.run_deferral_generation import request_seed


TEXT = "One sentence explains the topic clearly. A second sentence adds useful detail. A third sentence closes the passage with enough words for testing and scoring."


def _records():
    return [
        {"record_id": "cal", "corpus": "corpus-a", "group_id": "cal-group", "text": TEXT, "provenance_label": "human", "partition": "calibration"},
        {"record_id": "pilot", "corpus": "corpus-a", "group_id": "pilot-group", "text": TEXT + " Extra pilot words.", "provenance_label": "human", "partition": "pilot"},
    ]


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def score(self, text):
        self.calls.append(text)
        return {"canonical_ai_score": 0.75, "native_score": 0.75, "input_token_count": len(text.split()), "truncated": False}


def _fixture(root: Path):
    paths = DeferralPaths.from_root(root / "study")
    manifest = prepare_pilot_manifest(_records(), paths, calibration_cap=1, pilot_cap=1, width=20, block_size=2, endpoint_revisions={endpoint: "rev-1" for endpoint in ENDPOINT_ROLES})
    counts = {endpoint: {"original": 10, "wrap_80": 11, "sentence_blocks_2": 12, "sentence_per_paragraph": 13} for endpoint in ("radar_roberta_large__vicuna7b_training", "mage_longformer__paper")}
    lock_human_token_panels(paths, {"pilot": counts})
    prepare_generation_requests(paths, {"pilot": "A safe source topic"}, generator_families=(("fake", "family-rev"), ("fake2", "family-rev2"), ("fake3", "family-rev3")), target_length=25)
    requests = []
    with paths.generation_csv.open(encoding="utf-8", newline="") as handle:
        requests = list(csv.DictReader(handle))
    generated = "Generated text about the source topic with enough words for the locked output envelope. It has another sentence for testing. A third sentence keeps every locked reflow variant eligible and distinct."
    token_counts = {endpoint: {"original": 20, "wrap_80": 21, "sentence_blocks_2": 22, "sentence_per_paragraph": 23} for endpoint in ("radar_roberta_large__vicuna7b_training", "mage_longformer__paper")}
    outputs = [{
        **row, "seed": request_seed(int(row["seed"]), row["request_id"], 0),
        "text": generated, "attempt": 0, "token_counts": json.dumps(token_counts),
    } for row in requests]
    output_csv = root / "generated.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outputs[0]))
        writer.writeheader(); writer.writerows(outputs)
    import_generation_outputs(paths, outputs)
    candidates = root / "human_candidates.csv"
    with candidates.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "corpus", "group_id", "text", "provenance_label", "partition"])
        writer.writeheader(); writer.writerows(_records())
    return paths, candidates, output_csv


class RunDeferralScoringTests(unittest.TestCase):
    def test_registry_hash_checks_human_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, candidates, output = _fixture(Path(temporary))
            with candidates.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["text"] = "tampered"
            with candidates.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
            with self.assertRaises(ValueError):
                build_verified_registry(paths, candidates, output)

    def test_calibration_resume_and_real_repeat_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, candidates, output = _fixture(Path(temporary))
            adapter = FakeAdapter()
            journal = Path(temporary) / "journal.sqlite"
            canonical = Path(temporary) / "scores.csv"
            result = run_stage(paths, candidates, output, stage="calibration", endpoint="radar_roberta_large__vicuna7b_training", journal_path=journal, canonical_output=canonical, adapter_factory=lambda endpoint: adapter)
            self.assertEqual(result["completed"], 1)
            self.assertEqual(len(adapter.calls), 1)
            result = run_stage(paths, candidates, output, stage="calibration", endpoint="radar_roberta_large__vicuna7b_training", journal_path=journal, canonical_output=canonical, adapter_factory=lambda endpoint: adapter)
            self.assertEqual(result["skipped"], 1)
            self.assertEqual(len(adapter.calls), 1)

    def test_conditional_refuses_without_locked_worklist(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, candidates, output = _fixture(Path(temporary))
            with self.assertRaises(RuntimeError):
                run_stage(paths, candidates, output, stage="conditional", endpoint="radar_roberta_large__vicuna7b_training", journal_path=Path(temporary) / "journal.sqlite", canonical_output=Path(temporary) / "scores.csv", adapter_factory=lambda endpoint: FakeAdapter())

    def test_original_repeat_is_a_distinct_real_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, candidates, output = _fixture(Path(temporary))
            lock_forecasts(paths.threshold_lock, {"threshold": 0.5})
            panel = json.loads(paths.panel_lock.read_text(encoding="utf-8"))["payload"]["panels"][0]
            original_scores = {
                ("pilot", "radar_roberta_large__vicuna7b_training"): 0.9,
                (panel["ai_record_id"], "radar_roberta_large__vicuna7b_training"): 0.9,
            }
            build_conditional_worklist(paths, original_scores, thresholds={"radar_roberta_large__vicuna7b_training": 0.5}, sentinel_per_corpus_label=1)
            adapter = FakeAdapter()
            journal = Path(temporary) / "journal.sqlite"
            canonical = Path(temporary) / "scores.csv"
            run_stage(paths, candidates, output, stage="originals", endpoint="radar_roberta_large__vicuna7b_training", journal_path=journal, canonical_output=canonical, adapter_factory=lambda endpoint: adapter)
            before = len(adapter.calls)
            result = run_stage(paths, candidates, output, stage="conditional", endpoint="radar_roberta_large__vicuna7b_training", journal_path=journal, canonical_output=canonical, adapter_factory=lambda endpoint: adapter)
            self.assertGreater(result["completed"], 0)
            self.assertGreater(len(adapter.calls), before)
            connection = sqlite3.connect(journal)
            try:
                repeat_count = connection.execute("SELECT COUNT(*) FROM score_requests WHERE variant_id='original_repeat' AND status='success'").fetchone()[0]
            finally:
                connection.close()
            self.assertGreater(repeat_count, 0)


if __name__ == "__main__":
    unittest.main()
