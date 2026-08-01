from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fprint.core import (
    StudyDB, TextRecord, assign_grouped_partitions, deduplicate, grouping_key, make_probe_triplet,
    repeated_signature_samples, triplet_fits,
)


class ProbeSemanticsTests(unittest.TestCase):
    def test_contractions_are_unambiguous_and_accept_curly_apostrophes(self):
        text = "I can’t wait. You won't wait. He’s gone. It's done."
        triplet = make_probe_triplet("contraction_expansion", text, "seed", min_sites=2)
        self.assertIsNotNone(triplet)
        assert triplet
        self.assertIn("cannot", triplet.high)
        self.assertIn("will not", triplet.high)
        self.assertIn("He’s gone", triplet.high)
        self.assertIn("It's done", triplet.high)
        self.assertEqual(triplet.eligible_sites, 2)

    def test_sentence_splits_require_eight_words_on_each_side(self):
        long = "One two three four five six seven eight, and nine ten eleven twelve thirteen fourteen fifteen sixteen."
        short = "One two three, and four five six."
        self.assertIsNotNone(make_probe_triplet("sentence_splitting", long * 4, "seed"))
        self.assertIsNone(make_probe_triplet("sentence_splitting", short * 4, "seed"))

    def test_paragraph_resegmentation_preserves_existing_boundaries(self):
        text = "First sentence. Second sentence.\n\nThird sentence. Fourth sentence. Fifth sentence."
        triplet = make_probe_triplet("paragraph_resegmentation", text, "seed", min_sites=1)
        self.assertIsNotNone(triplet)
        assert triplet
        self.assertIn("Second sentence.\n\nThird sentence.", triplet.low)
        self.assertEqual(triplet.eligible_sites, 3)

    def test_repetition_is_limited_to_one_through_three_tokens(self):
        text = "yes yes. one two one two. a b c a b c. a b c d a b c d."
        triplet = make_probe_triplet("adjacent_repetition_removal", text, "seed", min_sites=1)
        self.assertIsNotNone(triplet)
        assert triplet
        self.assertEqual(triplet.eligible_sites, 3)
        self.assertIn("a b c d a b c d", triplet.high)

    def test_empty_tokenizer_set_is_rejected(self):
        triplet = make_probe_triplet("contraction_expansion", "can't can't can't can't", "seed")
        assert triplet
        self.assertFalse(triplet_fits(triplet, []))
        with tempfile.TemporaryDirectory() as directory:
            db = StudyDB(Path(directory) / "test.sqlite3")
            with self.assertRaises(ValueError):
                db.valid_primary_triplets([])
            db.close()

    def test_primary_selection_requires_every_tokenizer_and_one_record_per_group(self):
        triplet = make_probe_triplet("contraction_expansion", "can't can't can't can't", "seed")
        assert triplet
        with tempfile.TemporaryDirectory() as directory:
            db = StudyDB(Path(directory) / "test.sqlite3")
            records = [
                TextRecord("a", "pmc", triplet.original, "same"),
                TextRecord("b", "pmc", triplet.original, "same"),
                TextRecord("c", "pmc", triplet.original, "other"),
            ]
            db.add_records(records, {record.record_id: "anchor_candidates" for record in records})
            db.add_probe_triplets([(record.record_id, record.corpus, triplet) for record in records])
            rows = list(db.probe_triplets())
            for triplet_id, record_id, *_ in rows:
                db.add_probe_token_check(triplet_id, "one", (4, 4, 4), True)
                if record_id != "c":
                    db.add_probe_token_check(triplet_id, "two", (4, 4, 4), True)
            selected = db.valid_primary_triplets(["one", "two"])
            self.assertEqual(len(selected), 1)
            self.assertNotEqual(selected[0], next(row[0] for row in rows if row[1] == "c"))
            db.close()

    def test_stack_exchange_sentinel_user_falls_back_to_post(self):
        self.assertEqual(
            grouping_key("stack_exchange", {"site_id": "math", "user_id": "-1", "post_id": "42"}),
            "site_post:math:42",
        )
        with self.assertRaises(ValueError):
            grouping_key("stack_exchange", {"site_id": "math", "user_id": "nan", "post_id": "null"})
        with self.assertRaises(ValueError):
            grouping_key("pmc", {"article_id": "None"})

    def test_raid_only_duplicates_keep_one_reference(self):
        text = "The same retained RAID reference appears twice."
        kept, audit = deduplicate([
            TextRecord("raid:1", "raid_threshold", text, "raid:1", 1, True),
            TextRecord("raid:0", "raid_threshold", text, "raid:0", 0, True),
        ])
        self.assertEqual([record.record_id for record in kept], ["raid:0"])
        self.assertEqual({row["action"] for row in audit}, {"keep", "drop_duplicate"})

    def test_duplicate_record_ids_are_rejected_before_dedup(self):
        with self.assertRaisesRegex(ValueError, "Duplicate record_id"):
            deduplicate([
                TextRecord("same", "pmc", "first", "one"),
                TextRecord("same", "pmc", "second", "two"),
            ])

    def test_asap_group_cannot_span_essay_sets(self):
        records = [
            TextRecord("a", "asap_aes", "a", "student_id:7", stratum="1"),
            TextRecord("b", "asap_aes", "b", "student_id:7", stratum="2"),
        ]
        with self.assertRaisesRegex(ValueError, "multiple strata"):
            assign_grouped_partitions(records)

    def test_records_beyond_fixed_partitions_become_anchor_candidates(self):
        total = 64 + 500 + 1500 + 1000 + 2000 + 1
        assignments = assign_grouped_partitions([
            TextRecord(str(index), "pmc", "text", f"article:{index}")
            for index in range(total)
        ])
        self.assertEqual(list(assignments.values()).count("anchor_candidates"), 1)

    def test_grouped_signature_samples_use_one_record_per_group(self):
        records = [
            TextRecord(str(index), "pmc", "text", f"group:{index // 2}")
            for index in range(1001)
        ]
        samples = repeated_signature_samples(records, sizes=(250,), draws=1)
        chosen = samples[(0, 250)]
        self.assertEqual(len({records[int(record_id)].group_id for record_id in chosen}), 250)

    def test_external_target_reserves_writers_and_has_no_source_partition(self):
        records = [
            TextRecord(
                str(index), "bawe", "text", f"student:{index // 10}",
                stratum="SS",
            )
            for index in range(5000)
        ]
        assignments = assign_grouped_partitions(records)
        signature = [
            record for record in records
            if assignments[record.record_id] == "signature"
        ]
        self.assertEqual(len({record.group_id for record in signature}), 250)
        self.assertGreaterEqual(len(signature), 1000)
        self.assertFalse(
            {"source_summary", "source_model"} & set(assignments.values())
        )


if __name__ == "__main__":
    unittest.main()
