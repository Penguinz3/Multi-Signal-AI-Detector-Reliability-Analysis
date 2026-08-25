import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.prepare_deferral_inputs import (
    canonical_human_rows,
    source_topics,
    tokenize_human_rows,
    write_inputs,
)


TEXT = "One sentence explains the topic clearly. A second sentence adds useful detail. A third sentence closes the passage with enough words for testing."


def _write_fixture(root: Path) -> None:
    (root / "raw" / "pmc").mkdir(parents=True)
    (root / "raw" / "stackexchange_20221005").mkdir(parents=True)
    rows = {
        "blog_authorship": [{"record_id": "author:0000001:000", "text": TEXT, "author_id": "author"}],
        "pmc": [{"record_id": "PMC1:chunk000", "text": TEXT, "article_id": "PMC1"}],
        "stack_exchange": [{"record_id": "7:000", "text": TEXT, "site_id": "site", "user_id": "u", "post_id": "7"}],
        "wikitext_103": [{"record_id": "wiki:1", "text": TEXT, "article_id": "1", "title": "A source article"}],
    }
    for corpus, values in rows.items():
        with (root / f"{corpus}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (root / "raw" / "pmc" / "batch_000.xml").write_text(
        '<root><article><front><article-meta><article-id pub-id-type="pmc">PMC1</article-id><title-group><article-title>PMC source title</article-title></title-group></article-meta></front></article></root>',
        encoding="utf-8",
    )
    (root / "raw" / "stackexchange_20221005" / "Posts.xml").write_text(
        '<posts><row Id="7" PostTypeId="1" Title="Stack source question" /></posts>', encoding="utf-8"
    )


class PrepareDeferralInputsTests(unittest.TestCase):
    def test_topics_are_source_metadata_and_never_passage_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            corpora = ("blog_authorship", "pmc", "stack_exchange", "wikitext_103")
            rows = canonical_human_rows(root, corpora=corpora)
            topics = source_topics(rows, root, corpora=corpora)
            self.assertEqual(topics["PMC1:chunk000"]["topic"], "PMC source title")
            self.assertEqual(topics["7:000"]["topic"], "Stack source question")
            self.assertEqual(topics["wiki:1"]["topic"], "A source article")
            self.assertTrue(all(not entry["used_passage_text"] for entry in topics.values()))
            self.assertTrue(all(entry["topic"] != TEXT for entry in topics.values()))

    def test_grouped_db_preserves_frozen_group_and_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            db = root / "grouped.sqlite3"
            connection = sqlite3.connect(db)
            connection.execute("CREATE TABLE records(record_id TEXT, corpus TEXT, group_id TEXT, partition_name TEXT, text TEXT)")
            connection.execute("INSERT INTO records VALUES(?,?,?,?,?)", ("PMC1:chunk000", "pmc", "article:PMC1", "pilot", TEXT))
            connection.commit()
            connection.close()
            rows = canonical_human_rows(root, db, corpora=("blog_authorship", "pmc", "stack_exchange", "wikitext_103"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["group_id"], "article:PMC1")
            self.assertEqual(rows[0]["partition"], "pilot")

    def test_token_panel_uses_all_active_variants_and_two_endpoints(self):
        class FakeTokenizer:
            def token_count(self, text):
                return len(str(text).split())
        rows = [{"record_id": "r", "corpus": "pmc", "group_id": "g", "text": TEXT}]
        counts = tokenize_human_rows(rows, FakeTokenizer(), FakeTokenizer())
        self.assertEqual(set(counts["r"]), {"radar_roberta_large__vicuna7b_training", "mage_longformer__paper"})
        self.assertEqual(set(counts["r"]["radar_roberta_large__vicuna7b_training"]), {"original", "wrap_80", "sentence_blocks_2", "sentence_per_paragraph"})

    def test_write_outputs_are_reproducible_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            output = root / "out"
            summary = write_inputs(root, output, corpora=("blog_authorship", "pmc", "stack_exchange", "wikitext_103"))
            self.assertEqual(summary["rows"], 4)
            self.assertTrue((output / "human_candidates.csv").is_file())
            self.assertEqual(len(json.loads((output / "source_topics.json").read_text(encoding="utf-8"))), 4)
            self.assertEqual(json.loads((output / "human_token_counts.json").read_text(encoding="utf-8")), {})

    def test_asap_prompt_metadata_and_prefixed_grouped_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw").mkdir()
            with (root / "asap_aes.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["record_id", "text", "essay_id", "essay_set", "prompt_name"])
                writer.writeheader()
                writer.writerow({
                    "record_id": "essay-1:chunk00", "text": TEXT,
                    "essay_id": "essay-1", "essay_set": "7",
                    "prompt_name": "Write about the benefits of public transport.",
                })
            db = root / "grouped.sqlite3"
            connection = sqlite3.connect(db)
            connection.execute("CREATE TABLE records(record_id TEXT, corpus TEXT, group_id TEXT, partition_name TEXT, text TEXT)")
            connection.execute("INSERT INTO records VALUES(?,?,?,?,?)", (
                "asap_aes:essay-1:chunk00", "asap_aes", "essay:essay-1", "pilot", TEXT,
            ))
            connection.commit()
            connection.close()
            rows = canonical_human_rows(root, db, corpora=("asap_aes",))
            self.assertEqual(rows[0]["record_id"], "asap_aes:essay-1:chunk00")
            topics = source_topics(rows, root, corpora=("asap_aes",))
            topic = topics[rows[0]["record_id"]]
            self.assertEqual(topic["prompt_id"], "7")
            self.assertEqual(topic["prompt_text"], "Write about the benefits of public transport.")
            self.assertFalse(topic["used_passage_text"])
            self.assertNotEqual(topic["topic"], TEXT)


if __name__ == "__main__":
    unittest.main()
