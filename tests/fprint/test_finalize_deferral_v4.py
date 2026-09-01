import unittest

from fprint.deferral import CanonicalRecord, MAGE_ENDPOINT, RADAR_ENDPOINT
from tools.finalize_deferral_v4 import select_replacements


def _record(record_id, corpus, group, words):
    sentence = " ".join(["word"] * max(20, words // 4)) + "."
    text = " ".join([sentence] * 4)
    return CanonicalRecord(record_id, corpus, group, text, "human")


def _counts(value=100):
    panel = {
        "original": value,
        "wrap_80": value,
        "sentence_blocks_2": value,
        "sentence_per_paragraph": value,
    }
    return {RADAR_ENDPOINT: panel, MAGE_ENDPOINT: panel}


class FinalizeDeferralV4Tests(unittest.TestCase):
    def test_replacements_are_same_corpus_unused_group_and_deterministic(self):
        used = _record("used", "stack_exchange", "g-used", 120)
        close = _record("close", "stack_exchange", "g-close", 120)
        farther = _record("farther", "stack_exchange", "g-far", 132)
        other = _record("other", "wikitext_103", "g-other", 120)
        records = [used, close, farther, other]
        counts = {record.record_id: _counts() for record in records}
        topics = {record.record_id: {"topic": f"Topic {record.record_id}"} for record in records}
        requests = {
            "failed": {
                "request_id": "failed", "record_id": "old", "corpus": "stack_exchange",
                "target_length": len(close.text.split()),
            }
        }
        failures = [{"request_id": "failed"}]
        kwargs = dict(
            records=records, token_counts=counts, topics=topics, failures=failures,
            old_requests=requests, used_groups={(used.corpus, used.group_id)},
            seed=7, token_cap=460, max_words=300,
        )
        first = select_replacements(**kwargs)
        second = select_replacements(**kwargs)
        self.assertEqual(first["failed"].record_id, "close")
        self.assertEqual(first["failed"].record_id, second["failed"].record_id)

    def test_any_over_cap_variant_rejects_the_complete_candidate(self):
        rejected = _record("rejected", "stack_exchange", "g-rejected", 120)
        accepted = _record("accepted", "stack_exchange", "g-accepted", 125)
        bad = _counts()
        bad[RADAR_ENDPOINT]["sentence_per_paragraph"] = 461
        selected = select_replacements(
            [rejected, accepted],
            {"rejected": bad, "accepted": _counts()},
            {"rejected": "Rejected topic", "accepted": "Accepted topic"},
            [{"request_id": "failed"}],
            {"failed": {"corpus": "stack_exchange", "target_length": len(rejected.text.split())}},
            set(), seed=7, token_cap=460, max_words=300,
        )
        self.assertEqual(selected["failed"].record_id, "accepted")


if __name__ == "__main__":
    unittest.main()
