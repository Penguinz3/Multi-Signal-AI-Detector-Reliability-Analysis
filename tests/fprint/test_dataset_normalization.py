import unittest

from tools.normalize_news_reports_wiki import deterministic_sample, passage_chunks
from tools.filter_token_lengths import tokenizer_lengths


class DatasetNormalizationTests(unittest.TestCase):
    def test_chunks_stay_inside_frozen_word_range(self):
        text = " ".join(f"Sentence {index} has enough ordinary words to test deterministic splitting." for index in range(90))
        chunks = passage_chunks(text)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(100 <= len(chunk.split()) <= 300 for chunk in chunks))

    def test_sample_is_order_independent(self):
        rows = [{"record_id": str(index)} for index in range(20)]
        self.assertEqual(
            deterministic_sample(rows, "record_id", 5),
            deterministic_sample(reversed(rows), "record_id", 5),
        )

    def test_token_lengths_never_request_truncation(self):
        class FakeTokenizer:
            def __call__(self, texts, **options):
                self.options = options
                return {"input_ids": [[1] * len(text.split()) for text in texts]}

        tokenizer = FakeTokenizer()
        self.assertEqual(tokenizer_lengths(tokenizer, ["one two", "three"]), [2, 1])
        self.assertFalse(tokenizer.options["truncation"])


if __name__ == "__main__":
    unittest.main()
