from __future__ import annotations

import csv
import html
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.normalize_authored_social import (
    MAX_WORDS,
    MIN_WORDS,
    chunks,
    normalize_blog,
    normalize_gutenberg,
    normalize_stack,
)


def prose(sentences: int = 12, words: int = 12) -> str:
    return " ".join(
        " ".join(f"word{number}" for number in range(words)) + "."
        for _ in range(sentences)
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class NormalizeAuthoredSocialTests(unittest.TestCase):
    def test_chunk_bounds_drop_oversize_sentence(self):
        counts = Counter()
        result = list(chunks(prose() + " " + "x " * (MAX_WORDS + 1), counts))
        self.assertTrue(result)
        self.assertTrue(all(MIN_WORDS <= len(row.split()) <= MAX_WORDS for row in result))
        self.assertEqual(counts["oversize_sentences_dropped"], 1)

    def test_gutenberg_strips_boilerplate_and_preserves_author_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.csv"
            with catalog.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "Text#", "Type", "Issued", "Title", "Language",
                        "Authors", "Subjects", "LoCC", "Bookshelves",
                    ),
                )
                writer.writeheader()
                writer.writerow({
                    "Text#": "42", "Type": "Text", "Language": "en",
                    "Authors": "Example, Ada, 1900-1980",
                })
            (root / "42.txt").write_text(
                "license boilerplate\n*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
                + prose()
                + "\n*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\nfooter",
                encoding="utf-8",
            )
            output = root / "gutenberg.csv"
            provenance = normalize_gutenberg(catalog, root, output, 10, 7)
            rows = read_csv(output)
            self.assertEqual(list(rows[0]), ["record_id", "text", "author_id", "book_id"])
            self.assertEqual(rows[0]["author_id"], "Example, Ada, 1900-1980")
            self.assertNotIn("boilerplate", rows[0]["text"])
            self.assertEqual(json.loads(provenance.read_text())["counts"]["books_read"], 1)

    def test_stack_drops_code_quotes_and_community_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = f"<p>{prose()}</p><pre>{'secret ' * 120}</pre><blockquote>{'quote ' * 120}</blockquote>"
            rows = [
                {
                    "Id": "10", "PostTypeId": "2", "OwnerUserId": "77",
                    "Body": body, "ContentLicense": "CC BY-SA 4.0",
                },
                {
                    "Id": "11", "PostTypeId": "2", "OwnerUserId": "88",
                    "Body": body, "CommunityOwnedDate": "2020-01-01",
                },
            ]
            posts = root / "Posts.xml"
            posts.write_text(
                "<posts>"
                + "".join(
                    "<row "
                    + " ".join(f'{key}="{html.escape(value, quote=True)}"' for key, value in row.items())
                    + " />"
                    for row in rows
                )
                + "</posts>",
                encoding="utf-8",
            )
            output = root / "stack.csv"
            provenance = normalize_stack(posts, "english.stackexchange.com", output, 10, 7)
            normalized = read_csv(output)
            self.assertEqual(list(normalized[0]), ["record_id", "text", "site_id", "user_id", "post_id"])
            self.assertEqual(normalized[0]["user_id"], "77")
            self.assertNotIn("secret", normalized[0]["text"])
            self.assertNotIn("quote", normalized[0]["text"])
            self.assertEqual(json.loads(provenance.read_text())["counts"]["community_owned"], 1)

    def test_blog_parquet_preserves_id_when_pyarrow_available(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is optional")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "blog.parquet"
            pq.write_table(pa.table({"id": [123], "text": [prose()]}), parquet)
            output = root / "blog.csv"
            normalize_blog(parquet, output, 10, 7)
            rows = read_csv(output)
            self.assertEqual(list(rows[0]), ["record_id", "text", "author_id"])
            self.assertEqual(rows[0]["author_id"], "123")


if __name__ == "__main__":
    unittest.main()
