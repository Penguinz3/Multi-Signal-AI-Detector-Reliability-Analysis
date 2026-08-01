from __future__ import annotations

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.normalize_raid_academic import (
    PMC_QUERY, normalize_asap, normalize_bawe, normalize_pmc, normalize_raid,
    normalize_raid_ai, passage_chunks,
)


def prose(words: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{index}" for index in range(words)) + "."


class AcademicNormalizerTests(unittest.TestCase):
    def test_chunks_are_never_short_or_long_and_drop_unmergeable_tail(self):
        chunks = passage_chunks(f"{prose(300)} {prose(80, 'tail')}")
        self.assertEqual([len(chunk.split()) for chunk in chunks], [300])
        self.assertTrue(all(100 <= len(chunk.split()) <= 350 for chunk in chunks))

    def test_raid_keeps_only_verified_human_and_one_passage_per_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raid.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "id", "source_id", "model", "attack", "domain", "generation",
                ])
                writer.writeheader()
                writer.writerow({
                    "id": "human", "source_id": "source", "model": "human",
                    "attack": "none", "domain": "news", "generation": prose(480),
                })
                writer.writerow({
                    "id": "ai", "source_id": "source", "model": "gpt4",
                    "attack": "none", "domain": "news", "generation": prose(150),
                })
            rows, stats = normalize_raid(path)
        self.assertEqual([row["record_id"] for row in rows], ["human"])
        self.assertEqual(len(rows[0]["text"].split()), 350)
        self.assertEqual(stats["not_verified_human"], 1)

    def test_raid_ai_pilot_excludes_human_and_attacks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raid.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "id", "source_id", "model", "attack", "domain", "generation",
                ])
                writer.writeheader()
                for identifier, model, attack in (
                    ("ai", "gpt4", "none"),
                    ("human", "human", "none"),
                    ("attack", "gpt4", "paraphrase"),
                ):
                    writer.writerow({
                        "id": identifier, "source_id": identifier, "model": model,
                        "attack": attack, "domain": "news", "generation": prose(150),
                    })
            rows, _ = normalize_raid_ai(path)
        self.assertEqual([row["record_id"] for row in rows], ["ai"])
        self.assertEqual(rows[0]["model"], "gpt4")

    def test_asap_preserves_essay_group_and_prompt_stratum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asap.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("__MACOSX/._ASAP_2_Final_github_train.csv", b"resource fork")
                archive.writestr(
                    "ASAP_2_Final_github_train.csv",
                    "essay_id,full_text,prompt_name\n"
                    f'e7,"{prose(500)}",Car-free cities\n',
                )
            rows, _ = normalize_asap(path)
        self.assertEqual({row["essay_id"] for row in rows}, {"e7"})
        self.assertEqual({row["essay_set"] for row in rows}, {"Car-free cities"})
        self.assertEqual(len(rows), 2)

    def test_bawe_preserves_writer_group_and_excludes_nonprose(self):
        def document(identifier: str) -> str:
            return f"""<TEI.2 id="_{identifier}">
<teiHeader><fileDesc><sourceDesc>
  <p n="level">2</p><p n="genre family">Essay</p>
  <p n="discipline">Education</p><p n="disciplinary group">SS</p>
</sourceDesc></fileDesc><profileDesc><person>
  <p n="student ID">writer7</p>
</person></profileDesc></teiHeader>
<text><body><div1 type="section"><p>{prose(120, identifier)}
  <quote>{prose(120, "quoted")}</quote>
  <formula>{prose(120, "formula")}</formula>
  <table>{prose(120, "table")}</table>
</p></div1><div1 type="bibliography"><p>{prose(120, "reference")}</p></div1>
</body></text></TEI.2>"""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bawe.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for identifier in ("paper_a", "paper_b"):
                    archive.writestr(
                        f"download/CORPUS_UTF-8/{identifier}.xml",
                        document(identifier),
                    )
            rows, stats = normalize_bawe(path)
        self.assertEqual({row["student_id"] for row in rows}, {"writer7"})
        self.assertEqual({row["disciplinary_group"] for row in rows}, {"SS"})
        self.assertEqual({row["writer_stratum"] for row in rows}, {"SS"})
        self.assertEqual({row["document_id"] for row in rows}, {"paper_a", "paper_b"})
        self.assertEqual(stats["input_documents"], 2)
        self.assertTrue(all(term not in row["text"] for row in rows for term in (
            "quoted0", "formula0", "table0", "reference0",
        )))

    def test_pmc_requires_pre2020_ccby_and_ignores_tables(self):
        valid_body = prose(120, "body")
        table_text = prose(120, "table")
        xml = f"""<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <front><article-meta>
      <article-id pub-id-type="pmcid">123</article-id>
      <article-id pub-id-type="pmid">456</article-id>
      <pub-date><year>2018</year></pub-date>
      <permissions><license xmlns:xlink="http://www.w3.org/1999/xlink"
        xlink:href="https://creativecommons.org/licenses/by/4.0/"/></permissions>
      <contrib-group><contrib contrib-type="author"><name>
        <surname>Smith</surname><given-names>Alex</given-names>
      </name></contrib></contrib-group>
    </article-meta></front>
    <body><sec><p>{valid_body}</p>
      <table-wrap><caption><p>{table_text}</p></caption></table-wrap>
    </sec></body>
  </article>
  <article>
    <front><article-meta>
      <article-id pub-id-type="pmc">999</article-id>
      <pub-date><year>2021</year></pub-date>
      <permissions><license xmlns:xlink="http://www.w3.org/1999/xlink"
        xlink:href="https://creativecommons.org/licenses/by/4.0/"/></permissions>
    </article-meta></front><body><p>{valid_body}</p>
      <ref-list><ref><year>1900</year></ref></ref-list>
    </body>
  </article>
</pmc-articleset>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.xml"
            path.write_text(xml, encoding="utf-8")
            rows, stats = normalize_pmc([path])
        self.assertEqual({row["article_id"] for row in rows}, {"PMC123"})
        self.assertNotIn("table0", rows[0]["text"])
        self.assertEqual(rows[0]["author_cluster"], "name:smitha")
        self.assertEqual(stats["date_rejected"], 1)
        self.assertIn("2019/12/31[pubdate]", PMC_QUERY)


if __name__ == "__main__":
    unittest.main()
