from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fprint.core import ProbeTriplet, StudyDB, TextRecord
from fprint.detectors import SPECS
from fprint.forecasting import _connect, _probe_rows


class ForecastBuilderTests(unittest.TestCase):
    def test_panel_rejects_whole_probe_triplet_if_any_detector_does_not_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.sqlite3"
            db = StudyDB(path)
            record = TextRecord("pmc:1", "pmc", "original", "author:1")
            db.add_records([record], {record.record_id: "anchor_candidates"})
            triplet = ProbeTriplet(
                "contraction_expansion", "original", "low", "high", 4, 1, 4,
            )
            db.add_probe_triplets([(record.record_id, record.corpus, triplet)])
            triplet_id = db.connection.execute("SELECT triplet_id FROM probe_triplets").fetchone()[0]
            for detector, spec in SPECS.items():
                db.add_probe_token_check(triplet_id, detector, (10, 11, 12), detector != next(iter(SPECS)))
                for level, score in (("original", .2), ("low", .3), ("high", .6)):
                    db.add_score(
                        record.record_id, detector, spec.dependency_group,
                        {
                            "native_score": score, "canonical_ai_score": score,
                            "input_token_count": 10, "effective_token_count": 10,
                            "max_tokens": 512, "truncated": False,
                            "runtime_ms": 1, "failure": None,
                        },
                        f"contraction_expansion:{level}",
                    )
            db.close()
            connection = _connect(path)
            cdfs = {detector: [0.0, .5, 1.0] for detector in SPECS}
            self.assertEqual(_probe_rows(connection, tuple(SPECS), cdfs), ())
            connection.close()

            db = StudyDB(path)
            for detector in SPECS:
                db.add_probe_token_check(triplet_id, detector, (10, 11, 12), True)
            db.close()
            connection = _connect(path)
            rows = _probe_rows(connection, tuple(SPECS), cdfs)
            connection.close()
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0].slopes), set(SPECS))


if __name__ == "__main__":
    unittest.main()
