from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fprint.core import PROBES, ProbeTriplet, StudyDB, TextRecord
from fprint.detectors import SPECS
from fprint.forecasting import (
    ProbeRow, SourceExample, _connect, _joint_bootstrap, _probe_rows,
    _split_half_diagnostics,
)


class ForecastBuilderTests(unittest.TestCase):
    def test_split_half_reports_sparse_cells_and_overall_profile_stability(self):
        rows = {
            ("source_a", probe): (
                ProbeRow("a", "source_a", probe, "g1", {"detector": 1.0}, (0.0, .25, 1.0), {"detector": (0.0, .25, 1.0)}),
                ProbeRow("b", "source_a", probe, "g2", {"detector": 1.0}, (0.0, .25, 1.0), {"detector": (0.0, .25, 1.0)}),
            )
            for probe in PROBES
        }
        diagnostics = _split_half_diagnostics(
            rows, ("source_a", "sparse_source"), ("detector",)
        )["detector"]
        self.assertEqual(diagnostics[PROBES[0]]["excluded_corpora"][0]["corpus"], "sparse_source")
        self.assertEqual(diagnostics["overall_profile_stability"]["valid_pairs"], 100)
        self.assertAlmostEqual(diagnostics["overall_profile_stability"]["mean_cosine"], 1.0)

    def test_joint_bootstrap_rejects_nonpreregistered_replicate_count(self):
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            _joint_bootstrap(
                None, [], {}, {}, "bawe", (), {}, (), (), (), (), (), (),
                replicates=2,
            )

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

    def test_joint_bootstrap_locks_main_uncertainty_and_all_replicates(self):
        detector = "openai_roberta_base__gpt2_legacy"
        target_ids = tuple(f"target:{index}" for index in range(250))
        draws = {
            (draw, size): target_ids[:size]
            for draw in range(20) for size in (50, 100, 250)
        }
        forecasts = [
            {
                "target_corpus": "bawe", "detector_config": detector,
                "operating_fpr": fpr, "signature_size": size,
                "draw": draw, "model": "main", "prediction": .2,
            }
            for fpr in (.05, .01) for draw in range(20) for size in (50, 100, 250)
        ]
        features = {record_id: (.1,) for record_id in target_ids}
        source_model = tuple(
            SourceExample(f"source:{index}", "pmc", f"group:{index}", (.2,), {detector: score})
            for index, score in enumerate((.1, .9))
        )
        source_summary = tuple(
            SourceExample(f"summary:{index}", "pmc", f"summary-group:{index}", None, {detector: score})
            for index, score in enumerate((.1, .9))
        )
        probes = tuple(
            ProbeRow(
                f"{probe}:{index}", "pmc", probe, f"anchor:{probe}:{index}",
                {detector: .1}, (0.0, .25, 1.0),
                {detector: (.1, .2, .3)},
            )
            for probe in PROBES for index in range(2)
        )
        raid = tuple({detector: index / 20} for index in range(20))
        with patch("fprint.forecasting.fit_forecaster", side_effect=lambda observations, targets, *args, **kwargs: [.2] * len(targets)):
            artifact = _joint_bootstrap(
                None, forecasts, {"0.05:main": .1, "0.01:main": .1},
                draws, "bawe", target_ids, features, ("pmc",), (detector,),
                source_model, source_summary, probes, raid, replicates=100,
            )
        self.assertEqual(artifact["replicates"], 100)
        self.assertEqual(len(artifact["conditional"]), 120)
        self.assertTrue(all(row["uncertainty_status"] == "joint_cluster_bootstrap_v1" for row in forecasts))
        self.assertTrue(all(summary["n"] == 100 for summary in artifact["conditional"].values()))


if __name__ == "__main__":
    unittest.main()
