from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fprint.core import StudyDB, TextRecord, jeffreys_posterior, lock_forecasts
from fprint.detectors import SPECS
from fprint.privileged import (
    build_privileged_comparator,
    build_privileged_plan,
    verify_privileged_plan,
)
from fprint.workflow import fold_paths


class PrivilegedWorkflowTests(unittest.TestCase):
    def _fold(self, root: Path) -> tuple[object, list[str]]:
        paths = fold_paths(root, "pmc")
        paths.root.mkdir(parents=True)
        ids = [f"pmc:{index:03d}" for index in range(250)]
        db = StudyDB(paths.database)
        records = [TextRecord(record_id, "pmc", f"text {index}", f"group:{index}") for index, record_id in enumerate(ids)]
        db.add_records(records, {record_id: "signature" for record_id in ids})
        db.close()
        artifact = paths.root / "artifacts" / "zero" / "ids.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(json.dumps({
            "schema_version": 1,
            "data_ids": {"signature:pmc": ids},
            "draw_ids": {
                "draw:0:n:50": ids[:50],
                "draw:0:n:100": ids[:100],
                "draw:0:n:250": ids,
            },
        }, sort_keys=True), encoding="utf-8")
        artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        manifest = {
            "target_corpus": "pmc",
            "id_artifacts": {str(artifact.resolve()): artifact_hash},
            "panel_revisions": {detector: {} for detector in SPECS},
            "panel_sha256": "a" * 64,
            "thresholds_sha256": "b" * 64,
            "thresholds": {
                "detectors": {
                    detector: {"thresholds": {"0.05": .5, "0.01": .8}}
                    for detector in SPECS
                }
            },
            "code_commit": "c" * 40,
        }
        digest = lock_forecasts(paths.zero_lock, {"manifest": manifest, "forecasts": []})
        paths.state.write_text(json.dumps({
            "target_corpus": "pmc", "phase": "zero_locked", "zero_lock_sha256": digest,
        }), encoding="utf-8")
        return paths, ids

    def test_plan_is_exactly_nested_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, ids = self._fold(root)
            with patch("fprint.privileged.assert_all_target_locks"):
                plan_path = build_privileged_plan(root, "pmc")
            plan = verify_privileged_plan(root, "pmc")
            self.assertEqual(plan["sizes"]["25"], ids[:25])
            self.assertEqual(plan["sizes"]["50"], plan["sizes"]["100"][:50])
            self.assertEqual(plan["sizes"]["100"], plan["sizes"]["250"][:100])

            envelope = json.loads(plan_path.read_text(encoding="utf-8"))
            envelope["payload"]["sizes"]["25"][0] = "arbitrary:id"
            plan_path.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                verify_privileged_plan(root, "pmc")

    def test_comparator_uses_all_four_nested_sizes_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, ids = self._fold(root)
            with patch("fprint.privileged.assert_all_target_locks"):
                build_privileged_plan(root, "pmc")
            db = StudyDB(paths.database)
            for index, record_id in enumerate(ids):
                for detector, spec in SPECS.items():
                    db.add_score(record_id, detector, spec.dependency_group, {
                        "native_score": .9 if index < 10 else .1,
                        "canonical_ai_score": .9 if index < 10 else .1,
                        "truncated": False,
                        "failure": None,
                    })
            db.close()
            state = json.loads(paths.state.read_text(encoding="utf-8"))
            state["phase"] = "signature_scored"
            paths.state.write_text(json.dumps(state), encoding="utf-8")
            with (
                patch("fprint.privileged.assert_all_target_locks"),
                patch("fprint.privileged.lock_privileged_forecasts") as lock,
            ):
                output = build_privileged_comparator(root, "pmc")
            comparator = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(comparator["comparators"]), len(SPECS) * 2 * 4)
            first = next(row for row in comparator["comparators"] if row["signature_size"] == 25)
            self.assertEqual(first["flagged"], 10)
            self.assertAlmostEqual(first["posterior_mean"], jeffreys_posterior(10, 25)[0])
            lock.assert_called_once()

            output.unlink()
            paths.privileged_lock.unlink(missing_ok=True)
            db = StudyDB(paths.database)
            db.connection.execute(
                "UPDATE scores SET truncated=1 WHERE record_id=? AND detector_config=?",
                (ids[0], next(iter(SPECS))),
            )
            db.connection.commit()
            self.assertEqual(
                db.missing_partition_scores("signature", "pmc", tuple(SPECS), set(ids)),
                {next(iter(SPECS)): 1},
            )
            db.close()
            with patch("fprint.privileged.assert_all_target_locks"):
                with self.assertRaisesRegex(RuntimeError, "incomplete or invalid"):
                    build_privileged_comparator(root, "pmc")


if __name__ == "__main__":
    unittest.main()
