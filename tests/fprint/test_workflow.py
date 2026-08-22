from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fprint.workflow import (
    assert_prelock_database,
    build_forecast_manifest,
    build_threshold_artifact,
    fold_paths,
    initialize_fold,
    lock_privileged_forecasts,
    lock_zero_score_forecasts,
    mark_signature_scored,
    mark_test_scored,
)


def make_database(path: Path, rows: tuple[tuple[str, str], ...] = ()) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE records(record_id TEXT PRIMARY KEY, corpus TEXT NOT NULL);
        CREATE TABLE scores(record_id TEXT NOT NULL);
        """
    )
    connection.executemany("INSERT INTO records VALUES(?,?)", rows)
    connection.commit()
    connection.close()


def add_score(path: Path, record_id: str) -> None:
    connection = sqlite3.connect(path)
    connection.execute("INSERT INTO scores VALUES(?)", (record_id,))
    connection.commit()
    connection.close()


def calibration() -> dict:
    retained = [(f"r{i}", f"{i:064x}") for i in range(10_000)]
    scores = {"detector": {record_id: i / 10_000 for i, (record_id, _) in enumerate(retained)}}
    return build_threshold_artifact(retained, scores)


def manifest(paths, suffix: str = "a") -> dict:
    artifacts = {}
    for name, content in (
        (f"features-{suffix}.json", suffix),
        (f"profiles-{suffix}.json", "profile"),
        (f"ids-{suffix}.json", "ids"),
        (f"forecasts-{suffix}.json", "forecasts"),
        (f"uncertainty-{suffix}.json", "uncertainty"),
    ):
        path = (paths.root / name).resolve()
        path.write_text(content, encoding="utf-8")
        artifacts[name] = {str(path): hashlib.sha256(content.encode()).hexdigest()}
    return build_forecast_manifest(
        paths=paths,
        data_ids={"source": ["s1"], "target": ["t1"]},
        draw_ids={"draw:0:size:1": ["t1"]},
        panel_revisions={
            "detector": {
                "model_revision": "1" * 40,
                "tokenizer_revision": "2" * 40,
            }
        },
        thresholds=calibration(),
        selected_c={"main": 0.1},
        feature_artifacts=artifacts[f"features-{suffix}.json"],
        profile_artifacts=artifacts[f"profiles-{suffix}.json"],
        id_artifacts=artifacts[f"ids-{suffix}.json"],
        forecast_artifacts=artifacts[f"forecasts-{suffix}.json"],
        uncertainty_artifacts=artifacts[f"uncertainty-{suffix}.json"],
        code_commit="c" * 40,
    )


class WorkflowTests(unittest.TestCase):
    def test_fold_database_paths_are_target_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            left = fold_paths(Path(directory), "pmc")
            right = fold_paths(Path(directory), "gutenberg")
            self.assertNotEqual(left.database, right.database)
            self.assertEqual(left.database.name, "fprint.sqlite3")
            with self.assertRaises(ValueError):
                fold_paths(Path(directory), "../escape")

    def test_prelock_database_rejects_any_target_corpus_score(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "fold.sqlite3"
            make_database(database, (("s", "source"), ("t", "target")))
            add_score(database, "s")
            assert_prelock_database(database, "target", ("source",))
            add_score(database, "t")
            with self.assertRaises(RuntimeError):
                assert_prelock_database(database, "target", ("source",))

    def test_threshold_artifact_binds_retained_raid_records(self):
        first = calibration()
        retained = [(f"r{i}", f"{i:064x}") for i in range(10_000)]
        retained[0] = ("r0", "f" * 64)
        scores = {"detector": {record_id: i / 10_000 for i, (record_id, _) in enumerate(retained)}}
        second = build_threshold_artifact(retained, scores)
        self.assertEqual(set(first["fprs"]), {"0.05", "0.01"})
        self.assertNotEqual(first["retained_raid_sha256"], second["retained_raid_sha256"])

    def test_manifest_binds_database_draws_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = initialize_fold(Path(directory), "target")
            make_database(paths.database)
            first = manifest(paths)
            changed = build_forecast_manifest(
                paths=paths,
                data_ids={"source": ["s1"], "target": ["t1", "t2"]},
                draw_ids={"draw:0:size:1": ["t2"]},
                panel_revisions=first["panel_revisions"],
                thresholds=first["thresholds"],
                selected_c=first["selected_c"],
                feature_artifacts=first["feature_artifacts"],
                profile_artifacts=first["profile_artifacts"],
                id_artifacts=first["id_artifacts"],
                forecast_artifacts=first["forecast_artifacts"],
                uncertainty_artifacts=first["uncertainty_artifacts"],
                code_commit=first["code_commit"],
            )
            self.assertNotEqual(first["data_ids_sha256"], changed["data_ids_sha256"])
            self.assertNotEqual(first["draw_ids_sha256"], changed["draw_ids_sha256"])
            broken = dict(first)
            broken["thresholds_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                lock_zero_score_forecasts(paths, broken, {}, ("source",))

    def test_all_zero_and_privileged_locks_are_global_barriers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = ("a", "b")
            paths = {target: initialize_fold(root, target) for target in targets}
            for item in paths.values():
                make_database(item.database)
            zero_manifests = {target: manifest(paths[target]) for target in targets}
            lock_zero_score_forecasts(paths["a"], zero_manifests["a"], {}, ("source",))
            with self.assertRaises(RuntimeError):
                mark_signature_scored(root, targets, "a")
            lock_zero_score_forecasts(paths["b"], zero_manifests["b"], {}, ("source",))
            mark_signature_scored(root, targets, "a")
            mark_signature_scored(root, targets, "b")

            lock_privileged_forecasts(paths["a"], manifest(paths["a"]), {})
            with self.assertRaises(RuntimeError):
                mark_test_scored(root, targets, "a")
            lock_privileged_forecasts(paths["b"], manifest(paths["b"]), {})
            mark_test_scored(root, targets, "a")

    def test_zero_lock_rejects_tampered_builder_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = initialize_fold(Path(directory), "target")
            make_database(paths.database)
            built = manifest(paths)
            feature_path = Path(next(iter(built["feature_artifacts"])))
            feature_path.write_text("tampered", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                lock_zero_score_forecasts(paths, built, {}, ("source",))


if __name__ == "__main__":
    unittest.main()
