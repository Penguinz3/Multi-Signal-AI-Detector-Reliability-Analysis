from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from fprint.detectors import SPECS


def migrate(old_root: Path, new_root: Path, ai_reference: Path) -> dict[str, object]:
    old_db = old_root / "state" / "fprint.sqlite3"
    new_db = new_root / "state" / "fprint.sqlite3"
    ai_hash = hashlib.sha256(ai_reference.read_bytes()).hexdigest()
    pilot_source = old_root / "state" / "pilots"
    pilot_destination = new_root / "state" / "pilots"
    pilot_stage = new_root / "state" / "pilots.migration-tmp"
    output = new_root / "state" / "cache_migration.json"
    temporary = output.with_suffix(".json.tmp")
    if pilot_destination.exists() or pilot_stage.exists() or output.exists() or temporary.exists():
        raise RuntimeError("Destination migration artifacts already exist")
    pilot_paths = sorted(pilot_source.glob("*.json"))
    if {path.stem for path in pilot_paths} != set(SPECS):
        raise RuntimeError("Expected exactly one pilot report per admitted detector")
    for path in pilot_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("admitted") or report.get("ai_reference_sha256") != ai_hash:
            raise RuntimeError(f"Pilot is not reusable: {path.name}")
    connection = sqlite3.connect(new_db)
    connection.execute("ATTACH DATABASE ? AS olddb", (str(old_db),))
    try:
        counts = connection.execute(
            "SELECT (SELECT COUNT(*) FROM main.records), (SELECT COUNT(*) FROM olddb.records)"
        ).fetchone()
        deltas = connection.execute(
            """SELECT
               (SELECT COUNT(*) FROM (SELECT record_id FROM main.records EXCEPT SELECT record_id FROM olddb.records)),
               (SELECT COUNT(*) FROM (SELECT record_id FROM olddb.records EXCEPT SELECT record_id FROM main.records))"""
        ).fetchone()
        mismatches = connection.execute(
            """SELECT COUNT(*) FROM main.records n JOIN olddb.records o USING(record_id)
               WHERE n.text_hash<>o.text_hash OR n.group_id<>o.group_id OR n.corpus<>o.corpus"""
        ).fetchone()[0]
        stable_role_mismatches = connection.execute(
            """SELECT COUNT(*) FROM main.records n JOIN olddb.records o USING(record_id)
               WHERE o.partition_name IN ('technical_pilot','source_summary','source_model','threshold_reference')
                 AND n.partition_name<>o.partition_name"""
        ).fetchone()[0]
        probe_mismatches = connection.execute(
            """SELECT COUNT(*) FROM main.probe_triplets n JOIN olddb.probe_triplets o USING(triplet_id)
               WHERE n.record_id<>o.record_id OR n.probe<>o.probe OR n.original_text<>o.original_text
                 OR n.low_text<>o.low_text OR n.high_text<>o.high_text
                 OR n.low_intensity<>o.low_intensity OR n.high_intensity<>o.high_intensity"""
        ).fetchone()[0]
        if counts[0] != counts[1] or deltas != (0, 0) or mismatches or stable_role_mismatches or probe_mismatches:
            raise RuntimeError("Old and new prelock databases are not safe for exact cache reuse")
        if connection.execute("SELECT COUNT(*) FROM main.scores").fetchone()[0]:
            raise RuntimeError("Destination score cache is not empty")
        if connection.execute("SELECT COUNT(*) FROM main.probe_token_checks").fetchone()[0]:
            raise RuntimeError("Destination probe-check cache is not empty")
        pilot_stage.mkdir(parents=True)
        for path in pilot_paths:
            shutil.copy2(path, pilot_stage / path.name)
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """INSERT INTO main.scores SELECT s.* FROM olddb.scores s
                   JOIN main.records n USING(record_id)
                   WHERE s.variant_id='original'
                     AND n.partition_name IN ('threshold_reference','source_summary','source_model')"""
            )
            connection.execute(
                """INSERT OR IGNORE INTO main.scores SELECT s.* FROM olddb.scores s
                   JOIN main.probe_triplets p ON p.record_id=s.record_id
                   JOIN olddb.probe_triplets o ON o.triplet_id=p.triplet_id
                   WHERE s.variant_id IN (
                     p.probe||':original', p.probe||':low', p.probe||':high')"""
            )
            connection.execute(
                """INSERT INTO main.probe_token_checks SELECT c.*
                   FROM olddb.probe_token_checks c
                   JOIN main.probe_triplets p USING(triplet_id)"""
            )
            target_scores = connection.execute(
                """SELECT COUNT(*) FROM main.scores s JOIN main.records r USING(record_id)
                   WHERE r.partition_name IN ('signature','test')"""
            ).fetchone()[0]
            if target_scores:
                raise RuntimeError("Migration copied target signature/test scores")
            audit = {
                "schema_version": 1,
                "old_database": str(old_db.resolve()),
                "new_database": str(new_db.resolve()),
                "record_count": counts[0],
                "record_set_delta": list(deltas),
                "content_group_corpus_mismatches": mismatches,
                "stable_role_mismatches": stable_role_mismatches,
                "probe_content_mismatches": probe_mismatches,
                "copied_scores": connection.execute("SELECT COUNT(*) FROM main.scores").fetchone()[0],
                "copied_probe_checks": connection.execute(
                    "SELECT COUNT(*) FROM main.probe_token_checks"
                ).fetchone()[0],
                "target_signature_test_scores": target_scores,
                "copied_pilots": [path.name for path in pilot_paths],
                "ai_reference_sha256": ai_hash,
            }
            pilot_stage.replace(pilot_destination)
            temporary.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(output)
            connection.commit()
        except Exception:
            connection.rollback()
            if pilot_destination.exists():
                shutil.rmtree(pilot_destination)
            if output.exists():
                output.unlink()
            raise
    finally:
        connection.close()
        if pilot_stage.exists():
            shutil.rmtree(pilot_stage)
        if temporary.exists():
            temporary.unlink()
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--ai-reference", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.old_root, args.new_root, args.ai_reference), indent=2, sort_keys=True))
