from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median

from .core import (
    PROBES, STUDY_CORPORA, TARGET_CORPORA, StudyDB, TextRecord,
    assign_grouped_partitions, deduplicate, grouping_key,
    make_probe_triplet, storage_root,
)
from .detectors import SPECS, build_adapter, validate_labeled_pilot, validate_specs
from .final_evaluation import run_final_evaluation
from .forecasting import build_zero_forecasts
from .privileged import (
    build_privileged_comparator, build_privileged_plan, verify_privileged_plan,
)
from .workflow import (
    assert_all_target_locks, assert_target_score_allowed, build_threshold_artifact,
    fold_paths, initialize_fold, mark_signature_scored, mark_test_scored,
)


def _parse_corpus(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use corpus_name=path.csv")
    name, path = value.split("=", 1)
    return name.casefold(), Path(path)


def _read_corpora(inputs: list[tuple[str, Path]]) -> list[TextRecord]:
    records: list[TextRecord] = []
    for corpus, path in inputs:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for order, row in enumerate(csv.DictReader(handle)):
                if not row.get("text"):
                    continue
                raw_id = str(row.get("record_id") or "").strip()
                record_id = f"{corpus}:{raw_id or order}"
                stratum_column = {
                    "asap_aes": "essay_set",
                    "bawe": "writer_stratum",
                }.get(corpus)
                stratum = str(row.get(stratum_column) or "").strip() if stratum_column else ""
                if stratum_column and not stratum:
                    raise ValueError(f"{corpus} requires {stratum_column} for stratification")
                records.append(TextRecord(record_id, corpus, row["text"], grouping_key(corpus, row), order, False, stratum))
    return records


def _read_threshold_reference(path: Path) -> list[TextRecord]:
    records = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for order, row in enumerate(csv.DictReader(handle)):
            if row.get("text"):
                raw_id = str(row.get("record_id") or "").strip()
                record_id = f"raid_threshold:{raw_id or order}"
                records.append(TextRecord(record_id, "raid_threshold", row["text"], record_id, order, True))
    if len(records) < 10000:
        raise ValueError("Independent RAID human threshold reference requires at least 10,000 texts")
    return records


def prepare(args: argparse.Namespace) -> None:
    root = storage_root()
    root.mkdir(parents=True, exist_ok=True)
    clean, audit = deduplicate(_read_corpora(args.corpus) + _read_threshold_reference(args.threshold_reference))
    evaluation = [record for record in clean if not record.is_threshold_reference]
    references = [record for record in clean if record.is_threshold_reference]
    if len(references) < 10000:
        raise ValueError("Independent RAID human threshold reference has fewer than 10,000 texts after deduplication")
    partitions = assign_grouped_partitions(evaluation, args.seed)
    partitions.update({record.record_id: "threshold_reference" for record in references})
    candidate_counts = {}
    triplets = []
    anchor_records = {}
    for record in evaluation:
        if partitions[record.record_id] != "anchor_candidates":
            continue
        anchor_records.setdefault((record.corpus, record.group_id), record)
    for record in anchor_records.values():
        for probe in PROBES:
            key = (record.corpus, probe)
            if candidate_counts.get(key, 0) >= args.probe_candidates:
                continue
            triplet = make_probe_triplet(probe, record.text, record.record_id)
            if triplet:
                triplets.append((record.record_id, record.corpus, triplet))
                candidate_counts[key] = candidate_counts.get(key, 0) + 1
    database_paths = [root / "state" / "fprint.sqlite3"]
    for target in TARGET_CORPORA:
        database_paths.append(initialize_fold(root, target).database)
    for database_path in database_paths:
        db = StudyDB(database_path)
        db.add_records(clean, partitions)
        db.add_probe_triplets(triplets)
        db.close()
    audit_path = root / "state" / "dedup_audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as handle:
        for row in audit:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Prepared {len(clean)} records; audit: {audit_path}")


def build_zero(args: argparse.Namespace) -> None:
    outputs = build_zero_forecasts(
        storage_root(), args.target_corpus, args.threshold_artifact,
        args.output_dir, args.admitted_detectors,
    )
    print("Built zero-score artifacts: " + ", ".join(f"{name}={path}" for name, path in outputs.items()))


def _score(
    args: argparse.Namespace,
    partitions: list[str],
    *,
    detector_ids: tuple[str, ...] | None = None,
    database: Path | None = None,
    include_corpus: str | None = None,
    include_corpora: tuple[str, ...] | None = None,
    exclude_corpus: str | None = None,
    allowed_ids: set[str] | None = None,
) -> tuple[int, int]:
    db = StudyDB(database or storage_root() / "state" / "fprint.sqlite3")
    detector_ids = detector_ids or (args.detector,)
    adapters = {
        detector: build_adapter(detector, args.device, args.mage_repo)
        for detector in detector_ids
    }
    completed, skipped = 0, 0
    for record_id, text in db.records(
        partitions,
        include_corpus=include_corpus,
        include_corpora=include_corpora,
        exclude_corpus=exclude_corpus,
    ):
        if allowed_ids is not None and record_id not in allowed_ids:
            continue
        for detector, adapter in adapters.items():
            if db.has_score(record_id, detector):
                skipped += 1
                continue
            try:
                result = adapter.score(text)
            except Exception as error:
                spec = SPECS[detector]
                payload = {
                    "failure": f"{type(error).__name__}: {error}",
                    "native_score": None, "canonical_ai_score": None,
                    "input_token_count": adapter.token_count(text), "effective_token_count": 0,
                    "max_tokens": spec.max_tokens, "truncated": False, "runtime_ms": 0.0,
                }
            else:
                payload = asdict(result)
            db.add_score(record_id, detector, SPECS[detector].dependency_group, payload)
            completed += 1
    db.close()
    print(f"Scored {completed}; resumed/skipped {skipped}.")
    return completed, skipped


def _score_probes(
    args: argparse.Namespace,
    database: Path,
    include_corpora: tuple[str, ...] | None = None,
    detector_ids: tuple[str, ...] | None = None,
) -> None:
    db = StudyDB(database)
    detector_ids = detector_ids or (args.detector,)
    adapters = {
        detector: build_adapter(detector, args.device, args.mage_repo)
        for detector in detector_ids
    }
    completed, rejected, skipped = 0, 0, 0
    for triplet_id, record_id, probe, original, low, high in db.probe_triplets(
        include_corpora=include_corpora,
    ):
        for detector, adapter in adapters.items():
            spec = SPECS[detector]
            capacity = min(spec.max_tokens - 32, 460)
            counts = tuple(adapter.token_count(text) for text in (original, low, high))
            fits = max(counts) <= capacity
            db.add_probe_token_check(triplet_id, detector, counts, fits)
            if not fits:
                rejected += 1
                continue
            for intensity, text in (("original", original), ("low", low), ("high", high)):
                variant_id = f"{probe}:{intensity}"
                if db.has_score(record_id, detector, variant_id):
                    skipped += 1
                    continue
                try:
                    payload = asdict(adapter.score(text))
                except Exception as error:
                    payload = {
                        "failure": f"{type(error).__name__}: {error}",
                        "native_score": None, "canonical_ai_score": None,
                        "input_token_count": adapter.token_count(text), "effective_token_count": 0,
                        "max_tokens": spec.max_tokens, "truncated": False, "runtime_ms": 0.0,
                    }
                db.add_score(record_id, detector, spec.dependency_group, payload, variant_id)
                completed += 1
    db.close()
    print(f"Probe scores {completed}; triplets rejected {rejected}; resumed/skipped {skipped}.")


def pilot(args: argparse.Namespace) -> None:
    validate_specs()
    root = storage_root()
    db = StudyDB(root / "state" / "fprint.sqlite3")
    humans = list(db.connection.execute(
        """WITH ranked AS (
             SELECT record_id,text,corpus,
                    ROW_NUMBER() OVER(PARTITION BY corpus ORDER BY record_id) position
             FROM records WHERE partition_name='technical_pilot'
           )
           SELECT record_id,text FROM ranked ORDER BY position,corpus LIMIT 50"""
    ))
    db.close()
    with args.ai_reference.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"record_id", "text"} <= set(reader.fieldnames or ()):
            raise ValueError("AI pilot reference requires record_id and text columns.")
        ais = sorted(
            ((row["record_id"].strip(), row["text"].strip()) for row in reader),
            key=lambda item: item[0],
        )[:50]
    if len(humans) != 50 or len(ais) != 50 or any(not value for row in ais for value in row):
        raise ValueError("Technical pilot requires 50 non-empty human and 50 non-empty AI passages.")

    output = root / "state" / "pilots" / f"{args.detector}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "detector_config": args.detector,
        "human_count": len(humans),
        "ai_count": len(ais),
        "ai_reference_sha256": hashlib.sha256(args.ai_reference.read_bytes()).hexdigest(),
        "admitted": False,
    }
    try:
        adapter = build_adapter(args.detector, args.device, args.mage_repo)
        results = []
        scores = {"human": [], "ai": []}
        repeats = {"human": [], "ai": []}
        for label, rows in (("human", humans), ("ai", ais)):
            for record_id, text in rows:
                first = adapter.score(text)
                if first.failure or first.truncated:
                    raise RuntimeError(f"{record_id} failed or truncated during the technical pilot.")
                if hasattr(adapter, "scorer"):
                    adapter.scorer.sequence.cache_clear()
                repeat = adapter.score(text)
                if repeat.failure or repeat.truncated:
                    raise RuntimeError(f"{record_id} failed or truncated during repeat inference.")
                scores[label].append(float(first.canonical_ai_score))
                repeats[label].append(float(repeat.canonical_ai_score))
                results.append({
                    "label": label,
                    "record_id": record_id,
                    "first": asdict(first),
                    "repeat_score": repeat.canonical_ai_score,
                    "repeat_runtime_ms": repeat.runtime_ms,
                })
        runtimes = [
            float(result["first"]["runtime_ms"])
            for result in results
        ]
        report.update({
            "max_repeat_score_difference": max(
                abs(first - repeat)
                for first, repeat in zip(
                    [*scores["human"], *scores["ai"]],
                    [*repeats["human"], *repeats["ai"]],
                )
            ),
            "human_score_median": median(scores["human"]),
            "ai_score_median": median(scores["ai"]),
            "runtime_median_ms": median(runtimes),
            "runtime_max_ms": max(runtimes),
            "records": results,
        })
        validate_labeled_pilot(
            args.detector, scores["human"], scores["ai"],
            repeats["human"], repeats["ai"],
        )
        report["admitted"] = True
    except Exception as error:
        report["failure"] = f"{type(error).__name__}: {error}"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(output)
        raise
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(f"Pilot admitted {args.detector}: {output}")


def score_source(args: argparse.Namespace) -> None:
    detector_ids = tuple(dict.fromkeys((args.detector, *args.paired_detector)))
    if len({SPECS[detector].dependency_group for detector in detector_ids}) != 1:
        raise ValueError("Paired source detectors must share one inference backend")
    root = storage_root()
    master = root / "state" / "fprint.sqlite3"
    _score(
        args, ["source_summary", "source_model"], database=master,
        include_corpora=STUDY_CORPORA, detector_ids=detector_ids,
    )
    _score_probes(args, master, STUDY_CORPORA, detector_ids)
    paths = fold_paths(storage_root(), args.target_corpus)
    db = StudyDB(paths.database)
    for detector in detector_ids:
        db.import_source_results(master, detector, args.target_corpus)
    db.close()
    print(f"Imported leakage-safe source cache into fold: {args.target_corpus}")


def calibrate(args: argparse.Namespace) -> None:
    _score(args, ["threshold_reference"])


def threshold_artifact(args: argparse.Namespace) -> None:
    db = StudyDB(storage_root() / "state" / "fprint.sqlite3")
    retained, scores = db.threshold_inputs(args.detectors)
    db.close()
    artifact = build_threshold_artifact(retained, scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, sort_keys=True, indent=2), encoding="utf-8")
    print(f"Wrote threshold artifact: {args.output}")


def score_target(args: argparse.Namespace) -> None:
    root = storage_root()
    assert_target_score_allowed(root, TARGET_CORPORA, args.target_corpus, args.partition)
    paths = fold_paths(root, args.target_corpus)
    plan = verify_privileged_plan(root, args.target_corpus)
    if set(args.admitted_detectors) != set(plan["admitted_detectors"]):
        raise ValueError("Admitted detector panel does not match the locked privileged plan")
    detector_ids = tuple(dict.fromkeys((args.detector, *args.paired_detector)))
    if len({SPECS[detector].dependency_group for detector in detector_ids}) != 1:
        raise ValueError("Paired target detectors must share one inference backend")
    if args.partition == "privileged_signature":
        allowed_ids = set(plan["sizes"]["250"])
        completed, skipped = _score(
            args, ["signature"], database=paths.database,
            include_corpus=args.target_corpus, allowed_ids=allowed_ids,
            detector_ids=detector_ids,
        )
        if completed + skipped != 250 * len(detector_ids):
            raise RuntimeError("Not all 250 privileged target IDs belong to the target signature partition")
        db = StudyDB(paths.database)
        missing = db.missing_partition_scores(
            "signature", args.target_corpus, args.admitted_detectors, allowed_ids,
        )
        db.close()
        if missing:
            print(f"Privileged signature awaiting admitted detectors: {missing}")
        else:
            mark_signature_scored(root, TARGET_CORPORA, args.target_corpus)
    else:
        _score(
            args, ["test"], database=paths.database, include_corpus=args.target_corpus,
            detector_ids=detector_ids,
        )
        db = StudyDB(paths.database)
        missing = db.missing_partition_scores("test", args.target_corpus, args.admitted_detectors)
        db.close()
        if missing:
            print(f"Target test awaiting admitted detectors: {missing}")
        else:
            mark_test_scored(root, TARGET_CORPORA, args.target_corpus)


def prepare_privileged(args: argparse.Namespace) -> None:
    print(f"Locked privileged target plan: {build_privileged_plan(storage_root(), args.target_corpus)}")


def build_privileged(args: argparse.Namespace) -> None:
    print(f"Locked privileged comparator: {build_privileged_comparator(storage_root(), args.target_corpus)}")


def evaluate(args: argparse.Namespace) -> None:
    run_final_evaluation(storage_root(), args.output_dir)
    print(f"Wrote final evaluation: {args.output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fprint", description="Fixed-threshold detector FPR forecasting")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--corpus", type=_parse_corpus, action="append", required=True)
    prepare_parser.add_argument("--threshold-reference", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=20260729)
    prepare_parser.add_argument("--probe-candidates", type=int, default=200)
    prepare_parser.set_defaults(func=prepare)
    def scoring_arguments(stage):
        stage.add_argument("--detector", choices=sorted(SPECS), required=True)
        stage.add_argument("--device", type=int, default=0)
        stage.add_argument("--mage-repo")
    pilot_parser = sub.add_parser("pilot")
    scoring_arguments(pilot_parser)
    pilot_parser.add_argument("--ai-reference", type=Path, required=True)
    pilot_parser.set_defaults(func=pilot)
    source_parser = sub.add_parser("score-source")
    scoring_arguments(source_parser)
    source_parser.add_argument("--paired-detector", action="append", choices=sorted(SPECS), default=[])
    source_parser.add_argument("--target-corpus", choices=TARGET_CORPORA, required=True)
    source_parser.set_defaults(func=score_source)
    calibrate_parser = sub.add_parser("calibrate")
    scoring_arguments(calibrate_parser)
    calibrate_parser.set_defaults(func=calibrate)
    artifact_parser = sub.add_parser("threshold-artifact")
    artifact_parser.add_argument("--detectors", nargs="+", choices=sorted(SPECS), required=True)
    artifact_parser.add_argument("--output", type=Path, required=True)
    artifact_parser.set_defaults(func=threshold_artifact)
    build_parser_ = sub.add_parser("build-zero-forecasts")
    build_parser_.add_argument("--target-corpus", choices=TARGET_CORPORA, required=True)
    build_parser_.add_argument("--threshold-artifact", type=Path, required=True)
    build_parser_.add_argument("--output-dir", type=Path)
    build_parser_.add_argument(
        "--admitted-detectors", nargs="+", choices=sorted(SPECS),
        default=sorted(SPECS),
    )
    build_parser_.set_defaults(func=build_zero)
    plan_parser = sub.add_parser("prepare-privileged")
    plan_parser.add_argument("--target-corpus", choices=TARGET_CORPORA, required=True)
    plan_parser.set_defaults(func=prepare_privileged)
    target_parser = sub.add_parser("score-target")
    target_parser.add_argument("--target-corpus", choices=TARGET_CORPORA, required=True)
    target_parser.add_argument("--partition", choices=("privileged_signature", "test"), required=True)
    target_parser.add_argument("--admitted-detectors", nargs="+", choices=sorted(SPECS), required=True)
    scoring_arguments(target_parser)
    target_parser.add_argument("--paired-detector", action="append", choices=sorted(SPECS), default=[])
    target_parser.set_defaults(func=score_target)
    comparator_parser = sub.add_parser("build-privileged-comparator")
    comparator_parser.add_argument("--target-corpus", choices=TARGET_CORPORA, required=True)
    comparator_parser.set_defaults(func=build_privileged)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
