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
    lock_forecasts, make_probe_triplet, storage_root, verify_lock,
)
from .detectors import SPECS, build_adapter, validate_labeled_pilot, validate_specs
from .conformance import (
    evaluate_fault_audit, import_score_table, prepare_fault_audit, score_fault_audit,
)
from .final_evaluation import run_final_evaluation
from .fingerprint_geometry import write_fingerprint_geometry
from .forecasting import build_zero_forecasts
from .privileged import (
    build_privileged_comparator, build_privileged_plan, verify_privileged_plan,
)
from .workflow import (
    assert_all_target_locks, assert_target_score_allowed, build_threshold_artifact,
    fold_paths, initialize_fold, mark_signature_scored, mark_test_scored,
)
from .deferral import (
    ENDPOINT_ROLES, PROBES as DEFERRAL_PROBES, DeferralPaths,
    assemble_evaluation_rows, authorize_final_stage, build_conditional_worklist,
    build_reflow_variants, calibrate_radar_threshold, export_manual_audit,
    import_canonical_scores, import_generation_outputs, import_manual_audit,
    lock_human_token_panels, prepare_generation_requests, prepare_pilot_manifest,
    read_canonical_scores, validate_mage_effective_input_hashes,
)
from .deferral_evaluation import evaluate_pilot as evaluate_deferral_rows


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


def analyze_geometry(args: argparse.Namespace) -> None:
    report = write_fingerprint_geometry(args.storage_root, args.evaluation, args.output_dir)
    identification = report["leave_one_corpus_out_detector_identification"]
    print(
        "Fingerprint geometry written; cosine detector identification "
        f"{identification['cosine_accuracy']:.1%}."
    )


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


def prepare_conformance(args: argparse.Namespace) -> None:
    manifest = prepare_fault_audit(args.source_root, args.audit_root, args.config, args.evaluation)
    print(
        f"Locked {len(manifest['triplet_ids'])} discovery triplets and "
        f"{len(manifest['confirmation_candidate_ids'])} prospective candidates at {args.audit_root.resolve()}"
    )


def score_conformance(args: argparse.Namespace) -> None:
    if args.import_score_table:
        print(f"Imported {import_score_table(args.audit_root, args.import_score_table)} canonical score rows.")
        return
    if not args.endpoint or not args.fault:
        raise ValueError("Provide --endpoint and --fault, or --import-score-table")
    counts = score_fault_audit(
        args.audit_root, args.endpoint, args.fault,
        device=args.device, mage_repo=args.mage_repo, source_kind=args.source_kind,
    )
    print("Fault-audit scoring: " + ", ".join(f"{key}={value}" for key, value in counts.items()))


def evaluate_conformance(args: argparse.Namespace) -> None:
    report = evaluate_fault_audit(args.audit_root, args.output_dir)
    gate = report["success_gates"]
    print(
        f"Fault audit evaluated: claim={gate['permitted_primary_claim']}; "
        f"artifacts={report['artifacts']}"
    )


def _json_payload(path: Path) -> object:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _rows_payload(path: Path) -> list[dict[str, object]]:
    if path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    payload = _json_payload(path)
    if isinstance(payload, dict):
        payload = payload.get("rows", payload.get("outputs", payload.get("records")))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON row list or CSV: {path}")
    return [dict(row) for row in payload]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_binding_path(paths: DeferralPaths) -> Path:
    return paths.root / "locks" / "protocol_binding.json"


def _protocol_binding(paths: DeferralPaths, config: Path) -> dict[str, object]:
    files = {
        "config": config.resolve(),
        "deferral": Path(__file__).with_name("deferral.py").resolve(),
        "deferral_evaluation": Path(__file__).with_name("deferral_evaluation.py").resolve(),
        "core": Path(__file__).with_name("core.py").resolve(),
        "detectors": Path(__file__).with_name("detectors.py").resolve(),
    }
    return {
        "stage": "selective_deferral_protocol_binding",
        "pilot_lock_sha256": verify_lock(paths.lock)["sha256"],
        "files": {name: {"path": str(path), "sha256": _file_digest(path)} for name, path in files.items()},
    }


def _verify_deferral_protocol(paths: DeferralPaths) -> dict[str, object]:
    binding_path = _protocol_binding_path(paths)
    if not binding_path.exists():
        raise RuntimeError("Selective-deferral protocol binding is missing")
    envelope = verify_lock(binding_path)
    payload = envelope["payload"]
    if payload.get("pilot_lock_sha256") != verify_lock(paths.lock)["sha256"]:
        raise RuntimeError("Protocol binding references a different pilot lock")
    for row in payload.get("files", {}).values():
        path = Path(row["path"])
        if not path.exists() or _file_digest(path) != row["sha256"]:
            raise RuntimeError(f"Protocol-bound file changed: {path}")
    return envelope


def prepare_deferral(args: argparse.Namespace) -> None:
    config = _json_payload(args.config)
    if not isinstance(config, dict) or config.get("study_stage") != "pilot_only":
        raise ValueError("Deferral config must declare study_stage=pilot_only")
    paths = DeferralPaths.from_root(args.study_root)
    token_counts = _json_payload(args.human_token_counts)
    if not isinstance(token_counts, dict):
        raise ValueError("Human token counts must be a record-keyed JSON object")
    manifest = prepare_pilot_manifest(
        args.records, paths,
        calibration_cap=int(config["calibration"]["human_groups_total"]),
        pilot_cap=int(config["pilot"]["human_groups_total"]),
        seed=int(config["seed"]),
        endpoint_revisions={endpoint: SPECS[endpoint].revision for endpoint in ENDPOINT_ROLES},
        candidate_token_counts=token_counts,
        token_cap=int(config["common_token_ceiling"]),
    )
    selected_ids = {str(row["record_id"]) for row in manifest["pilot"]}
    lock_human_token_panels(
        paths, {record_id: counts for record_id, counts in token_counts.items() if record_id in selected_ids},
        cap=int(config["common_token_ceiling"]),
    )
    generation = _json_payload(args.generation_spec)
    topics = _json_payload(args.topic_map)
    if not isinstance(generation, dict) or not isinstance(topics, dict):
        raise ValueError("Generation specification and topic map must be JSON objects")
    requests = prepare_generation_requests(
        paths, topics,
        generator_families=generation.get("generator_families"),
        prompt_template=str(generation.get("prompt_template", "")),
        decoding=generation.get("decoding"),
        seed=int(generation.get("seed", config["seed"])),
        retry=int(generation.get("retry", 0)),
        target_length=int(generation["target_length"]),
    )
    binding_path = _protocol_binding_path(paths)
    lock_forecasts(binding_path, _protocol_binding(paths, args.config))
    print(f"Locked {len(manifest['calibration'])} calibration humans, {len(manifest['pilot'])} pilot humans, and {len(requests)} AI requests.")


def import_deferral_panel(args: argparse.Namespace) -> None:
    paths = DeferralPaths.from_root(args.study_root)
    _verify_deferral_protocol(paths)
    token_counts = _json_payload(args.token_counts) if args.token_counts else None
    panels = import_generation_outputs(paths, args.outputs, token_counts=token_counts, token_cap=460)
    print(f"Locked {len(panels)} AI panels.")


def validate_deferral(args: argparse.Namespace) -> None:
    paths = DeferralPaths.from_root(args.study_root)
    _verify_deferral_protocol(paths)
    if args.judgments:
        result = import_manual_audit(paths, args.judgments, probe=args.probe, count=args.count, minimum_valid=args.minimum_valid)
        print(f"Locked {args.probe} manual validation: {result['valid']}/{result['count']} valid.")
    else:
        if not args.text_table:
            raise ValueError("--text-table is required when exporting manual-audit examples")
        rows = export_manual_audit(paths, probe=args.probe, count=args.count, texts=args.text_table)
        print(f"Exported {len(rows)} {args.probe} audit rows to {paths.manual_audit_csv_for(args.probe)}")


def calibrate_deferral(args: argparse.Namespace) -> None:
    paths = DeferralPaths.from_root(args.study_root)
    _verify_deferral_protocol(paths)
    payload = calibrate_radar_threshold(paths, read_canonical_scores(args.score_table))
    print(f"Locked RADAR 5% threshold at {payload['threshold']:.12g} from {payload['reference_count']} humans.")


def score_deferral_originals(args: argparse.Namespace) -> None:
    paths = DeferralPaths.from_root(args.study_root)
    _verify_deferral_protocol(paths)
    imported = import_canonical_scores(args.score_table, paths)
    threshold = verify_lock(paths.threshold_lock)["payload"]["threshold"]
    originals = {
        (row.record_id, row.endpoint): row
        for row in imported
        if row.variant_id == "original" and row.endpoint == "radar_roberta_large__vicuna7b_training"
    }
    work = build_conditional_worklist(
        paths, originals,
        thresholds={"radar_roberta_large__vicuna7b_training": float(threshold)},
        sentinel_per_corpus_label=25,
    )
    print(f"Imported {len(imported)} score rows and wrote {len(work)} conditional scoring requests to {paths.worklist}.")


def score_deferral_positives(args: argparse.Namespace) -> None:
    paths = DeferralPaths.from_root(args.study_root)
    _verify_deferral_protocol(paths)
    rows = import_canonical_scores(args.score_table, paths)
    print(f"Canonical score cache now contains {len(rows)} rows.")


def evaluate_deferral(args: argparse.Namespace) -> None:
    paths = DeferralPaths.from_root(args.study_root)
    protocol = _verify_deferral_protocol(paths)
    text_rows = _rows_payload(args.text_table)
    rows = assemble_evaluation_rows(paths, text_rows)
    manual = []
    for probe in DEFERRAL_PROBES:
        lock = paths.manual_audit_lock_for(probe)
        if not lock.exists():
            raise RuntimeError(f"Manual audit lock is missing for {probe}")
        item = verify_lock(lock)["payload"]
        if item.get("count") != 300 or item.get("minimum_valid") != 297:
            raise RuntimeError(f"Manual audit policy mismatch for {probe}")
        manual.append(item)
    mage_invariant = True
    for row in rows:
        text = str(row["text"])
        variants = {variant.variant_id: variant.text for variant in build_reflow_variants(text)}
        validate_mage_effective_input_hashes(text, variants)
    validation = {
        "manual_gate": all(item["valid"] >= item["minimum_valid"] for item in manual),
        "automated_gate": True,
        "mage_gate": mage_invariant,
    }
    output_dir = args.output_dir or paths.results
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "pilot_evaluation.json"
    report = evaluate_deferral_rows(
        rows, validation_summary=validation,
        bootstrap_replicates=10000,
        seed=20260824, output_path=report_path,
    )
    gate_payload = {
        "stage": "selective_deferral_pilot_gate",
        "protocol_binding_sha256": protocol["sha256"],
        "report_sha256": _file_digest(report_path),
        "passed": bool(report["gates"]["passed"]),
        "failures": report["gates"]["failures"],
    }
    lock_forecasts(paths.root / "locks" / "pilot_gate.json", gate_payload)
    print(f"Pilot evaluated: passed={gate_payload['passed']}; report={report_path}")


def lock_deferral_final(args: argparse.Namespace) -> None:
    paths = DeferralPaths.from_root(args.study_root)
    _verify_deferral_protocol(paths)
    gate = verify_lock(paths.root / "locks" / "pilot_gate.json")["payload"]
    if gate.get("passed") is not True:
        raise RuntimeError("Final protocol cannot lock because the pilot did not pass")
    digest = authorize_final_stage(paths, {
        "status": "pilot_passed", "passed": True,
        "pilot_gate_sha256": verify_lock(paths.root / "locks" / "pilot_gate.json")["sha256"],
        "radar_specific_protocol_sha256": _file_digest(args.protocol),
        "detector_general_claim_blocked": True,
    })
    print(f"Locked RADAR-specific final protocol authorization: {digest}")


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
    geometry_parser = sub.add_parser("analyze-fingerprint-geometry")
    geometry_parser.add_argument("--storage-root", type=Path, required=True)
    geometry_parser.add_argument("--evaluation", type=Path, required=True)
    geometry_parser.add_argument("--output-dir", type=Path, required=True)
    geometry_parser.set_defaults(func=analyze_geometry)
    fault_prepare = sub.add_parser(
        "prepare-fault-audit",
        help="Create and hash-lock the isolated behavioral conformance challenge",
    )
    fault_prepare.add_argument("--source-root", type=Path, required=True)
    fault_prepare.add_argument("--audit-root", type=Path, required=True)
    fault_prepare.add_argument("--config", type=Path, default=Path("fault_audit_config.json"))
    fault_prepare.add_argument("--evaluation", type=Path, required=True)
    fault_prepare.set_defaults(func=prepare_conformance)
    fault_score = sub.add_parser(
        "score-fault-audit",
        help="Resumably score inference faults or import a canonical score table",
    )
    fault_score.add_argument("--audit-root", type=Path, required=True)
    fault_score.add_argument("--endpoint", choices=sorted(SPECS))
    fault_score.add_argument("--fault")
    fault_score.add_argument("--device", type=int, default=0)
    fault_score.add_argument("--mage-repo")
    fault_score.add_argument(
        "--source-kind", choices=("all", "discovery", "confirmation_candidate"), default="all",
    )
    fault_score.add_argument("--import-score-table", type=Path)
    fault_score.set_defaults(func=score_conformance)
    fault_evaluate = sub.add_parser(
        "evaluate-fault-audit",
        help="Run nested held-out-corpus change detection and abstaining diagnosis",
    )
    fault_evaluate.add_argument("--audit-root", type=Path, required=True)
    fault_evaluate.add_argument("--output-dir", type=Path)
    fault_evaluate.set_defaults(func=evaluate_conformance)
    deferral_prepare = sub.add_parser(
        "prepare-deferral-pilot",
        help="Select, validate, and lock the isolated human/AI deferral pilot",
    )
    deferral_prepare.add_argument("--records", type=Path, required=True, help="Canonical human CSV")
    deferral_prepare.add_argument("--study-root", type=Path, required=True)
    deferral_prepare.add_argument("--config", type=Path, default=Path("deferral_config.json"))
    deferral_prepare.add_argument("--topic-map", type=Path, required=True)
    deferral_prepare.add_argument("--generation-spec", type=Path, required=True)
    deferral_prepare.add_argument("--human-token-counts", type=Path, required=True)
    deferral_prepare.set_defaults(func=prepare_deferral)
    deferral_import = sub.add_parser(
        "import-deferral-ai-panel",
        help="Import outputs matching the locked provider-neutral generation requests",
    )
    deferral_import.add_argument("--study-root", type=Path, required=True)
    deferral_import.add_argument("--outputs", type=Path, required=True)
    deferral_import.add_argument("--token-counts", type=Path)
    deferral_import.set_defaults(func=import_deferral_panel)
    deferral_validate = sub.add_parser(
        "validate-deferral-probes",
        help="Export or lock one preregistered manual probe audit",
    )
    deferral_validate.add_argument("--study-root", type=Path, required=True)
    deferral_validate.add_argument("--probe", choices=DEFERRAL_PROBES, required=True)
    deferral_validate.add_argument("--judgments", type=Path)
    deferral_validate.add_argument("--text-table", type=Path)
    deferral_validate.add_argument("--count", type=int, default=300)
    deferral_validate.add_argument("--minimum-valid", type=int, default=297)
    deferral_validate.set_defaults(func=validate_deferral)
    deferral_calibrate = sub.add_parser(
        "calibrate-deferral-thresholds",
        help="Lock the RADAR 5%% threshold from exactly 2,000 calibration humans",
    )
    deferral_calibrate.add_argument("--study-root", type=Path, required=True)
    deferral_calibrate.add_argument("--score-table", type=Path, required=True)
    deferral_calibrate.set_defaults(func=calibrate_deferral)
    deferral_originals = sub.add_parser(
        "score-deferral-originals",
        help="Import original RADAR scores and emit the conditional positive-only worklist",
    )
    deferral_originals.add_argument("--study-root", type=Path, required=True)
    deferral_originals.add_argument("--score-table", type=Path, required=True)
    deferral_originals.set_defaults(func=score_deferral_originals)
    deferral_positives = sub.add_parser(
        "score-deferral-positives",
        help="Import completed RADAR reflow, MAGE, LogRank, and sentinel scores",
    )
    deferral_positives.add_argument("--study-root", type=Path, required=True)
    deferral_positives.add_argument("--score-table", type=Path, required=True)
    deferral_positives.set_defaults(func=score_deferral_positives)
    deferral_evaluate = sub.add_parser(
        "evaluate-deferral-pilot",
        help="Run frozen leave-one-corpus-out triage evaluation and lock its gate",
    )
    deferral_evaluate.add_argument("--study-root", type=Path, required=True)
    deferral_evaluate.add_argument("--text-table", type=Path, required=True)
    deferral_evaluate.add_argument("--output-dir", type=Path)
    deferral_evaluate.set_defaults(func=evaluate_deferral)
    deferral_final = sub.add_parser(
        "lock-deferral-final",
        help="After a passed pilot, lock a RADAR-specific final protocol without scoring it",
    )
    deferral_final.add_argument("--study-root", type=Path, required=True)
    deferral_final.add_argument("--protocol", type=Path, required=True)
    deferral_final.set_defaults(func=lock_deferral_final)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
