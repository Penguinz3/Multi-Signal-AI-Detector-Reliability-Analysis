"""Prepare a score-blind completion screen and lock its complete-case panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from fprint.core import lock_forecasts, verify_lock
from fprint.deferral import (
    DEV_CORPORA,
    DeferralPaths,
    _sha256,
    lock_human_token_panels,
    verify_generation_lock,
    verify_pilot_lock,
)
from tools.run_deferral_generation import _is_complete_passage, _word_count, request_seed


REQUEST_FIELDS = (
    "request_id", "record_id", "corpus", "generator_family",
    "generator_revision", "prompt", "prompt_sha256", "seed", "retry",
    "target_length", "min_word_count", "max_word_count", "decoding",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{number}") from error
    return tuple(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_requests(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "decoding": json.dumps(row["decoding"], sort_keys=True)})


def _validate_checkpoint(
    rows: Sequence[Mapping[str, object]], requests: Mapping[str, Mapping[str, object]],
) -> None:
    seen = set()
    for row in rows:
        request_id = str(row.get("request_id", ""))
        if request_id in seen or request_id not in requests:
            raise RuntimeError(f"Invalid or duplicate checkpoint request: {request_id}")
        seen.add(request_id)
        request = requests[request_id]
        if (
            str(row.get("generator_family")) != str(request["generator_family"])
            or str(row.get("generator_revision")) != str(request["generator_revision"])
            or int(row.get("retry", -1)) != int(request["retry"])
            or int(row.get("target_length", -1)) != int(request["target_length"])
        ):
            raise RuntimeError(f"Checkpoint provenance mismatch: {request_id}")
        attempt = int(row.get("attempt", -1))
        if not 0 <= attempt <= int(request["retry"]) or int(row.get("seed", -1)) != request_seed(
            int(request["seed"]), request_id, attempt,
        ):
            raise RuntimeError(f"Checkpoint seed mismatch: {request_id}")
        text = str(row.get("text", ""))
        count = _word_count(text)
        if not int(request["min_word_count"]) <= count <= int(request["max_word_count"]) or not _is_complete_passage(text):
            raise RuntimeError(f"Checkpoint text violates its locked envelope: {request_id}")


def prepare_screen(source: Path, screen: Path) -> dict[str, object]:
    if screen.exists():
        raise FileExistsError(f"Completion-screen root already exists: {screen}")
    source_paths = DeferralPaths.from_root(source)
    pilot = verify_pilot_lock(source_paths)
    generation = verify_generation_lock(source_paths)
    checkpoint_path = source / "generation" / "checkpoint.jsonl"
    checkpoint = _jsonl(checkpoint_path)
    requests = {str(row["request_id"]): row for row in generation["payload"]["requests"]}
    _validate_checkpoint(checkpoint, requests)
    if source_paths.score_table.exists() or (source / "generation" / "accepted_outputs.csv").exists():
        raise RuntimeError("Completion screening must begin before detector scores or a final panel")
    if not 0 < len(checkpoint) < len(requests):
        raise RuntimeError("Source checkpoint is not an incomplete generation run")

    copies = (
        source_paths.manifest, source_paths.generation_json, source_paths.generation_csv,
        source_paths.lock, source_paths.human_token_lock, source_paths.generation_lock,
        source / "locks" / "protocol_binding.json", source / "locks" / "v4_amendment.json",
    )
    for path in copies:
        relative = path.relative_to(source)
        destination = screen / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    copied_checkpoint = screen / "generation" / "checkpoint.jsonl"
    copied_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint_path, copied_checkpoint)
    if _file_sha256(copied_checkpoint) != _file_sha256(checkpoint_path):
        raise RuntimeError("Completion-screen checkpoint copy is not byte-identical")

    payload = {
        "stage": "score_blind_generation_completion_screen",
        "policy": "attempt_each_unfinished_locked_request_once_and_log_all_exhausted_requests",
        "source_root": str(source.resolve()),
        "source_pilot_lock_sha256": pilot["sha256"],
        "source_generation_lock_sha256": generation["sha256"],
        "source_checkpoint_sha256": _file_sha256(checkpoint_path),
        "starting_accepted": len(checkpoint),
        "locked_requests": len(requests),
        "detector_score_rows": 0,
        "final_panel_forbidden": True,
        "failure_log": str((screen / "generation" / "failures.jsonl").resolve()),
    }
    lock_forecasts(screen / "locks" / "completion_screen.json", payload)
    return payload


def finalize_complete_case(
    source: Path, screen: Path, destination: Path, token_counts_path: Path,
    repository: Path,
) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(f"Complete-case root already exists: {destination}")
    source_paths = DeferralPaths.from_root(source)
    screen_paths = DeferralPaths.from_root(screen)
    source_pilot = verify_pilot_lock(source_paths)
    source_generation = verify_generation_lock(source_paths)
    if verify_pilot_lock(screen_paths)["sha256"] != source_pilot["sha256"]:
        raise RuntimeError("Completion screen uses a different pilot lock")
    if verify_generation_lock(screen_paths)["sha256"] != source_generation["sha256"]:
        raise RuntimeError("Completion screen uses a different generation lock")
    source_v4_amendment = verify_lock(source / "locks" / "v4_amendment.json")
    if verify_lock(screen / "locks" / "v4_amendment.json")["sha256"] != source_v4_amendment["sha256"]:
        raise RuntimeError("Completion screen uses a different v4 amendment lock")
    screen_envelope = verify_lock(screen / "locks" / "completion_screen.json")
    screen_protocol = screen_envelope["payload"]
    source_checkpoint = source / "generation" / "checkpoint.jsonl"
    source_checkpoint_rows = _jsonl(source_checkpoint)
    if (
        screen_protocol.get("stage") != "score_blind_generation_completion_screen"
        or Path(str(screen_protocol.get("source_root", ""))).resolve() != source.resolve()
        or screen_protocol.get("source_pilot_lock_sha256") != source_pilot["sha256"]
        or screen_protocol.get("source_generation_lock_sha256") != source_generation["sha256"]
        or screen_protocol.get("source_checkpoint_sha256") != _file_sha256(source_checkpoint)
        or int(screen_protocol.get("starting_accepted", -1)) != len(source_checkpoint_rows)
        or int(screen_protocol.get("locked_requests", -1)) != len(source_generation["payload"]["requests"])
        or int(screen_protocol.get("detector_score_rows", -1)) != 0
        or screen_protocol.get("final_panel_forbidden") is not True
    ):
        raise RuntimeError("Completion-screen lock is not bound to the actual v4 starting state")
    checkpoint_path = screen / "generation" / "checkpoint.jsonl"
    failures_path = screen / "generation" / "failures.jsonl"
    checkpoint, failures = _jsonl(checkpoint_path), _jsonl(failures_path)
    if not checkpoint_path.read_bytes().startswith(source_checkpoint.read_bytes()):
        raise RuntimeError("The completion checkpoint does not preserve the source checkpoint as an exact byte prefix")
    requests = source_generation["payload"]["requests"]
    requests_by_id = {str(row["request_id"]): row for row in requests}
    accepted_ids = {str(row["request_id"]) for row in checkpoint}
    failed_ids = {str(row["request_id"]) for row in failures}
    if accepted_ids & failed_ids or accepted_ids | failed_ids != set(requests_by_id):
        raise RuntimeError("Completion screen must partition every locked request")
    _validate_checkpoint(checkpoint, requests_by_id)
    for failure in failures:
        request_id = str(failure.get("request_id", ""))
        request = requests_by_id.get(request_id)
        if not request or (
            str(failure.get("record_id")) != str(request["record_id"])
            or str(failure.get("generator_family")) != str(request["generator_family"])
            or str(failure.get("generator_revision")) != str(request["generator_revision"])
            or int(failure.get("attempts", -1)) != int(request["retry"]) + 1
        ):
            raise RuntimeError(f"Failure-log provenance mismatch: {request_id}")
        diagnostics = failure.get("diagnostics", ())
        if len(diagnostics) != int(request["retry"]) + 1:
            raise RuntimeError(f"Failure-log diagnostics are incomplete: {request_id}")
        for attempt, diagnostic in enumerate(diagnostics):
            if (
                int(diagnostic.get("attempt", -1)) != attempt
                or int(diagnostic.get("seed", -1)) != request_seed(int(request["seed"]), request_id, attempt)
            ):
                raise RuntimeError(f"Failure-log attempt seed mismatch: {request_id}/{attempt}")
    if screen_paths.score_table.exists() or (screen / "generation" / "accepted_outputs.csv").exists():
        raise RuntimeError("Completion screen must contain neither detector scores nor a final panel")

    failed_record_ids = {str(requests_by_id[request_id]["record_id"]) for request_id in failed_ids}
    manifest = source_pilot["payload"]
    pilot_counts = Counter(str(row["record_id"]) for row in manifest["pilot"])
    if any(pilot_counts[record_id] != 1 for record_id in failed_record_ids):
        raise RuntimeError("Each failed record must occur exactly once in the source pilot")
    kept_pilot = [row for row in manifest["pilot"] if str(row["record_id"]) not in failed_record_ids]
    kept_requests = [row for row in requests if str(row["request_id"]) in accepted_ids]
    if len(kept_pilot) != len(kept_requests) or len(kept_pilot) != len(checkpoint):
        raise RuntimeError("Complete-case human, request, and accepted-output counts differ")
    kept_record_ids = [str(row["record_id"]) for row in kept_pilot]
    request_record_ids = [str(row["record_id"]) for row in kept_requests]
    if (
        len(set(kept_record_ids)) != len(kept_record_ids)
        or len(set(request_record_ids)) != len(request_record_ids)
        or set(kept_record_ids) != set(request_record_ids)
        or {str(row["request_id"]) for row in checkpoint} != {str(row["request_id"]) for row in kept_requests}
    ):
        raise RuntimeError("Complete-case humans, requests, and accepted rows are not one-to-one")
    corpus_counts = Counter(str(row["corpus"]) for row in kept_pilot)
    if set(corpus_counts) != set(DEV_CORPORA) or any(corpus_counts[corpus] < 1_000 for corpus in DEV_CORPORA):
        raise RuntimeError(f"Complete-case corpus minimum failed: {dict(corpus_counts)}")

    paths = DeferralPaths.from_root(destination)
    destination.mkdir(parents=True)
    attrition_rows = [
        {
            "request_id": request_id,
            "record_id": str(requests_by_id[request_id]["record_id"]),
            "corpus": str(requests_by_id[request_id]["corpus"]),
            "generator_family": str(requests_by_id[request_id]["generator_family"]),
            "target_length": int(requests_by_id[request_id]["target_length"]),
            "reason": str(next(row["reason"] for row in failures if str(row["request_id"]) == request_id)),
        }
        for request_id in sorted(failed_ids)
    ]
    new_manifest = dict(manifest)
    new_manifest["pilot"] = kept_pilot
    new_manifest["caps"] = {**manifest["caps"], "pilot_records": len(kept_pilot)}
    new_manifest["pilot_per_corpus"] = dict(sorted(corpus_counts.items()))
    new_manifest["complete_case_attrition"] = {
        "policy": "exclude_only_score_blind_generation_infeasible_pairs",
        "source_pilot_lock_sha256": source_pilot["sha256"],
        "excluded_count": len(attrition_rows),
        "excluded_sha256": _sha256(attrition_rows),
    }
    _write_json(paths.manifest, new_manifest)
    lock_forecasts(paths.lock, new_manifest)

    token_counts = _json(token_counts_path)
    kept_record_id_set = set(kept_record_ids)
    lock_human_token_panels(paths, {record_id: token_counts[record_id] for record_id in kept_record_id_set}, cap=460)
    human_tokens = verify_lock(paths.human_token_lock)

    generation_payload = dict(source_generation["payload"])
    generation_payload.pop("request_json_sha256", None)
    generation_payload.pop("request_csv_sha256", None)
    generation_payload["pilot_lock_sha256"] = verify_pilot_lock(paths)["sha256"]
    generation_payload["human_token_lock_sha256"] = human_tokens["sha256"]
    generation_payload["topic_map"] = {
        record_id: topic for record_id, topic in source_generation["payload"]["topic_map"].items()
        if record_id in kept_record_id_set
    }
    generation_payload["requests"] = kept_requests
    generation_payload["requests_sha256"] = _sha256(kept_requests)
    generation_payload["complete_case_attrition"] = {
        "excluded_count": len(attrition_rows), "excluded_sha256": _sha256(attrition_rows),
    }
    _write_json(paths.generation_json, kept_requests)
    _write_requests(paths.generation_csv, kept_requests)
    generation_payload["request_json_sha256"] = _file_sha256(paths.generation_json)
    generation_payload["request_csv_sha256"] = _file_sha256(paths.generation_csv)
    lock_forecasts(paths.generation_lock, generation_payload)

    destination_checkpoint = destination / "generation" / "checkpoint.jsonl"
    destination_checkpoint.parent.mkdir(parents=True)
    shutil.copyfile(checkpoint_path, destination_checkpoint)
    if _file_sha256(destination_checkpoint) != _file_sha256(checkpoint_path):
        raise RuntimeError("Complete-case checkpoint copy is not byte-identical")

    family_counts = Counter(
        (str(row["corpus"]), str(row["generator_family"])) for row in kept_requests
    )
    attrition = {
        "stage": "selective_deferral_score_blind_complete_case_lock",
        "policy": "exclude_only_score_blind_generation_infeasible_pairs",
        "detector_score_rows": 0,
        "source_root": str(source.resolve()),
        "screen_root": str(screen.resolve()),
        "screen_checkpoint_sha256": _file_sha256(checkpoint_path),
        "screen_failures_sha256": _file_sha256(failures_path),
        "source_pilot_lock_sha256": source_pilot["sha256"],
        "source_generation_lock_sha256": source_generation["sha256"],
        "source_v4_amendment_lock_sha256": source_v4_amendment["sha256"],
        "completion_screen_lock_sha256": screen_envelope["sha256"],
        "accepted_count": len(checkpoint),
        "excluded_count": len(attrition_rows),
        "excluded": attrition_rows,
        "excluded_sha256": _sha256(attrition_rows),
        "corpus_counts": dict(sorted(corpus_counts.items())),
        "corpus_generator_counts": [
            {"corpus": key[0], "generator_family": key[1], "count": value}
            for key, value in sorted(family_counts.items())
        ],
        "checkpoint_sha256": _file_sha256(destination_checkpoint),
        "pilot_lock_sha256": verify_pilot_lock(paths)["sha256"],
        "generation_lock_sha256": verify_generation_lock(paths)["sha256"],
    }
    lock_forecasts(destination / "locks" / "complete_case_attrition.json", attrition)

    bound_files = {
        "config": repository / "deferral_config.json",
        "generation_spec": repository / "deferral_generation_spec.json",
        "core": repository / "fprint" / "core.py",
        "deferral": repository / "fprint" / "deferral.py",
        "deferral_evaluation": repository / "fprint" / "deferral_evaluation.py",
        "detectors": repository / "fprint" / "detectors.py",
        "generation_runner": repository / "tools" / "run_deferral_generation.py",
        "scoring_runner": repository / "tools" / "run_deferral_scoring.py",
        "prepare_inputs": repository / "tools" / "prepare_deferral_inputs.py",
        "complete_case_finalizer": repository / "tools" / "finalize_deferral_complete_case.py",
    }
    protocol = {
        "stage": "selective_deferral_protocol_binding",
        "pilot_lock_sha256": verify_pilot_lock(paths)["sha256"],
        "generation_lock_sha256": verify_generation_lock(paths)["sha256"],
        "attrition_lock_sha256": verify_lock(destination / "locks" / "complete_case_attrition.json")["sha256"],
        "files": {
            name: {"path": str(path.resolve()), "sha256": _file_sha256(path)}
            for name, path in bound_files.items()
        },
    }
    lock_forecasts(destination / "locks" / "protocol_binding.json", protocol)
    return attrition


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-screen")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--screen-root", type=Path, required=True)
    final = subparsers.add_parser("finalize")
    final.add_argument("--source-root", type=Path, required=True)
    final.add_argument("--screen-root", type=Path, required=True)
    final.add_argument("--destination", type=Path, required=True)
    final.add_argument("--token-counts", type=Path, required=True)
    final.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    if args.command == "prepare-screen":
        payload = prepare_screen(args.source_root, args.screen_root)
        result = {"screen_root": str(args.screen_root.resolve()), **payload}
    else:
        payload = finalize_complete_case(
            args.source_root, args.screen_root, args.destination,
            args.token_counts, args.repository,
        )
        result = {
            "destination": str(args.destination.resolve()),
            "accepted_count": payload["accepted_count"],
            "excluded_count": payload["excluded_count"],
            "excluded_sha256": payload["excluded_sha256"],
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
