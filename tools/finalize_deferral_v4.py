"""Create the one-time v4 amendment after score-blind generation screening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from fprint.core import canonical_json, lock_forecasts, verify_lock
from fprint.deferral import (
    CanonicalRecord,
    DeferralPaths,
    _base_text,
    _record_payload,
    _render_generation_prompt,
    _sha256,
    _text_sha256,
    build_reflow_variants,
    lock_human_token_panels,
    read_canonical_table,
    validate_triplet_token_budget,
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
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        rows.append(row)
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


def _pilot_row(record: CanonicalRecord, *, width: int, block_size: int) -> dict[str, object]:
    variants = build_reflow_variants(record.text, width=width, block_size=block_size)
    return {
        **_record_payload(record, "pilot", normalized=True),
        "variants": [
            {
                "probe": variant.probe,
                "variant_id": variant.variant_id,
                "text_sha256": variant.text_sha256,
                "non_whitespace_sha256": variant.non_whitespace_sha256,
                "changed": variant.changed,
                "non_whitespace_preserved": variant.non_whitespace_preserved,
            }
            for variant in variants
        ],
    }


def select_replacements(
    records: Iterable[CanonicalRecord],
    token_counts: Mapping[str, Mapping[str, object]],
    topics: Mapping[str, object],
    failures: Sequence[Mapping[str, object]],
    old_requests: Mapping[str, Mapping[str, object]],
    used_groups: set[tuple[str, str]],
    *,
    seed: int,
    token_cap: int,
    max_words: int,
) -> dict[str, CanonicalRecord]:
    """Hash-tie-broken, same-corpus reserve selection with close length matching."""
    eligible: dict[str, list[CanonicalRecord]] = {}
    representatives: dict[tuple[str, str], CanonicalRecord] = {}
    for record in records:
        key = (record.corpus, record.group_id)
        if not record.is_human or key in used_groups or record.record_id not in topics:
            continue
        word_count = len(_base_text(record.text).split())
        if not 1 <= word_count <= max_words or record.record_id not in token_counts:
            continue
        try:
            build_reflow_variants(record.text)
            validate_triplet_token_budget(token_counts[record.record_id], cap=token_cap)
        except ValueError:
            continue
        topic_value = topics[record.record_id]
        topic = str(topic_value.get("topic", topic_value) if isinstance(topic_value, Mapping) else topic_value)
        if not topic.strip() or _text_sha256(_base_text(topic)) == _text_sha256(_base_text(record.text)):
            continue
        current = representatives.get(key)
        rank = _sha256({"seed": seed, "purpose": "v4_group_representative", "record_id": record.record_id})
        if current is None or (rank, record.record_id) < (
            _sha256({"seed": seed, "purpose": "v4_group_representative", "record_id": current.record_id}),
            current.record_id,
        ):
            representatives[key] = record
    for record in representatives.values():
        eligible.setdefault(record.corpus, []).append(record)

    selected: dict[str, CanonicalRecord] = {}
    reserved_groups = set(used_groups)
    ordered_failures = sorted(
        failures,
        key=lambda row: (
            str(old_requests[str(row["request_id"])]["corpus"]),
            int(old_requests[str(row["request_id"])]["target_length"]),
            str(row["request_id"]),
        ),
    )
    for failure in ordered_failures:
        request_id = str(failure["request_id"])
        request = old_requests[request_id]
        corpus = str(request["corpus"])
        target = int(request["target_length"])
        tolerance = max(15, int(math.ceil(target * 0.10)))
        candidates = [
            record for record in eligible.get(corpus, ())
            if (record.corpus, record.group_id) not in reserved_groups
            and abs(len(_base_text(record.text).split()) - target) <= tolerance
        ]
        if not candidates:
            raise RuntimeError(f"No close, group-disjoint reserve for failed request {request_id}")
        candidates.sort(key=lambda record: (
            abs(len(_base_text(record.text).split()) - target),
            _sha256({
                "seed": seed, "purpose": "v4_failure_replacement",
                "failed_request_id": request_id, "record_id": record.record_id,
            }),
            record.record_id,
        ))
        selected[request_id] = candidates[0]
        reserved_groups.add((candidates[0].corpus, candidates[0].group_id))
    return selected


def _replacement_request(
    old: Mapping[str, object], record: CanonicalRecord, topic_value: object,
    generation: Mapping[str, object],
) -> dict[str, object]:
    target = len(_base_text(record.text).split())
    fraction = float(generation["length_tolerance_fraction"])
    minimum = int(generation["length_tolerance_min_words"])
    tolerance = max(minimum, int(math.ceil(target * fraction)))
    low, high = max(1, target - tolerance), target + tolerance
    templates = generation["prompt_template"]
    template = str(templates.get(record.corpus, templates.get("default", "")))
    topic = str(topic_value.get("topic", topic_value) if isinstance(topic_value, Mapping) else topic_value)
    prompt = _render_generation_prompt(template, topic, target, low, high)
    prompt_hash = _text_sha256(prompt)
    identity = {
        "record_id": record.record_id,
        "corpus": record.corpus,
        "generator_family": str(old["generator_family"]),
        "generator_revision": str(old["generator_revision"]),
        "prompt_sha256": prompt_hash,
        "seed": int(old["seed"]),
        "retry": int(old["retry"]),
        "target_length": target,
        "min_word_count": low,
        "max_word_count": high,
        "decoding": dict(old["decoding"]),
    }
    return {"request_id": _sha256(identity), **identity, "prompt": prompt}


def create_v4(
    source_root: Path, screen_root: Path, destination: Path,
    candidates_path: Path, token_counts_path: Path, topics_path: Path,
    repository: Path,
) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    source_paths = DeferralPaths.from_root(source_root)
    screen_paths = DeferralPaths.from_root(screen_root)
    source_pilot = verify_pilot_lock(source_paths)
    source_generation = verify_generation_lock(source_paths)
    if verify_generation_lock(screen_paths)["sha256"] != source_generation["sha256"]:
        raise RuntimeError("Screening generation lock differs from v3")
    if verify_pilot_lock(screen_paths)["sha256"] != source_pilot["sha256"]:
        raise RuntimeError("Screening pilot lock differs from v3")
    if screen_paths.score_table.exists() or (screen_root / "generation" / "accepted_outputs.csv").exists():
        raise RuntimeError("Screening lane must contain neither detector scores nor a final panel")

    checkpoint_path = screen_root / "generation" / "checkpoint.jsonl"
    failures_path = screen_root / "generation" / "failures.jsonl"
    checkpoint, failures = _jsonl(checkpoint_path), _jsonl(failures_path)
    requests = source_generation["payload"]["requests"]
    old_requests = {str(row["request_id"]): row for row in requests}
    accepted_ids = {str(row["request_id"]) for row in checkpoint}
    failed_ids = {str(row["request_id"]) for row in failures}
    if accepted_ids & failed_ids or accepted_ids | failed_ids != set(old_requests):
        raise RuntimeError("Screening checkpoint and failures must partition all locked requests")
    if len(checkpoint) != 4_954 or len(failures) != 46:
        raise RuntimeError(f"Expected the frozen 4,954/46 screen, got {len(checkpoint)}/{len(failures)}")

    records = read_canonical_table(candidates_path)
    records_by_id = {record.record_id: record for record in records}
    token_counts = _json(token_counts_path)
    topics = _json(topics_path)
    manifest = source_pilot["payload"]
    source_rows = [_record_payload(record, record.partition or "input") for record in records]
    if _sha256(source_rows) != manifest["source_table_sha256"]:
        raise RuntimeError("Candidate table differs from the table bound by the v3 pilot lock")
    for partition in ("calibration", "pilot"):
        for locked in manifest[partition]:
            record_id = str(locked["record_id"])
            if record_id not in records_by_id:
                raise RuntimeError(f"Locked {partition} record is absent from the candidate table: {record_id}")
            expected = _record_payload(records_by_id[record_id], partition, normalized=True)
            if any(locked.get(field) != value for field, value in expected.items()):
                raise RuntimeError(f"Locked {partition} provenance differs from the candidate table: {record_id}")
    pilot_record_counts = Counter(str(row["record_id"]) for row in manifest["pilot"])
    failed_record_ids = [str(old_requests[request_id]["record_id"]) for request_id in failed_ids]
    if any(pilot_record_counts[record_id] != 1 for record_id in failed_record_ids):
        raise RuntimeError("Every failed record must occur exactly once in the v3 pilot")
    used_groups = {
        (str(row["corpus"]), records_by_id[str(row["record_id"])].group_id)
        for row in (*manifest["calibration"], *manifest["pilot"])
    }
    selected = select_replacements(
        records, token_counts, topics, failures, old_requests, used_groups,
        seed=int(manifest["seed"]), token_cap=460,
        max_words=int(manifest["max_paired_target_words"]),
    )
    replacement_by_record = {
        str(old_requests[request_id]["record_id"]): record
        for request_id, record in selected.items()
    }
    replacement_ids = [record.record_id for record in selected.values()]
    old_record_ids = {
        str(row["record_id"]) for row in (*manifest["calibration"], *manifest["pilot"])
    }
    if len(set(replacement_ids)) != len(failures) or set(replacement_ids) & old_record_ids:
        raise RuntimeError("Replacement records must be unique and unused by calibration/pilot")
    mapping = [
        {
            "failed_request_id": request_id,
            "failed_record_id": str(old_requests[request_id]["record_id"]),
            "replacement_record_id": record.record_id,
            "replacement_group_id": record.group_id,
            "corpus": record.corpus,
            "generator_family": str(old_requests[request_id]["generator_family"]),
            "failed_target_length": int(old_requests[request_id]["target_length"]),
            "replacement_target_length": len(_base_text(record.text).split()),
        }
        for request_id, record in sorted(selected.items())
    ]

    paths = DeferralPaths.from_root(destination)
    destination.mkdir(parents=True)
    width = int(manifest["transform"]["width"])
    block_size = int(manifest["transform"]["sentence_block_size"])
    new_manifest = dict(manifest)
    new_manifest["pilot"] = [
        _pilot_row(replacement_by_record[str(row["record_id"])], width=width, block_size=block_size)
        if str(row["record_id"]) in replacement_by_record else row
        for row in manifest["pilot"]
    ]
    if len(new_manifest["pilot"]) != 5_000 or len({str(row["record_id"]) for row in new_manifest["pilot"]}) != 5_000:
        raise RuntimeError("The v4 pilot must contain exactly 5,000 unique records")
    new_manifest["generation_failure_amendment"] = {
        "version": "v4_single_amendment",
        "policy": "replace_only_score_blind_generation_failures",
        "source_pilot_lock_sha256": source_pilot["sha256"],
        "source_generation_lock_sha256": source_generation["sha256"],
        "replacement_count": len(mapping),
        "mapping_sha256": _sha256(mapping),
    }
    _write_json(paths.manifest, new_manifest)
    lock_forecasts(paths.lock, new_manifest)

    pilot_ids = {str(row["record_id"]) for row in new_manifest["pilot"]}
    lock_human_token_panels(paths, {record_id: token_counts[record_id] for record_id in pilot_ids}, cap=460)
    human_token_envelope = verify_lock(paths.human_token_lock)

    replacements = {
        request_id: _replacement_request(old_requests[request_id], record, topics[record.record_id], source_generation["payload"])
        for request_id, record in selected.items()
    }
    new_requests = [replacements.get(str(row["request_id"]), row) for row in requests]
    new_requests.sort(key=lambda row: (str(row["corpus"]), str(row["record_id"]), str(row["generator_family"])))
    if len({str(row["request_id"]) for row in new_requests}) != len(new_requests):
        raise RuntimeError("v4 generation request IDs are not unique")
    old_counts = {}
    new_counts = {}
    for old, new in zip(
        sorted(requests, key=lambda row: (str(row["corpus"]), str(row["generator_family"]))),
        sorted(new_requests, key=lambda row: (str(row["corpus"]), str(row["generator_family"]))),
    ):
        old_key = (str(old["corpus"]), str(old["generator_family"]))
        new_key = (str(new["corpus"]), str(new["generator_family"]))
        old_counts[old_key] = old_counts.get(old_key, 0) + 1
        new_counts[new_key] = new_counts.get(new_key, 0) + 1
    if old_counts != new_counts:
        raise RuntimeError("v4 changed a corpus/generator quota")

    generation_payload = dict(source_generation["payload"])
    generation_payload.pop("request_json_sha256", None)
    generation_payload.pop("request_csv_sha256", None)
    generation_payload["pilot_lock_sha256"] = verify_pilot_lock(paths)["sha256"]
    generation_payload["human_token_lock_sha256"] = human_token_envelope["sha256"]
    generation_payload["topic_map"] = {
        record_id: str(topics[record_id].get("topic", topics[record_id]) if isinstance(topics[record_id], Mapping) else topics[record_id])
        for record_id in sorted(pilot_ids)
    }
    generation_payload["requests"] = new_requests
    generation_payload["requests_sha256"] = _sha256(new_requests)
    generation_payload["generation_failure_amendment"] = {
        "version": "v4_single_amendment", "mapping_sha256": _sha256(mapping),
    }
    _write_json(paths.generation_json, new_requests)
    _write_requests(paths.generation_csv, new_requests)
    generation_payload["request_json_sha256"] = _file_sha256(paths.generation_json)
    generation_payload["request_csv_sha256"] = _file_sha256(paths.generation_csv)
    lock_forecasts(paths.generation_lock, generation_payload)

    new_request_ids = {str(row["request_id"]) for row in new_requests}
    migrated = [row for row in checkpoint if str(row["request_id"]) in new_request_ids]
    if len(migrated) != len(checkpoint):
        raise RuntimeError("A successful screen request was unexpectedly changed")
    for row in migrated:
        request = old_requests[str(row["request_id"])]
        if (
            str(row.get("generator_family")) != str(request["generator_family"])
            or str(row.get("generator_revision")) != str(request["generator_revision"])
            or int(row.get("retry", -1)) != int(request["retry"])
            or int(row.get("target_length", -1)) != int(request["target_length"])
        ):
            raise RuntimeError(f"Successful checkpoint provenance mismatch: {row['request_id']}")
        attempt = int(row.get("attempt", -1))
        if not 0 <= attempt <= int(request["retry"]) or int(row.get("seed", -1)) != request_seed(
            int(request["seed"]), str(row["request_id"]), attempt,
        ):
            raise RuntimeError(f"Successful checkpoint seed mismatch: {row['request_id']}")
        text = str(row.get("text", ""))
        count = _word_count(text)
        if not int(request["min_word_count"]) <= count <= int(request["max_word_count"]) or not _is_complete_passage(text):
            raise RuntimeError(f"Successful checkpoint text is outside its locked envelope: {row['request_id']}")
        stored_counts = row.get("token_counts", "")
        if isinstance(stored_counts, str):
            stored_counts = json.loads(stored_counts) if stored_counts else None
        if not isinstance(stored_counts, Mapping):
            raise RuntimeError(f"Successful checkpoint lacks its token panel: {row['request_id']}")
        validate_triplet_token_budget(stored_counts, cap=460)
    generation_dir = destination / "generation"
    generation_dir.mkdir(parents=True)
    migrated_path = generation_dir / "checkpoint.jsonl"
    # The successful rows are already immutable and all remain valid v4 requests.
    # Preserve their exact serialized bytes; replacement generations append later.
    with migrated_path.open("xb") as handle:
        handle.write(checkpoint_path.read_bytes())

    amendment = {
        "stage": "selective_deferral_v4_generation_failure_amendment",
        "score_blind": True,
        "detector_score_rows": 0,
        "source_root": str(source_root.resolve()),
        "screen_root": str(screen_root.resolve()),
        "source_checkpoint_sha256": _file_sha256(checkpoint_path),
        "source_failures_sha256": _file_sha256(failures_path),
        "source_pilot_lock_sha256": source_pilot["sha256"],
        "source_generation_lock_sha256": source_generation["sha256"],
        "selection_rule": "same_corpus_unused_group_token_valid_close_length_then_seeded_sha256",
        "reused_outputs": len(migrated),
        "replacement_count": len(mapping),
        "mapping": mapping,
        "mapping_sha256": _sha256(mapping),
        "migrated_checkpoint_sha256": _file_sha256(migrated_path),
        "v4_pilot_lock_sha256": verify_pilot_lock(paths)["sha256"],
        "v4_generation_lock_sha256": verify_generation_lock(paths)["sha256"],
    }
    lock_forecasts(destination / "locks" / "v4_amendment.json", amendment)

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
        "v4_finalizer": repository / "tools" / "finalize_deferral_v4.py",
    }
    protocol = {
        "stage": "selective_deferral_protocol_binding",
        "pilot_lock_sha256": verify_pilot_lock(paths)["sha256"],
        "generation_lock_sha256": verify_generation_lock(paths)["sha256"],
        "amendment_lock_sha256": verify_lock(destination / "locks" / "v4_amendment.json")["sha256"],
        "files": {
            name: {"path": str(path.resolve()), "sha256": _file_sha256(path)}
            for name, path in bound_files.items()
        },
    }
    lock_forecasts(destination / "locks" / "protocol_binding.json", protocol)
    return amendment


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--token-counts", type=Path, required=True)
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    amendment = create_v4(
        args.source_root, args.screen_root, args.destination, args.candidates,
        args.token_counts, args.topics, args.repository,
    )
    print(json.dumps({
        "destination": str(args.destination.resolve()),
        "reused_outputs": amendment["reused_outputs"],
        "replacement_count": amendment["replacement_count"],
        "mapping_sha256": amendment["mapping_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
