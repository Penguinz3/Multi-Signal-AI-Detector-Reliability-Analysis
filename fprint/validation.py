from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from .conformance import (
    FaultSpec,
    _draw_triplets,
    _load_audit_state,
    _reference_distributions,
    _resolve_score,
    audit_paths,
)
from .core import lock_forecasts, verify_lock
from .core import make_probe_triplet
from .detectors import SPECS, _mage_preprocessor
from .operational import PROBES, analyze_score_maps


PRIMARY_ENDPOINTS = (
    "radar_roberta_large__vicuna7b_training",
    "mage_longformer__paper",
    "logrank__qwen2_5_0_5b_fp32",
)
REPLAY_FAMILIES = {"unchanged", "input_handling", "output_policy", "core_computation"}
PRIMARY_CORPORA = (
    "asap_aes", "blog_authorship", "cnn_dailymail", "govreport",
    "gutenberg", "pmc", "stack_exchange", "wikitext_103",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _bounded_text(text: str, max_words: int) -> str:
    return "".join(re.findall(r"\S+\s*", text)[:max_words]).strip()


def _condition_specs(seed: str) -> list[dict]:
    definitions = {
        "radar_roberta_large__vicuna7b_training": (
            ("unchanged", "unchanged", {}),
            ("input_handling", "newline_flatten", {}),
            ("precision", "bf16", {}),
            ("calibration", "logit_bias", {"bias": .25}),
            ("core_computation", "endpoint_replacement", {"replacement": "mage_longformer__paper"}),
            ("negative_control", "threshold_only", {"policy": "one_percent"}),
        ),
        "mage_longformer__paper": (
            ("unchanged", "unchanged", {}),
            ("input_handling", "preprocessor_disabled", {}),
            ("precision", "bf16", {}),
            ("calibration", "logit_bias", {"bias": .25}),
            ("core_computation", "endpoint_replacement", {"replacement": "radar_roberta_large__vicuna7b_training"}),
            ("negative_control", "threshold_only", {"policy": "one_percent"}),
        ),
        "logrank__qwen2_5_0_5b_fp32": (
            ("unchanged", "unchanged", {}),
            ("input_handling", "nfkc_whitespace", {}),
            ("precision", "bf16", {}),
            ("calibration", "logit_bias", {"bias": .25}),
            ("core_computation", "endpoint_replacement", {"replacement": "lastde__qwen2_5_0_5b_fp32"}),
            ("negative_control", "threshold_only", {"policy": "one_percent"}),
        ),
    }
    result = []
    for endpoint, conditions in definitions.items():
        for family, mode, parameters in conditions:
            code = hashlib.sha256(f"{seed}:{endpoint}:{family}:{mode}".encode()).hexdigest()[:12]
            result.append({
                "condition_code": code, "endpoint": endpoint, "family": family,
                "mode": mode, "parameters": parameters,
            })
    return result


def prepare_prospective_validation(
    source_root: Path,
    prior_fault_root: Path,
    validation_root: Path,
    *,
    seed: str = "fprint-prospective-operational-v1",
    candidates_per_cell: int = 75,
    minimum_candidates_per_cell: int = 10,
    minimum_sites: int = 2,
    maximum_words: int = 180,
) -> Path:
    """Freeze fresh group-disjoint candidates and opaque conditions before scoring."""
    if candidates_per_cell < 50 or minimum_candidates_per_cell < 3:
        raise ValueError("Candidate target must be at least 50 and sparse-cell minimum at least three")
    source_database = Path(source_root).resolve() / "folds" / "bawe" / "fprint.sqlite3"
    prior_paths = audit_paths(prior_fault_root)
    prior_envelope = verify_lock(prior_paths.lock)
    source = _readonly(source_database)
    prior = _readonly(prior_paths.database)
    try:
        used_groups = {
            str(row[0]) for row in source.execute(
                "SELECT DISTINCT r.group_id FROM probe_triplets p JOIN records r USING(record_id)"
            )
        }
        used_groups.update(str(row[0]) for row in prior.execute("SELECT DISTINCT group_id FROM audit_triplets"))
        records = [dict(row) for row in source.execute(
            """SELECT record_id,corpus,group_id,text FROM records
                WHERE partition_name='anchor_candidates' AND corpus IN ({})
                ORDER BY corpus,group_id,record_id""".format(",".join("?" for _ in PRIMARY_CORPORA)),
            PRIMARY_CORPORA,
        )]
    finally:
        source.close()
        prior.close()
    by_corpus = {corpus: [row for row in records if row["corpus"] == corpus] for corpus in PRIMARY_CORPORA}
    selected_groups = set(used_groups)
    candidates = []
    for corpus in PRIMARY_CORPORA:
        for probe in PROBES:
            ordered = sorted(by_corpus[corpus], key=lambda row: hashlib.sha256(
                f"{seed}:{corpus}:{probe}:{row['group_id']}:{row['record_id']}".encode()
            ).hexdigest())
            cell = []
            for row in ordered:
                group = str(row["group_id"])
                if group in selected_groups:
                    continue
                text = _bounded_text(str(row["text"]), maximum_words)
                triplet = make_probe_triplet(
                    probe, text, f"{seed}:{row['record_id']}:{probe}", minimum_sites,
                )
                if triplet is None:
                    continue
                triplet_id = hashlib.sha256(
                    f"{seed}:{row['record_id']}:{probe}:{hashlib.sha256(text.encode()).hexdigest()}".encode()
                ).hexdigest()
                payload = {
                    "triplet_id": triplet_id, "record_id": str(row["record_id"]),
                    "corpus": corpus, "group_id": group, "probe": probe,
                    "original_text": triplet.original, "low_text": triplet.low,
                    "high_text": triplet.high, "low_intensity": triplet.low_intensity,
                    "high_intensity": triplet.high_intensity,
                    "triplet_sha256": _digest([
                        triplet.original, triplet.low, triplet.high,
                        triplet.low_intensity, triplet.high_intensity,
                    ]),
                }
                cell.append(payload)
                selected_groups.add(group)
                if len(cell) == candidates_per_cell:
                    break
            candidates.extend(cell)
    candidate_counts = {
        f"{corpus}:{probe}": sum(row["corpus"] == corpus and row["probe"] == probe for row in candidates)
        for corpus in PRIMARY_CORPORA for probe in PROBES
    }
    full_corpora = [
        corpus for corpus in PRIMARY_CORPORA
        if all(candidate_counts[f"{corpus}:{probe}"] >= 50 for probe in PROBES)
    ]
    if len(full_corpora) < 4:
        raise RuntimeError(f"Only {len(full_corpora)} corpora support the preregistered full budget")
    validation_root = Path(validation_root).resolve()
    if validation_root.exists():
        raise FileExistsError(f"Validation root already exists: {validation_root}")
    validation_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{validation_root.name}-", dir=validation_root.parent))
    try:
        candidate_path = staging / "candidates.csv"
        with candidate_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)
        conditions = _condition_specs(seed)
        public_conditions = [
            {"condition_code": row["condition_code"], "endpoint": row["endpoint"]}
            for row in conditions
        ]
        manifest = {
            "schema_version": 1,
            "construct": "prospective_operational_black_box_validation",
            "seed": seed,
            "source_database": str(source_database),
            "source_database_sha256": _file_sha256(source_database),
            "prior_fault_manifest_sha256": prior_envelope["sha256"],
            "selection": {
                "partition": "anchor_candidates", "excluded_prior_groups": len(used_groups),
                "one_record_per_group_across_panel": True, "minimum_sites": minimum_sites,
                "maximum_words": maximum_words, "candidates_per_corpus_probe": candidates_per_cell,
                "minimum_candidates_per_corpus_probe": minimum_candidates_per_cell,
                "score_blind_hash_order": True,
                "source_is_globally_raid_deduplicated_grouped_final": True,
            },
            "corpora": list(PRIMARY_CORPORA), "probes": list(PROBES),
            "endpoints": list(PRIMARY_ENDPOINTS), "query_budgets": [10, 25, 50], "draws": 20,
            "candidate_rows": len(candidates), "candidate_table_sha256": _file_sha256(candidate_path),
            "candidate_counts": candidate_counts,
            "full_budget_candidate_corpora": full_corpora,
            "candidate_triplet_digest": _digest([
                [row["triplet_id"], row["record_id"], row["corpus"], row["group_id"], row["triplet_sha256"]]
                for row in candidates
            ]),
            "opaque_conditions": public_conditions,
            "success_gate": {
                "budget": 50, "minimum_sensitivity": .80,
                "maximum_unchanged_false_alarm_rate": .05,
                "minimum_endpoint_sensitivity": .70, "minimum_full_budget_corpora": 4,
            },
            "claim_boundary": "Detect observable behavioral departures and localize changed probe responses; do not infer authorship or exact internal cause.",
        }
        lock_forecasts(staging / "manifest.lock.json", manifest)
        lock_forecasts(staging / "condition_truth.private.lock.json", {
            "schema_version": 1, "parent_manifest_payload_sha256": _digest(manifest),
            "release_rule": "only_after_all_blinded_reports_are_hash_locked",
            "conditions": conditions,
        })
        staging.replace(validation_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validation_root


def lock_prospective_panel(validation_root: Path, mage_repo: Path) -> Path:
    """Token-validate every triplet first, then lock the first 50 valid groups per cell."""
    root = Path(validation_root).resolve()
    manifest = verify_lock(root / "manifest.lock.json")["payload"]
    if (root / "panel.lock.json").exists():
        verify_lock(root / "panel.lock.json")
        return root / "panel.lock.json"
    from transformers import AutoTokenizer

    preprocess = _mage_preprocessor(str(mage_repo))
    tokenizers = {
        endpoint: AutoTokenizer.from_pretrained(
            SPECS[endpoint].model_id, revision=SPECS[endpoint].tokenizer_revision, local_files_only=True,
        )
        for endpoint in PRIMARY_ENDPOINTS
    }
    with (root / "candidates.csv").open(encoding="utf-8-sig", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    selected, rejected = [], []
    for corpus in manifest["corpora"]:
        for probe in manifest["probes"]:
            cell_candidates = [item for item in candidates if item["corpus"] == corpus and item["probe"] == probe]
            target = min(50, len(cell_candidates))
            if not target:
                continue
            valid = []
            for row in cell_candidates:
                counts = {}
                for endpoint, tokenizer in tokenizers.items():
                    texts = [row[f"{level}_text"] for level in ("original", "low", "high")]
                    if endpoint == "mage_longformer__paper":
                        texts = [preprocess(text) for text in texts]
                    counts[endpoint] = [
                        len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
                        for text in texts
                    ]
                if any(max(values) > min(460, SPECS[endpoint].max_tokens - 32) for endpoint, values in counts.items()):
                    rejected.append({
                        "triplet_id": row["triplet_id"], "corpus": corpus, "probe": probe,
                        "reason": "full_triplet_capacity", "token_counts": json.dumps(counts, sort_keys=True),
                    })
                else:
                    valid.append((row, counts))
                if len(valid) == target:
                    break
            if len(valid) < min(10, target):
                raise RuntimeError(f"{corpus}/{probe} has only {len(valid)} all-endpoint-valid triplets")
            for row, counts in valid:
                row = dict(row)
                row["token_counts_json"] = json.dumps(counts, sort_keys=True)
                selected.append(row)
    panel_path = root / "panel.csv"
    with panel_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    rejection_path = root / "capacity_exclusions.csv"
    with rejection_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("triplet_id", "corpus", "probe", "reason", "token_counts"))
        writer.writeheader()
        writer.writerows(rejected)
    lock_forecasts(root / "panel.lock.json", {
        "schema_version": 1,
        "parent_manifest_sha256": _file_sha256(root / "manifest.lock.json"),
        "selection_rule": "first_50_frozen_hash_order_candidates_valid_for_every_endpoint_tokenizer",
        "rows": len(selected), "groups": len({row["group_id"] for row in selected}),
        "rows_per_corpus_probe": {
            f"{corpus}:{probe}": sum(row["corpus"] == corpus and row["probe"] == probe for row in selected)
            for corpus in manifest["corpora"] for probe in manifest["probes"]
        },
        "capacity_exclusions_sha256": _file_sha256(rejection_path),
        "triplet_ids": [row["triplet_id"] for row in selected],
    })
    return root / "panel.lock.json"


def _challenge(selected: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    rows = []
    for triplet in selected:
        for level, intensity in (
            ("original", 0.0),
            ("low", float(triplet["low_intensity"])),
            ("high", float(triplet["high_intensity"])),
        ):
            rows.append({
                "challenge_id": f"{triplet['triplet_id']}:{level}",
                "triplet_id": str(triplet["triplet_id"]),
                "probe": str(triplet["probe"]),
                "intensity": level,
                "intensity_value": str(intensity),
            })
    return rows


def _score_map(
    selected: Sequence[Mapping[str, object]],
    endpoint: str,
    fault: FaultSpec,
    faults: Mapping[str, FaultSpec],
    scores: Mapping[tuple[str, str, str, str], Mapping[str, object]],
    references: Mapping[str, Sequence[float]],
) -> dict[str, float] | None:
    result = {}
    for triplet in selected:
        for level in ("original", "low", "high"):
            resolved = _resolve_score(
                str(triplet["triplet_id"]), level, endpoint, fault, faults, scores, references,
            )
            if resolved is None:
                return None
            result[f"{triplet['triplet_id']}:{level}"] = resolved[0]
    return result


def _alarm_score(analysis: Mapping[str, object]) -> float:
    rule = analysis["decision_rule"]
    tolerance, multiplier = float(rule["absolute_tolerance"]), float(rule["noise_multiplier"])
    return max(
        float(cell["median_current_delta"])
        / max(tolerance, multiplier * float(cell["median_reference_repeat_delta"]), 1e-12)
        for cells in analysis["probe_results"].values()
        for cell in cells.values()
    )


def _auroc(rows: Sequence[Mapping[str, object]]) -> float:
    positive = [float(row["alarm_score"]) for row in rows if row["truth_changed"]]
    negative = [float(row["alarm_score"]) for row in rows if not row["truth_changed"]]
    if not positive or not negative:
        return 0.0
    wins = sum((left > right) + .5 * (left == right) for left in positive for right in negative)
    return wins / (len(positive) * len(negative))


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict:
    changed = [row for row in rows if row["truth_changed"]]
    unchanged = [row for row in rows if not row["truth_changed"]]
    return {
        "cases": len(rows),
        "changed_cases": len(changed),
        "unchanged_cases": len(unchanged),
        "auroc": _auroc(rows),
        "sensitivity": sum(row["status"] == "changed" for row in changed) / len(changed) if changed else 0.0,
        "unchanged_false_alarm_rate": (
            sum(row["status"] == "changed" for row in unchanged) / len(unchanged) if unchanged else 0.0
        ),
        "inconclusive_rate": sum(row["status"] == "inconclusive" for row in rows) / len(rows) if rows else 0.0,
    }


def replay_operational_validation(fault_audit_root: Path, output_dir: Path) -> Path:
    """Replay the production rule on prior locked scores; this is development evidence only."""
    paths = audit_paths(fault_audit_root)
    envelope = verify_lock(paths.lock)
    manifest = envelope["payload"]
    triplets, scores = _load_audit_state(paths)
    triplets = [row for row in triplets if row["source_kind"] == "discovery"]
    references = _reference_distributions(manifest)
    config = manifest["config"]
    faults = {row["fault_id"]: FaultSpec.from_mapping(row) for row in manifest["faults"]}
    candidates = [
        fault for fault in faults.values()
        if fault.family in REPLAY_FAMILIES and fault.mode != "threshold_policy"
    ]
    rule = {
        "probes": list(PROBES),
        "decision_rule": {
            "alpha": .05,
            "absolute_tolerance": .01,
            "noise_multiplier": 3.0,
            "minimum_affected_fraction": .20,
            "maximum_reference_noise": .02,
        },
    }
    predictions = []
    for budget in config["query_budgets"]:
        for draw in range(int(config["draws"])):
            for corpus in manifest["primary_corpora"]:
                selected = _draw_triplets(
                    triplets, str(corpus), PROBES, int(budget), draw, int(config["seed"]), True,
                )
                counts = {probe: sum(row["probe"] == probe for row in selected) for probe in PROBES}
                if min(counts.values(), default=0) < 3:
                    continue
                challenge = _challenge(selected)
                for endpoint in PRIMARY_ENDPOINTS:
                    unchanged = _score_map(
                        selected, endpoint, faults["unchanged"], faults, scores, references,
                    )
                    if unchanged is None:
                        continue
                    for fault in candidates:
                        if not fault.applies_to(endpoint):
                            continue
                        current = _score_map(selected, endpoint, fault, faults, scores, references)
                        if current is None:
                            continue
                        analysis = analyze_score_maps(rule, challenge, (unchanged, unchanged), current)
                        changed_cells = [
                            f"{probe}:{feature}"
                            for probe, cells in analysis["probe_results"].items()
                            for feature, cell in cells.items() if cell["changed"]
                        ]
                        predictions.append({
                            "budget": int(budget), "draw": draw, "corpus": str(corpus),
                            "endpoint": endpoint, "fault_id": fault.fault_id, "family": fault.family,
                            "truth_changed": fault.family != "unchanged", "status": analysis["status"],
                            "alarm_score": _alarm_score(analysis), "changed_cells": ";".join(changed_cells),
                            "punctuation_n": counts[PROBES[0]], "sentence_n": counts[PROBES[1]],
                            "paragraph_n": counts[PROBES[2]],
                        })
    if not predictions:
        raise RuntimeError("No complete replay cases were available")
    by_budget = {
        str(budget): _metrics([row for row in predictions if row["budget"] == budget])
        for budget in config["query_budgets"]
    }
    primary = [row for row in predictions if row["budget"] == 50]
    endpoint_metrics = {
        endpoint: _metrics([row for row in primary if row["endpoint"] == endpoint])
        for endpoint in PRIMARY_ENDPOINTS
    }
    family_metrics = {
        family: _metrics([row for row in primary if row["family"] in {"unchanged", family}])
        for family in ("input_handling", "output_policy", "core_computation")
    }
    full_budget_corpora = sorted({
        str(row["corpus"]) for row in primary
        if min(int(row[name]) for name in ("punctuation_n", "sentence_n", "paragraph_n")) >= 50
    })
    gate = {
        "minimum_sensitivity": .80,
        "maximum_false_alarm_rate": .05,
        "minimum_endpoint_sensitivity": .70,
        "minimum_full_budget_corpora": 4,
    }
    primary_metrics = _metrics(primary)
    gate["passed"] = (
        primary_metrics["sensitivity"] >= gate["minimum_sensitivity"]
        and primary_metrics["unchanged_false_alarm_rate"] <= gate["maximum_false_alarm_rate"]
        and min(row["sensitivity"] for row in endpoint_metrics.values()) >= gate["minimum_endpoint_sensitivity"]
        and len(full_budget_corpora) >= gate["minimum_full_budget_corpora"]
    )
    report = {
        "schema_version": 1,
        "construct": "operational_rule_retrospective_replay",
        "evidence_status": "development_replication_not_prospective_validation",
        "source_manifest_sha256": envelope["sha256"],
        "score_source": str(paths.database),
        "budgets": by_budget,
        "primary_budget_50": primary_metrics,
        "by_endpoint_at_50": endpoint_metrics,
        "by_family_at_50": family_metrics,
        "full_budget_corpora": full_budget_corpora,
        "success_gate": gate,
        "limitations": [
            "Scores and fault outcomes predate this production-rule replay.",
            "The two deterministic reference maps are identical rather than fresh endpoint reruns.",
            "Sparse corpus/probe cells use their full available count below the nominal budget.",
            "Threshold-only policy changes are unobservable from score tables and excluded.",
        ],
    }
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Validation output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        with (staging / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        (staging / "validation_metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        lock_forecasts(staging / "replay.lock.json", {
            "schema_version": 1,
            "source_manifest_sha256": envelope["sha256"],
            "metrics_sha256": _file_sha256(staging / "validation_metrics.json"),
            "predictions_sha256": _file_sha256(staging / "predictions.csv"),
        })
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir
