"""Pilot-only selective-deferral experiment primitives.

This module deliberately has no dependency on the forecasting or fault-audit
artifacts.  It owns the small amount of protocol state needed to ask a later
question: after an original detector-positive result, do a few deterministic
reflow checks justify retaining or deferring that decision?

The module does not score a detector.  Scoring is supplied through the
canonical score-table interface so local, hosted, and later commercial
endpoints can use the same locked analyzer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .core import canonical_json, lock_forecasts, verify_lock


SCHEMA_VERSION = 1
PILOT_STAGE = "selective_deferral_pilot"
# Variant IDs are part of the score-table contract.  Do not rename these
# without a locked protocol amendment.
PROBES = ("wrap_80", "sentence_blocks_2", "sentence_per_paragraph")
VARIANT_LEVEL = "high"
DEV_CORPORA = ("asap_aes", "blog_authorship", "stack_exchange", "wikitext_103")
CALIBRATION_PER_CORPUS = 500
PILOT_HUMAN_CAP = 5_000
GENERATOR_FAMILY_COUNT = 3

# These are intentionally operational endpoint IDs, not vague detector names.
# The role assignment is frozen in every pilot manifest.
RADAR_ENDPOINT = "radar_roberta_large__vicuna7b_training"
MAGE_ENDPOINT = "mage_longformer__paper"
LOGRANK_ENDPOINT = "logrank__qwen2_5_0_5b_fp32"
ENDPOINT_ROLES = {
    RADAR_ENDPOINT: "primary",
    MAGE_ENDPOINT: "preprocessing_invariance_negative_control",
    LOGRANK_ENDPOINT: "original_score_disagreement_only",
}

_REQUIRED_RECORD_FIELDS = ("record_id", "corpus", "group_id", "text")
_HUMAN_VALUES = {"human", "real", "authored", "human_written", "0", "false"}
_AI_VALUES = {"ai", "machine", "generated", "machine_generated", "1", "true"}


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    # This is intentionally byte-exact.  A score row must identify exactly the
    # text that was sent to an endpoint, including its line endings.
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _non_whitespace(text: str) -> str:
    return "".join(character for character in str(text) if not character.isspace())


def _non_whitespace_sha256(text: str) -> str:
    return hashlib.sha256(_non_whitespace(text).encode("utf-8")).hexdigest()


def _base_text(text: str) -> str:
    """The exact normalized input sent for the original and all reflows."""
    return " ".join(str(text).split())


@dataclass(frozen=True)
class DeferralPaths:
    """All pilot artifacts under one isolated storage root."""

    root: Path
    state: Path
    manifest: Path
    lock: Path
    score_table: Path
    worklist: Path
    results: Path
    authorization_lock: Path

    @classmethod
    def from_root(cls, root: Path | str) -> "DeferralPaths":
        root = Path(root).expanduser().resolve()
        return cls(
            root=root,
            state=root / "state",
            manifest=root / "state" / "pilot_manifest.json",
            lock=root / "locks" / "pilot_manifest.json",
            score_table=root / "state" / "canonical_scores.csv",
            worklist=root / "state" / "conditional_worklist.csv",
            results=root / "results",
            authorization_lock=root / "locks" / "pilot_authorization.json",
        )

    # Names used in handoffs and by downstream CLI code.
    @property
    def manifest_lock(self) -> Path:
        return self.lock

    @property
    def pilot_lock(self) -> Path:
        return self.lock

    @property
    def generation_csv(self) -> Path:
        return self.root / "state" / "generation_requests.csv"

    @property
    def generation_json(self) -> Path:
        return self.root / "state" / "generation_requests.json"

    @property
    def generation_lock(self) -> Path:
        return self.root / "locks" / "generation_requests.json"

    @property
    def panel_lock(self) -> Path:
        return self.root / "locks" / "generated_panel.json"

    @property
    def manual_audit_csv(self) -> Path:
        return self.root / "state" / "manual_audit.csv"

    @property
    def manual_audit_lock(self) -> Path:
        return self.root / "locks" / "manual_audit.json"

    def manual_audit_csv_for(self, probe: str) -> Path:
        return self.root / "state" / f"manual_audit_{probe}.csv"

    def manual_audit_lock_for(self, probe: str) -> Path:
        return self.root / "locks" / f"manual_audit_{probe}.json"

    @property
    def threshold_lock(self) -> Path:
        return self.root / "locks" / "radar_threshold.json"

    @property
    def worklist_lock(self) -> Path:
        return self.root / "locks" / "conditional_worklist.json"

    @property
    def human_token_lock(self) -> Path:
        return self.root / "locks" / "human_token_panels.json"


def deferral_paths(root: Path | str) -> DeferralPaths:
    return DeferralPaths.from_root(root)


@dataclass(frozen=True)
class CanonicalRecord:
    record_id: str
    corpus: str
    group_id: str
    text: str
    provenance_label: str
    partition: str = ""
    source_order: int = 0

    @property
    def text_sha256(self) -> str:
        return _text_sha256(self.text)

    @property
    def is_human(self) -> bool:
        return _label_is_human(self.provenance_label)


@dataclass(frozen=True)
class ReflowVariant:
    probe: str
    variant_id: str
    text: str
    text_sha256: str
    non_whitespace_sha256: str
    changed: bool
    non_whitespace_preserved: bool
    eligible: bool


@dataclass(frozen=True)
class ScoreRow:
    record_id: str
    variant_id: str
    endpoint: str
    detector_revision: str
    text_sha256: str
    provenance_label: str
    canonical_ai_score: float | None
    native_score: float | None = None
    input_token_count: int | None = None
    truncated: bool = False
    failure: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return self.record_id, self.variant_id, self.endpoint


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    record_id: str
    corpus: str
    generator_family: str
    generator_revision: str
    prompt: str
    prompt_sha256: str
    seed: int
    retry: int
    target_length: int
    min_word_count: int
    max_word_count: int
    decoding: Mapping[str, object]


def _label_is_human(label: object) -> bool:
    value = str(label or "").strip().casefold()
    if value in _HUMAN_VALUES:
        return True
    if value in _AI_VALUES:
        return False
    raise ValueError(f"Unsupported provenance label: {label!r}")


def _label(value: object) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError("Canonical rows require provenance_label or label")
    _label_is_human(value)
    return value


def _record_from_mapping(row: Mapping[str, object], index: int = 0) -> CanonicalRecord:
    missing = [field for field in _REQUIRED_RECORD_FIELDS if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Canonical input row is missing {', '.join(missing)}")
    label = row.get("provenance_label", row.get("label", ""))
    return CanonicalRecord(
        record_id=str(row["record_id"]).strip(),
        corpus=str(row["corpus"]).strip(),
        group_id=str(row["group_id"]).strip(),
        text=str(row["text"]),
        provenance_label=_label(label),
        partition=str(row.get("partition", row.get("split", "")) or "").strip(),
        source_order=int(row.get("source_order", index) or index),
    )


def read_canonical_table(table: Path | str | Iterable[Mapping[str, object]]) -> tuple[CanonicalRecord, ...]:
    """Read a CSV or mapping iterable and return fixed canonical records."""
    if isinstance(table, (str, Path)):
        with Path(table).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = list(table)
    records = tuple(_record_from_mapping(row, index) for index, row in enumerate(rows))
    if len({row.record_id for row in records}) != len(records):
        raise ValueError("Canonical input has duplicate record_id values")
    if any(not row.group_id for row in records):
        raise ValueError("Canonical input requires opaque group_id values")
    return tuple(sorted(records, key=lambda row: (row.corpus, row.group_id, row.record_id)))


def _seeded_record_hash(record: CanonicalRecord, seed: int, purpose: str) -> str:
    return hashlib.sha256(canonical_json({
        "seed": int(seed), "purpose": purpose, "corpus": record.corpus,
        "group_id": record.group_id, "record_id": record.record_id,
        "text_sha256": record.text_sha256,
    })).hexdigest()


def select_human_panel(
    records: Sequence[CanonicalRecord],
    *,
    seed: int = 20260824,
    calibration_per_corpus: int = CALIBRATION_PER_CORPUS,
    pilot_total: int = PILOT_HUMAN_CAP,
    dev_corpora: Sequence[str] = DEV_CORPORA,
    minimum_pilot_per_corpus: int = 1_000,
    pilot_per_corpus: Mapping[str, int] | None = None,
) -> tuple[tuple[CanonicalRecord, ...], tuple[CanonicalRecord, ...]]:
    """Select one hash-first human record per group, calibration before pilot."""
    corpora = tuple(str(corpus) for corpus in dev_corpora)
    if len(set(corpora)) != 4 or calibration_per_corpus <= 0 or pilot_total <= 0:
        raise ValueError("The pilot policy requires four corpora and positive caps")
    human = [record for record in records if record.is_human and record.corpus in corpora]
    representatives: dict[tuple[str, str], CanonicalRecord] = {}
    for record in human:
        key = (record.corpus, record.group_id)
        candidate_hash = _seeded_record_hash(record, seed, "group_representative")
        current = representatives.get(key)
        if current is None or (candidate_hash, record.record_id) < (
            _seeded_record_hash(current, seed, "group_representative"), current.record_id,
        ):
            representatives[key] = record
    by_corpus = {
        corpus: sorted(
            (record for (record_corpus, _), record in representatives.items() if record_corpus == corpus),
            key=lambda record: (_seeded_record_hash(record, seed, "calibration"), record.record_id),
        )
        for corpus in corpora
    }
    missing = [corpus for corpus in corpora if len(by_corpus[corpus]) < calibration_per_corpus]
    if missing:
        raise ValueError(f"Calibration shortfall for corpus/corpora: {missing}")
    calibration = tuple(
        record
        for corpus in corpora
        for record in by_corpus[corpus][:calibration_per_corpus]
    )
    calibration_groups = {(record.corpus, record.group_id) for record in calibration}
    remainder = [
        record for record in representatives.values()
        if (record.corpus, record.group_id) not in calibration_groups
    ]
    global_order = sorted(
        remainder,
        key=lambda record: (_seeded_record_hash(record, seed, "pilot_remainder"), record.corpus, record.record_id),
    )
    if pilot_per_corpus is not None:
        quotas = {str(corpus): int(value) for corpus, value in pilot_per_corpus.items()}
        if set(quotas) != set(corpora) or sum(quotas.values()) != pilot_total:
            raise ValueError("Pilot corpus quotas must cover four corpora and sum to the pilot cap")
        if any(value < minimum_pilot_per_corpus for value in quotas.values()):
            raise ValueError("Pilot corpus quota is below its minimum")
        pilot = tuple(
            record
            for corpus in corpora
            for record in [row for row in global_order if row.corpus == corpus][:quotas[corpus]]
        )
        counts = {corpus: sum(record.corpus == corpus for record in pilot) for corpus in corpora}
        if counts != quotas:
            raise ValueError(f"Pilot shortfall for locked corpus quotas: {counts}")
        return tuple(calibration), pilot
    if pilot_total % len(corpora) == 0:
        balanced_per_corpus = pilot_total // len(corpora)
        if balanced_per_corpus < minimum_pilot_per_corpus:
            raise ValueError("Balanced pilot cell is below its per-corpus minimum")
        pilot = tuple(
            record
            for corpus in corpora
            for record in [row for row in global_order if row.corpus == corpus][:balanced_per_corpus]
        )
        if len(pilot) != pilot_total:
            raise ValueError("Pilot shortfall: insufficient balanced corpus cells")
        return tuple(calibration), pilot
    required = []
    selected_ids: set[str] = set()
    for corpus in corpora:
        corpus_rows = [record for record in global_order if record.corpus == corpus]
        if len(corpus_rows) < minimum_pilot_per_corpus:
            raise ValueError(f"Pilot shortfall for corpus: {corpus}")
        required.extend(corpus_rows[:minimum_pilot_per_corpus])
        selected_ids.update(record.record_id for record in corpus_rows[:minimum_pilot_per_corpus])
    if pilot_total < len(required):
        raise ValueError("Pilot cap is smaller than its per-corpus minimum")
    pilot = required + [record for record in global_order if record.record_id not in selected_ids]
    pilot = tuple(pilot[:pilot_total])
    if len(pilot) != pilot_total:
        raise ValueError("Pilot shortfall: insufficient group-disjoint human records")
    counts = {corpus: sum(record.corpus == corpus for record in pilot) for corpus in corpora}
    if any(counts[corpus] < minimum_pilot_per_corpus for corpus in corpora):
        raise ValueError(f"Pilot minimum not met: {counts}")
    if {(record.corpus, record.group_id) for record in calibration} & {
        (record.corpus, record.group_id) for record in pilot
    }:
        raise RuntimeError("Calibration and pilot groups overlap")
    return tuple(calibration), tuple(pilot)


def line_wrap_variant(text: str, width: int = 80) -> str:
    """Wrap at word boundaries without changing any non-whitespace character."""
    if width < 2:
        raise ValueError("line-wrap width must be at least 2")
    words = _base_text(text).split()
    if not words:
        return ""
    return textwrap.fill(
        " ".join(words), width=width, break_long_words=False,
        break_on_hyphens=False, replace_whitespace=True,
    )


def _sentences(text: str) -> list[str]:
    # A deterministic punctuation-boundary splitter is enough for this pilot;
    # it does not normalize or rewrite the sentence contents.
    base = _base_text(text)
    return [piece for piece in re.split(r"(?<=[.!?]) (?=\S)", base) if piece]


def sentence_blocks_variant(text: str, block_size: int = 2) -> str:
    if block_size < 1:
        raise ValueError("sentence block size must be positive")
    sentences = _sentences(text)
    return "\n\n".join(
        " ".join(sentences[index:index + block_size])
        for index in range(0, len(sentences), block_size)
    )


def sentence_per_paragraph_variant(text: str) -> str:
    return "\n\n".join(_sentences(text))


# Explicit names mirror the frozen variant IDs for downstream adapters.
wrap_80_variant = line_wrap_variant
sentence_blocks_2_variant = sentence_blocks_variant


def reflow_variant(text: str, probe: str, *, width: int = 80, block_size: int = 2) -> str:
    if probe == "wrap_80":
        return line_wrap_variant(_base_text(text), width=width)
    if probe == "sentence_blocks_2":
        return sentence_blocks_variant(_base_text(text), block_size=block_size)
    if probe == "sentence_per_paragraph":
        return sentence_per_paragraph_variant(text)
    raise ValueError(f"Unknown active reflow probe: {probe}")


def validate_reflow_variant(original: str, variant: str, *, require_change: bool = True) -> dict[str, object]:
    preserved = _non_whitespace(original) == _non_whitespace(variant)
    changed = str(original) != str(variant)
    if not preserved:
        raise ValueError("Reflow changed the non-whitespace character sequence")
    if require_change and not changed:
        raise ValueError("Reflow variant did not change the text")
    return {
        "changed": changed,
        "non_whitespace_preserved": preserved,
        "eligible": changed and preserved,
        "original_text_sha256": _text_sha256(original),
        "variant_text_sha256": _text_sha256(variant),
    }


def build_reflow_variants(
    text: str,
    *,
    width: int = 80,
    block_size: int = 2,
    probes: Sequence[str] = PROBES,
    require_change: bool = True,
) -> tuple[ReflowVariant, ...]:
    base = _base_text(text)
    variants = []
    for probe in probes:
        variant = reflow_variant(base, probe, width=width, block_size=block_size)
        state = validate_reflow_variant(base, variant, require_change=require_change)
        variants.append(ReflowVariant(
            probe=probe,
            variant_id=probe,
            text=variant,
            text_sha256=_text_sha256(variant),
            non_whitespace_sha256=_non_whitespace_sha256(variant),
            changed=bool(state["changed"]),
            non_whitespace_preserved=bool(state["non_whitespace_preserved"]),
            eligible=bool(state["eligible"]),
        ))
    texts = (base, *(variant.text for variant in variants))
    if len(set(texts)) != len(texts):
        raise ValueError("Original and active reflow variants must be pairwise distinct")
    if len({_non_whitespace_sha256(value) for value in texts}) != 1:
        raise ValueError("Original and active reflows must share one non-whitespace SHA")
    return tuple(variants)


def _record_payload(record: CanonicalRecord, partition: str, *, normalized: bool = False) -> dict[str, object]:
    text = _base_text(record.text) if normalized else record.text
    return {
        "record_id": record.record_id,
        "corpus": record.corpus,
        "group_id": record.group_id,
        "partition": partition,
        "text_sha256": _text_sha256(text),
        "raw_text_sha256": record.text_sha256,
        "non_whitespace_sha256": _non_whitespace_sha256(text),
        "provenance_label": record.provenance_label,
        "word_count": len(_base_text(record.text).split()),
    }


def _group_disjoint(left: Sequence[CanonicalRecord], right: Sequence[CanonicalRecord]) -> bool:
    return not ({record.group_id for record in left} & {record.group_id for record in right})


def _select_partition(
    candidates: Sequence[CanonicalRecord],
    *,
    name: str,
    cap: int,
    used_groups: set[str],
) -> tuple[CanonicalRecord, ...]:
    selected: list[CanonicalRecord] = []
    for record in candidates:
        if record.group_id in used_groups:
            continue
        # A group is atomic: never partially assign it to calibration/pilot.
        group_records = [candidate for candidate in candidates if candidate.group_id == record.group_id]
        if len(selected) + len(group_records) > cap:
            continue
        selected.extend(group_records)
        used_groups.add(record.group_id)
        if len(selected) >= cap:
            break
    if not selected:
        raise ValueError(f"No group-disjoint records available for {name}")
    return tuple(sorted(selected, key=lambda row: (row.corpus, row.group_id, row.record_id)))


def prepare_pilot_manifest(
    table: Path | str | Iterable[Mapping[str, object]],
    paths: DeferralPaths,
    *,
    calibration_cap: int = 2_000,
    pilot_cap: int = PILOT_HUMAN_CAP,
    width: int = 80,
    block_size: int = 2,
    seed: int = 20260824,
    endpoint_revisions: Mapping[str, str] | None = None,
    candidate_token_counts: Mapping[str, Mapping[str, Mapping[str, int] | Sequence[int]]] | None = None,
    token_cap: int = 460,
    max_paired_target_words: int | None = None,
    pilot_per_corpus: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Create the locked, group-disjoint pilot challenge manifest.

    The manifest contains hashes and opaque IDs, never passage text.  The
    candidate ordering is the canonical `(corpus, group_id, record_id)` order,
    so rerunning preparation cannot silently draw a different panel.
    """
    records = read_canonical_table(table)
    if max_paired_target_words is not None and max_paired_target_words <= 0:
        raise ValueError("Maximum paired target words must be positive")
    human = tuple(
        record for record in records
        if record.is_human and (
            max_paired_target_words is None
            or len(_base_text(record.text).split()) <= max_paired_target_words
        )
    )
    if not human:
        raise ValueError("Pilot preparation requires human calibration/pilot candidates")
    explicit_cal = tuple(record for record in human if record.partition.casefold() == "calibration")
    explicit_pilot = tuple(record for record in human if record.partition.casefold() == "pilot")
    used: set[str] = set()
    if calibration_cap == len(DEV_CORPORA) * CALIBRATION_PER_CORPUS and pilot_cap == PILOT_HUMAN_CAP:
        eligible_records = []
        for record in human:
            if record.corpus not in DEV_CORPORA:
                continue
            try:
                build_reflow_variants(record.text, width=width, block_size=block_size)
            except ValueError:
                continue
            if candidate_token_counts is not None:
                if record.record_id not in candidate_token_counts:
                    raise ValueError(f"Missing candidate token panel: {record.record_id}")
                try:
                    validate_triplet_token_budget(candidate_token_counts[record.record_id], cap=token_cap)
                except ValueError:
                    continue
            eligible_records.append(record)
        calibration, pilot = select_human_panel(
            eligible_records, seed=seed, calibration_per_corpus=CALIBRATION_PER_CORPUS,
            pilot_total=pilot_cap, pilot_per_corpus=pilot_per_corpus,
        )
    elif explicit_cal or explicit_pilot:
        calibration = explicit_cal
        pilot = explicit_pilot
        if not calibration or not pilot:
            raise ValueError("Explicit partitions must contain both calibration and pilot records")
        if len(calibration) > calibration_cap or len(pilot) > pilot_cap:
            raise ValueError("Explicit partition exceeds its fixed cap")
        if not _group_disjoint(calibration, pilot):
            raise ValueError("Calibration and pilot partitions share a group")
    else:
        calibration = _select_partition(human, name="calibration", cap=calibration_cap, used_groups=used)
        pilot = _select_partition(human, name="pilot", cap=pilot_cap, used_groups=used)
    if not _group_disjoint(calibration, pilot):
        raise ValueError("Calibration and pilot partitions share a group")

    endpoint_revisions = dict(endpoint_revisions or {})
    for endpoint in ENDPOINT_ROLES:
        endpoint_revisions.setdefault(endpoint, "unspecified")
    pilot_rows = []
    ineligible = []
    for record in pilot:
        try:
            reflows = build_reflow_variants(
                record.text, width=width, block_size=block_size,
                probes=PROBES, require_change=True,
            )
        except ValueError as error:
            ineligible.append({"record_id": record.record_id, "reason": str(error)})
            reflows = ()
        base = _base_text(record.text)
        variants = [
            {
                "probe": variant.probe, "variant_id": variant.variant_id,
                "text_sha256": variant.text_sha256,
                "non_whitespace_sha256": variant.non_whitespace_sha256,
                "changed": variant.changed,
                "non_whitespace_preserved": variant.non_whitespace_preserved,
            }
            for variant in reflows
        ]
        pilot_rows.append({**_record_payload(record, "pilot", normalized=True), "variants": variants})
    canonical_rows = [_record_payload(record, record.partition or "input") for record in records]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": PILOT_STAGE,
        "seed": int(seed),
        "max_paired_target_words": max_paired_target_words,
        "pilot_per_corpus": None if pilot_per_corpus is None else {
            str(corpus): int(value) for corpus, value in sorted(pilot_per_corpus.items())
        },
        "source_table_sha256": _sha256(canonical_rows),
        "candidate_order": "corpus,group_id,record_id",
        "caps": {"calibration_records": int(calibration_cap), "pilot_records": int(pilot_cap)},
        "transform": {"width": int(width), "sentence_block_size": int(block_size)},
        "probes": list(PROBES),
        "variant_level": VARIANT_LEVEL,
        "endpoint_roles": dict(ENDPOINT_ROLES),
        "endpoint_revisions": endpoint_revisions,
        "calibration": [_record_payload(record, "calibration", normalized=True) for record in calibration],
        "pilot": pilot_rows,
        "ineligible": ineligible,
        "query_formula": "Q=N_original+3*N_positive",
        "final_stage": {"authorization_required": True, "sealed": False},
    }
    paths.root.mkdir(parents=True, exist_ok=True)
    if paths.lock.exists():
        existing = verify_pilot_lock(paths)["payload"]
        if canonical_json(existing) != canonical_json(manifest):
            raise RuntimeError("Existing pilot lock disagrees with the requested manifest")
        return existing
    if paths.manifest.exists() or paths.score_table.exists() or paths.worklist.exists():
        raise RuntimeError("Pilot state exists without an immutable pilot lock")
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    lock_forecasts(paths.lock, manifest)
    return manifest


def verify_pilot_lock(paths: DeferralPaths) -> dict[str, object]:
    if not paths.lock.exists():
        raise RuntimeError("Pilot manifest lock is missing")
    envelope = verify_lock(paths.lock)
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or payload.get("stage") != PILOT_STAGE:
        raise RuntimeError("Not a selective-deferral pilot lock")
    if paths.manifest.exists():
        try:
            copy = json.loads(paths.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Pilot manifest copy cannot be read") from error
        if canonical_json(copy) != canonical_json(payload):
            raise RuntimeError("Pilot manifest copy disagrees with immutable lock")
    return envelope


def _normalize_generator_families(
    generator_families: Mapping[str, str] | Sequence[Mapping[str, object] | Sequence[str]] | None,
) -> tuple[tuple[str, str], ...]:
    if generator_families is None:
        raise ValueError("Three immutable generator family/revision pairs must be supplied")
    elif isinstance(generator_families, Mapping):
        pairs = tuple((str(family), str(revision)) for family, revision in generator_families.items())
    else:
        normalized = []
        for item in generator_families:
            if isinstance(item, Mapping):
                family = item.get("family", item.get("generator_family", item.get("id", "")))
                revision = item.get("revision", item.get("generator_revision", ""))
                normalized.append((str(family), str(revision)))
            else:
                values = tuple(item)
                if len(values) != 2:
                    raise ValueError("Generator family entries require family and revision")
                normalized.append((str(values[0]), str(values[1])))
        pairs = tuple(normalized)
    if len(pairs) != GENERATOR_FAMILY_COUNT or len(set(pairs)) != GENERATOR_FAMILY_COUNT:
        raise ValueError("Exactly three unique generator family/revision pairs are required")
    if any(not family or not revision for family, revision in pairs):
        raise ValueError("Generator family and revision cannot be empty")
    return tuple(sorted(pairs))


def _render_generation_prompt(
    template: str, topic: str, target_length: int,
    min_word_count: int, max_word_count: int,
) -> str:
    if not str(template).strip():
        raise ValueError("Generation prompt template is required")
    forbidden = ("{text}", "{human_text}", "{record_text}", "{source_text}")
    if any(token in template for token in forbidden):
        raise ValueError("Generation prompt may not interpolate human text")
    try:
        prompt = template.format(
            topic=str(topic), target_length=int(target_length),
            min_word_count=int(min_word_count), max_word_count=int(max_word_count),
        )
    except (KeyError, ValueError) as error:
        raise ValueError("Prompt template may use only topic and frozen length fields") from error
    if not prompt.strip():
        raise ValueError("Rendered generation prompt is empty")
    return prompt


def _generation_attempt_seed(locked_seed: int, request_id: str, attempt: int) -> int:
    payload = f"{int(locked_seed)}:{request_id}:{int(attempt)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647


def prepare_generation_requests(
    paths: DeferralPaths,
    topic_map: Mapping[str, object],
    *,
    generator_families: Mapping[str, str] | Sequence[Mapping[str, object] | Sequence[str]] | None = None,
    prompt_template: str | Mapping[str, str] = "Write an original passage about {topic} in approximately {target_length} words.",
    decoding: Mapping[str, object] | None = None,
    seed: int = 20260824,
    retry: int = 0,
    target_length: int | None = None,
    length_tolerance_fraction: float = 0.10,
    length_tolerance_min_words: int = 15,
) -> tuple[GenerationRequest, ...]:
    """Prepare and lock provider-neutral requests without human passage text."""
    pilot_lock = verify_pilot_lock(paths)
    if not paths.human_token_lock.exists():
        raise RuntimeError("Human reflow token panels must be locked before generation requests")
    token_envelope = verify_lock(paths.human_token_lock)
    if token_envelope["payload"].get("pilot_lock_sha256") != pilot_lock["sha256"]:
        raise RuntimeError("Human token panels are bound to a different pilot lock")
    if not isinstance(topic_map, Mapping) or not topic_map:
        raise ValueError("A non-empty record-to-topic map is required")
    if retry < 0 or (target_length is not None and target_length <= 0):
        raise ValueError("Retry must be nonnegative and target length positive")
    if not 0 <= length_tolerance_fraction <= 1 or length_tolerance_min_words < 0:
        raise ValueError("Invalid generation length tolerance")
    families = _normalize_generator_families(generator_families)
    decoding_payload = dict(decoding or {"temperature": 0.7, "top_p": 0.95})
    pilot_rows = [row for row in pilot_lock["payload"].get("pilot", ()) if set(PROBES) <= {
        str(variant["variant_id"]) for variant in row.get("variants", ())
    }]
    if len(pilot_rows) != len(pilot_lock["payload"].get("pilot", ())):
        raise RuntimeError("Every locked pilot human must have a complete active reflow panel")
    by_corpus: dict[str, list[Mapping[str, object]]] = {}
    for row in pilot_rows:
        corpus = str(row["corpus"])
        if str(row["record_id"]) not in topic_map:
            raise ValueError(f"Topic map is missing record: {row['record_id']}")
        by_corpus.setdefault(corpus, []).append(row)
    request_rows: list[GenerationRequest] = []
    family_order = tuple(family for family, _ in families)
    if isinstance(prompt_template, Mapping):
        prompt_templates = {str(key): str(value) for key, value in prompt_template.items()}
    else:
        prompt_templates = {"default": str(prompt_template)}
    for corpus in sorted(by_corpus):
        template = prompt_templates.get(corpus, prompt_templates.get("default", ""))
        rows = sorted(
            by_corpus[corpus],
            key=lambda row: (
                _sha256({"seed": int(seed), "purpose": "generator_assignment", "record_id": str(row["record_id"])}),
                str(row["record_id"]),
            ),
        )
        for index, row in enumerate(rows):
            family, revision = families[index % len(families)]
            topic_value = topic_map[str(row["record_id"])]
            topic = str(topic_value.get("topic", topic_value) if isinstance(topic_value, Mapping) else topic_value)
            if not topic.strip() or _text_sha256(_base_text(topic)) == str(row["text_sha256"]):
                raise ValueError(f"Topic must be non-empty and cannot reproduce the human passage: {row['record_id']}")
            paired_length = int(row.get("word_count", 0))
            request_length = int(target_length if target_length is not None else paired_length)
            if request_length <= 0:
                raise ValueError(f"Missing paired human word count: {row['record_id']}")
            tolerance = max(
                int(length_tolerance_min_words),
                int(math.ceil(request_length * float(length_tolerance_fraction))),
            )
            min_word_count = max(1, request_length - tolerance)
            max_word_count = request_length + tolerance
            prompt = _render_generation_prompt(
                template, topic, request_length, min_word_count, max_word_count,
            )
            prompt_hash = _text_sha256(prompt)
            identity = {
                "record_id": str(row["record_id"]), "corpus": corpus,
                "generator_family": family, "generator_revision": revision,
                "prompt_sha256": prompt_hash, "seed": int(seed), "retry": int(retry),
                "target_length": request_length, "min_word_count": min_word_count,
                "max_word_count": max_word_count, "decoding": decoding_payload,
            }
            request_rows.append(GenerationRequest(
                request_id=_sha256(identity), record_id=str(row["record_id"]), corpus=corpus,
                generator_family=family, generator_revision=revision, prompt=prompt,
                prompt_sha256=prompt_hash, seed=int(seed), retry=int(retry),
                target_length=request_length, min_word_count=min_word_count,
                max_word_count=max_word_count, decoding=decoding_payload,
            ))
        counts = {family: sum(request.generator_family == family for request in request_rows if request.corpus == corpus) for family in family_order}
        if max(counts.values(), default=0) - min(counts.values(), default=0) > 1:
            raise RuntimeError(f"Generator assignment is not balanced within {corpus}: {counts}")
    request_rows = sorted(request_rows, key=lambda row: (row.corpus, row.record_id, row.generator_family))
    if len({row.request_id for row in request_rows}) != len(request_rows):
        raise RuntimeError("Generation request IDs are not unique")
    rows_payload = [
        {
            "request_id": row.request_id, "record_id": row.record_id, "corpus": row.corpus,
            "generator_family": row.generator_family, "generator_revision": row.generator_revision,
            "prompt": row.prompt, "prompt_sha256": row.prompt_sha256, "seed": row.seed,
            "retry": row.retry, "target_length": row.target_length,
            "min_word_count": row.min_word_count, "max_word_count": row.max_word_count,
            "decoding": dict(row.decoding),
        }
        for row in request_rows
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "generation_requests",
        "pilot_lock_sha256": pilot_lock["sha256"],
        "human_token_lock_sha256": token_envelope["sha256"],
        "generator_families": [{"family": family, "revision": revision} for family, revision in families],
        "topic_map": {
            str(key): str(value.get("topic", value) if isinstance(value, Mapping) else value)
            for key, value in sorted(topic_map.items())
            if str(key) in {str(row["record_id"]) for row in pilot_rows}
        },
        "prompt_template": prompt_templates,
        "decoding": decoding_payload,
        "seed": int(seed), "retry": int(retry),
        "target_length": "paired_human" if target_length is None else int(target_length),
        "length_tolerance_fraction": float(length_tolerance_fraction),
        "length_tolerance_min_words": int(length_tolerance_min_words),
        "requests": rows_payload,
        "requests_sha256": _sha256(rows_payload),
    }
    if paths.generation_lock.exists():
        existing = verify_generation_lock(paths)["payload"]
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError("Existing generation-request lock disagrees with requested protocol")
        return tuple(_generation_request_from_mapping(row) for row in existing["requests"])
    paths.generation_json.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = paths.generation_json.with_suffix(".json.tmp")
    json_temporary.write_text(json.dumps(rows_payload, sort_keys=True, indent=2), encoding="utf-8")
    fields = ("request_id", "record_id", "corpus", "generator_family", "generator_revision", "prompt", "prompt_sha256", "seed", "retry", "target_length", "min_word_count", "max_word_count", "decoding")
    csv_temporary = paths.generation_csv.with_suffix(".csv.tmp")
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows_payload:
            writer.writerow({**row, "decoding": json.dumps(row["decoding"], sort_keys=True)})
    json_temporary.replace(paths.generation_json)
    csv_temporary.replace(paths.generation_csv)
    payload["request_json_sha256"] = _file_sha256(paths.generation_json)
    payload["request_csv_sha256"] = _file_sha256(paths.generation_csv)
    lock_forecasts(paths.generation_lock, payload)
    return tuple(request_rows)


def _generation_request_from_mapping(row: Mapping[str, object]) -> GenerationRequest:
    decoding = row.get("decoding", {})
    if isinstance(decoding, str):
        decoding = json.loads(decoding)
    return GenerationRequest(
        request_id=str(row["request_id"]), record_id=str(row["record_id"]), corpus=str(row["corpus"]),
        generator_family=str(row["generator_family"]), generator_revision=str(row["generator_revision"]),
        prompt=str(row["prompt"]), prompt_sha256=str(row["prompt_sha256"]), seed=int(row["seed"]),
        retry=int(row["retry"]), target_length=int(row["target_length"]),
        min_word_count=int(row.get("min_word_count", row["target_length"])),
        max_word_count=int(row.get("max_word_count", row["target_length"])), decoding=dict(decoding),
    )


def verify_generation_lock(paths: DeferralPaths) -> dict[str, object]:
    if not paths.generation_lock.exists():
        raise RuntimeError("Generation-request lock is missing")
    envelope = verify_lock(paths.generation_lock)
    payload = envelope["payload"]
    if payload.get("pilot_lock_sha256") != verify_pilot_lock(paths)["sha256"]:
        raise RuntimeError("Generation requests are bound to a different pilot lock")
    if _sha256(payload.get("requests", ())) != payload.get("requests_sha256"):
        raise RuntimeError("Generation request hash mismatch")
    for path, field_name in ((paths.generation_json, "request_json_sha256"), (paths.generation_csv, "request_csv_sha256")):
        if not path.exists() or _file_sha256(path) != payload.get(field_name):
            raise RuntimeError(f"Generation request file mismatch: {path.name}")
    return envelope


def mage_effective_input(text: str) -> str:
    """Pinned MAGE preprocessing used for the negative-control equality check."""
    return " ".join(str(text).split())


def mage_effective_input_hash(text: str) -> str:
    return _text_sha256(mage_effective_input(text))


def validate_mage_effective_input_hashes(original: str, variants: Mapping[str, str] | Sequence[str]) -> str:
    hashes = [mage_effective_input_hash(original)]
    hashes.extend(
        mage_effective_input_hash(text)
        for text in (variants.values() if isinstance(variants, Mapping) else variants)
    )
    if len(set(hashes)) != 1:
        raise ValueError("MAGE effective-input hashes differ across whitespace-only reflows")
    return hashes[0]


def _output_rows(outputs: Path | str | Iterable[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    if isinstance(outputs, (str, Path)):
        with Path(outputs).open("r", encoding="utf-8-sig", newline="") as handle:
            return tuple(csv.DictReader(handle))
    return tuple(outputs)


def import_generation_outputs(
    paths: DeferralPaths,
    outputs: Path | str | Iterable[Mapping[str, object]],
    *,
    token_counts: Mapping[str, Mapping[str, Mapping[str, int] | Sequence[int]]] | None = None,
    token_cap: int = 460,
) -> tuple[dict[str, object], ...]:
    """Validate one output per locked request and lock a symmetric AI panel."""
    requests_lock = verify_generation_lock(paths)["payload"]
    request_map = {str(row["request_id"]): row for row in requests_lock["requests"]}
    rows = _output_rows(outputs)
    if len(rows) != len(request_map):
        raise ValueError(f"Expected exactly one output per locked request: {len(request_map)}")
    seen: set[str] = set()
    panels: list[dict[str, object]] = []
    for row in rows:
        request_id = str(row.get("request_id", ""))
        request = request_map.get(request_id)
        if request is None or request_id in seen:
            raise ValueError(f"Unknown or duplicate generation request: {request_id}")
        seen.add(request_id)
        for field_name in ("generator_family", "generator_revision", "retry"):
            if str(row.get(field_name, "")) != str(request[field_name]):
                raise ValueError(f"Generation provenance mismatch for {request_id}: {field_name}")
        if "target_length" in row and str(row["target_length"]) != str(request["target_length"]):
            raise ValueError(f"Generation provenance mismatch for {request_id}: target_length")
        attempt = int(row.get("attempt", 0) or 0)
        if attempt < 0 or attempt > int(request["retry"]):
            raise ValueError(f"Generation retry attempt is outside the frozen policy: {request_id}")
        if "seed" in row and str(row["seed"]) != str(_generation_attempt_seed(int(request["seed"]), request_id, attempt)):
            raise ValueError(f"Generation provenance mismatch for {request_id}: seed")
        if "decoding" in row and row["decoding"] not in (None, ""):
            decoding = row["decoding"]
            if isinstance(decoding, str):
                decoding = json.loads(decoding)
            if canonical_json(decoding) != canonical_json(request["decoding"]):
                raise ValueError(f"Generation provenance mismatch for {request_id}: decoding")
        text = str(row.get("text", row.get("generated_text", "")))
        if not text.strip():
            raise ValueError(f"Empty generated output: {request_id}")
        output_word_count = len(_base_text(text).split())
        if not int(request["min_word_count"]) <= output_word_count <= int(request["max_word_count"]):
            raise ValueError(f"Generated output violates frozen length tolerance: {request_id}")
        if re.search(r"[.!?][\"')\]]*$", text.strip()) is None:
            raise ValueError(f"Generated output is not a complete passage: {request_id}")
        selected_word_count = int(row.get("selected_word_count", output_word_count) or output_word_count)
        raw_word_count = int(row.get("raw_word_count", selected_word_count) or selected_word_count)
        prefix_rank = int(row.get("prefix_rank", 0) or 0)
        if selected_word_count != output_word_count or raw_word_count < selected_word_count or prefix_rank < 0:
            raise ValueError(f"Generation prefix provenance mismatch: {request_id}")
        variants = build_reflow_variants(text, probes=PROBES)
        by_variant = {variant.variant_id: variant for variant in variants}
        panel_counts = None
        if token_counts is not None:
            panel_counts = token_counts.get(request_id)
        elif row.get("token_counts"):
            panel_counts = row["token_counts"]
            if isinstance(panel_counts, str):
                panel_counts = json.loads(panel_counts)
        if panel_counts is None:
            raise ValueError(f"Missing atomic RADAR/MAGE token counts for {request_id}")
        validate_triplet_token_budget(panel_counts, cap=token_cap)
        pair_id = _sha256({"stage": PILOT_STAGE, "source_record_id": request["record_id"]})
        ai_record_id = f"ai:{request_id}"
        panels.append({
            "request_id": request_id,
            "pair_id": pair_id,
            "human_record_id": str(request["record_id"]),
            "ai_record_id": ai_record_id,
            "corpus": str(request["corpus"]),
            "generator_family": str(request["generator_family"]),
            "generator_revision": str(request["generator_revision"]),
            "target_length": int(request["target_length"]),
            "output_word_count": output_word_count,
            "raw_word_count": raw_word_count,
            "prefix_rank": prefix_rank,
            "prefix_used": bool(int(row.get("prefix_used", raw_word_count != selected_word_count) or 0)),
            "attempt": attempt,
            "base_text_sha256": _text_sha256(_base_text(text)),
            "non_whitespace_sha256": _non_whitespace_sha256(text),
            "mage_effective_input_sha256": mage_effective_input_hash(text),
            "variants": {
                key: {
                    "text_sha256": by_variant[key].text_sha256,
                    "non_whitespace_sha256": by_variant[key].non_whitespace_sha256,
                }
                for key in PROBES
            },
        })
    if seen != set(request_map):
        raise ValueError("Generation outputs are incomplete")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "generated_panel",
        "generation_lock_sha256": _file_sha256(paths.generation_lock),
        "token_cap": int(token_cap),
        "panels": sorted(panels, key=lambda row: str(row["request_id"])),
        "panel_sha256": _sha256(sorted(panels, key=lambda row: str(row["request_id"]))),
    }
    if paths.panel_lock.exists():
        existing = verify_lock(paths.panel_lock)["payload"]
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError("Existing generated-panel lock disagrees with imported outputs")
    else:
        lock_forecasts(paths.panel_lock, payload)
    return tuple(payload["panels"])


def _locked_panels(paths: DeferralPaths) -> tuple[Mapping[str, object], ...]:
    if not paths.panel_lock.exists():
        return ()
    payload = verify_lock(paths.panel_lock)["payload"]
    if payload.get("generation_lock_sha256") != _file_sha256(paths.generation_lock):
        raise RuntimeError("Generated panel is bound to a different generation lock")
    panels = tuple(payload.get("panels", ()))
    if _sha256(sorted(panels, key=lambda row: str(row["request_id"]))) != payload.get("panel_sha256"):
        raise RuntimeError("Generated panel hash mismatch")
    return panels


def _manifest_variant_lookup(paths: DeferralPaths, manifest: Mapping[str, object], probe: str) -> dict[str, tuple[str, str]]:
    expected: dict[str, tuple[str, str]] = {}
    for row in manifest.get("pilot", ()):
        for variant in row.get("variants", ()):
            if str(variant.get("variant_id")) == probe:
                expected[str(row["record_id"])] = (
                    str(row["text_sha256"]), str(variant["text_sha256"]),
                )
    for panel in _locked_panels(paths):
        expected[str(panel["ai_record_id"])] = (
            str(panel["base_text_sha256"]), str(panel["variants"][probe]["text_sha256"]),
        )
    return expected


def export_manual_audit(
    paths: DeferralPaths,
    *,
    probe: str,
    texts: Path | str | Iterable[Mapping[str, object]],
    count: int = 300,
    seed: int = 20260824,
) -> tuple[dict[str, object], ...]:
    """Export exactly ``count`` opaque audit items for one probe."""
    if probe not in PROBES or count <= 0:
        raise ValueError("Manual audit requires a locked probe and positive count")
    manifest = verify_pilot_lock(paths)["payload"]
    lookup = _manifest_variant_lookup(paths, manifest, probe)
    if len(lookup) < count:
        raise ValueError(f"Manual audit shortfall for {probe}: {len(lookup)}/{count}")
    records = sorted(
        lookup.items(),
        key=lambda item: hashlib.sha256(canonical_json({"seed": seed, "record_id": item[0]})).hexdigest(),
    )[:count]
    panels = _locked_panels(paths)
    request_to_ai = {str(panel["request_id"]): str(panel["ai_record_id"]) for panel in panels}
    text_map: dict[str, str] = {}
    for row in _output_rows(texts):
        record_id = str(row.get("record_id") or request_to_ai.get(str(row.get("request_id", "")), ""))
        text = _base_text(str(row.get("text", row.get("generated_text", ""))))
        if record_id and text:
            if record_id in text_map:
                raise ValueError(f"Duplicate manual-audit text: {record_id}")
            text_map[record_id] = text
    output = []
    for record_id, hashes in records:
        if record_id not in text_map or _text_sha256(text_map[record_id]) != hashes[0]:
            raise ValueError(f"Manual-audit text is missing or does not match its lock: {record_id}")
        transform = manifest.get("transform", {})
        variant = reflow_variant(
            text_map[record_id], probe,
            width=int(transform.get("width", 80)),
            block_size=int(transform.get("sentence_block_size", 2)),
        )
        if _text_sha256(variant) != hashes[1]:
            raise ValueError(f"Manual-audit variant does not match its lock: {record_id}")
        output.append({
            "record_id": record_id, "probe": probe,
            "original_text": text_map[record_id], "variant_text": variant,
            "original_text_sha256": hashes[0], "variant_text_sha256": hashes[1],
            "valid": "", "audit_label": "",
        })
    rows = tuple(output)
    audit_csv = paths.manual_audit_csv_for(probe)
    if audit_csv.exists():
        raise RuntimeError("Manual audit export already exists")
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "record_id", "probe", "original_text", "variant_text",
        "original_text_sha256", "variant_text_sha256", "valid", "audit_label",
    )
    with audit_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def import_manual_audit(
    paths: DeferralPaths,
    table: Path | str | Iterable[Mapping[str, object]],
    *,
    probe: str,
    count: int = 300,
    minimum_valid: int | None = None,
) -> dict[str, object]:
    if probe not in PROBES:
        raise ValueError("Unknown manual-audit probe")
    minimum_valid = count - 3 if minimum_valid is None else int(minimum_valid)
    rows = _output_rows(table)
    if len(rows) != count:
        raise ValueError(f"Manual audit requires exactly {count} rows")
    expected = _manifest_variant_lookup(paths, verify_pilot_lock(paths)["payload"], probe)
    seen = set()
    valid = 0
    normalized = []
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if record_id in seen or record_id not in expected:
            raise ValueError(f"Manual audit row is not a unique locked item: {record_id}")
        seen.add(record_id)
        if str(row.get("probe", "")) != probe:
            raise ValueError("Manual audit probe mismatch")
        original_hash, variant_hash = expected[record_id]
        if str(row.get("original_text_sha256")) != original_hash or str(row.get("variant_text_sha256")) != variant_hash:
            raise ValueError(f"Manual audit text hash mismatch: {record_id}")
        if str(row.get("valid", "")).casefold() not in {"0", "1", "false", "true", "no", "yes", "invalid", "valid"}:
            raise ValueError(f"Manual audit requires an explicit valid judgment: {record_id}")
        is_valid = str(row.get("valid", "")).casefold() in {"1", "true", "yes", "valid"}
        valid += int(is_valid)
        normalized.append({**dict(row), "valid": int(is_valid)})
    if valid < minimum_valid:
        raise ValueError(f"Manual audit validity floor not met: {valid}/{count}")
    payload = {
        "schema_version": SCHEMA_VERSION, "stage": "manual_audit_validation",
        "pilot_lock_sha256": verify_pilot_lock(paths)["sha256"], "probe": probe,
        "count": count, "valid": valid, "minimum_valid": minimum_valid,
        "rows_sha256": _sha256(normalized),
    }
    audit_lock = paths.manual_audit_lock_for(probe)
    if audit_lock.exists():
        existing = verify_lock(audit_lock)["payload"]
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError("Existing manual-audit validation lock disagrees")
    else:
        lock_forecasts(audit_lock, payload)
    return payload


def _expected_variants(
    manifest: Mapping[str, object],
    panels: Sequence[Mapping[str, object]] = (),
) -> dict[tuple[str, str], str]:
    expected: dict[tuple[str, str], str] = {}
    for row in manifest.get("calibration", ()):
        expected[(str(row["record_id"]), "original")] = str(row["text_sha256"])
    for row in manifest.get("pilot", ()):
        record_id = str(row["record_id"])
        expected[(record_id, "original")] = str(row["text_sha256"])
        expected[(record_id, "original_repeat")] = str(row["text_sha256"])
        for variant in row.get("variants", ()):
            expected[(record_id, str(variant["variant_id"]))] = str(variant["text_sha256"])
    for panel in panels:
        record_id = str(panel["ai_record_id"])
        expected[(record_id, "original")] = str(panel["base_text_sha256"])
        expected[(record_id, "original_repeat")] = str(panel["base_text_sha256"])
        for probe in PROBES:
            expected[(record_id, probe)] = str(panel["variants"][probe]["text_sha256"])
    return expected


def _score_row_from_mapping(row: Mapping[str, object]) -> ScoreRow:
    required = ("record_id", "variant_id", "endpoint", "detector_revision", "text_sha256", "provenance_label")
    missing = [field for field in required if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Canonical score row is missing {', '.join(missing)}")
    value = row.get("canonical_ai_score", "")
    score = None if value in (None, "", "NA", "na", "null") else float(value)
    native = row.get("native_score", "")
    native_score = None if native in (None, "", "NA", "na", "null") else float(native)
    return ScoreRow(
        str(row["record_id"]), str(row["variant_id"]), str(row["endpoint"]),
        str(row["detector_revision"]), str(row["text_sha256"]),
        _label(row["provenance_label"]), score, native_score,
        None if row.get("input_token_count", "") in (None, "") else int(row["input_token_count"]),
        str(row.get("truncated", "0")).casefold() in {"1", "true", "yes"},
        str(row.get("failure", "") or ""),
    )


def read_canonical_scores(table: Path | str | Iterable[Mapping[str, object]]) -> tuple[ScoreRow, ...]:
    if isinstance(table, (str, Path)):
        with Path(table).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = list(table)
    return tuple(row if isinstance(row, ScoreRow) else _score_row_from_mapping(row) for row in rows)


def validate_canonical_scores(
    rows: Iterable[ScoreRow | Mapping[str, object]],
    manifest: Mapping[str, object],
    *,
    endpoint_revisions: Mapping[str, str] | None = None,
    require_originals: bool = False,
    panels: Sequence[Mapping[str, object]] = (),
) -> tuple[ScoreRow, ...]:
    expected = _expected_variants(manifest, panels)
    revisions = dict(manifest.get("endpoint_revisions", {}))
    revisions.update(endpoint_revisions or {})
    normalized = tuple(row if isinstance(row, ScoreRow) else _score_row_from_mapping(row) for row in rows)
    seen: set[tuple[str, str, str]] = set()
    manifest_labels = {
        str(row["record_id"]): str(row["provenance_label"])
        for row in (*manifest.get("calibration", ()), *manifest.get("pilot", ()))
    }
    manifest_labels.update({str(panel["ai_record_id"]): "ai" for panel in panels})
    for row in normalized:
        if row.key in seen:
            raise ValueError(f"Duplicate canonical score row: {row.key}")
        seen.add(row.key)
        expected_hash = expected.get((row.record_id, row.variant_id))
        if expected_hash is None:
            raise ValueError(f"Score row is not in the locked pilot worklist: {row.key}")
        if row.text_sha256 != expected_hash:
            raise ValueError(f"Text hash mismatch for score row: {row.key}")
        if (
            row.record_id not in manifest_labels
            or _label_is_human(row.provenance_label) != _label_is_human(manifest_labels[row.record_id])
        ):
            raise ValueError(f"Provenance label mismatch for score row: {row.key}")
        expected_revision = revisions.get(row.endpoint)
        if expected_revision is None:
            raise ValueError(f"Endpoint is not in the locked panel: {row.endpoint}")
        if row.detector_revision != expected_revision:
            raise ValueError(f"Detector revision mismatch for {row.endpoint}")
    if require_originals:
        expected_originals = {(record_id, endpoint) for record_id, _ in expected for endpoint in revisions}
        actual_originals = {(row.record_id, row.endpoint) for row in normalized if row.variant_id == "original"}
        if not expected_originals <= actual_originals:
            raise ValueError("Canonical score table is missing original scores")
    return tuple(sorted(normalized, key=lambda row: row.key))


def import_canonical_scores(
    table: Path | str | Iterable[Mapping[str, object]],
    paths: DeferralPaths,
    *,
    endpoint_revisions: Mapping[str, str] | None = None,
    require_originals: bool = False,
) -> tuple[ScoreRow, ...]:
    manifest = verify_pilot_lock(paths)["payload"]
    imported = validate_canonical_scores(
        read_canonical_scores(table), manifest,
        endpoint_revisions=endpoint_revisions, require_originals=False,
        panels=_locked_panels(paths),
    )
    merged = {row.key: row for row in read_canonical_scores(paths.score_table)} if paths.score_table.exists() else {}
    for row in imported:
        if row.key in merged and merged[row.key] != row:
            raise RuntimeError(f"Existing canonical score disagrees with imported row: {row.key}")
        merged[row.key] = row
    rows = validate_canonical_scores(
        merged.values(), manifest,
        endpoint_revisions=endpoint_revisions, require_originals=require_originals,
        panels=_locked_panels(paths),
    )
    paths.score_table.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "record_id", "variant_id", "endpoint", "detector_revision", "text_sha256",
        "provenance_label", "canonical_ai_score", "native_score", "input_token_count",
        "truncated", "failure",
    )
    temporary = paths.score_table.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})
    temporary.replace(paths.score_table)
    return rows


def _score_values(scores: Mapping[object, object]) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for key, value in scores.items():
        if isinstance(key, tuple) and len(key) == 2:
            record_id, endpoint = map(str, key)
        else:
            raise ValueError("Original scores must be keyed by (record_id, endpoint)")
        if isinstance(value, ScoreRow):
            if value.canonical_ai_score is None:
                continue
            values[(record_id, endpoint)] = float(value.canonical_ai_score)
        else:
            values[(record_id, endpoint)] = float(value)
    return values


def query_accounting(n_original: int, n_positive: int) -> int:
    if n_original < 0 or n_positive < 0 or n_positive > n_original:
        raise ValueError("Require 0 <= N_positive <= N_original")
    return int(n_original + 3 * n_positive)


def radar_positive(score: float, threshold: float) -> bool:
    """Frozen decision rule: equality at the threshold is not positive."""
    return float(score) > float(threshold)


def calibrate_radar_threshold(
    paths: DeferralPaths,
    scores: Mapping[str, object] | Iterable[ScoreRow | Mapping[str, object]],
) -> dict[str, object]:
    """Lock the empirical 95th percentile from exactly 2,000 human scores."""
    manifest = verify_pilot_lock(paths)["payload"]
    calibration = tuple(manifest.get("calibration", ()))
    if len(calibration) != 4 * CALIBRATION_PER_CORPUS:
        raise ValueError("RADAR threshold calibration requires exactly 2,000 calibration records")
    corpus_counts = {
        corpus: sum(str(row.get("corpus")) == corpus for row in calibration)
        for corpus in DEV_CORPORA
    }
    if any(count != CALIBRATION_PER_CORPUS for count in corpus_counts.values()):
        raise ValueError(f"RADAR threshold calibration requires 500 records per corpus: {corpus_counts}")
    expected = {str(row["record_id"]): str(row["text_sha256"]) for row in calibration}
    values: dict[str, float] = {}
    if isinstance(scores, Mapping):
        for record_id, value in scores.items():
            if isinstance(value, ScoreRow):
                if value.endpoint != RADAR_ENDPOINT or value.variant_id != "original":
                    raise ValueError("Threshold scores must be RADAR original scores")
                if value.text_sha256 != expected.get(str(record_id)) or not value.provenance_label.casefold() in _HUMAN_VALUES:
                    raise ValueError(f"Invalid threshold provenance/hash: {record_id}")
                if value.canonical_ai_score is None:
                    raise ValueError("Threshold score is missing")
                values[str(record_id)] = float(value.canonical_ai_score)
            else:
                values[str(record_id)] = float(value)
    else:
        rows = tuple(scores)
        for row in rows:
            score_row = row if isinstance(row, ScoreRow) else _score_row_from_mapping(row)
            if score_row.endpoint != RADAR_ENDPOINT or score_row.variant_id != "original":
                raise ValueError("Threshold scores must be RADAR original scores")
            if score_row.record_id in values or score_row.record_id not in expected:
                raise ValueError("Threshold score IDs are duplicate or outside calibration")
            if score_row.text_sha256 != expected[score_row.record_id] or not score_row.provenance_label.casefold() in _HUMAN_VALUES:
                raise ValueError(f"Invalid threshold provenance/hash: {score_row.record_id}")
            if score_row.canonical_ai_score is None:
                raise ValueError("Threshold score is missing")
            values[score_row.record_id] = float(score_row.canonical_ai_score)
    if set(values) != set(expected) or len(values) != 2_000:
        raise ValueError(f"Incomplete RADAR threshold reference: {len(values)}/2000")
    ordered = sorted(values.values())
    index = max(0, min(len(ordered) - 1, int((0.95 * len(ordered) + 0.9999999999)) - 1))
    threshold_value = float(ordered[index])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "radar_threshold",
        "pilot_lock_sha256": verify_pilot_lock(paths)["sha256"],
        "endpoint": RADAR_ENDPOINT,
        "reference_count": len(ordered),
        "reference_score_sha256": _sha256(sorted(values.items())),
        "quantile": 0.95,
        "threshold": threshold_value,
        "decision_rule": "strict_score_gt_threshold",
    }
    if paths.threshold_lock.exists():
        existing = verify_lock(paths.threshold_lock)["payload"]
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError("Existing RADAR threshold lock disagrees")
    else:
        lock_forecasts(paths.threshold_lock, payload)
    return payload


def validate_triplet_token_budget(
    token_counts: Mapping[str, Mapping[str, int] | Sequence[int]],
    *,
    cap: int = 460,
    required_endpoints: Sequence[str] = (RADAR_ENDPOINT, MAGE_ENDPOINT),
) -> bool:
    """Atomically accept a complete original-plus-reflow panel.

    A missing endpoint, missing variant, or one over-cap member rejects the
    complete panel.  This function is intentionally independent of a model
    tokenizer: callers pass the counts produced by each pinned tokenizer.
    """
    if cap <= 0:
        raise ValueError("token cap must be positive")
    expected = ("original", *PROBES)
    for endpoint in required_endpoints:
        if endpoint not in token_counts:
            raise ValueError(f"Token panel is missing {endpoint}")
        counts = token_counts[endpoint]
        if isinstance(counts, Mapping):
            if set(counts) != set(expected):
                raise ValueError(f"Token panel is incomplete for {endpoint}")
            values = [int(counts[level]) for level in expected]
        else:
            values = [int(value) for value in counts]
            if len(values) != len(expected):
                raise ValueError(f"Token panel is incomplete for {endpoint}")
        if any(value < 0 or value > cap for value in values):
            raise ValueError("Full reflow panel rejected by common token cap")
    return True


triplet_fits_token_budget = validate_triplet_token_budget


def lock_human_token_panels(
    paths: DeferralPaths,
    token_counts: Mapping[str, Mapping[str, Mapping[str, int] | Sequence[int]]],
    *,
    cap: int = 460,
) -> dict[str, object]:
    """Atomically validate every selected human pilot panel under both tokenizers."""
    pilot_lock = verify_pilot_lock(paths)
    expected = {str(row["record_id"]) for row in pilot_lock["payload"].get("pilot", ())}
    if set(token_counts) != expected:
        raise ValueError(f"Human token panels must exactly cover the pilot: {len(token_counts)}/{len(expected)}")
    normalized: list[dict[str, object]] = []
    for record_id in sorted(expected):
        validate_triplet_token_budget(token_counts[record_id], cap=cap)
        panel_counts = {}
        for endpoint, counts in sorted(token_counts[record_id].items()):
            values = counts.items() if isinstance(counts, Mapping) else zip(("original", *PROBES), counts)
            panel_counts[endpoint] = {variant: int(value) for variant, value in values}
        normalized.append({
            "record_id": record_id,
            "counts": panel_counts,
        })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "human_token_panels",
        "pilot_lock_sha256": pilot_lock["sha256"],
        "token_cap": int(cap),
        "panels_sha256": _sha256(normalized),
        "record_count": len(normalized),
    }
    if paths.human_token_lock.exists():
        existing = verify_lock(paths.human_token_lock)["payload"]
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError("Existing human token-panel lock disagrees")
    else:
        lock_forecasts(paths.human_token_lock, payload)
    return payload


def build_conditional_worklist(
    paths: DeferralPaths,
    original_scores: Mapping[tuple[str, str], object] | Iterable[ScoreRow],
    *,
    thresholds: Mapping[str, float],
    sentinel_per_corpus_label: int = 0,
) -> tuple[dict[str, object], ...]:
    """Return the RADAR-gated worklist in fixed query order.

    RADAR originals are scored for every pilot passage.  Strictly positive
    RADAR results (score ``>`` threshold) receive three RADAR reflow queries;
    MAGE and LogRank are then queried on those same original passages only.
    Their endpoint-specific thresholds never gate work and neither receives
    transformed variants.  The ``Q=N_original+3*N_positive`` accounting refers
    to RADAR queries; the two original-only diagnostic queries are reported in
    the worklist but are not counted in Q.
    """
    manifest = verify_pilot_lock(paths)["payload"]
    panels = _locked_panels(paths)
    expected = _expected_variants(manifest, panels)
    if not isinstance(original_scores, Mapping):
        original_scores = {
            (row.record_id, row.endpoint): row
            for row in original_scores if row.variant_id == "original"
        }
    values = _score_values(original_scores)
    revisions = {str(key): str(value) for key, value in manifest["endpoint_revisions"].items()}
    records = {str(row["record_id"]): row for row in manifest.get("pilot", ())}
    for panel in panels:
        records[str(panel["ai_record_id"])] = {
            "record_id": str(panel["ai_record_id"]),
            "corpus": str(panel["corpus"]),
            "provenance_label": "ai",
            "variants": [
                {
                    "variant_id": probe,
                    "text_sha256": str(panel["variants"][probe]["text_sha256"]),
                }
                for probe in PROBES
            ],
        }
    work: list[dict[str, object]] = []
    if RADAR_ENDPOINT not in thresholds:
        raise ValueError(f"Missing positive threshold for {RADAR_ENDPOINT}")
    positives: list[str] = []
    for record_id in sorted(records):
        original_key = (record_id, "original")
        if original_key not in expected or (record_id, RADAR_ENDPOINT) not in values:
            raise ValueError(f"Missing RADAR original score for {record_id}")
        work.append({
            "record_id": record_id, "variant_id": "original", "endpoint": RADAR_ENDPOINT,
            "detector_revision": revisions[RADAR_ENDPOINT], "text_sha256": expected[original_key],
            "provenance_label": records[record_id]["provenance_label"],
        })
        if values[(record_id, RADAR_ENDPOINT)] <= float(thresholds[RADAR_ENDPOINT]):
            continue
        available = {str(variant["variant_id"]): variant for variant in records[record_id].get("variants", ())}
        if set(PROBES) - set(available):
            raise ValueError(f"Positive record lacks an eligible three-probe panel: {record_id}")
        positives.append(record_id)
        for probe in PROBES:
            work.append({
                "record_id": record_id, "variant_id": probe, "endpoint": RADAR_ENDPOINT,
                "detector_revision": revisions[RADAR_ENDPOINT],
                "text_sha256": available[probe]["text_sha256"],
                "provenance_label": records[record_id]["provenance_label"],
            })
    for record_id in positives:
        for endpoint in (MAGE_ENDPOINT, LOGRANK_ENDPOINT):
            work.append({
                "record_id": record_id, "variant_id": "original", "endpoint": endpoint,
                "detector_revision": revisions[endpoint],
                "text_sha256": expected[(record_id, "original")],
                "provenance_label": records[record_id]["provenance_label"],
            })
    if sentinel_per_corpus_label < 0:
        raise ValueError("Sentinel count cannot be negative")
    sentinel_cells: dict[tuple[str, bool], list[str]] = {}
    for record_id in positives:
        row = records[record_id]
        cell = (str(row["corpus"]), _label_is_human(row["provenance_label"]))
        sentinel_cells.setdefault(cell, []).append(record_id)
    for cell in sorted(sentinel_cells):
        selected = sorted(
            sentinel_cells[cell],
            key=lambda record_id: hashlib.sha256(canonical_json({
                "purpose": "original_repeat", "record_id": record_id,
            })).hexdigest(),
        )[:sentinel_per_corpus_label]
        for record_id in selected:
            work.append({
                "record_id": record_id, "variant_id": "original_repeat", "endpoint": RADAR_ENDPOINT,
                "detector_revision": revisions[RADAR_ENDPOINT],
                "text_sha256": expected[(record_id, "original_repeat")],
                "provenance_label": records[record_id]["provenance_label"],
            })
    accounting = query_accounting(len(records), len(positives))
    result = tuple(work)
    if paths.worklist.exists():
        with paths.worklist.open("r", encoding="utf-8", newline="") as handle:
            existing = tuple(csv.DictReader(handle))
        if tuple(dict(row) for row in existing) != result:
            raise RuntimeError("Existing conditional worklist disagrees with locked scores")
    else:
        paths.worklist.parent.mkdir(parents=True, exist_ok=True)
        fields = tuple(result[0]) if result else (
            "record_id", "variant_id", "endpoint", "detector_revision",
            "text_sha256", "provenance_label",
        )
        with paths.worklist.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result)
    worklist_payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "conditional_worklist",
        "pilot_lock_sha256": verify_pilot_lock(paths)["sha256"],
        "panel_lock_sha256": _file_sha256(paths.panel_lock) if paths.panel_lock.exists() else None,
        "threshold_lock_sha256": _file_sha256(paths.threshold_lock),
        "rows_sha256": _sha256(result),
        "worklist_csv_sha256": _file_sha256(paths.worklist),
        "row_count": len(result),
        "radar_original_count": len(records),
        "radar_positive_count": len(positives),
        "radar_query_count": accounting,
    }
    if paths.worklist_lock.exists():
        existing_lock = verify_lock(paths.worklist_lock)["payload"]
        if canonical_json(existing_lock) != canonical_json(worklist_payload):
            raise RuntimeError("Existing conditional-worklist lock disagrees")
    else:
        lock_forecasts(paths.worklist_lock, worklist_payload)
    return result


def verify_conditional_worklist(paths: DeferralPaths) -> tuple[dict[str, object], ...]:
    if not paths.worklist.exists() or not paths.worklist_lock.exists():
        raise RuntimeError("Locked conditional worklist is required")
    payload = verify_lock(paths.worklist_lock)["payload"]
    if payload.get("pilot_lock_sha256") != verify_pilot_lock(paths)["sha256"]:
        raise RuntimeError("Conditional worklist references a different pilot lock")
    current_panel_hash = _file_sha256(paths.panel_lock) if paths.panel_lock.exists() else None
    if payload.get("panel_lock_sha256") != current_panel_hash:
        raise RuntimeError("Conditional worklist references a different generated panel")
    if payload.get("threshold_lock_sha256") != _file_sha256(paths.threshold_lock):
        raise RuntimeError("Conditional worklist references a different threshold")
    if payload.get("worklist_csv_sha256") != _file_sha256(paths.worklist):
        raise RuntimeError("Conditional worklist CSV hash mismatch")
    with paths.worklist.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    if payload.get("rows_sha256") != _sha256(rows) or payload.get("row_count") != len(rows):
        raise RuntimeError("Conditional worklist rows disagree with lock")
    return rows


def assemble_evaluation_rows(
    paths: DeferralPaths,
    texts: Path | str | Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Join locked provenance and complete conditional scores for pilot evaluation."""
    if not paths.threshold_lock.exists() or not paths.score_table.exists():
        raise RuntimeError("Threshold and canonical score table are required for evaluation")
    threshold_payload = verify_lock(paths.threshold_lock)["payload"]
    threshold_value = float(threshold_payload["threshold"])
    panels = _locked_panels(paths)
    if not panels:
        raise RuntimeError("A locked generated panel is required for evaluation")
    rows = _output_rows(texts)
    by_record: dict[str, str] = {}
    request_to_ai = {str(panel["request_id"]): str(panel["ai_record_id"]) for panel in panels}
    for row in rows:
        record_id = str(row.get("record_id") or request_to_ai.get(str(row.get("request_id", "")), ""))
        text = str(row.get("text", row.get("generated_text", "")))
        if not record_id or not text.strip() or record_id in by_record:
            raise ValueError("Evaluation texts require unique locked record_id/request_id and non-empty text")
        by_record[record_id] = _base_text(text)

    metadata: dict[str, dict[str, object]] = {}
    human_manifest = {
        str(row["record_id"]): row for row in verify_pilot_lock(paths)["payload"].get("pilot", ())
    }
    for panel in panels:
        human_id = str(panel["human_record_id"])
        ai_id = str(panel["ai_record_id"])
        if human_id not in human_manifest:
            raise RuntimeError(f"Generated panel references unknown human record: {human_id}")
        shared = {
            "pair_id": str(panel["pair_id"]),
            "corpus": str(panel["corpus"]),
            "generator_family": str(panel["generator_family"]),
        }
        metadata[human_id] = {**shared, "label": 1, "text_sha256": str(human_manifest[human_id]["text_sha256"])}
        metadata[ai_id] = {**shared, "label": 0, "text_sha256": str(panel["base_text_sha256"])}

    scores = {row.key: row for row in read_canonical_scores(paths.score_table)}
    output: list[dict[str, object]] = []
    for record_id, meta in sorted(metadata.items()):
        original = scores.get((record_id, "original", RADAR_ENDPOINT))
        if original is None or original.canonical_ai_score is None:
            raise ValueError(f"Missing RADAR original score: {record_id}")
        if original.failure or original.truncated:
            raise ValueError(f"Invalid RADAR original score: {record_id}")
        if not radar_positive(original.canonical_ai_score, threshold_value):
            continue
        if record_id not in by_record or _text_sha256(by_record[record_id]) != meta["text_sha256"]:
            raise ValueError(f"Evaluation text is missing or does not match its locked hash: {record_id}")
        required = {
            "radar_wrap_80": ("wrap_80", RADAR_ENDPOINT),
            "radar_sentence_blocks_2": ("sentence_blocks_2", RADAR_ENDPOINT),
            "radar_sentence_per_paragraph": ("sentence_per_paragraph", RADAR_ENDPOINT),
            "mage_original": ("original", MAGE_ENDPOINT),
            "logrank_original": ("original", LOGRANK_ENDPOINT),
        }
        values: dict[str, float] = {}
        for name, (variant, endpoint) in required.items():
            score = scores.get((record_id, variant, endpoint))
            if score is None or score.canonical_ai_score is None or score.failure or score.truncated:
                raise ValueError(f"Missing or invalid conditional score {name}: {record_id}")
            values[name] = float(score.canonical_ai_score)
        repeat = scores.get((record_id, "original_repeat", RADAR_ENDPOINT))
        output.append({
            "record_id": record_id,
            **meta,
            "text": by_record[record_id],
            "radar_threshold": threshold_value,
            "radar_original": float(original.canonical_ai_score),
            **values,
            **({"radar_original_repeat": float(repeat.canonical_ai_score)} if repeat and repeat.canonical_ai_score is not None and not repeat.failure and not repeat.truncated else {}),
        })
    return tuple(output)


def require_pilot_authorization(paths: DeferralPaths, stage: str = "final") -> dict[str, object]:
    """Refuse any sealed/final stage until an explicit pilot authorization lock."""
    if stage not in {"final", "sealed", "confirmation", "deployment"}:
        raise ValueError(f"Unknown protected stage: {stage}")
    if not paths.authorization_lock.exists():
        raise RuntimeError(
            f"Refusing {stage} stage: pilot authorization lock is required; pilot results are not yet authorized"
        )
    envelope = verify_lock(paths.authorization_lock)
    payload = envelope.get("payload", {})
    if payload.get("stage") != "pilot_authorization" or payload.get("pilot_lock_sha256") != _file_sha256(paths.lock):
        raise RuntimeError("Pilot authorization lock does not bind to the current pilot lock")
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping) or summary.get("passed") is not True or summary.get("status") != "pilot_passed":
        raise RuntimeError("Pilot authorization requires summary passed=True and status=pilot_passed")
    return envelope


def authorize_final_stage(paths: DeferralPaths, summary: Mapping[str, object]) -> str:
    """Explicitly authorize later work; never called implicitly by preparation."""
    verify_pilot_lock(paths)
    if summary.get("passed") is not True or summary.get("status") != "pilot_passed":
        raise ValueError("Authorization requires summary passed=True and status=pilot_passed")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "pilot_authorization",
        "pilot_lock_sha256": _file_sha256(paths.lock),
        "summary": dict(summary),
    }
    return lock_forecasts(paths.authorization_lock, payload)


# Friendly alias for callers that want an assertion-shaped name.
assert_final_stage_authorized = require_pilot_authorization


__all__ = [
    "CALIBRATION_PER_CORPUS", "CanonicalRecord",
    "DEV_CORPORA", "DeferralPaths", "ENDPOINT_ROLES", "GenerationRequest",
    "GENERATOR_FAMILY_COUNT", "LOGRANK_ENDPOINT", "MAGE_ENDPOINT",
    "PILOT_HUMAN_CAP", "PILOT_STAGE", "PROBES", "RADAR_ENDPOINT", "ReflowVariant",
    "ScoreRow", "assert_final_stage_authorized", "authorize_final_stage",
    "build_conditional_worklist", "build_reflow_variants", "deferral_paths",
    "assemble_evaluation_rows",
    "calibrate_radar_threshold", "export_manual_audit", "import_canonical_scores",
    "import_generation_outputs", "import_manual_audit", "line_wrap_variant",
    "lock_human_token_panels",
    "mage_effective_input", "mage_effective_input_hash", "prepare_generation_requests",
    "prepare_pilot_manifest", "query_accounting", "radar_positive",
    "read_canonical_scores", "read_canonical_table", "reflow_variant",
    "require_pilot_authorization", "select_human_panel", "sentence_blocks_2_variant",
    "sentence_blocks_variant", "sentence_per_paragraph_variant",
    "triplet_fits_token_budget", "validate_canonical_scores", "validate_mage_effective_input_hashes",
    "verify_conditional_worklist",
    "validate_reflow_variant", "validate_triplet_token_budget", "verify_generation_lock",
    "verify_pilot_lock", "wrap_80_variant",
]
