from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from statistics import median
from typing import Callable, Iterable, Iterator, Mapping, Sequence

STUDY_VERSION = "2026.07"
WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
SENTENCE_EDGE_RE = re.compile(r"[.!?]|\n\s*\n")
CONTRACTIONS = {
    "aren't": "are not", "can't": "cannot", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "i'm": "I am", "isn't": "is not", "shouldn't": "should not", "they're": "they are",
    "wasn't": "was not", "we're": "we are", "weren't": "were not",
    "won't": "will not", "wouldn't": "would not", "you're": "you are",
}
CONTRACTIONS |= {key.replace("'", "\u2019"): value for key, value in tuple(CONTRACTIONS.items())}
DISCOURSE = {
    "however": "nevertheless", "therefore": "consequently",
    "moreover": "furthermore", "for example": "for instance",
    "in addition": "additionally", "on the other hand": "conversely",
}
PARTITION_SIZES = {
    "technical_pilot": 64,
    "source_summary": 500,
    "source_model": 1500,
    "signature": 1000,
    "test": 2000,
}
GROUP_COLUMNS = {
    "blog_authorship": ("author_id",),
    "gutenberg": ("author_id", "book_id"),
    "stack_exchange": ("site_id", "user_id", "post_id"),
    "asap_aes": ("student_id", "essay_id"),
    "pmc": ("article_id",),
    "cnn_dailymail": ("article_id",),
    "govreport": ("report_id",),
    "wikitext_103": ("article_id",),
    "bawe": ("student_id",),
}
STUDY_CORPORA = (
    "blog_authorship", "gutenberg", "stack_exchange", "asap_aes",
    "pmc", "cnn_dailymail", "govreport", "wikitext_103",
)
EXTERNAL_VALIDATION_CORPORA = ("bawe",)
TARGET_CORPORA = STUDY_CORPORA + EXTERNAL_VALIDATION_CORPORA
FORECAST_MODELS = (
    "source_fpr", "text_only", "profile_only", "detector_id_text", "detector_id_x_text",
    "source_fpr_id_text", "profile_text", "main",
)
PROBES = (
    "contraction_expansion", "punctuation_normalization", "sentence_splitting",
    "discourse_markers", "paragraph_resegmentation", "adjacent_repetition_removal",
)


def storage_root() -> Path:
    root = os.environ.get("FPRINT_STORAGE_ROOT")
    if not root:
        raise RuntimeError("Set FPRINT_STORAGE_ROOT to a large, non-system drive.")
    return Path(root).expanduser().resolve()


def canonical_text(text: str) -> str:
    return SPACE_RE.sub(" ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def words(text: str) -> list[str]:
    return WORD_RE.findall(canonical_text(text))


def shingles(text: str, width: int = 5) -> set[str]:
    tokens = words(text)
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + width]) for i in range(len(tokens) - width + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def minhash_signature(items: set[str], permutations: int = 128) -> tuple[int, ...]:
    try:
        from datasketch import MinHash
    except ImportError as error:
        raise RuntimeError("Global deduplication requires the 'datasketch' package") from error
    signature = MinHash(num_perm=permutations, seed=20260729)
    if items:
        signature.update_batch(item.encode("utf-8") for item in items)
    return tuple(int(value) for value in signature.hashvalues)


def lsh_candidates(signatures: Sequence[tuple[int, ...]], rows: int = 8) -> Iterator[tuple[int, int]]:
    """Yield candidate pairs; exact Jaccard applies the preregistered .80 cutoff."""
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    yielded: set[tuple[int, int]] = set()
    for index, signature in enumerate(signatures):
        for band, start in enumerate(range(0, len(signature), rows)):
            key = (band, signature[start:start + rows])
            for other in buckets[key]:
                pair = (other, index)
                if pair not in yielded:
                    yielded.add(pair)
                    yield pair
            buckets[key].append(index)


def _minhash_candidates(items: Sequence[set[str]], cutoff: float) -> Iterator[tuple[int, int]]:
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as error:
        raise RuntimeError("Global deduplication requires the 'datasketch' package") from error
    lsh = MinHashLSH(threshold=cutoff, num_perm=128)
    for right, shingles_ in enumerate(items):
        signature = MinHash(num_perm=128, seed=20260729)
        if shingles_:
            signature.update_batch(item.encode("utf-8") for item in shingles_)
        for left in lsh.query(signature):
            yield int(left), right
        lsh.insert(str(right), signature)


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


@dataclass(frozen=True)
class TextRecord:
    record_id: str
    corpus: str
    text: str
    group_id: str
    source_order: int = 0
    is_threshold_reference: bool = False
    stratum: str = ""


def deduplicate(records: Sequence[TextRecord], cutoff: float = .80) -> tuple[list[TextRecord], list[dict]]:
    """Global exact + near dedup; RAID collisions and cross-corpus components are removed."""
    if not records:
        return [], []
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("Duplicate record_id in input")
    uf = UnionFind(len(records))
    exact: dict[str, int] = {}
    shingle_sets = [shingles(record.text) for record in records]
    for i, record in enumerate(records):
        digest = text_hash(record.text)
        if digest in exact:
            uf.union(i, exact[digest])
        else:
            exact[digest] = i
    try:
        candidates = _minhash_candidates(shingle_sets, cutoff)
        first = next(candidates, None)
    except RuntimeError:
        if len(records) > 1_000:
            raise
        # ponytail: exact quadratic fallback is only for dependency-free smoke tests.
        candidates = ((left, right) for right in range(len(records)) for left in range(right))
    else:
        candidates = chain((), candidates) if first is None else chain((first,), candidates)
    for left, right in candidates:
        if uf.find(left) != uf.find(right) and jaccard(shingle_sets[left], shingle_sets[right]) >= cutoff:
            uf.union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        components[uf.find(i)].append(i)
    kept: list[TextRecord] = []
    audit: list[dict] = []
    for indices in components.values():
        members = [records[i] for i in indices]
        corpora = {member.corpus for member in members}
        raid_collision = (
            any(member.is_threshold_reference for member in members)
            and any(not member.is_threshold_reference for member in members)
        )
        cross_corpus = len(corpora) > 1
        if raid_collision or cross_corpus:
            action = "drop_raid_collision" if raid_collision else "drop_cross_corpus_component"
            audit.extend({"record_id": member.record_id, "action": action} for member in members)
            continue
        winner = min(members, key=lambda item: (item.source_order, item.record_id))
        kept.append(winner)
        for member in members:
            audit.append({"record_id": member.record_id, "action": "keep" if member == winner else "drop_duplicate"})
    return sorted(kept, key=lambda item: (item.corpus, item.source_order, item.record_id)), audit


def _group_value(value: object) -> str:
    value = str(value or "").strip()
    return "" if value.casefold() in {"", "-1", "-1.0", "0", "0.0", "na", "n/a", "none", "null", "nan", "<na>", "unknown"} else value


def grouping_key(corpus: str, row: Mapping[str, object]) -> str:
    if corpus.casefold() == "stack_exchange":
        site, user, post = (_group_value(row.get(column)) for column in ("site_id", "user_id", "post_id"))
        if site and user:
            return f"site_user:{site}:{user}"
        if site and post:
            return f"site_post:{site}:{post}"
        raise ValueError("stack_exchange requires site_id plus user_id or post_id")
    columns = GROUP_COLUMNS.get(corpus.casefold())
    if not columns:
        raise ValueError(f"Unknown corpus: {corpus}")
    for column in columns:
        value = _group_value(row.get(column))
        if value:
            return f"{column}:{value}"
    raise ValueError(f"{corpus} row has none of the required grouping fields: {columns}")


def assign_grouped_partitions(records: Sequence[TextRecord], seed: int = 20260729) -> dict[str, str]:
    """Assign whole groups, never documents, to fixed disjoint partitions."""
    by_corpus: dict[str, dict[str, list[TextRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_corpus[record.corpus][record.group_id].append(record)
    assignments: dict[str, str] = {}
    for corpus, groups in by_corpus.items():
        by_stratum: dict[str, list[str]] = defaultdict(list)
        for key, members in groups.items():
            strata = {member.stratum for member in members}
            if len(strata) != 1:
                raise ValueError(f"{corpus} group {key!r} spans multiple strata: {sorted(strata)}")
            by_stratum[strata.pop()].append(key)
        for stratum, keys in by_stratum.items():
            random.Random(f"{seed}:{corpus}:{stratum}").shuffle(keys)
        keys = []
        while any(by_stratum.values()):
            for stratum in sorted(by_stratum):
                if by_stratum[stratum]:
                    keys.append(by_stratum[stratum].pop())
        if corpus in EXTERNAL_VALIDATION_CORPORA:
            index = 0
            technical_records = 0
            while technical_records < PARTITION_SIZES["technical_pilot"] and index < len(keys):
                key = keys[index]
                index += 1
                technical_records += len(groups[key])
                assignments.update((record.record_id, "technical_pilot") for record in groups[key])
            signature_keys = keys[index:index + 250]
            index += len(signature_keys)
            for key in signature_keys:
                assignments.update((record.record_id, "signature") for record in groups[key])
            test_records = 0
            while test_records < PARTITION_SIZES["test"] and index < len(keys):
                key = keys[index]
                index += 1
                test_records += len(groups[key])
                assignments.update((record.record_id, "test") for record in groups[key])
            for key in keys[index:]:
                assignments.update((record.record_id, "anchor_candidates") for record in groups[key])
            signature_records = sum(len(groups[key]) for key in signature_keys)
            if (
                len(signature_keys) < 250
                or signature_records < PARTITION_SIZES["signature"]
                or test_records < PARTITION_SIZES["test"]
            ):
                raise ValueError(f"{corpus} lacks independent grouped target records")
            continue
        remaining = dict(PARTITION_SIZES)
        current = iter(PARTITION_SIZES)
        partition = next(current)
        for key in keys:
            while partition in remaining and remaining[partition] <= 0:
                try:
                    partition = next(current)
                except StopIteration:
                    partition = "anchor_candidates"
                    break
            for record in groups[key]:
                assignments[record.record_id] = partition
            if partition in remaining:
                remaining[partition] -= len(groups[key])
        shortages = {name: count for name, count in remaining.items() if count > 0}
        if shortages:
            raise ValueError(f"{corpus} lacks grouped records for partitions: {shortages}")
    return assignments


def repeated_signature_samples(
    record_ids: Sequence[str] | Sequence[TextRecord],
    sizes: Sequence[int] = (50, 100, 250),
    draws: int = 20,
    seed: int = 20260729,
) -> dict[tuple[int, int], tuple[str, ...]]:
    if record_ids and isinstance(record_ids[0], TextRecord):
        records = list(record_ids)
        if len(records) < PARTITION_SIZES["signature"]:
            raise ValueError("Grouped signature pool must contain at least 1,000 records")
        ids = [record.record_id for record in records]
        groups = [record.group_id for record in records]
    else:
        ids = list(record_ids)
        groups = ids
    if len(set(ids)) != len(ids) or len(ids) < max(sizes):
        raise ValueError("Signature pool must contain enough unique records")
    by_group: dict[str, list[str]] = defaultdict(list)
    for record_id, group_id in zip(ids, groups):
        by_group[group_id].append(record_id)
    if len(by_group) < max(sizes):
        raise ValueError("Signature pool must contain enough unique groups")
    samples: dict[tuple[int, int], tuple[str, ...]] = {}
    for draw in range(draws):
        rng = random.Random(f"{seed}:signature:{draw}")
        group_order = list(by_group)
        rng.shuffle(group_order)
        permutation = [rng.choice(by_group[group]) for group in group_order]
        for size in sizes:
            samples[(draw, size)] = tuple(permutation[:size])
    return samples


@dataclass(frozen=True)
class ProbeTriplet:
    probe: str
    original: str
    low: str
    high: str
    eligible_sites: int
    low_sites: int
    high_sites: int

    @property
    def low_intensity(self) -> float:
        return self.low_sites / self.eligible_sites

    @property
    def high_intensity(self) -> float:
        return self.high_sites / self.eligible_sites


def _selected(count: int, fraction: float, seed: str) -> set[int]:
    amount = max(1, math.ceil(count * fraction))
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    return set(indices[:amount])


def _replace_sites(text: str, pattern: re.Pattern[str], replacements: Mapping[str, str], fraction: float, seed: str) -> tuple[str, int, int]:
    matches = list(pattern.finditer(text))
    chosen = _selected(len(matches), fraction, seed) if matches else set()
    pieces, cursor = [], 0
    for i, match in enumerate(matches):
        pieces.append(text[cursor:match.start()])
        value = match.group(0)
        replacement = replacements[value.casefold()]
        if i in chosen:
            if value[:1].isupper():
                replacement = replacement[:1].upper() + replacement[1:]
            pieces.append(replacement)
        else:
            pieces.append(value)
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces), len(matches), len(chosen)


def _probe_variant(probe: str, text: str, fraction: float, seed: str) -> tuple[str, int, int]:
    if probe == "contraction_expansion":
        pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, CONTRACTIONS)) + r")\b", re.I)
        return _replace_sites(text, pattern, CONTRACTIONS, fraction, seed)
    if probe == "discourse_markers":
        pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, DISCOURSE)) + r")\b", re.I)
        return _replace_sites(text, pattern, DISCOURSE, fraction, seed)
    if probe == "punctuation_normalization":
        replacements = {"—": "-", "–": "-", "…": "...", "“": '"', "”": '"', "‘": "'", "’": "'"}
        pattern = re.compile("|".join(map(re.escape, replacements)))
        return _replace_sites(text, pattern, replacements, fraction, seed)
    if probe == "sentence_splitting":
        pattern = re.compile(r"[,;:]\s+(?=(?:and|but|yet|so|because|although|while)\b)", re.I)
        matches = []
        for match in pattern.finditer(text):
            left = SENTENCE_EDGE_RE.split(text[:match.start()])[-1]
            right = SENTENCE_EDGE_RE.split(text[match.end():], maxsplit=1)[0]
            if len(words(left)) >= 8 and len(words(right)) >= 8:
                matches.append(match)
        chosen = _selected(len(matches), fraction, seed) if matches else set()
        parts, cursor = [], 0
        for i, match in enumerate(matches):
            parts.append(text[cursor:match.start()])
            parts.append(". " if i in chosen else match.group(0))
            cursor = match.end()
        parts.append(text[cursor:])
        return "".join(parts), len(matches), len(chosen)
    if probe == "paragraph_resegmentation":
        matches = [match for match in SENTENCE_RE.finditer(text) if not re.search(r"\n\s*\n", match.group(0))]
        chosen = _selected(len(matches), fraction, seed) if matches else set()
        parts, cursor = [], 0
        for i, match in enumerate(matches):
            parts.extend((text[cursor:match.start()], "\n\n" if i in chosen else match.group(0)))
            cursor = match.end()
        parts.append(text[cursor:])
        return "".join(parts), len(matches), len(chosen)
    if probe == "adjacent_repetition_removal":
        pattern = re.compile(r"\b(\w+(?:\s+\w+){0,2})\s+\1\b", re.I)
        matches = list(pattern.finditer(text))
        chosen = _selected(len(matches), fraction, seed) if matches else set()
        parts, cursor = [], 0
        for i, match in enumerate(matches):
            parts.append(text[cursor:match.start()])
            parts.append(match.group(1) if i in chosen else match.group(0))
            cursor = match.end()
        parts.append(text[cursor:])
        return "".join(parts), len(matches), len(chosen)
    raise ValueError(f"Unknown probe: {probe}")


def make_probe_triplet(probe: str, text: str, seed: str, min_sites: int = 4) -> ProbeTriplet | None:
    low, sites, low_count = _probe_variant(probe, text, .25, f"{seed}:low")
    high, high_sites, high_count = _probe_variant(probe, text, 1.0, f"{seed}:high")
    if sites < min_sites or high_sites != sites or low == high:
        return None
    return ProbeTriplet(probe, text, low, high, sites, low_count, high_count)


def triplet_fits(triplet: ProbeTriplet, tokenizers: Sequence[tuple[str, Callable[[str], int], int]]) -> bool:
    return bool(tokenizers) and all(counter(text) <= capacity for _, counter, capacity in tokenizers for text in (triplet.original, triplet.low, triplet.high))


def empirical_cdf(reference_scores: Sequence[float], score: float) -> float:
    if not reference_scores:
        raise ValueError("Reference scores are empty")
    ordered = sorted(reference_scores)
    import bisect
    return bisect.bisect_right(ordered, score) / len(ordered)


def threshold(reference_scores: Sequence[float], false_positive_rate: float) -> float:
    if not 0 < false_positive_rate < 1 or not reference_scores:
        raise ValueError("Need scores and a false-positive rate in (0, 1)")
    ordered = sorted(reference_scores)
    return ordered[min(len(ordered) - 1, math.ceil((1 - false_positive_rate) * len(ordered)) - 1)]


def slope(intensities: Sequence[float], values: Sequence[float]) -> float:
    if len(intensities) != len(values) or len(values) < 2:
        raise ValueError("Slope requires equally sized sequences with at least two values")
    x_mean, y_mean = sum(intensities) / len(intensities), sum(values) / len(values)
    denominator = sum((x - x_mean) ** 2 for x in intensities)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(intensities, values)) / denominator if denominator else 0.0


def backend_macro(values: Mapping[str, Mapping[str, float]]) -> float:
    """Average configs within dependency groups, then average groups."""
    grouped = [sum(configs.values()) / len(configs) for configs in values.values() if configs]
    if not grouped:
        raise ValueError("No backend values")
    return sum(grouped) / len(grouped)


def exact_sign_flip(improvements: Sequence[float]) -> float:
    """One-sided exact randomization p-value for positive mean improvement."""
    if not improvements:
        raise ValueError("No improvements")
    observed = sum(improvements)
    extreme = 0
    for mask in range(1 << len(improvements)):
        value = sum(item if mask & (1 << i) else -item for i, item in enumerate(improvements))
        extreme += value >= observed - 1e-15
    return extreme / (1 << len(improvements))


def jeffreys_posterior(flagged: int, total: int) -> tuple[float, float, float]:
    if total <= 0 or not 0 <= flagged <= total:
        raise ValueError("Expected 0 <= flagged <= total and total > 0")
    alpha, beta = flagged + .5, total - flagged + .5
    mean = alpha / (alpha + beta)
    try:
        from scipy.stats import beta as beta_dist
        low, high = beta_dist.ppf([.025, .975], alpha, beta)
    except ImportError:
        # ponytail: normal approximation only supports dependency-free smoke tests.
        sd = math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
        low, high = max(0.0, mean - 1.96 * sd), min(1.0, mean + 1.96 * sd)
    return mean, float(low), float(high)


def canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def lock_forecasts(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json(payload)
    digest = hashlib.sha256(body).hexdigest()
    envelope = {"sha256": digest, "payload": payload}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(envelope, handle, sort_keys=True, indent=2, ensure_ascii=False)
    return digest


def verify_lock(path: Path) -> dict:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(canonical_json(envelope["payload"])).hexdigest()
    if actual != envelope.get("sha256"):
        raise RuntimeError(f"Forecast lock hash mismatch: {path}")
    return envelope


def validate_forecast_payload(
    payload: Mapping[str, object],
    corpora: Sequence[str] = STUDY_CORPORA,
    detectors: Sequence[str] | None = None,
    sizes: Sequence[int] = (50, 100, 250),
    draws: int = 20,
    models: Sequence[str] = FORECAST_MODELS,
) -> None:
    if detectors is None:
        from .detectors import SPECS
        declared = payload.get("admitted_detectors")
        if not isinstance(declared, list) or len(declared) != len(set(declared)):
            raise ValueError("Forecast payload requires unique admitted_detectors")
        if any(detector not in SPECS for detector in declared):
            raise ValueError("Forecast payload names an unknown detector configuration")
        detectors = tuple(declared)
        groups = {SPECS[detector].dependency_group for detector in detectors}
        if len(detectors) < 4 or groups != {"openai_roberta", "radar", "mage", "qwen25_shared"}:
            raise ValueError("Admitted panel requires >=4 configurations across all four dependency groups")
    expected = {
        (corpus, detector, size, draw, model)
        for corpus in corpora for detector in detectors for size in sizes
        for draw in range(draws) for model in models
    }
    entries = payload.get("forecasts")
    if not isinstance(entries, list):
        raise ValueError("Forecast payload requires a 'forecasts' list")
    actual = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Every forecast must be an object")
        key = (
            entry.get("target_corpus"), entry.get("detector_config"),
            entry.get("signature_size"), entry.get("draw"), entry.get("model"),
        )
        if key in actual:
            raise ValueError(f"Duplicate forecast cell: {key}")
        prediction = entry.get("prediction")
        if not isinstance(prediction, (int, float)) or not 0 <= prediction <= 1:
            raise ValueError(f"Invalid forecast prediction for {key}")
        actual.add(key)
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise ValueError(f"Incomplete forecast lock: missing={len(missing)}, extra={len(extra)}")


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY, corpus TEXT NOT NULL, group_id TEXT NOT NULL,
    partition_name TEXT NOT NULL, text_hash TEXT NOT NULL, text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scores (
    record_id TEXT NOT NULL, variant_id TEXT NOT NULL DEFAULT 'original',
    detector_config TEXT NOT NULL, detector_group TEXT NOT NULL,
    native_score REAL, canonical_ai_score REAL, input_token_count INTEGER,
    effective_token_count INTEGER, max_tokens INTEGER, truncated INTEGER NOT NULL DEFAULT 0,
    runtime_ms REAL, failure TEXT, adapter_json TEXT NOT NULL,
    PRIMARY KEY(record_id, variant_id, detector_config)
);
CREATE TABLE IF NOT EXISTS thresholds (
    detector_config TEXT NOT NULL, fpr REAL NOT NULL, threshold REAL NOT NULL,
    reference_hash TEXT NOT NULL, PRIMARY KEY(detector_config, fpr)
);
CREATE TABLE IF NOT EXISTS probe_triplets (
    triplet_id TEXT PRIMARY KEY, record_id TEXT NOT NULL, corpus TEXT NOT NULL,
    probe TEXT NOT NULL, original_text TEXT NOT NULL, low_text TEXT NOT NULL,
    high_text TEXT NOT NULL, eligible_sites INTEGER NOT NULL,
    low_intensity REAL NOT NULL, high_intensity REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS probe_token_checks (
    triplet_id TEXT NOT NULL, detector_config TEXT NOT NULL, fits INTEGER NOT NULL,
    original_tokens INTEGER, low_tokens INTEGER, high_tokens INTEGER,
    PRIMARY KEY(triplet_id, detector_config)
);
"""


class StudyDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(SCHEMA)
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('study_version',?)",
            (STUDY_VERSION,),
        )
        self.connection.commit()

    def add_records(self, records: Sequence[TextRecord], partitions: Mapping[str, str]) -> None:
        rows = [(r.record_id, r.corpus, r.group_id, partitions[r.record_id], text_hash(r.text), r.text) for r in records]
        if len({row[0] for row in rows}) != len(rows):
            raise ValueError("Duplicate record_id in input")
        with self.connection:
            self.connection.executemany("INSERT INTO records VALUES(?,?,?,?,?,?)", rows)

    def add_probe_triplets(self, rows: Sequence[tuple[str, str, ProbeTriplet]]) -> None:
        payload = []
        for record_id, corpus, triplet in rows:
            triplet_id = hashlib.sha256(f"{record_id}:{triplet.probe}".encode()).hexdigest()
            payload.append((
                triplet_id, record_id, corpus, triplet.probe, triplet.original,
                triplet.low, triplet.high, triplet.eligible_sites,
                triplet.low_intensity, triplet.high_intensity,
            ))
        with self.connection:
            self.connection.executemany("INSERT OR REPLACE INTO probe_triplets VALUES(?,?,?,?,?,?,?,?,?,?)", payload)

    def scored_target_count(self) -> int:
        query = """
        SELECT COUNT(*) FROM scores s JOIN records r USING(record_id)
        WHERE r.partition_name IN ('signature','test')
        """
        return int(self.connection.execute(query).fetchone()[0])

    def assert_forecast_can_lock(self) -> None:
        if self.scored_target_count():
            raise RuntimeError("Target scores exist; restore a clean pre-lock database before forecasting.")

    def records(
        self,
        partitions: Sequence[str],
        *,
        include_corpus: str | None = None,
        include_corpora: Sequence[str] | None = None,
        exclude_corpus: str | None = None,
    ) -> Iterator[tuple[str, str]]:
        filters = sum(bool(value) for value in (include_corpus, include_corpora, exclude_corpus))
        if filters > 1:
            raise ValueError("Choose one corpus filter")
        placeholders = ",".join("?" for _ in partitions)
        query = f"SELECT record_id,text FROM records WHERE partition_name IN ({placeholders}) ORDER BY record_id"
        parameters: tuple[object, ...] = tuple(partitions)
        if include_corpus:
            query = query.replace(" ORDER BY", " AND corpus=? ORDER BY")
            parameters += (include_corpus,)
        elif include_corpora:
            corpus_placeholders = ",".join("?" for _ in include_corpora)
            query = query.replace(" ORDER BY", f" AND corpus IN ({corpus_placeholders}) ORDER BY")
            parameters += tuple(include_corpora)
        elif exclude_corpus:
            query = query.replace(" ORDER BY", " AND corpus<>? ORDER BY")
            parameters += (exclude_corpus,)
        yield from self.connection.execute(query, parameters)

    def has_score(self, record_id: str, detector_config: str, variant_id: str = "original") -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM scores WHERE record_id=? AND variant_id=? AND detector_config=?",
            (record_id, variant_id, detector_config),
        ).fetchone()
        return row is not None

    def add_score(self, record_id: str, detector_config: str, detector_group: str, payload: Mapping[str, object], variant_id: str = "original") -> None:
        adapter_json = json.dumps(payload, sort_keys=True, default=str)
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO scores(
                    record_id,variant_id,detector_config,detector_group,native_score,
                    canonical_ai_score,input_token_count,effective_token_count,max_tokens,
                    truncated,runtime_ms,failure,adapter_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record_id, variant_id, detector_config, detector_group,
                    payload.get("native_score"), payload.get("canonical_ai_score"),
                    payload.get("input_token_count"), payload.get("effective_token_count"),
                    payload.get("max_tokens"), int(bool(payload.get("truncated"))),
                    payload.get("runtime_ms"), payload.get("failure"), adapter_json,
                ),
            )

    def probe_triplets(
        self,
        *,
        include_corpora: Sequence[str] | None = None,
        exclude_corpus: str | None = None,
    ) -> Iterator[tuple]:
        if include_corpora and exclude_corpus:
            raise ValueError("Choose include_corpora or exclude_corpus, not both")
        query = """SELECT triplet_id,record_id,probe,original_text,low_text,high_text
                   FROM probe_triplets"""
        parameters: tuple[object, ...] = ()
        if include_corpora:
            placeholders = ",".join("?" for _ in include_corpora)
            query += f" WHERE corpus IN ({placeholders})"
            parameters = tuple(include_corpora)
        elif exclude_corpus:
            query += " WHERE corpus<>?"
            parameters = (exclude_corpus,)
        query += " ORDER BY corpus,probe,record_id"
        yield from self.connection.execute(query, parameters)

    def add_probe_token_check(
        self,
        triplet_id: str,
        detector_config: str,
        counts: tuple[int, int, int],
        fits: bool,
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO probe_token_checks VALUES(?,?,?,?,?,?)",
                (triplet_id, detector_config, int(fits), *counts),
            )

    def valid_primary_triplets(self, detector_configs: Sequence[str], limit_per_corpus_probe: int = 50) -> list[str]:
        """Return triplets fitting every active detector; sparse groups remain unavailable."""
        detector_configs = tuple(dict.fromkeys(detector_configs))
        if not detector_configs:
            raise ValueError("Primary probes require at least one active tokenizer")
        placeholders = ",".join("?" for _ in detector_configs)
        query = f"""
        WITH valid AS (
          SELECT p.triplet_id,p.corpus,p.probe,r.group_id,
                 COUNT(DISTINCT c.detector_config) checked,
                 MIN(c.fits) fits
          FROM probe_triplets p
          JOIN probe_token_checks c USING(triplet_id)
          JOIN records r USING(record_id)
          WHERE c.detector_config IN ({placeholders})
          GROUP BY p.triplet_id,p.corpus,p.probe,r.group_id
          HAVING checked=? AND fits=1
        ), one_per_group AS (
          SELECT *,ROW_NUMBER() OVER(
            PARTITION BY corpus,probe,group_id ORDER BY triplet_id
          ) group_position
          FROM valid
        ), ranked AS (
          SELECT *,ROW_NUMBER() OVER(PARTITION BY corpus,probe ORDER BY triplet_id) position
          FROM one_per_group WHERE group_position=1
        )
        SELECT triplet_id FROM ranked WHERE position<=?
        """
        parameters = (*detector_configs, len(detector_configs), limit_per_corpus_probe)
        return [row[0] for row in self.connection.execute(query, parameters)]

    def threshold_inputs(
        self,
        detector_configs: Sequence[str],
    ) -> tuple[list[tuple[str, str]], dict[str, dict[str, float]]]:
        retained = [
            (record_id, digest)
            for record_id, digest in self.connection.execute(
                "SELECT record_id,text_hash FROM records WHERE partition_name='threshold_reference' ORDER BY record_id"
            )
        ]
        expected = {record_id for record_id, _ in retained}
        scores: dict[str, dict[str, float]] = {}
        for detector in detector_configs:
            rows = self.connection.execute(
                """SELECT s.record_id,s.canonical_ai_score
                   FROM scores s JOIN records r USING(record_id)
                   WHERE r.partition_name='threshold_reference'
                     AND s.variant_id='original' AND s.detector_config=?
                     AND s.failure IS NULL AND s.canonical_ai_score IS NOT NULL""",
                (detector,),
            )
            scores[detector] = {record_id: float(score) for record_id, score in rows}
            if set(scores[detector]) != expected:
                raise RuntimeError(f"Incomplete threshold-reference scores for {detector}")
        return retained, scores

    def import_source_results(
        self,
        source_database: Path,
        detector_config: str,
        excluded_corpus: str,
    ) -> None:
        source = sqlite3.connect(source_database)
        try:
            score_rows = source.execute(
                """SELECT s.* FROM scores s JOIN records r USING(record_id)
                   WHERE s.detector_config=? AND r.corpus<>?
                     AND r.partition_name IN ('source_summary','source_model','anchor_candidates')""",
                (detector_config, excluded_corpus),
            ).fetchall()
            check_rows = source.execute(
                """SELECT c.* FROM probe_token_checks c
                   JOIN probe_triplets p USING(triplet_id)
                   WHERE c.detector_config=? AND p.corpus<>?""",
                (detector_config, excluded_corpus),
            ).fetchall()
        finally:
            source.close()
        with self.connection:
            self.connection.executemany(
                "INSERT OR IGNORE INTO scores VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                score_rows,
            )
            self.connection.executemany(
                "INSERT OR IGNORE INTO probe_token_checks VALUES(?,?,?,?,?,?)",
                check_rows,
            )

    def missing_partition_scores(
        self,
        partition: str,
        corpus: str,
        detector_configs: Sequence[str],
        record_ids: set[str] | None = None,
    ) -> dict[str, int]:
        query = "SELECT record_id FROM records WHERE partition_name=? AND corpus=?"
        expected = {row[0] for row in self.connection.execute(query, (partition, corpus))}
        if record_ids is not None:
            if not record_ids <= expected:
                raise ValueError("Requested record IDs are outside the target partition")
            expected = record_ids
        missing = {}
        for detector in detector_configs:
            scored = {
                row[0] for row in self.connection.execute(
                    """SELECT s.record_id FROM scores s
                       JOIN records r USING(record_id)
                       WHERE r.partition_name=? AND r.corpus=?
                         AND s.variant_id='original' AND s.detector_config=?
                         AND s.failure IS NULL AND s.canonical_ai_score IS NOT NULL""",
                    (partition, corpus, detector),
                )
            }
            missing[detector] = len(expected - scored)
        return {detector: count for detector, count in missing.items() if count}

    def close(self) -> None:
        self.connection.close()
