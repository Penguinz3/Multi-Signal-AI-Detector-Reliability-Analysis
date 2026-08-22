from __future__ import annotations

import hashlib
from collections import Counter
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np

from .core import canonical_json


FORECAST_ID_FIELDS = (
    "target_corpus",
    "detector_config",
    "operating_fpr",
    "signature_size",
    "draw",
    "model",
)


def _groups(group_ids: Sequence[Hashable]) -> tuple[tuple[int, ...], ...]:
    grouped: dict[Hashable, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(index)
    if not grouped:
        raise ValueError("At least one group is required")
    return tuple(tuple(indices) for indices in grouped.values())


def cluster_bootstrap_indices(
    group_ids: Sequence[Hashable],
    *,
    seed: int,
    replicates: int = 100,
) -> tuple[tuple[int, ...], ...]:
    """Resample clusters, retaining every row of each sampled cluster."""
    groups = _groups(group_ids)
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    return tuple(
        tuple(index for draw in rng.integers(len(groups), size=len(groups)) for index in groups[int(draw)])
        for _ in range(replicates)
    )


def summarize_replicates(
    values: Sequence[float],
    *,
    confidence: float = .90,
) -> dict[str, float | int]:
    """Return a deterministic percentile band plus the replicate mean and SD."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty sequence of finite numbers")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(array, (tail, 1.0 - tail), method="linear")
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1 if len(array) > 1 else 0)),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": float(confidence),
    }


def forecast_identity_hashes(
    forecast: Mapping[str, object],
    signature_record_ids: Sequence[str],
) -> tuple[str, str]:
    """Hash the canonical forecast cell and its order-independent signature IDs."""
    missing = [field for field in FORECAST_ID_FIELDS if field not in forecast]
    if missing:
        raise ValueError(f"Forecast identity is missing fields: {missing}")
    if not signature_record_ids or len(signature_record_ids) != len(set(signature_record_ids)):
        raise ValueError("Signature record IDs must be non-empty and unique")
    identity = {field: forecast[field] for field in FORECAST_ID_FIELDS}
    forecast_id = hashlib.sha256(canonical_json(identity)).hexdigest()
    signature_hash = hashlib.sha256(canonical_json(sorted(signature_record_ids))).hexdigest()
    return forecast_id, signature_hash


def split_half_pairs(
    group_ids: Sequence[Hashable],
    *,
    seed: int,
    pairs: int = 100,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Generate seeded disjoint half-samples without splitting any group."""
    groups = _groups(group_ids)
    if len(groups) < 2:
        raise ValueError("At least two groups are required")
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    rng = np.random.default_rng(seed)
    midpoint = len(groups) // 2
    result = []
    for _ in range(pairs):
        order = rng.permutation(len(groups))
        left = tuple(index for group in order[:midpoint] for index in groups[int(group)])
        right = tuple(index for group in order[midpoint:] for index in groups[int(group)])
        result.append((left, right))
    return tuple(result)


def validate_replicate_completeness(
    replicate_ids: Iterable[int],
    *,
    expected: int = 100,
) -> tuple[int, ...]:
    """Require exactly one result for every replicate ID from 0 to expected-1."""
    if expected <= 0:
        raise ValueError("expected must be positive")
    ids = tuple(replicate_ids)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
        raise ValueError("Replicate IDs must be integers")
    counts = Counter(ids)
    required = set(range(expected))
    if len(ids) != expected or set(ids) != required or any(count != 1 for count in counts.values()):
        missing = sorted(required - set(ids))
        extra = sorted(set(ids) - required)
        duplicates = sorted(value for value, count in counts.items() if count > 1)
        raise ValueError(
            f"Incomplete replicates: {len(ids)}/{expected}; "
            f"missing={missing}, extra={extra}, duplicates={duplicates}"
        )
    return tuple(sorted(ids))
