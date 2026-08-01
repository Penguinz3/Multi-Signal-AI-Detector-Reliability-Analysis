from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any, Callable, Sequence


MAGE_REPOSITORY_REVISION = "6d11f851184b9f04166f952ddc1f47727f36710f"
LASTDE_IMPLEMENTATION_REVISION = "ead6939e0e9382f9ce5aa1b33b936ee6c4e0605d"
RADAR_DEPLOYMENT_REVISION = "0b17bd20035a44d5bd4df265c03d088ea9389960"
RADAR_ORIENTATION_SOURCE = (
    "https://huggingface.co/spaces/TrustSafeAI/RADAR-AI-Text-Detector/blob/"
    f"{RADAR_DEPLOYMENT_REVISION}/app.py#L18-L21"
)


@dataclass(frozen=True)
class DetectorSpec:
    config_id: str
    method_family: str
    dependency_group: str
    model_id: str
    ai_label: int | None
    revision: str
    tokenizer_revision: str
    max_tokens: int = 512
    precision: str = "fp32"
    quantization: str = "none"
    score_orientation: str = "higher_is_more_ai_like"
    implementation_revision: str | None = None
    preprocessing_revision: str | None = None
    candidate: bool = False
    orientation_source: str | None = None


SPECS = {
    "openai_roberta_base__gpt2_legacy": DetectorSpec(
        "openai_roberta_base__gpt2_legacy", "supervised_legacy", "openai_roberta",
        "openai-community/roberta-base-openai-detector", 0,
        "6cba99c003b711c7fe94f8a3aa2be35a792cb6fa",
        "6cba99c003b711c7fe94f8a3aa2be35a792cb6fa",
    ),
    "radar_roberta_large__vicuna7b_training": DetectorSpec(
        "radar_roberta_large__vicuna7b_training", "supervised_adversarial", "radar",
        "TrustSafeAI/RADAR-Vicuna-7B", 0,
        "4ff1f23a69a36aa1df47b0933be6279f1b896c9b",
        "4ff1f23a69a36aa1df47b0933be6279f1b896c9b",
        orientation_source=RADAR_ORIENTATION_SOURCE,
    ),
    "mage_longformer__paper": DetectorSpec(
        "mage_longformer__paper", "supervised_cross_domain", "mage",
        "yaful/MAGE", 0,
        "0d82ca0fdf6ebef5babb813cc11bd8eb2552c846",
        "0d82ca0fdf6ebef5babb813cc11bd8eb2552c846",
        preprocessing_revision=MAGE_REPOSITORY_REVISION,
    ),
    "logrank__qwen2_5_0_5b_fp32": DetectorSpec(
        "logrank__qwen2_5_0_5b_fp32", "zero_shot_logrank", "qwen25_shared",
        "Qwen/Qwen2.5-0.5B", None,
        "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "060db6499f32faf8b98477b0a26969ef7d8b9987", 512, "fp32", "none",
    ),
    "lastde__qwen2_5_0_5b_fp32": DetectorSpec(
        "lastde__qwen2_5_0_5b_fp32", "zero_shot_lastde", "qwen25_shared",
        "Qwen/Qwen2.5-0.5B", None,
        "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "060db6499f32faf8b98477b0a26969ef7d8b9987", 512, "fp32", "none",
        implementation_revision=LASTDE_IMPLEMENTATION_REVISION,
    ),
}


@dataclass
class ScoreRecord:
    detector_config: str
    method_family: str
    dependency_group: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    quantization: str
    dtype: str
    device_map: str
    input_token_count: int
    effective_token_count: int
    max_tokens: int
    truncated: bool
    native_score: float | None
    canonical_ai_score: float | None
    runtime_ms: float
    failure: str | None
    text_hash: str
    cache_hash: str | None = None
    software_commit: str | None = None
    precision: str | None = None
    score_orientation: str = "higher_is_more_ai_like"
    implementation_revision: str | None = None
    preprocessing_revision: str | None = None
    attention_implementation: str | None = None
    candidate: bool = False
    orientation_source: str | None = None


def _require_pinned_revision(revision: str, label: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError(f"{label} must be an immutable 40-character Git revision.")


def _model_metadata(model: Any, fallback_device_map: str) -> tuple[str, str]:
    dtype = str(next(model.parameters()).dtype)
    device_map = getattr(model, "hf_device_map", fallback_device_map)
    if not isinstance(device_map, str):
        device_map = json.dumps(device_map, sort_keys=True, default=str)
    return dtype, device_map


def _require_finite(torch: Any, value: Any, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{label} contains non-finite values; score rejected.")


@lru_cache(maxsize=1)
def _software_commit() -> str | None:
    try:
        revision = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else None


class SequenceClassifierAdapter:
    def __init__(self, spec: DetectorSpec, device: int = -1, preprocessor: Callable[[str], str] | None = None):
        if spec.ai_label is None:
            raise ValueError(f"{spec.config_id} requires an explicit AI label mapping from its pinned config.")
        _require_pinned_revision(spec.revision, f"{spec.config_id} model")
        _require_pinned_revision(spec.tokenizer_revision, f"{spec.config_id} tokenizer")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.tokenizer_revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(spec.model_id, revision=spec.revision)
        self.device = device
        if device >= 0:
            self.model.to(f"cuda:{device}")
        self.model.eval()
        self.preprocessor = preprocessor
        if spec.config_id == "mage_longformer__paper" and preprocessor is None:
            raise RuntimeError(f"MAGE requires deployment.preprocess from pinned commit {MAGE_REPOSITORY_REVISION}.")

    def token_count(self, text: str) -> int:
        text = self.preprocessor(text) if self.preprocessor else text
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def score(self, text: str) -> ScoreRecord:
        import torch
        started = time.perf_counter()
        model_text = self.preprocessor(text) if self.preprocessor else text
        count = len(self.tokenizer(model_text, add_special_tokens=True, truncation=False)["input_ids"])
        if count > self.spec.max_tokens:
            raise ValueError(f"Primary analysis forbids truncation: {count}>{self.spec.max_tokens}")
        encoded = self.tokenizer(model_text, return_tensors="pt", padding=False, truncation=False)
        if self.device >= 0:
            encoded = {key: value.to(f"cuda:{self.device}") for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits
            _require_finite(torch, logits, "Classifier logits")
            probabilities = logits.float().softmax(-1)[0]
            _require_finite(torch, probabilities, "Classifier probabilities")
        score = float(probabilities[self.spec.ai_label].cpu())
        if not math.isfinite(score):
            raise FloatingPointError("Classifier score is non-finite; score rejected.")
        dtype, device_map = _model_metadata(
            self.model, f"cuda:{self.device}" if self.device >= 0 else "cpu",
        )
        return ScoreRecord(
            self.spec.config_id, self.spec.method_family, self.spec.dependency_group,
            self.spec.model_id, self.spec.revision, self.spec.tokenizer_revision,
            self.spec.quantization, dtype, device_map,
            count, count, self.spec.max_tokens, False, score, score,
            (time.perf_counter() - started) * 1000, None,
            hashlib.sha256(text.encode()).hexdigest(),
            software_commit=_software_commit(),
            precision=self.spec.precision,
            score_orientation=self.spec.score_orientation,
            implementation_revision=self.spec.implementation_revision,
            preprocessing_revision=self.spec.preprocessing_revision,
            candidate=self.spec.candidate,
            orientation_source=self.spec.orientation_source,
        )


class CausalTokenScorer:
    """One pinned observer pass reused by LogRank and the official Lastde statistic."""

    def __init__(self, spec: DetectorSpec):
        _require_pinned_revision(spec.revision, "observer model")
        _require_pinned_revision(spec.tokenizer_revision, "observer tokenizer")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        kwargs: dict[str, Any] = {
            "revision": spec.revision,
            "device_map": "auto",
            "attn_implementation": "eager",
            "torch_dtype": torch.float32,
        }
        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.tokenizer_revision)
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs).eval()
        self.model.config.use_cache = False

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    @lru_cache(maxsize=1024)
    def sequence(self, text: str) -> dict[str, Any]:
        import torch
        encoded = self.tokenizer(
            text, return_tensors="pt", padding=False, truncation=False,
            return_token_type_ids=False,
        )
        count = int(encoded["input_ids"].shape[1])
        if count > self.spec.max_tokens:
            raise ValueError(f"Primary analysis forbids truncation: {count}>{self.spec.max_tokens}")
        if count < 2:
            raise ValueError("No scored tokens")
        device = self.model.get_input_embeddings().weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]
        with torch.inference_mode():
            logits = self.model(**encoded, use_cache=False).logits[:, :-1, :].float()
        _require_finite(torch, logits, "Observer logits")
        targets = input_ids[:, 1:]
        target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ranks = (logits > target_logits.unsqueeze(-1)).sum(-1).add(1).squeeze(0).cpu()
        log_probs = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(0).squeeze(-1).cpu()
        _require_finite(torch, ranks, "Observer token ranks")
        _require_finite(torch, log_probs, "Observer token log probabilities")
        rank_values, probability_values = ranks.numpy(), log_probs.numpy()
        payload = {"ranks": rank_values, "log_probs": probability_values, "token_count": count}
        raw = rank_values.tobytes() + probability_values.tobytes()
        payload["cache_hash"] = hashlib.sha256(raw).hexdigest()
        return payload


def logrank(sequence: dict[str, Any]) -> float:
    import numpy as np
    ranks = np.asarray(sequence["ranks"], dtype=float)
    if not len(ranks):
        raise ValueError("No scored tokens")
    return -float(np.log(ranks).mean())  # Higher means more AI-like.


def lastde(sequence: dict[str, Any], embed_size: int = 3, tau_prime: int = 5) -> float:
    """Published open-source Lastde statistic from pinned commit ead6939e."""
    import torch
    values = torch.as_tensor(sequence["log_probs"], dtype=torch.float32).reshape(1, -1, 1)
    token_count = values.shape[1]
    epsilon = 10 * token_count
    multiscale = []
    for tau in range(1, tau_prime + 1):
        scaled = values.unfold(1, tau, 1).mean(dim=3)
        if scaled.shape[1] <= embed_size:
            raise ValueError("Text is too short for Lastde")
        orbits = scaled.unfold(1, embed_size, 1)
        similarities = torch.nn.functional.cosine_similarity(orbits[:, :-1], orbits[:, 1:], dim=-1)
        entropies = []
        for sample in range(similarities.shape[-1]):
            hist = torch.histc(similarities[..., sample].float(), bins=epsilon, min=-1, max=1)
            probabilities = hist / hist.sum()
            entropy = -torch.nansum(probabilities * torch.log(probabilities)) / torch.log(torch.tensor(epsilon))
            entropies.append(entropy)
        multiscale.append(torch.stack(entropies))
    dispersion = torch.stack(multiscale).std(dim=0).mean()
    score = float(values.mean() / dispersion)
    if not math.isfinite(score):
        raise ValueError("Lastde returned a non-finite statistic")
    return score


class StatisticalAdapter:
    def __init__(self, spec: DetectorSpec, scorer: CausalTokenScorer):
        self.spec = spec
        self.scorer = scorer
        self.precision = spec.precision

    def token_count(self, text: str) -> int:
        return self.scorer.token_count(text)

    def score(self, text: str) -> ScoreRecord:
        started = time.perf_counter()
        sequence = self.scorer.sequence(text)
        if self.spec.method_family == "zero_shot_logrank":
            native = logrank(sequence)
        elif self.spec.method_family == "zero_shot_lastde":
            native = lastde(sequence)
        else:
            raise ValueError(self.spec.method_family)
        count = int(sequence["token_count"])
        dtype, device_map = _model_metadata(self.scorer.model, "auto")
        return ScoreRecord(
            self.spec.config_id, self.spec.method_family, self.spec.dependency_group,
            self.spec.model_id, self.spec.revision, self.spec.tokenizer_revision,
            self.spec.quantization, dtype, device_map,
            count, count, self.spec.max_tokens, False, native, native,
            (time.perf_counter() - started) * 1000, None,
            hashlib.sha256(text.encode()).hexdigest(), sequence["cache_hash"],
            software_commit=_software_commit(),
            precision=self.spec.precision,
            score_orientation=self.spec.score_orientation,
            implementation_revision=self.spec.implementation_revision,
            preprocessing_revision=self.spec.preprocessing_revision,
            attention_implementation="eager",
            candidate=self.spec.candidate,
            orientation_source=self.spec.orientation_source,
        )


_TOKEN_SCORERS: dict[tuple[str, str, str, str], CausalTokenScorer] = {}


def _shared_token_scorer(spec: DetectorSpec) -> CausalTokenScorer:
    key = (spec.model_id, spec.revision, spec.tokenizer_revision, spec.precision)
    if key not in _TOKEN_SCORERS:
        _TOKEN_SCORERS[key] = CausalTokenScorer(spec)
    return _TOKEN_SCORERS[key]


def _mage_preprocessor(mage_repo: str) -> Callable[[str], str]:
    repo = Path(mage_repo).resolve()
    if not (repo / "deployment" / "__init__.py").is_file():
        raise RuntimeError("--mage-repo must be the official MAGE repository checkout.")
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--", "deployment"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Unable to verify the MAGE repository revision.") from error
    if revision != MAGE_REPOSITORY_REVISION:
        raise RuntimeError(
            f"MAGE checkout must be exactly {MAGE_REPOSITORY_REVISION}; found {revision or 'unknown'}."
        )
    if dirty:
        raise RuntimeError("MAGE deployment preprocessing has local modifications.")
    previous_modules = {
        name: module for name, module in sys.modules.items()
        if name == "deployment" or name.startswith("deployment.")
    }
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    try:
        for name in previous_modules:
            sys.modules.pop(name, None)
        preprocessor = importlib.import_module("deployment").preprocess
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.pop(0)
        for name in tuple(sys.modules):
            if name == "deployment" or name.startswith("deployment."):
                sys.modules.pop(name)
        sys.modules.update(previous_modules)
    return preprocessor


def build_adapter(config_id: str, device: int = -1, mage_repo: str | None = None):
    if config_id not in SPECS:
        raise ValueError(f"Unknown detector configuration: {config_id}")
    spec = SPECS[config_id]
    if spec.method_family.startswith("supervised"):
        preprocessor = None
        if spec.config_id == "mage_longformer__paper":
            if not mage_repo:
                raise RuntimeError(f"--mage-repo must point to pinned MAGE commit {MAGE_REPOSITORY_REVISION}.")
            preprocessor = _mage_preprocessor(mage_repo)
        return SequenceClassifierAdapter(spec, device, preprocessor)
    return StatisticalAdapter(spec, _shared_token_scorer(spec))


def assert_orientation_pilot(
    config_id: str,
    ai_anchor_scores: Sequence[float],
    human_anchor_scores: Sequence[float],
) -> None:
    """Fail the technical pilot unless labeled anchors confirm higher-is-AI orientation."""
    if config_id not in SPECS:
        raise ValueError(f"Unknown detector configuration: {config_id}")
    if len(ai_anchor_scores) < 3 or len(human_anchor_scores) < 3:
        raise ValueError("Orientation pilot requires at least three AI and three human anchors")
    scores = [*ai_anchor_scores, *human_anchor_scores]
    if not all(math.isfinite(float(score)) for score in scores):
        raise FloatingPointError("Orientation pilot contains non-finite scores")
    if median(ai_anchor_scores) <= median(human_anchor_scores):
        raise RuntimeError(f"{config_id} failed its higher-is-more-AI orientation pilot")


def validate_labeled_pilot(
    config_id: str,
    human_scores: Sequence[float],
    ai_scores: Sequence[float],
    human_repeats: Sequence[float],
    ai_repeats: Sequence[float],
    tolerance: float = 1e-6,
) -> float:
    fields = (human_scores, ai_scores, human_repeats, ai_repeats)
    if any(len(values) != 50 for values in fields):
        raise ValueError("Technical pilot requires exactly 50 human and 50 AI passages.")
    flat = [float(value) for values in fields for value in values]
    if not all(math.isfinite(value) for value in flat):
        raise FloatingPointError("Technical pilot contains non-finite scores.")
    if max(human_scores) == min(human_scores) or max(ai_scores) == min(ai_scores):
        raise RuntimeError("Technical pilot requires score variation in both labeled classes.")
    differences = [
        abs(float(first) - float(repeat))
        for first, repeat in zip(
            [*human_scores, *ai_scores], [*human_repeats, *ai_repeats],
        )
    ]
    worst = max(differences)
    if worst > tolerance:
        raise RuntimeError(
            f"{config_id} is not deterministic within tolerance: {worst}>{tolerance}"
        )
    assert_orientation_pilot(config_id, ai_scores, human_scores)
    return worst


def validate_specs() -> None:
    groups = {spec.dependency_group for spec in SPECS.values()}
    if len(SPECS) < 4 or len(groups) < 4:
        raise RuntimeError("Admission gate requires >=4 configurations across all 4 dependency groups.")
    for config_id, spec in SPECS.items():
        if config_id != spec.config_id:
            raise RuntimeError(f"Detector key and precision-specific ID differ for {config_id}.")
        _require_pinned_revision(spec.revision, f"{config_id} model")
        _require_pinned_revision(spec.tokenizer_revision, f"{config_id} tokenizer")
        if spec.method_family.startswith("supervised") and spec.ai_label not in {0, 1}:
            raise RuntimeError(f"Resolve label orientation for {spec.config_id} before the pilot.")
        if spec.score_orientation != "higher_is_more_ai_like":
            raise RuntimeError(f"{config_id} lacks canonical AI-score orientation.")
        if spec.precision not in config_id and spec.method_family.startswith("zero_shot"):
            raise RuntimeError(f"{config_id} must encode its precision.")
    if SPECS["radar_roberta_large__vicuna7b_training"].ai_label != 0:
        raise RuntimeError("Pinned RADAR class 0 is the AI-generated orientation.")
    if SPECS["radar_roberta_large__vicuna7b_training"].orientation_source != RADAR_ORIENTATION_SOURCE:
        raise RuntimeError("RADAR checkpoint orientation source is not pinned.")
    if SPECS["mage_longformer__paper"].preprocessing_revision != MAGE_REPOSITORY_REVISION:
        raise RuntimeError("MAGE preprocessing revision is not pinned.")
    observers = [
        spec for spec in SPECS.values()
        if spec.dependency_group == "qwen25_shared"
    ]
    if len(observers) != 2 or len({
        (spec.model_id, spec.revision, spec.tokenizer_revision, spec.precision)
        for spec in observers
    }) != 1:
        raise RuntimeError("LogRank and Lastde must share one pinned fp32 observer.")
