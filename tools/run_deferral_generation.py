"""Run the locked, provider-neutral deferral generation requests locally.

The runner is intentionally small and sequential: one pinned Hugging Face
family is resident at a time, every accepted output is checkpointed before
the next request, and the final CSV is already in the shape consumed by
``fprint import-deferral-panel``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


IMPORT_FIELDS = (
    "request_id", "generator_family", "generator_revision", "retry", "attempt",
    "seed", "target_length", "raw_word_count", "selected_word_count",
    "prefix_rank", "prefix_used", "decoding", "text", "token_counts",
)


class GenerationBackend(Protocol):
    def generate(self, prompt: str, *, seed: int, target_length: int,
                 min_word_count: int, max_word_count: int,
                 decoding: Mapping[str, object]) -> str: ...


@dataclass(frozen=True)
class LockedRequest:
    request_id: str
    generator_family: str
    generator_revision: str
    retry: int
    seed: int
    target_length: int
    min_word_count: int
    max_word_count: int
    decoding: Mapping[str, object]
    prompt: str
    record_id: str = ""


class GenerationFailure(RuntimeError):
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_artifacts(model_path: Path | str, artifact_files: Mapping[str, object]) -> None:
    root = Path(model_path).resolve()
    if not artifact_files:
        raise ValueError(f"No pinned artifact files for {root.name}")
    for relative, expected in sorted(artifact_files.items()):
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe model artifact path: {relative}")
        path = (root / relative_path).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Missing pinned model artifact: {path}")
        if _file_sha256(path) != str(expected).casefold():
            raise RuntimeError(f"Pinned model artifact hash mismatch: {path}")


def request_seed(locked_seed: int, request_id: str, attempt: int) -> int:
    """Derive a stable, independent 32-bit generation seed."""
    payload = f"{int(locked_seed)}:{request_id}:{int(attempt)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647


def _decode_field(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def read_locked_requests(path: Path | str) -> tuple[LockedRequest, ...]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Locked generation request CSV is empty")
    output: list[LockedRequest] = []
    seen: set[str] = set()
    required = {"request_id", "generator_family", "generator_revision", "retry", "seed", "target_length", "min_word_count", "max_word_count", "decoding", "prompt"}
    for row in rows:
        missing = sorted(field for field in required if not str(row.get(field, "")).strip())
        if missing:
            raise ValueError(f"Generation request is missing {missing}")
        request_id = str(row["request_id"])
        if request_id in seen:
            raise ValueError(f"Duplicate locked request: {request_id}")
        seen.add(request_id)
        retry = int(row["retry"])
        minimum = int(row["min_word_count"])
        maximum = int(row["max_word_count"])
        if retry < 0 or minimum <= 0 or maximum < minimum:
            raise ValueError(f"Invalid retry or word-count envelope: {request_id}")
        prompt = str(row["prompt"])
        if not prompt.strip():
            raise ValueError(f"Empty locked prompt: {request_id}")
        decoding = _decode_field(row["decoding"])
        if not isinstance(decoding, Mapping):
            raise ValueError(f"Locked decoding must be a JSON object: {request_id}")
        output.append(LockedRequest(
            request_id=request_id,
            generator_family=str(row["generator_family"]),
            generator_revision=str(row["generator_revision"]),
            retry=retry,
            seed=int(row["seed"]),
            target_length=int(row["target_length"]),
            min_word_count=minimum,
            max_word_count=maximum,
            decoding=dict(decoding),
            prompt=prompt,
            record_id=str(row.get("record_id", "")),
        ))
    return tuple(output)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % 4_294_967_296)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _passage_only(text: str) -> str:
    """Remove model reasoning wrappers while retaining only assistant text."""
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.I | re.S)
    value = re.sub(r"</?think>", "", value, flags=re.I)
    value = re.sub(r"^\s*(?:assistant|Answer)\s*:\s*", "", value, flags=re.I)
    return value.strip()


def _word_count(text: str) -> int:
    return len(str(text).split())


def _is_complete_passage(text: str) -> bool:
    return bool(re.search(r"[.!?][\"')\]]*$", str(text).strip()))


def _fit_word_envelope(
    text: str, *, target_length: int, min_word_count: int, max_word_count: int,
) -> str:
    """Keep a valid passage or remove only an overlong/incomplete suffix."""
    candidates = _word_envelope_candidates(
        text, target_length=target_length, min_word_count=min_word_count,
        max_word_count=max_word_count,
    )
    return candidates[0] if candidates else _passage_only(text)


def _word_envelope_candidates(
    text: str, *, target_length: int, min_word_count: int, max_word_count: int,
) -> tuple[str, ...]:
    """Return complete sentence prefixes ordered by paired-length proximity."""
    value = _passage_only(text)
    candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    if min_word_count <= _word_count(value) <= max_word_count and _is_complete_passage(value):
        candidates.append((abs(_word_count(value) - target_length), -_word_count(value), value))
        seen.add(value)
    for match in re.finditer(r"[.!?][\"')\]]*(?=\s|$)", value):
        prefix = value[:match.end()].strip()
        count = _word_count(prefix)
        if min_word_count <= count <= max_word_count and prefix not in seen:
            candidates.append((abs(count - target_length), -count, prefix))
            seen.add(prefix)
    return tuple(row[2] for row in sorted(candidates))


class HuggingFaceBackend:
    """One exact family/revision loaded in BF16 on one CUDA device."""

    def __init__(
        self, family: str, revision: str, *, device: int = 0,
        model_path: Path | str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("Local generation requires CUDA; use the fake backend for tests")
        self.family = str(family)
        self.revision = str(revision)
        self.device_name = f"cuda:{int(device)}"
        source = str(Path(model_path).resolve()) if model_path is not None else self.family
        revision_kwargs = {} if model_path is not None else {"revision": self.revision}
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=model_path is not None, **revision_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16, local_files_only=model_path is not None,
            attn_implementation="eager",
            **revision_kwargs,
        ).to(self.device_name).eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _inputs(self, prompt: str, *, disable_thinking: bool = False):
        if disable_thinking and self.family.casefold().startswith("qwen/"):
            prompt = f"{prompt}\n/no_think"
        messages = [{"role": "user", "content": str(prompt)}]
        template = getattr(self.tokenizer, "apply_chat_template", None)
        if template is None:
            return self.tokenizer(prompt, return_tensors="pt")
        common = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        try:
            return template(messages, enable_thinking=False, **common)
        except TypeError:
            try:
                return template(messages, thinking=False, **common)
            except TypeError:
                return template(messages, **common)

    def generate(self, prompt: str, *, seed: int, target_length: int,
                 min_word_count: int, max_word_count: int,
                 decoding: Mapping[str, object]) -> str:
        import torch
        _set_seed(seed)
        runtime_decoding = dict(decoding)
        disable_thinking = bool(runtime_decoding.pop("disable_thinking", False))
        encoded = self._inputs(prompt, disable_thinking=disable_thinking)
        if hasattr(encoded, "to"):
            encoded = encoded.to(self.device_name)
        elif isinstance(encoded, Mapping):
            encoded = {key: value.to(self.device_name) for key, value in encoded.items()}
        if isinstance(encoded, Mapping):
            input_ids = encoded["input_ids"]
        else:
            # ``apply_chat_template`` returns a tensor unless return_dict=True
            # is requested; generation still expects keyword model inputs.
            input_ids = encoded
            encoded = {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}
        runtime_decoding.pop("max_length", None)
        runtime_decoding.setdefault("min_new_tokens", max(1, int(min_word_count * 1.35)))
        runtime_decoding.setdefault("max_new_tokens", max(32, int(max_word_count * 1.60)))
        if "temperature" in runtime_decoding:
            runtime_decoding["do_sample"] = float(runtime_decoding["temperature"]) > 0
        runtime_decoding.setdefault("do_sample", True)
        with torch.inference_mode():
            generated = self.model.generate(**encoded, **runtime_decoding)
        continuation = generated[0, input_ids.shape[-1]:]
        return _passage_only(self.tokenizer.decode(continuation, skip_special_tokens=True))


class FakeBackend:
    """Deterministic backend used only by focused tests and dry runs."""

    def __init__(self, family: str, revision: str):
        self.family, self.revision = family, revision
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, *, seed: int, target_length: int,
                 min_word_count: int, max_word_count: int,
                 decoding: Mapping[str, object]) -> str:
        self.calls.append((prompt, seed))
        words = ["generated"] * max(1, min(max_word_count, target_length))
        words[-1] += "."
        return " ".join(words)


def _load_checkpoint(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, object]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # interrupted append: discard only the incomplete tail
            raise ValueError(f"Corrupt generation checkpoint line {index + 1}")
        request_id = str(row.get("request_id", ""))
        if not request_id or request_id in completed:
            raise ValueError(f"Invalid or duplicate checkpoint row: {request_id}")
        completed[request_id] = row
    return completed


def _append_checkpoint(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _csv_row(
    request: LockedRequest, text: str, attempt: int, seed: int,
    token_counts: Mapping[str, object] | None = None,
    *, raw_word_count: int | None = None, prefix_rank: int = 0,
) -> dict[str, object]:
    selected_word_count = _word_count(text)
    raw_count = selected_word_count if raw_word_count is None else int(raw_word_count)
    return {
        "request_id": request.request_id,
        "generator_family": request.generator_family,
        "generator_revision": request.generator_revision,
        "retry": request.retry,
        "attempt": attempt,
        "seed": seed,
        "target_length": request.target_length,
        "raw_word_count": raw_count,
        "selected_word_count": selected_word_count,
        "prefix_rank": int(prefix_rank),
        "prefix_used": int(raw_count != selected_word_count or prefix_rank != 0),
        "decoding": json.dumps(dict(request.decoding), sort_keys=True),
        "text": text,
        "token_counts": "" if token_counts is None else json.dumps(token_counts, sort_keys=True),
    }


def run_generation(
    requests_csv: Path | str,
    output_csv: Path | str,
    checkpoint: Path | str,
    *,
    backend_factory: Callable[[str, str], GenerationBackend] | None = None,
    lock_verifier: Callable[[], object] | None = None,
    panel_counter: Callable[[str], Mapping[str, object]] | None = None,
    continue_on_failure: bool = False,
    failure_log: Path | str | None = None,
) -> tuple[dict[str, object], ...]:
    """Generate all locked requests, resuming from accepted checkpoint rows."""
    if continue_on_failure != (failure_log is not None):
        raise ValueError("Screening mode requires both continue_on_failure and failure_log")
    if lock_verifier is not None:
        lock_verifier()
    requests = read_locked_requests(requests_csv)
    request_by_id = {request.request_id: request for request in requests}
    checkpoint_path = Path(checkpoint)
    completed = _load_checkpoint(checkpoint_path)
    failures = _load_checkpoint(Path(failure_log)) if failure_log is not None else {}
    unknown = set(completed) - set(request_by_id)
    unknown_failures = set(failures) - set(request_by_id)
    if unknown or unknown_failures:
        raise ValueError(f"Checkpoint contains unknown requests: {sorted(unknown | unknown_failures)[:3]}")
    if set(completed) & set(failures):
        raise ValueError("A request cannot be both accepted and generation-infeasible")
    for request_id, row in completed.items():
        request = request_by_id[request_id]
        if str(row.get("generator_family")) != request.generator_family or str(row.get("generator_revision")) != request.generator_revision or int(row.get("retry", -1)) != request.retry:
            raise ValueError(f"Checkpoint provenance mismatch: {request_id}")
        attempt = int(row.get("attempt", -1))
        if attempt < 0 or attempt > request.retry or int(row.get("seed", -1)) != request_seed(request.seed, request_id, attempt):
            raise ValueError(f"Checkpoint seed/attempt mismatch: {request_id}")
        if int(row.get("target_length", -1)) != request.target_length:
            raise ValueError(f"Checkpoint target length mismatch: {request_id}")
        count = _word_count(str(row.get("text", "")))
        if not request.min_word_count <= count <= request.max_word_count or not _is_complete_passage(str(row.get("text", ""))):
            raise ValueError(f"Checkpoint contains out-of-range text: {request_id}")
        if int(row.get("selected_word_count", count)) != count or int(row.get("raw_word_count", count)) < count:
            raise ValueError(f"Checkpoint word-count provenance mismatch: {request_id}")
        if panel_counter is not None:
            expected_counts = json.dumps(panel_counter(str(row["text"])), sort_keys=True)
            actual_counts = row.get("token_counts", "")
            if isinstance(actual_counts, str):
                actual_counts = json.loads(actual_counts) if actual_counts else None
            if actual_counts is None or json.dumps(actual_counts, sort_keys=True) != expected_counts:
                raise ValueError(f"Checkpoint token panel mismatch: {request_id}")
    for request_id, row in failures.items():
        request = request_by_id[request_id]
        if (
            str(row.get("record_id")) != request.record_id
            or str(row.get("generator_family")) != request.generator_family
            or str(row.get("generator_revision")) != request.generator_revision
            or int(row.get("attempts", -1)) != request.retry + 1
        ):
            raise ValueError(f"Failure-log provenance mismatch: {request_id}")
    factory = backend_factory or (lambda family, revision: HuggingFaceBackend(family, revision))
    all_rows: dict[str, dict[str, object]] = dict(completed)
    started = time.monotonic()
    total = len(requests)
    print(f"Generation progress: {len(all_rows)}/{total} accepted", flush=True)
    grouped: dict[tuple[str, str], list[LockedRequest]] = {}
    for request in requests:
        if request.request_id not in completed and request.request_id not in failures:
            grouped.setdefault((request.generator_family, request.generator_revision), []).append(request)
    for (family, revision), batch in grouped.items():
        backend = factory(family, revision)
        for request in batch:
            accepted = None
            diagnostics = []
            for attempt in range(request.retry + 1):
                seed = request_seed(request.seed, request.request_id, attempt)
                raw_text = backend.generate(
                    request.prompt, seed=seed, target_length=request.target_length,
                    min_word_count=request.min_word_count,
                    max_word_count=request.max_word_count, decoding=request.decoding,
                )
                candidates = _word_envelope_candidates(
                    raw_text,
                    target_length=request.target_length,
                    min_word_count=request.min_word_count,
                    max_word_count=request.max_word_count,
                )
                raw_word_count = _word_count(_passage_only(raw_text))
                token_valid_candidates = 0
                for prefix_rank, text in enumerate(candidates):
                    try:
                        token_counts = panel_counter(text) if panel_counter is not None else None
                    except ValueError:
                        continue
                    token_valid_candidates += 1
                    accepted = _csv_row(
                        request, text, attempt, seed, token_counts,
                        raw_word_count=raw_word_count, prefix_rank=prefix_rank,
                    )
                    _append_checkpoint(checkpoint_path, accepted)
                    all_rows[request.request_id] = accepted
                    if len(all_rows) % 25 == 0 or len(all_rows) == total:
                        elapsed = max(time.monotonic() - started, 1e-9)
                        rate = (len(all_rows) - len(completed)) / elapsed
                        remaining = (total - len(all_rows)) / rate if rate > 0 else float("inf")
                        print(
                            f"Generation progress: {len(all_rows)}/{total} accepted; "
                            f"{rate:.3f}/s; ETA {remaining / 3600:.1f}h",
                            flush=True,
                        )
                    break
                diagnostics.append({
                    "attempt": attempt,
                    "seed": seed,
                    "raw_word_count": raw_word_count,
                    "candidate_word_counts": [_word_count(text) for text in candidates],
                    "token_valid_candidates": token_valid_candidates,
                })
                if accepted is not None:
                    break
            if accepted is None:
                message = f"No output within locked word envelope after {request.retry + 1} attempts: {request.request_id}"
                if not continue_on_failure:
                    raise GenerationFailure(message)
                failure_row = {
                    "request_id": request.request_id,
                    "record_id": request.record_id,
                    "generator_family": request.generator_family,
                    "generator_revision": request.generator_revision,
                    "attempts": request.retry + 1,
                    "diagnostics": diagnostics,
                    "reason": message,
                }
                _append_checkpoint(Path(failure_log), failure_row)
                failures[request.request_id] = failure_row
        del backend
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    ordered = tuple(all_rows[request.request_id] for request in requests if request.request_id in all_rows)
    if len(ordered) + len(failures) != total:
        raise RuntimeError("Generation screening did not account for every locked request")
    if failures:
        return ordered
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=IMPORT_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in IMPORT_FIELDS} for row in ordered)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output_path)
    return ordered


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, help="Verify the generation CSV against its immutable study lock")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model-root", type=Path, help="Directory containing one exact local folder per model basename")
    parser.add_argument("--generation-spec", type=Path, help="Protocol-bound generator specification with artifact hashes")
    parser.add_argument("--mage-repo", type=Path, help="Pinned MAGE repository used for atomic token-panel checks")
    parser.add_argument("--fake", action="store_true", help="Use the deterministic test backend")
    parser.add_argument("--continue-on-failure", action="store_true", help="Score-blind screening: log exhausted requests and continue")
    parser.add_argument("--failure-log", type=Path, help="Append-only JSONL for exhausted screening requests")
    args = parser.parse_args(argv)
    specifications = {}
    if not args.fake:
        if not args.generation_spec or not args.model_root:
            raise ValueError("--generation-spec and --model-root are required for real generation")
        generation_spec = json.loads(args.generation_spec.read_text(encoding="utf-8"))
        if generation_spec.get("dtype") != "bfloat16" or generation_spec.get("attention_implementation") != "eager":
            raise ValueError("Generation specification must pin BF16 with eager attention")
        specifications = {
            str(row["family"]): row
            for row in generation_spec.get("generator_families", ())
        }
    def local_backend(family: str, revision: str) -> HuggingFaceBackend:
        model_path = args.model_root / family.rsplit("/", 1)[-1] if args.model_root else None
        if model_path is not None and not model_path.is_dir():
            raise FileNotFoundError(f"Missing pinned local model directory: {model_path}")
        specification = specifications.get(family)
        if not specification or str(specification.get("revision")) != revision:
            raise ValueError(f"Generator family/revision is absent from the pinned specification: {family}")
        verify_model_artifacts(model_path, specification.get("artifact_files", {}))
        return HuggingFaceBackend(
            family, revision, device=args.device, model_path=model_path,
        )
    factory = (lambda family, revision: FakeBackend(family, revision)) if args.fake else local_backend
    lock_verifier = None
    if args.study_root:
        from fprint.core import verify_lock
        from fprint.deferral import DeferralPaths, verify_generation_lock
        paths = DeferralPaths.from_root(args.study_root)
        if paths.generation_csv.resolve() != args.requests.resolve():
            raise ValueError("--requests must be the locked study generation_requests.csv")
        def lock_verifier():
            verify_generation_lock(paths)
            binding = verify_lock(paths.root / "locks" / "protocol_binding.json")["payload"]
            locked_spec = binding.get("files", {}).get("generation_spec", {})
            if not args.generation_spec or Path(locked_spec.get("path", "")).resolve() != args.generation_spec.resolve():
                raise RuntimeError("--generation-spec is not the protocol-bound file")
            if _file_sha256(args.generation_spec) != locked_spec.get("sha256"):
                raise RuntimeError("Generation specification hash mismatch")
    panel_counter = None
    if not args.fake:
        if not args.mage_repo:
            raise ValueError("--mage-repo is required for real generation")
        from fprint.deferral import validate_triplet_token_budget
        from tools.prepare_deferral_inputs import build_pinned_token_counters, tokenize_text_panel
        radar_counter, mage_counter = build_pinned_token_counters(args.mage_repo)
        def panel_counter(text_value: str):
            counts = tokenize_text_panel(text_value, radar_counter, mage_counter)
            validate_triplet_token_budget(counts)
            return counts
    rows = run_generation(
        args.requests, args.output, args.checkpoint, backend_factory=factory,
        lock_verifier=lock_verifier, panel_counter=panel_counter,
        continue_on_failure=args.continue_on_failure, failure_log=args.failure_log,
    )
    print(json.dumps({
        "completed": len(rows),
        "failed": len(_load_checkpoint(args.failure_log)) if args.failure_log else 0,
        "output": str(args.output) if args.output.exists() else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
