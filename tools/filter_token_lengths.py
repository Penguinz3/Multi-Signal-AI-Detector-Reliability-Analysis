from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fprint.detectors import SPECS, _mage_preprocessor


ACTIVE_TOKENIZERS = (
    "openai_roberta_base__gpt2_legacy",
    "radar_roberta_large__vicuna7b_training",
    "mage_longformer__paper",
    "logrank__qwen2_5_0_5b_fp32",
)


def tokenizer_lengths(tokenizer, texts: list[str]) -> list[int]:
    encoded = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
    return [len(ids) for ids in encoded["input_ids"]]


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mage-repo", required=True)
    parser.add_argument("--ceiling", type=int, default=460)
    parser.add_argument("--minimum-reference", type=int, default=10_000)
    parser.add_argument("--minimum-evaluation", type=int, default=6_000)
    args = parser.parse_args()
    inputs = sorted(args.input_dir.glob("*.csv"))
    if not inputs:
        raise FileNotFoundError(f"No normalized CSV files in {args.input_dir}")
    rows_by_file: dict[Path, list[dict[str, str]]] = {}
    for path in inputs:
        with path.open(encoding="utf-8", newline="") as handle:
            rows_by_file[path] = list(csv.DictReader(handle))
    excluded: dict[str, dict[str, int]] = {}
    preprocess_mage = _mage_preprocessor(args.mage_repo)
    for detector in ACTIVE_TOKENIZERS:
        spec = SPECS[detector]
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id, revision=spec.tokenizer_revision, trust_remote_code=False
        )
        for path, rows in rows_by_file.items():
            for start in range(0, len(rows), 128):
                batch = rows[start:start + 128]
                texts = [preprocess_mage(row["text"]) if spec.dependency_group == "mage"
                         else row["text"] for row in batch]
                for row, length in zip(batch, tokenizer_lengths(tokenizer, texts)):
                    if length > args.ceiling:
                        excluded.setdefault(row["record_id"], {})[detector] = length
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    retained_by_file = {}
    failures = []
    for source, rows in rows_by_file.items():
        retained = [row for row in rows if row["record_id"] not in excluded]
        if source.stem == "raid_human" and len(retained) < args.minimum_reference:
            failures.append(f"Only {len(retained)} RAID rows survive the common token ceiling")
        if source.stem != "raid_human" and len(retained) < args.minimum_evaluation:
            failures.append(
                f"Only {len(retained)} {source.stem} rows survive the common token ceiling"
            )
        retained_by_file[source] = retained
        counts[source.stem] = {"input": len(rows), "retained": len(retained)}
    if failures:
        raise RuntimeError("; ".join(failures))
    for source, retained in retained_by_file.items():
        destination = args.output_dir / source.name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(retained[0]))
            writer.writeheader()
            writer.writerows(retained)
        temporary.replace(destination)
    audit = {
        "ceiling": args.ceiling,
        "tokenizers": {
            detector: {
                "model": SPECS[detector].model_id,
                "revision": SPECS[detector].tokenizer_revision,
            }
            for detector in ACTIVE_TOKENIZERS
        },
        "counts": counts,
        "excluded": excluded,
    }
    payload = json.dumps(audit, indent=2, sort_keys=True)
    (args.output_dir / "token_length_audit.json").write_text(payload, encoding="utf-8")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
