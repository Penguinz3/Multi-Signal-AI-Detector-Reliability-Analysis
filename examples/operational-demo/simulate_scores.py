import csv
import json
import sys
from pathlib import Path


challenge_path, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=False)
with challenge_path.open(encoding="utf-8-sig", newline="") as handle:
    challenge = list(csv.DictReader(handle))

for run_id, offset, changed in (
    ("reference-a", 0.0, False),
    ("reference-b", .001, False),
    ("current-changed", .0005, True),
):
    with (output_dir / f"{run_id}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("challenge_id", "canonical_ai_score", "truncated", "failure"))
        for index, row in enumerate(challenge):
            score = .20 + (index % 7) * .01 + offset
            if changed and row["probe"] == "paragraph_resegmentation" and row["intensity"] != "original":
                score += .20
            writer.writerow((row["challenge_id"], score, "false", ""))
    (output_dir / f"{run_id}-metadata.json").write_text(json.dumps({
        "version": "demo-v2" if changed else "demo-v1",
        "configuration": "synthetic-default",
        "threshold_policy": "unchanged",
        "collected_at_utc": "2026-09-02T12:00:00Z",
    }, indent=2) + "\n", encoding="utf-8")
