from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from tools.normalize_raid_academic import PMC_EFETCH_URL, PMC_ESEARCH_URL, PMC_QUERY


def fetch(url: str, parameters: dict[str, str]) -> bytes:
    request = urllib.request.Request(
        url + "?" + urllib.parse.urlencode(parameters),
        headers={"User-Agent": "fprint_dataset_prep/2026.07"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--articles", type=int, default=1_200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = {"db": "pmc", "tool": "fprint_dataset_prep", "email": args.email}
    search = json.loads(fetch(PMC_ESEARCH_URL, {
        **common, "term": PMC_QUERY, "retmode": "json",
        "retmax": str(args.articles), "sort": "pub date",
    }))
    identifiers = search["esearchresult"]["idlist"]
    if len(identifiers) < args.articles:
        raise RuntimeError(f"PMC search returned only {len(identifiers)} IDs")
    (args.output_dir / "ids.json").write_text(
        json.dumps({"query": PMC_QUERY, "ids": identifiers}, indent=2), encoding="utf-8"
    )
    for start in range(0, len(identifiers), 200):
        batch = identifiers[start:start + 200]
        destination = args.output_dir / f"batch_{start // 200:03d}.xml"
        if destination.is_file() and destination.stat().st_size:
            continue
        destination.write_bytes(fetch(PMC_EFETCH_URL, {
            **common, "id": ",".join(batch), "retmode": "xml",
        }))
        print(f"{min(start + 200, len(identifiers))}/{len(identifiers)}", flush=True)
        time.sleep(.4)


if __name__ == "__main__":
    main()
