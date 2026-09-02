# FPRINT

**Behavioral conformance testing for black-box AI-text detectors**

FPRINT records how an approved detector responds to a locked set of controlled
text probes, then checks a later run for meaningful behavioral changes. It needs
only detector scores—not model weights, source code, or vendor integration.

FPRINT answers:

- does the current endpoint still match its approved reference behavior?
- which punctuation, sentence-splitting, paragraph, or raw-score responses changed?
- should the endpoint be revalidated before continued high-stakes use?

It does **not** determine authorship, adjudicate an individual accusation,
estimate deployment accuracy, or identify the detector's exact internal change.

## Product status

The repository contains two deliberately separate layers:

- **Operational beta:** a vendor-neutral, local reference/current workflow. Its
  repeat-noise decision rule is transparent and fail-closed. A retrospective
  replay reached 0.964 AUROC and 88.5% sensitivity with no unchanged false
  alarms, but missed its per-endpoint gate; prospective validation is underway.
- **Frozen research artifact:** controlled-fault experiments achieved 0.974
  macro AUROC and 94.8% sensitivity with no unchanged false alarms. Prospective
  confirmation achieved 0.933 AUROC and 86.6% sensitivity. Fault-family
  diagnosis failed, so only change detection and probe-level localization are
  supported.

## Operational workflow

Start with a diverse, approved CSV containing `record_id,text`. FPRINT creates
three variants per eligible record and locks every challenge ID and text hash.

```powershell
python -m fprint init-audit `
  --records <approved-records.csv> `
  --audit-root <new-audit-directory> `
  --endpoint <detector-and-configuration-id>

python -m fprint export-challenge `
  --audit-root <audit-directory> `
  --output-dir <new-export-directory>
```

Query the detector twice while it is in its approved reference state and once
for the current state. Enter scores in the exported template. Canonical scores
must be finite values from 0 to 1, with larger values meaning more AI-like.

```powershell
python -m fprint import-run --audit-root <audit-directory> `
  --run-id reference-a --role reference --scores <reference-a.csv> `
  --metadata <reference-a-metadata.json>

python -m fprint import-run --audit-root <audit-directory> `
  --run-id reference-b --role reference --scores <reference-b.csv> `
  --metadata <reference-b-metadata.json>

python -m fprint import-run --audit-root <audit-directory> `
  --run-id current --role current --scores <current.csv> `
  --metadata <current-metadata.json>

python -m fprint compare-runs --audit-root <audit-directory> `
  --reference reference-a reference-b --current current `
  --output-dir <new-report-directory>
```

Metadata must record `version`, `configuration`, `threshold_policy`, and
`collected_at_utc`. The score table must explicitly report `truncated` and
`failure`; failed or truncated queries reject the complete run. The
report contains aggregate deltas and hashes but no source passages. A noisy
reference produces `inconclusive`; a detected departure produces `changed` and
`revalidation_required: true`.

Run the complete [synthetic operational demo](examples/operational-demo/README.md)
or view the [research-result report demo](examples/fault-audit-demo/README.md).

## Research artifact

The original controlled-fault workflow remains available through the
`prepare-fault-audit`, `score-fault-audit`, `evaluate-fault-audit`, and
`package-fault-audit` commands. Its protocol and limitations are documented in
[the frozen protocol](docs/fault_audit_protocol.md) and
[results](docs/fault_audit_results.md). The immutable Git tag is
`fprint-research-v1`.

## Install and test

The operational black-box workflow uses the Python standard library. Local
open-model scoring additionally uses the pinned packages in `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests\fprint -p "test_*.py" -v
```

See [production boundaries and release gates](docs/production_release.md) and
the [operational validation record](docs/operational_validation.md).
