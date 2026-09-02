# FPRINT

**Multi-Probe Behavioral Conformance Testing for Black-Box AI-Text Detectors**

FPRINT checks whether an AI-text detector still behaves like a previously
audited reference. It replays controlled text probes, detects behavioral
departures, and shows which probe responses contributed to the alarm without
requiring access to the detector's internals.

FPRINT is a system-assurance tool. It is **not** an authorship detector and must
not be used to decide whether an individual person used AI.

## Validated result

The completed forecast-locked study reached 0.974 macro AUROC and 0.948
sensitivity with no unchanged false alarms in held-out-corpus evaluation. A
prospective new-group confirmation reached 0.933 macro AUROC and 0.866
sensitivity, again with no unchanged false alarms. Coarse fault-family diagnosis
did not pass its gate, so the supported claim is behavioral change detection and
probe-level localization—not identification of a detector's exact internal cause.

## Local release workflow

```powershell
python -m fprint fault-audit-status --audit-root <locked-audit-root>

python -m fprint package-fault-audit `
  --audit-root <locked-audit-root> `
  --output-dir <new-release-directory>
```

The packaging command refuses incomplete audits and existing output directories.
It publishes a standalone HTML report, aggregate public evaluation summary,
versioned import contracts, and a manifest containing every artifact hash. Raw
passages, credentials, caches, model weights, and local paths are excluded.

See the [demonstration bundle](examples/fault-audit-demo/README.md), the
[study protocol](docs/fault_audit_protocol.md), and the
[production boundary](docs/production_release.md).

## Reproduce the checks

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests\fprint -p "test_*.py" -v
```

The earlier fixed-threshold FPR-forecasting experiment is retained as a negative
result and scope boundary. Historical multi-signal work remains in `archive/`.
