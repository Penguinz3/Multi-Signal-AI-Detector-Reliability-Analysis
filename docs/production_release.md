# FPRINT Production Release

## Product boundary

FPRINT is a vendor-neutral behavioral conformance tool for AI-text detector
endpoints. It records a locked reference, replays controlled probes, detects a
measurable departure, and reports which probe responses contributed to the
alarm.

It does not determine whether a passage was written by AI, estimate a deployed
false-positive rate, identify an internal software defect, or adjudicate an
individual accusation.

## Operational interface

An operator creates a new audit from approved records, exports its locked
challenge, records two reference repeats, records a current run, and compares
them:

1. `init-audit`
2. `export-challenge`
3. `import-run --role reference` twice
4. `import-run --role current`
5. `compare-runs`

Challenge IDs, text hashes, detector identity, decision settings, score-table
hashes, and run roles are locked. Imports must contain every challenge ID
exactly once and canonical scores must be finite values in `[0,1]`. Existing
audit, run, export, and report destinations are never overwritten.

Each run must record its version, configuration, threshold policy, and UTC
collection time. Any reported failure or truncation rejects the complete run;
variants are never silently compared on different visible text.

The two reference runs estimate repeat noise. The comparison tests changes in
unmodified scores, low/high probe shifts, and response slopes across punctuation
normalization, sentence splitting, and paragraph resegmentation. It uses a
Bonferroni-adjusted paired sign rule plus frozen practical-effect requirements.
A reference whose 95th-percentile repeat disagreement exceeds the frozen limit
returns `inconclusive`.

This operational rule is an **engineering beta**. The controlled-fault research
supports the underlying feature design, but the rule has not yet been validated
on real external vendor updates. Institutions must conduct that validation
before treating its alarm as production assurance.

## Research release interface

The frozen research pipeline remains the source of truth for its controlled-fault
preparation, scoring, and evaluation. Its packaging layer adds:

- versioned external-score and evaluation contracts;
- a blank canonical score-table template;
- a fail-closed audit readiness command;
- a deterministic, standalone HTML report containing no raw passages.

The operator packages a completed audit with one command:

```powershell
python -m fprint package-fault-audit `
  --audit-root <locked-audit-root> `
  --output-dir <new-release-directory>
```

Packaging is fail-closed: the audit must be complete, the evaluation must match
the public schema, and the destination must not exist. Files are assembled in a
temporary sibling directory and exposed only after the release manifest is
written.

This keeps vendor credentials and endpoint-specific clients outside the core.
An institution or auditor may collect scores through an authorized mechanism
and import them through the frozen table contract.

## Intended use

1. Approve a detector configuration and create a reference audit.
2. Record its endpoint identity, settings, threshold policy, and lock digest.
3. Re-run the same locked challenge after a vendor, model, or configuration
   change, or on a scheduled interval.
4. Review the change alarm and per-probe contributions.
5. Require revalidation before renewed high-stakes reliance.

Education is one important application: an institution could use FPRINT during
procurement and periodic revalidation. The current study does not evaluate a
school deployment, GPTZero, or Turnitin, so these remain implications rather
than empirical claims.

## Public-release gates

- All existing and production tests pass in the pinned research environment.
- The example report reproduces byte-for-byte from the same evaluation JSON.
- Invalid report schemas, non-finite metrics, malformed lock digests, and
  incomplete audits fail closed.
- Reports contain aggregate metrics and integrity identifiers, never raw text.
- Documentation states the individual-adjudication prohibition prominently.
- The public repository excludes caches, model weights, credentials, and local
  research data.

Dashboard, hosted execution, user accounts, billing, and direct commercial API
clients are deferred until a real operator requires them.
