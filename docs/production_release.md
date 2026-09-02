# FPRINT Production Release

## Product boundary

FPRINT is a vendor-neutral behavioral conformance tool for AI-text detector
endpoints. It records a locked reference, replays controlled probes, detects a
measurable departure, and reports which probe responses contributed to the
alarm.

It does not determine whether a passage was written by AI, estimate a deployed
false-positive rate, identify an internal software defect, or adjudicate an
individual accusation.

## Release interface

The research pipeline remains the source of truth for preparation, scoring, and
evaluation. The production layer adds only:

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

## Intended operational use

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
