# Black-Box AI-Detector Fault Audit

## Scope

This follow-up study tests whether a previously recorded behavioral fingerprint can detect an observable departure from an AI-text detector's audited reference behavior, localize the changed probe responses, and assign a coarse operational family. “Fault” means a behavioral conformance failure; it is not proof of a source-code defect or evidence about an undisclosed proprietary implementation.

The completed fixed-threshold FPR-forecasting experiment remains immutable and is not extended here. Its negative forecasting result is a scope boundary. Formatting-induced threshold crossings motivate the conformance probes, but they are not a second primary outcome.

## Frozen design

The archive actually contains 1,114—not 1,200—selected triplets for the three retained probes. Twenty-one of the 24 corpus/probe cells contain 50 group-disjoint anchors; ASAP-AES punctuation normalization contains 6, CNN/DailyMail sentence splitting 35, and PMC sentence splitting 23. The implementation does not manufacture extra eligibility or reuse protected groups. Budgets therefore mean “up to 10, 25, or 50 eligible groups per probe,” and every output records its effective per-probe count. This amendment was made before fault outputs were generated. The stable reference endpoints are RADAR, MAGE, and LogRank. The fault manifest freezes record and group IDs, detector revisions, transformations, severities, seeds, query budgets, folds, features, abstention policy, and success gates before fault scores are generated.

Faults are declared in [`fault_audit_config.json`](../fault_audit_config.json):

- input handling: newline flattening, full whitespace collapse, and NFKC plus whitespace normalization;
- output policy: monotone recalibration, temperature remapping, and a 5%-to-1% decision policy change;
- core computation: supervised endpoint replacement, LogRank-to-Lastde replacement, and LogRank-to-mean-log-probability replacement;
- unknown cases: predeclared input-plus-output and core-plus-output combinations.

An independent confirmation panel is formed prospectively from 50 unused paragraph-resegmentation groups in each of nine corpora. A triplet that fails the frozen capacity rule remains rejected rather than being replaced post hoc. Unused non-paragraph records remain untouched.

## Isolation and locking

The source root is opened by SQLite in read-only mode. The fault audit writes only to a separate root. Preparation creates a challenge database and an exclusive SHA-256 lock; later stages verify the lock and refuse unlocked triplets, endpoints, or fault IDs. Existing scores are copied with their provenance and a frozen-row digest. A changed lock envelope is a hard failure.

Example:

```powershell
python -m fprint.cli prepare-fault-audit `
  --source-root F:\Research\FPRINT-storage-grouped-final `
  --audit-root F:\Research\FPRINT-fault-audit `
  --config .\fault_audit_config.json `
  --evaluation F:\Research\FPRINT-storage-grouped-final\results\final\final_evaluation.json
```

## Scoring

`score-fault-audit` is resumable. Input and mean-log-probability faults require inference. Monotone output and threshold faults are derived from frozen scores. Endpoint replacements reuse exact frozen scores when available. The earlier archive retained Qwen score hashes but not the underlying arrays, so new Qwen passes are stored once in an audit-local compressed rank/log-probability cache and reused across LogRank, Lastde, and mean-log-probability. Every original/low/high triplet is rejected for an endpoint-fault pair if any member exceeds its scoring capacity; members are never independently truncated.

```powershell
python -m fprint.cli score-fault-audit `
  --audit-root F:\Research\FPRINT-fault-audit `
  --endpoint radar_roberta_large__vicuna7b_training `
  --fault input_newline_flatten
```

External black-box results can use the same analyzer through `--import-score-table`. The canonical CSV requires `triplet_id`, `intensity`, `audited_endpoint`, `fault_id`, `effective_endpoint`, `native_score`, and `canonical_ai_score`; optional runtime, token-count, truncation, and failure columns retain provenance. This is the future GPTZero/Turnitin interface. No commercial API integration is part of this phase.

## Evaluation

```powershell
python -m fprint.cli evaluate-fault-audit `
  --audit-root F:\Research\FPRINT-fault-audit
```

The evaluator produces raw unperturbed-score summaries, monotone-resistant within-run rank geometry, and their combination. For every held-out corpus it recomputes scaling, unchanged centroids, the 5% alarm threshold, family centroids, and distance/margin abstention rules using training corpora only. Family centroids also exclude the tested fault variant. Draws are group-aware at budgets 10, 25, and 50.

Each machine-readable prediction is `unchanged`, `changed`, or `inconclusive` and includes its alarm distance, training threshold, likely coarse family, per-probe contributions, raw-score change, and `revalidation_required`. Combined multi-fault cases are successful only when rejected or marked inconclusive.

The primary gates are those frozen in the config. If detection passes but diagnosis does not, the permitted claim is limited to change detection and probe-level localization. A report never estimates deployment false-positive rates or identifies exact proprietary internals.
