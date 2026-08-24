# Black-Box Detector Fault Audit: Locked Results

## Result

The prospective fault audit supports **behavioral change detection and probe-level localization**. It does not support reliable fault-family attribution.

At the preregistered primary budget of 50 base passages per probe, the combined diagnostic achieved:

- macro AUROC: **0.974**;
- macro sensitivity: **94.8%** at an unchanged false-alarm rate of **0%**;
- family AUROC: **1.000** for core-computation changes, **0.979** for input-handling changes, and **0.943** for output-policy changes;
- higher AUROC than the raw-score baseline in **7 of 8** held-out corpora, with an exact one-sided corpus-level sign-flip p-value of **0.0078**.

The 1,000-replicate corpus/endpoint/group/draw hierarchical bootstrap gave a 95% interval of **0.955–0.987** for macro AUROC and **0.910–0.975** for macro sensitivity. The prospective nine-corpus paragraph-resegmentation confirmation achieved macro AUROC **0.933**, sensitivity **86.6%**, and unchanged false-alarm rate **0%**.

## What the feature channels show

The combined channel outperformed raw-score summaries overall (AUROC **0.974** versus **0.881**). The benefit was concentrated in input-handling faults: rank-invariant probe geometry reached AUROC **0.979**, compared with **0.701** for raw summaries. Conversely, rank-invariant geometry was intentionally insensitive to monotone output remapping (AUROC **0.500**), while raw and combined features reached **0.943**. This division is consistent with the frozen feature design: probe geometry detects altered input response, while score-distribution summaries detect output-policy change.

Sample efficiency was strong but monotonic: combined AUROC/sensitivity were **0.949/89.7%** at 10 passages, **0.966/93.3%** at 25, and **0.974/94.8%** at 50.

## Failed claims

Coarse diagnosis did not meet its preregistered gate. Combined diagnosis macro-F1 was **0.426**, below 0.65, and was also lower than the raw baseline's **0.477**. Only **14.6%** of declared multi-fault cases were rejected or marked inconclusive, far below the required 90%. Therefore the study must not claim that it reliably distinguishes input handling, output policy, and core computation for an individual alarm.

The permitted report is: the audited endpoint measurably departed from its reference behavior, with probe-level contributions indicating where the behavioral response changed. It does not estimate deployment false-positive rates, prove a source-code defect, or identify exact proprietary internals.

## Integrity record

- Discovery set: **1,114** existing, group-disjoint probe triplets across eight primary corpora.
- Prospective confirmation: **50** score-blind paragraph-resegmentation groups from each of nine corpora.
- Final score table: **76,215** rows, **0** failures, **0** truncated rows.
- Compact observer cache: **6,314** token-sequence records.
- Required endpoint/fault pairs: **15 of 15** complete.
- Existing forecast-study artifacts remained unmodified.
- Three numbered, hash-chained amendments preserve and explain an inference dispatch correction, a paired-feature evaluation correction, and the retry of 2,286 failed accelerator rows. Invalid preliminary evaluations remain archived under the audit root.

Canonical machine-readable results are written to `F:\Research\FPRINT-fault-audit\results\fault_audit_evaluation.json`; observation-level predictions are written to `F:\Research\FPRINT-fault-audit\results\fault_audit_predictions.csv`.
