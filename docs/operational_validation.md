# Operational Validation

## Development replay

The production alarm rule was replayed end to end on the previously locked
controlled-fault scores. This is development evidence, not an independent
prospective result, because the scores and fault outcomes already existed.

At 50 passages per probe, the replay produced:

- AUROC 0.964;
- sensitivity 88.5%;
- unchanged false-alarm rate 0%;
- RADAR sensitivity 98.4%;
- LogRank sensitivity 100%;
- MAGE sensitivity 67.2%.

The replay did not pass the stricter product gate because MAGE was below the
predeclared 70% per-endpoint sensitivity floor. Most misses were upstream
whitespace changes that affected too few MAGE passages to satisfy the frozen
20% materiality rule. The gate is not lowered after seeing this result.

The machine-readable local result is generated with
`replay-operational-validation`. It explicitly labels itself
`development_replication_not_prospective_validation`.

## Prospective confirmation

A new score-blind panel has been locked under
`F:\Research\FPRINT-operational-validation`. It uses groups that were absent
from the earlier probe and fault panels, preserves author/user/article/book/
report grouping, and binds the globally RAID-deduplicated grouped source
database by SHA-256.

The locked panel contains 968 triplets. Four corpora support the complete
50-per-probe primary budget: CNN/DailyMail, GovReport, Stack Exchange, and
WikiText-103. Sparse cells are retained with their actual counts; eligibility
is never manufactured by reusing groups or lowering the frozen two-site probe
minimum. Gutenberg has no independent full panel remaining and is unavailable
for the three-probe prospective analysis.

Every selected original/low/high triplet passed the pinned RADAR, MAGE, and
LogRank tokenizer limits as a unit. If any member had exceeded capacity, the
whole triplet would have been excluded before fault scoring. The condition
truth table is separately locked and must remain hidden from the analyzer
until all blind reports are hash-locked.

The prospective test keeps the 50-passage gate at:

- at least 80% overall sensitivity;
- at most 5% unchanged false alarms;
- at least 70% sensitivity for every endpoint;
- at least four full-budget corpora.

Passing the retrospective replay alone cannot remove the engineering-beta
label. Only the prospective panel can supply that evidence.

## Locked execution

Prospective execution has three separate, fail-closed stages:

1. `lock-operational-validation-scoring` binds the committed scoring and
   evaluation code, the panel and private condition locks, the source score
   database, and each endpoint's frozen human-reference empirical CDF before
   new inference begins.
2. `score-operational-validation` produces one resumable, hash-locked run for
   an endpoint, opaque condition code, and run role. The unchanged condition
   receives two independent reference reruns plus a current rerun; every other
   condition receives only a current run and reuses those endpoint references.
3. `evaluate-operational-validation` writes and hash-locks every blinded alarm
   before opening the private condition truth, then publishes the prospective
   metrics and success-gate decision.

The audited baseline remains FP32. BF16 is intentionally one hidden precision
change rather than a silent redefinition of the reference. Any missing score,
failure, truncation, duplicate run, altered panel row, or mismatched lock stops
the pipeline and prevents a completion artifact.

The first scoring preflight exposed that the original panel lock froze row IDs
and counts but omitted the `panel.csv` byte hash. No detector was loaded. A
separate `lock-operational-validation-integrity-amendment` stage preserves the
original locks and binds their exact panel bytes plus the corrected code before
scoring; it refuses to run after any score row or completed run exists. The
pre-score database integration check then exposed and corrected a column-count
defect before the first score was written; the final amendment is chained to
the earlier amendment rather than replacing it.

After unchanged, input, precision, and threshold-control RADAR runs were
locked, the first offline-calibration preflight found that the text router did
not classify `logit_bias` as an identity transform. It failed before writing a
calibration row. A score-preserving execution patch records the exact pre-patch
rows and completed-run hashes, adds that routing case plus stricter code/run
provenance checks, and is locked before collection resumes or truth is opened.
