# Selective Deferral from Formatting-Response Fingerprints

## Frozen pilot question

Among passages already accused by a fixed AI-text detector threshold, can three
deterministic formatting-response deltas distinguish human false accusations
from correctly accused AI passages better than a query-matched classifier that
does not receive the individual deltas?

This is an accusation-triage experiment. It does not improve the underlying
detector, estimate deployment prevalence, prove why a detector behaves as it
does, or validate commercial products that were not tested.

## Pre-outcome feasibility amendment

The preserved human archive had already collapsed most natural whitespace, so
the earlier canonical-whitespace and paragraph-collapse probes were ineligible.
The pinned MAGE deployment preprocessor also removes line breaks and collapses
whitespace. Before generating or scoring pilot outcomes, the protocol was
therefore corrected as follows:

- RADAR is the sole formatting-fingerprint endpoint and the only endpoint used
  for the pilot go/no-go decision.
- MAGE is a preprocessing-invariance negative control. Its effective input must
  be identical for every view; it is not evidence of detector replication.
- LogRank contributes one original score to the disagreement baseline and is
  never queried on the formatting variants.
- Detector-general final evaluation is blocked until a successful RADAR pilot
  and a separately selected whitespace-sensitive learned endpoint are covered
  by a new prospective lock.

The correction prevents a guaranteed MAGE null effect from being presented as
a failed replication and preserves the actual construct: detector response to
reflow alone.

## Inputs and transformations

Human and AI originals are normalized symmetrically with
`" ".join(text.split())`. Every probe begins from that base and must preserve
the exact ordered sequence of non-whitespace characters:

1. `wrap_80`: word-boundary wrapping at 80 characters, with no long-word or
   hyphen breaking.
2. `sentence_blocks_2`: split after `.`, `!`, or `?` followed by a space;
   place consecutive sentence units into two-sentence paragraphs.
3. `sentence_per_paragraph`: use the same split and place every sentence unit
   in its own paragraph.

A panel is eligible only when all three variants differ from the original and
from one another, all four non-whitespace hashes match, and all four views fit
the common 460-token ceiling. The whole panel is rejected if any view fails.

## Data separation

The pilot uses Blog Authorship, PMC, Stack Exchange, and WikiText-103. Exactly
2,000 human groups (500 per corpus) form a threshold-only calibration pool.
Exactly 5,000 additional human groups form the pilot, with at least 1,000 per
corpus. One record is selected per author/user/article/source group, calibration
is selected first, and all ordering is fixed by a seeded content hash.

Three immutable generator families must be supplied in a separate generation
specification. Assignment is balanced within corpus, prompts are locked before
generation, and provider-specific generation remains outside this repository.
Imported AI outputs must match their opaque locked request, revision, decoding
settings, retry policy, and provenance.

## Conditional scoring

The frozen RADAR threshold is the empirical 95th percentile of the independent
2,000-human calibration scores. A passage is positive only when its canonical
AI score is strictly greater than the threshold. All 10,000 pilot originals are
scored with RADAR. Only RADAR-positive passages receive the three transformed
RADAR queries. MAGE and LogRank receive only the original RADAR-positive texts.

For RADAR, query use is exactly

`N_original + 3 * N_original_positive`.

Original-repeat sentinels measure scoring noise. GPU scoring stages run
sequentially.

## Models and evaluation

The fingerprint model is a fixed balanced L2 logistic regression (`C=1`) with
training-fold median imputation and standardization. Its inputs, in order, are
the original RADAR margin and the three signed transformed-minus-original
deltas.

The fixed comparator receives original detector scores, absolute standardized
detector disagreements, surface features, character 3--5-gram TF-IDF, and the
mean RADAR score over the same four queries. It never receives the individual
signed deltas. All preprocessing and fitting occur inside leave-one-corpus-out
training folds.

The primary estimand is extra human-false-accusation removal at an oracle,
interpolated operating point that retains 90% of correctly accused AI passages.
It is a ranking comparison, not a deployable cutoff. The paired bootstrap
resamples complete source pairs within corpus and generator strata and
recomputes the operating point in every replicate.

## Pilot gate and stopping rule

The pilot advances only if every preregistered validity, event, noise, effect,
bootstrap, and cross-corpus gate in `deferral_config.json` passes. In
particular, the fingerprint model must improve false-accusation removal by at
least 7.5 percentage points over the fixed comparator, with a one-sided 80%
bootstrap lower bound above zero and positive improvement in at least three of
four corpora.

Failure stops the study without probe substitution, threshold relaxation,
post-hoc model expansion, or additional sampling. A passed pilot permits only
a RADAR-specific final-protocol lock. This implementation intentionally has no
final scoring command.
