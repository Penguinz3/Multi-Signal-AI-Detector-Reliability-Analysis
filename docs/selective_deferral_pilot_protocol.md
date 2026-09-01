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

A second pre-lock execution audit found that a common 250-word generation
target would expose corpus and provenance through passage length, and that PMC
did not contain enough completely unused groups for the required disjoint
calibration and pilot samples. Before any request, threshold, or outcome lock
was written, the protocol was therefore amended to:

- use only the untouched `anchor_candidates` source partition;
- replace PMC with ASAP-AES, yielding four corpora with adequate unused groups
  and adding a direct student-writing setting;
- match every AI request to its paired human base-text word count;
- require generated output to fall within the larger of 10% or 15 words of
  that frozen target, with at most two deterministic retries; and
- use a frozen genre-specific prompt for student essays, personal blogs,
  forum responses, and encyclopedia prose.

Generator-family assignment is a seeded hash-balanced allocation within each
corpus. These changes remove trivial length and prior-record reuse shortcuts.
The three local BF16 generator families are SmolLM2, OLMo 2, and Granite 3.3.
Generation is sequential with pinned per-request seeds and eager attention; this
avoids batch-coupled sampling and fused-kernel instability on the local GPU.
Qwen3 was rejected during the pre-lock engineering check because its no-think
chat path repeatedly emitted duplicated reasoning/final-answer material and
failed the frozen word-count envelope; no Qwen3 pilot output was retained.

A pre-lock warm-resident BF16 benchmark used two discarded warmups followed by
ten discarded 200-word requests per retained family. All 30 fitted passages
passed the word-envelope and terminal-completeness checks. Median generation
times were 25.5 seconds for SmolLM2, 27.7 seconds for OLMo 2, and 41.6 seconds
for Granite 3.3 on the local RTX 3070 Laptop GPU. These texts are not pilot
records and were never detector-scored.

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

The pilot uses ASAP-AES, Blog Authorship, Stack Exchange, and WikiText-103,
drawn only from records unused by the completed study. Exactly
2,000 human groups (500 per corpus) form a threshold-only calibration pool.
Exactly 5,000 additional human groups form the pilot, with 1,250 per corpus.
One record is selected per author/user/article/source group, calibration
is selected first, and all ordering is fixed by a seeded content hash.

Three immutable generator families are pinned in a separate generation
specification. Assignment is hash-randomized and balanced within corpus;
genre-specific prompts, paired target lengths, tolerances, and retry limits are
locked before generation. Imported AI outputs must match their opaque locked
request, revision, decoding settings, retry policy, length envelope, and
provenance. An output is accepted only when it ends as a complete passage and
its original plus all three reflow variants jointly pass the pinned RADAR and
MAGE tokenizer ceiling; failure of any view rejects that generation attempt.
If a raw generation exceeds its locked word envelope, ends with an incomplete
suffix, or its nearest-length prefix exceeds either tokenizer ceiling, the
runner selects the nearest complete-sentence prefix that satisfies both the
word envelope and the pinned RADAR/MAGE token ceilings. It never crops below
the minimum, invents a sentence ending, or uses detector scores to select text;
outputs without a valid prefix are retried.

The first locked execution root was stopped after four generated passages and
before any detector score was computed. The fifth request exhausted its three
attempts because the nearest-length prefixes used 473--495 RADAR tokens even
though earlier complete prefixes remained inside the frozen word envelope and
token ceiling. That root remains preserved as an aborted pre-outcome run. This
score-blind fitting correction was committed before creating a fresh execution
root and does not alter the human selection, prompts, seeds, retry count,
threshold, probes, models, or success gates.

The second locked execution root was stopped after 66 generated passages and
before detector scoring when a 347-word Granite request had no complete prefix
that jointly satisfied its frozen 312-word minimum and the 460-token ceiling;
the closest admissible prefix was 316 words and 465 tokens. The final execution
therefore restricts both selected human sources and their paired AI targets to
at most 300 words while retaining the stricter 460-token ceiling. This removes
the structural incompatibility rather than relaxing truncation protection. The
two stopped roots remain preserved, and neither contributes pilot records.
The final pilot quotas are fixed at 1,400 ASAP-AES, 1,000 Blog Authorship,
1,200 Stack Exchange, and 1,400 WikiText groups after 500 calibration groups
per corpus. Macro-weighted evaluation prevents unequal raw corpus sizes from
dominating the result. Generation reserves at least 1.35 tokens per minimum
requested word, then applies the frozen complete-prefix and tokenizer checks;
this score-free control passed the discarded 300-word boundary case that had
failed without a minimum token budget.
A final discarded stress matrix covered all three generator families by all
four corpus-specific prompt genres at the 300-word boundary. All 12 cells
passed on attempt zero; selected lengths were 289--309 words and the maximum
four-view token count was 451. Stress texts were never retained or scored.

The v3 execution later stopped after 2,781 accepted passages when one
190-word Stack Exchange/Granite request exhausted its three locked attempts
without a complete output inside the 171--209 word envelope. No detector was
scored. The v3 root and checkpoint are preserved as an aborted pre-outcome
run. Before any detector score, the remaining locked v3 requests were run once
in a separate score-blind feasibility-screening lane. That lane accounted for
all 5,000 requests: 4,954 passed and 46 exhausted their locked attempts (20
Stack Exchange and 26 WikiText; 44 Granite, one OLMo, and one SmolLM2). It
retained successful texts, recorded every failure in an append-only log, and
did not emit a final pilot panel.

The single v4 amendment replaced all and only those 46 failed pairs. Reserves
were required to come from an unused group in the same corpus, retain a full
token-valid reflow panel, have a valid source topic, and fall within the failed
request's frozen length tolerance; absolute length difference and then a
seeded SHA-256 rank resolved selection. Generator family and corpus quotas were
preserved one-for-one. The mean absolute target-length difference was 1.67
words (maximum 16). All 4,954 compatible outputs passed independent provenance,
seed, word-envelope, completeness, and stored token-panel checks; their JSONL
checkpoint was copied byte-for-byte. The replacement mapping is locked at
SHA-256 `cbc990d9b055122ad1f261207a953857e0defa6f39b50b7b48bcfb3d9d67dc4a`.
Detector settings, probes, thresholds, models, evaluation, and success gates
remain unchanged, and no detector was scored before the amendment lock.

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
