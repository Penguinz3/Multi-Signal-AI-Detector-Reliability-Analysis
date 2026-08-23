# Fingerprint geometry pivot: analysis record

Date: 2026-08-22

## Status and disclosure

This diagnostic analysis was specified after the preregistered forecast gate was
evaluated. It is therefore a transparent secondary analysis, not a replacement
confirmatory hypothesis. The forecast-locked result remains primary: the full
fingerprint model did not improve unseen-corpus FPR forecasting over detector
identity × target-text features.

## Revised research question

> Do operational perturbation fingerprints identify a detector's local response
> geometry, and is that geometry sufficient to transport fixed-threshold human
> false-positive tail risk across corpora?

The six probes are human-source-provenance-preserving deterministic edits: no
passage is replaced with LLM-generated text. They are not assumed to be perfectly
meaning-preserving interventions. A probe is called panel-complete only when the
frozen artifacts contain a panel-valid corpus slope for all five configurations
in all eight primary corpora; this rule uses availability, not stability,
identification, or FPR outcomes. The three probes meeting it are punctuation
normalization, sentence splitting, and paragraph resegmentation. The other probes
remain in probe-specific analyses with their eligibility counts reported.

## Results

### Local response geometry is detector-discriminative

Using only the three panel-complete slopes, leave-one-corpus-out nearest-centroid
identification recovered the detector configuration in 30/40 cells (75.0%; chance
20%; corpus-cluster bootstrap 90% interval 67.5–82.5%). A detector-label
permutation test with 10,000 structured permutations gave Monte-Carlo p = 0.00010
under the +1 convention. This is evidence about the five tested configurations,
not arbitrary detector-family classification.

As a normalization sensitivity, raw canonical-score slopes produced 62.5% cosine
identification. Raw-score Euclidean identification was 77.5%, but raw detector
scales are not intrinsically comparable, so that number is descriptive only. The
cosine result indicates that the separation is not solely an artifact of RAID-CDF
normalization.

After collapsing LogRank and Lastde into their shared Qwen dependency group,
four-way identification was 87.5% by both cosine and Euclidean distance (chance
25%). Discriminability was nevertheless probe-dependent: leave-one-probe-out
cosine accuracy was 72.5% without punctuation normalization, 52.5% without
sentence splitting, and 40.0% without paragraph resegmentation.

### Reliability is configuration-dependent

Full-source mean split-half profile cosine was:

| Configuration | Mean cosine |
|---|---:|
| LogRank–Qwen | 0.993 |
| RADAR | 0.993 |
| MAGE | 0.959 |
| Lastde–Qwen | 0.412 |
| OpenAI RoBERTa legacy | 0.199 |

Fingerprints therefore cannot be described as uniformly stable. The sharp
difference between the two methods sharing Qwen is consistent with the detection
statistic and implementation contributing beyond the shared observer backbone;
it is not a causal component ablation.

### Operational edits can change fixed-threshold decisions

At the frozen nominal 5% operating point, RADAR exhibited an unadjusted descriptive
monotone low-to-high pattern:

| Probe | Low: human→AI | High: human→AI | High 90% hierarchical-bootstrap interval |
|---|---:|---:|---:|
| Paragraph resegmentation | 11.8% | 18.8% | 11.5–26.5% |
| Sentence splitting | 5.9% | 13.4% | 6.0–22.8% |
| Contraction expansion | 4.2% | 5.6% | 0.9–11.1% |

These are paired transformations of the same human-source passages. They measure
decision instability under non-generative operational editing, not ordinary FPR
on independently sampled documents and not evidence about every detector.
No individual directional detector × probe × intensity cell survived Holm
adjustment across the complete 60-test family at the 5% operating point. The
effect sizes and hierarchical-bootstrap intervals must therefore be presented as
descriptive secondary findings, with the complete table rather than selected
cells in the supplement.

### Local geometry is not a transportable risk representation

Pooling 140 dependent within-detector corpus pairs produced an apparently moderate
association between fingerprint distance and FPR distance (Spearman rho 0.395 at
the 5% operating point). That aggregate is misleading. After ranking and centering
within detector configuration, rho was 0.021 (corpus-label permutation p = 0.813).
At the 1% operating point, the conditioned association was rho = -0.094 (p =
0.361).

The prospective forecast test agrees with this dissociation. At the 5% operating
point, the main model's backend-macro MAE was 0.04246 at signature size 100 and
0.04217 at size 250, versus 0.04070 and 0.04046 for detector identity × target
features. It won in only 3/8 corpora at both sizes; the exact sign-flip p-value was
0.957.

The locked 1% sensitivity analysis reached the same conclusion: main-model MAE
was 0.01137 and 0.01112 at signature sizes 100 and 250, versus 0.01067 and 0.01039
for detector identity × target features. The main model won in 2/8 and 3/8
corpora, respectively; the exact sign-flip p-value was 0.977.

## Strongest supported thesis

> Operational perturbation fingerprints can identify how a detector reacts to
> controlled edits, but they are not risk certificates: detector-discriminative
> local response geometry is neither uniformly reliable nor sufficient to
> transport fixed-threshold false-positive risk into an unseen corpus.

This is a local-identifiability versus global-risk-identifiability result. The
metamorphic decision audit is the central practical contribution of the secondary
pivot analysis; the preregistered locked forecast remains primary in the study
record and prospectively falsifies the stronger sufficiency claim.

## Remaining checks before manuscript lock

1. Manually validate a blinded sample of transformed passages for readability and
   semantic acceptability before calling any probe meaning-preserving.
2. Produce the complete detector × probe × intensity tables and manuscript figures.
3. Present the post-outcome timing of this secondary analysis prominently.

Reproducible artifacts are written to `outputs/fingerprint_geometry/` by:

```powershell
python -m fprint analyze-fingerprint-geometry `
  --storage-root F:\Research\FPRINT-storage-grouped-final `
  --evaluation F:\Research\FPRINT-storage-grouped-final\results\final\final_evaluation.json `
  --output-dir outputs\fingerprint_geometry
```
