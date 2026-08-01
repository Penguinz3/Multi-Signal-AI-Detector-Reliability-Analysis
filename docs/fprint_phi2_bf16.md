# Protocol amendment: Phi-2 bf16

Date: 2026-07-29

The technical pilot rejected both Phi-2 fp16 configurations: all 533 LogRank
and all 533 Lastde attempts produced non-finite logits. The failed score rows
remain in the study database as audit evidence.

Before calibration or source/target scoring, the Phi-2 observer precision was
amended experimentally from fp16 to bf16:

- `logrank__phi2_2_7b_bf16`
- `lastde__phi2_2_7b_bf16`

The model, tokenizer, revisions, eager-attention setting, token ceiling,
statistics, and shared backend group remained unchanged. Deterministic
PyTorch/CUBLAS execution reduced but did not eliminate repeat drift:

- LogRank maximum repeat-score difference: `0.015363047714110567`
- Lastde maximum repeat-score difference: `51.925079345703125`

Both exceed the frozen `1e-6` tolerance, so neither bf16 configuration is
admitted. Int8 also remains unadmitted because there is no valid unquantized
reference for its equivalence gate. The study remains blocked at three valid
configurations across three backend groups.

The pilot gate was also corrected to require exactly 50 verified-human and
50 verified-AI passages, repeated inference within tolerance, score variation
in each class, no failures, no truncation, and the pinned higher-is-more-AI
orientation. No target or test passage may be used by this gate.
