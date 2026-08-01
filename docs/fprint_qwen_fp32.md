# Protocol amendment: Qwen2.5-0.5B fp32 observer

Date: 2026-07-30

Phi-2 fp16 failed with non-finite logits, and Phi-2 bf16 failed the frozen
repeatability tolerance. Those artifacts remain preserved.

Before calibration or any source/target scoring, the shared observer candidate
was replaced with the Apache-2.0 Qwen2.5-0.5B base model at immutable revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, using fp32:

- `logrank__qwen2_5_0_5b_fp32`
- `lastde__qwen2_5_0_5b_fp32`

LogRank and Lastde retain their definitions and share the same token-probability
backend. The common tokenizer ceiling and all grouped partitions must be rebuilt
before the labeled 50-human/50-AI repeated pilot. No target or test score may be
produced before at least one Qwen configuration passes and the panel spans four
backend groups.
