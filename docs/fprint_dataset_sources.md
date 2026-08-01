# FPRINT dataset sources

Raw text remains local on `F:`. Exact input and output SHA-256 values are stored
beside each normalized CSV in its provenance JSON.

| Corpus | Frozen source | Admission notes |
|---|---|---|
| RAID human reference | `https://dataset.raid-bench.xyz/train_none.csv` | Official no-attack release; retain `model=human`, `attack=none`; underlying source-text terms vary, so do not redistribute. |
| PMC | NCBI PMC ESearch/EFetch APIs | English, open-access CC BY/CC0, pre-2020, non-preprint/non-retracted articles; group by PMCID. |
| ASAP student essays | `scrosseye/ASAP_2.0`, official training ZIP | CC BY 4.0; `essay_id` grouping and `prompt_name` stratification. This intentionally replaces the original 2012 Kaggle ASAP-AES release, whose competition terms and authentication are unsuitable for a reproducible public pipeline. |
| Project Gutenberg | Official `pg_catalog.csv.gz` plus `gutenberg.pglaf.org` mirror | Deterministic one-book-per-author subset; English text, non-anonymous authors; Project Gutenberg boilerplate removed. Public-domain status is United States-specific. |
| Blog Authorship | `tasksource/blog_authorship_corpus` Parquet conversion `689fcafb8e93edb6f2340edf7197a8ae3cf6aa3d` | Preserves original blogger ID. Original permission is non-commercial research use; do not redistribute normalized text. |
| Stack Exchange | Internet Archive item `stackexchange_20221005`, `english.stackexchange.com.7z` | Pre-ChatGPT snapshot of the official data dump; questions/answers grouped by owner user. Preserve post IDs and row-specific CC BY-SA attribution if redistributed. |
| CNN/DailyMail | `abisee/cnn_dailymail` Parquet conversion `690bb95a2ac2c5a99d7bde63ac1401539ddd3967`, config `3.0.0`, test | Apache 2.0 dataset release; one deterministic passage per article. |
| GovReport | `launch/gov_report` Parquet conversion `c0b3f7bd48f480f34a572beff5f110fc6c0f11c4`, `plain_text` train shard 0 | CC BY 4.0; one deterministic passage per report. |
| WikiText-103 | `Salesforce/wikitext` Parquet conversion `3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e`, `wikitext-103-raw-v1` train | CC BY-SA; article headers reconstruct source-article groups. |
| BAWE external validation | Oxford Text Archive handle `20.500.12024/2539`, UTF-8 TEI XML | External target only; group by student ID and stratify by disciplinary group. Corpus files and normalized text must not be redistributed. Use is research-only, resulting projects/publications must be reported to the BAWE team, and the prescribed acknowledgement must appear in the paper. |

All evaluation passages are 100–350 whitespace-delimited words. Final admitted
files additionally require at most 460 tokens under every pinned active
detector tokenizer, including MAGE preprocessing. `fprint prepare` then performs
global exact and near-duplicate removal across RAID and all evaluation corpora,
including the BAWE external target.
