# FPRINT: Fixed-Threshold Detector False-Positive Forecasting

This repository now contains the implementation of the FPRINT study. The legacy
multi-signal analysis remains under `src/` and `archive/`; it is not used by the
new pipeline.

FPRINT forecasts a detector configuration's human false-positive rate on an
unscored target corpus from fixed-threshold source behavior, operational probe
responses, and an unscored target text signature. The design is leave-one-corpus
out; it does not automatically claim leave-one-domain-out transfer.

## Storage and environment

Keep large state, model caches, and the virtual environment on `F:`:

```powershell
$env:FPRINT_STORAGE_ROOT = 'F:\Research\FPRINT-storage'
$env:HF_HOME = 'F:\Research\FPRINT-storage\hf'
$env:HF_HUB_CACHE = 'F:\Research\FPRINT-storage\hf\hub'
$env:TRANSFORMERS_CACHE = 'F:\Research\FPRINT-storage\hf\transformers'
py -3.12 -m venv F:\Research\FPRINT\.venv
F:\Research\FPRINT\.venv\Scripts\python.exe -m pip install -r F:\Research\FPRINT\requirements.txt
F:\Research\FPRINT\.venv\Scripts\python.exe -m pip install --force-reinstall -r F:\Research\FPRINT\requirements-gpu.txt
```

The second command replaces PyPI's CPU-only Windows wheel with the tested
CUDA 13.0 build.

The code refuses to infer with mutable model revisions. MAGE additionally
requires its official repository checked out at commit
`6d11f851184b9f04166f952ddc1f47727f36710f` and supplied with `--mage-repo`.

## Stages

```powershell
F:\Research\FPRINT\.venv\Scripts\python.exe -m fprint prepare `
  --threshold-reference F:\Research\FPRINT-storage\data\raid_human.csv `
  --corpus pmc=F:\Research\FPRINT-storage\data\pmc.csv `
  --corpus asap_aes=F:\Research\FPRINT-storage\data\asap_aes.csv `
  --corpus gutenberg=F:\Research\FPRINT-storage\data\gutenberg.csv `
  --corpus blog_authorship=F:\Research\FPRINT-storage\data\blog_authorship.csv `
  --corpus stack_exchange=F:\Research\FPRINT-storage\data\stack_exchange.csv `
  --corpus cnn_dailymail=F:\Research\FPRINT-storage\data\cnn_dailymail.csv `
  --corpus govreport=F:\Research\FPRINT-storage\data\govreport.csv `
  --corpus wikitext_103=F:\Research\FPRINT-storage\data\wikitext_103.csv `
  --corpus bawe=F:\Research\FPRINT-storage\data\bawe.csv

F:\Research\FPRINT\.venv\Scripts\python.exe -m fprint pilot `
  --detector openai_roberta_base__gpt2_legacy `
  --ai-reference F:\Research\FPRINT-storage\data\raid_ai_pilot.csv

F:\Research\FPRINT\.venv\Scripts\python.exe -m fprint calibrate `
  --detector openai_roberta_base__gpt2_legacy

F:\Research\FPRINT\.venv\Scripts\python.exe -m fprint score-source `
  --target-corpus pmc --detector openai_roberta_base__gpt2_legacy

F:\Research\FPRINT\.venv\Scripts\python.exe -m fprint forecast `
  --phase zero --target-corpus pmc --manifest pmc-manifest.json `
  --forecasts pmc-zero-forecasts.json

F:\Research\FPRINT\.venv\Scripts\python.exe -m fprint score-target `
  --target-corpus pmc --partition privileged_signature `
  --record-ids pmc-privileged-250.txt `
  --detector openai_roberta_base__gpt2_legacy `
  --admitted-detectors openai_roberta_base__gpt2_legacy `
    radar_roberta_large__vicuna7b_training mage_longformer__paper `
    logrank__qwen2_5_0_5b_fp32 lastde__qwen2_5_0_5b_fp32

F:\Research\FPRINT\.venv\Scripts\python.exe -m fprint evaluate `
  --rows locked-forecast-evaluation.csv --output evaluation.json
```

`prepare` performs global exact and five-word-shingle near-deduplication before
partitioning, including RAID-to-evaluation collisions. Groups are kept intact at
author, user, student, article, book, report, or source-article level where
available. ASAP-AES allocation is prompt-stratified.

Source detector inference is cached once, then copied into isolated outer-fold
databases with the held-out corpus excluded. Probe triplets use deterministic
25% and 100% site transformations. A primary
triplet is rejected if its original, low, or high member exceeds any active
adapter's token capacity. Forecast locking validates the complete Cartesian set
of corpora, detector configurations, signature sizes, 20 draws, and preregistered
models before target scoring can start. BAWE is an external target only and
never contributes source quantities or model fitting. All nine zero-score locks
must exist before any privileged target score, and all nine privileged locks
must exist before any held-out test score. The preregistered success gate remains
the original eight-corpus Computers analysis; BAWE is reported separately as
external educational validation.

Historical Phi-2 fp16 and bf16 failures are preserved in
`docs/protocol_amendment_phi2_bf16.md`. The replacement observer is the pinned
Qwen2.5-0.5B base model in fp32; calibration remains blocked until its labeled
repeatability pilot passes.

Run dependency-free checks with:

```powershell
python -m unittest discover -s tests -v
```

The frozen numerical design is in `fprint_config.json`. Core safeguards and
their runnable checks are in `fprint/` and `tests/test_fprint.py`.

---

# Legacy Multi-Signal AI-Detector Reliability Analysis

## Overview

This project is a multi-signal research pipeline for studying **false positives in AI-text detection** — cases where human writing gets mislabeled as AI-generated.

The goal is not just to build another AI detector, but to analyze how and where commonly used AI-detection systems fail. The project stress-tests detector behavior across large sets of text chunks and uses statistical and feature-space analysis to study which measurable writing patterns are associated with misclassification.

The pipeline uses corpus processing, sentiment scoring, entropy/repetition features, structural-text features, n-gram Markov modeling, statistical testing, regression analysis, Bayesian analysis, PCA/clustering, detector-failure analysis, and generated plots.

## Research Question

The main question behind the project is:

> When do AI-text detectors wrongly label human writing as AI-generated, and what measurable text-level signals appear around those false positives?

This matters because AI detectors are often used in academic and educational settings, where false positives can lead to human writers being wrongly accused of using AI.

## Project Structure

```text
SentimentPolarityAIDetection/
├── analysis.py                         # Main analysis script
├── bayesian_regression.py              # Bayesian regression models
├── build_corpus_chunks.py              # Corpus chunking from PMC XML
├── pmc_parser.py                       # PMC/JATS XML parser
├── regression_evidence.py              # Regression analysis utilities
├── sentiment_regression_models.py      # OLS regression models
├── statistical_tests_runner.py         # Chi-square and statistical tests
├── corpus_chunks.csv                   # Chunked corpus data
├── corpus_with_results.csv             # Full analysis results
├── detection_summary.csv               # Detector-output summary
├── positive_detections.csv             # Detector-positive classifications
├── chunk_index.json                    # Chunk indexing data
├── test_chunks.csv                     # Test dataset chunks
├── test_chunk_index.json               # Test chunk indexing
├── top_bigrams.csv                     # Top bigram features
├── data_sample/                        # Sample data
├── docs/                               # Documentation
├── outputs/                            # Generated outputs
├── plots/                              # Generated visualizations
├── stats/                              # Statistical outputs
│   ├── bayesian_*.csv                  # Bayesian analysis results
│   ├── chi_square_*.csv                # Chi-square test results
│   ├── logit_*.csv                     # Logistic regression results
│   ├── ols_*.csv                       # OLS regression results
│   ├── model_metrics.csv               # Model-performance metrics
│   └── sentiment_regression_*.csv      # Sentiment-regression outputs
└── appendices/
    ├── APPENDIX_B_FULL_STATS.md        # Full statistical results
    ├── CODE_MANIFEST.md                # Reproducibility guide
    ├── CODEBOOK.md                     # Variable definitions
    └── RESEARCH_INSTRUMENT.md          # Research methodology
```

## Methodology

The project is organized around a reproducible analysis pipeline.

### 1. Corpus Processing

The pipeline processes PubMed Central articles and converts them into text chunks. Each chunk is indexed and stored with metadata so the analysis can be reproduced and inspected later.

### 2. Feature Extraction

For each text chunk, the project computes multiple writing signals, including:

* sentiment polarity
* entropy-based features
* repetition metrics
* structural-text features
* lexical features
* n-gram patterns
* Markov-style transition patterns
* detector probability outputs

These signals are used to compare human and AI-generated writing and to study detector behavior.

### 3. AI-Detector Stress Testing

The project applies AI-detection models to large sets of text chunks and records detector outputs, confidence scores, and positive classifications.

The main focus is on false positives: cases where human-authored writing is flagged as AI-generated.

### 4. Statistical Analysis

The project uses statistical testing and modeling to study relationships between text-level features and detector classifications.

This includes:

* chi-square tests
* OLS regression
* logistic regression
* Bayesian regression
* model-performance summaries
* regression diagnostics

### 5. Feature-Space and Failure-Mode Analysis

The project uses PCA, clustering, feature analysis, and generated visualizations to inspect whether false-positive cases form recognizable patterns in feature space.

The goal is to better characterize when detector outputs should not be trusted.

## Dependencies

The project uses Python and the following packages:

* pandas
* numpy
* nltk
* torch
* transformers
* scipy
* matplotlib
* tqdm

Install dependencies:

```bash
pip install pandas numpy nltk torch transformers scipy matplotlib tqdm
```

Download required NLTK data:

```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

## Data Preparation

Place raw PMC XML files in:

```text
data/raw/PMC/
```

Then build the chunked corpus:

```bash
python build_corpus_chunks.py --pmc-dir ../data/raw/PMC --out corpus_chunks.csv --sentences-per-chunk 8
```

This creates `corpus_chunks.csv`, which contains chunked text data and metadata.

Expected columns include:

* `pmcid`
* `title`
* `abstract`
* `chunk_id`
* `chunk_text`
* `word_count`
* `sentence_count`

## Running the Analysis

Run the main pipeline:

```bash
python analysis.py corpus_chunks.csv
```

The script:

1. loads corpus chunks
2. computes text-level features
3. calculates sentiment scores
4. applies AI-detection models
5. records detector outputs
6. saves checkpoints
7. generates statistical outputs and plots

Primary output:

```text
corpus_with_results.csv
```

## Statistical Analysis

Run chi-square tests:

```bash
python statistical_tests_runner.py
```

Generated files include:

* `stats/chi_square_contingency.csv`
* `stats/chi_square_expected.csv`
* `stats/chi_square_stats.csv`
* `detection_summary.csv`

Run regression models:

```bash
python sentiment_regression_models.py
```

Generated files include:

* `stats/ols_coefficients.csv`
* `stats/ols_summary.txt`
* `stats/sentiment_regression_vif.csv`

Run Bayesian analysis:

```bash
python bayesian_regression.py
```

Bayesian posterior estimates and probability summaries are saved in the `stats/` directory.

## Outputs

Main outputs include:

* `corpus_chunks.csv` — chunked input corpus
* `corpus_with_results.csv` — complete feature and detector-output table
* `detection_summary.csv` — detector behavior summary
* `positive_detections.csv` — chunks flagged as AI-generated
* `stats/` — statistical outputs
* `plots/` — generated visualizations
* `appendices/` — methodology, codebook, and reproducibility documentation

## Reproducibility

The project includes:

* chunk indexes
* checkpointed processing
* saved intermediate outputs
* documented statistical results
* a code manifest
* a codebook
* a research methodology document

These files are intended to make the analysis easier to inspect, rerun, and extend.

## Limitations

This project does not claim to solve AI-text detection. Detector behavior depends on the dataset, model, text length, writing domain, thresholds, and preprocessing choices.

The project should be read as an empirical analysis of detector reliability and false-positive behavior under the tested conditions.

## License

This project is released under the MIT License.

## Contact

For questions or issues, please open an issue on the repository.
