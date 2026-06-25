# Multi-Signal AI-Detector Reliability Analysis

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
