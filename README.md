# Multi-Signal-AI-Detector-Reliability-Analysis

## Overview

This project implements a comprehensive analysis framework for detecting AI-generated text through sentiment polarity patterns. The research investigates whether AI-generated content exhibits distinct sentiment characteristics compared to human-written text, using a combination of lexical features, sentiment analysis, and statistical modeling.

The analysis pipeline processes PubMed Central (PMC) articles, extracts text chunks, computes sentiment scores using multiple methods, applies AI detection algorithms, and performs rigorous statistical testing to identify significant differences between human and AI-generated content.

## Project Structure

```
SentimentPolarityAIDetection/
├── anlysis.py                          # Main analysis script
├── bayesian_regression.py             # Bayesian regression models
├── build_corpus_chunks.py             # Corpus chunking from PMC XML
├── pmc_parser.py                      # PMC/JATS XML parser
├── regression_evidence.py             # Regression analysis utilities
├── sentiment_regression_models.py     # OLS regression models
├── statistical_tests_runner.py        # Chi-square and statistical tests
├── corpus_chunks.csv                  # Chunked corpus data
├── corpus_with_results.csv            # Analysis results
├── detection_summary.csv              # Detection statistics
├── positive_detections.csv            # Positive detection results
├── chunk_index.json                   # Chunk indexing data
├── test_chunks.csv                    # Test dataset chunks
├── test_chunk_index.json              # Test chunk indexing
├── top_bigrams.csv                    # Top bigram features
├── appendices/                        # Documentation and manifests
│   ├── APPENDIX_B_FULL_STATS.md       # Statistical results
│   ├── CODE_MANIFEST.md               # Code reproducibility guide
│   ├── CODEBOOK.md                    # Variable definitions
│   └── RESEARCH_INSTRUMENT.md         # Research methodology
├── checkpoints/                       # Analysis checkpoints
│   ├── corpus_with_results.partial.csv
│   └── run_status.txt
├── plots/                             # Generated visualizations
├── research_env/                      # Python virtual environment
├── stats/                             # Statistical outputs
│   ├── bayesian_*.csv                 # Bayesian analysis results
│   ├── chi_square_*.csv               # Chi-square test results
│   ├── logit_*.csv                    # Logistic regression results
│   ├── ols_*.csv                      # OLS regression results
│   ├── model_metrics.csv              # Model performance metrics
│   └── sentiment_regression_*.csv     # Sentiment regression outputs
└── __pycache__/                       # Python bytecode cache
```

## Dependencies

The project requires the following Python packages:

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- nltk: Natural language processing toolkit
- torch: PyTorch deep learning framework
- transformers: Hugging Face transformers for AI detection
- scipy: Scientific computing and statistical functions
- matplotlib: Data visualization
- tqdm: Progress bars for long-running operations

## Installation

1. Clone or download the project repository to your local machine.

2. Navigate to the project directory:
   ```
   cd SentimentPolarityAIDetection
   ```

3. Activate the provided virtual environment:
   ```
   research_env\Scripts\activate  # Windows
   # or
   source research_env/bin/activate  # Linux/Mac
   ```

4. If the virtual environment is not set up, create and activate a new one:
   ```
   python -m venv research_env
   research_env\Scripts\activate  # Windows
   pip install pandas numpy nltk torch transformers scipy matplotlib tqdm
   ```

5. Download required NLTK data:
   ```
   python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
   ```

## Data Preparation

### Corpus Construction

The analysis requires PMC article data in XML format. Place raw PMC XML files in a `data/raw/PMC/` directory structure.

1. Run the corpus builder:
   ```
   python build_corpus_chunks.py --pmc-dir ../data/raw/PMC --out corpus_chunks.csv --sentences-per-chunk 8
   ```

   This creates `corpus_chunks.csv` with text chunks of approximately 8 sentences each.

### Alternative: Use Pre-built Corpus

If you have existing corpus data, ensure it's formatted as `corpus_chunks.csv` with the following columns:
- pmcid: PubMed Central ID
- title: Article title
- abstract: Article abstract
- chunk_id: Unique chunk identifier
- chunk_text: Text content of the chunk
- word_count: Number of words in chunk
- sentence_count: Number of sentences in chunk

## Running the Analysis

The main analysis is performed by `anlysis.py`, which:

1. Loads the corpus chunks
2. Computes lexical features (word count, sentence count, etc.)
3. Calculates sentiment scores using VADER and TextBlob
4. Applies AI detection using transformer models
5. Generates plots and saves results

Execute the analysis:
```
python anlysis.py corpus_chunks.csv
```

Or using the virtual environment explicitly:
```
research_env\Scripts\python.exe -u anlysis.py corpus_chunks.csv
```

The script will:
- Process chunks in batches
- Save progress to checkpoints
- Generate intermediate results
- Create visualization plots
- Output final results to `corpus_with_results.csv`

## Statistical Analysis

### Chi-Square Tests

Run statistical significance tests:
```
python statistical_tests_runner.py
```

This generates:
- `stats/chi_square_contingency.csv`: Contingency tables
- `stats/chi_square_expected.csv`: Expected frequencies
- `stats/chi_square_stats.csv`: Test statistics and p-values
- `detection_summary.csv`: Detection performance summary

### Regression Models

Run sentiment regression analysis:
```
python sentiment_regression_models.py
```

This produces three OLS regression models:
1. Human Features only
2. Hybrid Features only
3. Combined Features

Outputs include:
- `stats/ols_coefficients.csv`: Regression coefficients
- `stats/ols_summary.txt`: Model summaries
- `stats/sentiment_regression_vif.csv`: Variance Inflation Factors

### Bayesian Analysis

For Bayesian regression models:
```
python bayesian_regression.py
```

Generates posterior distributions and probability estimates in the `stats/` directory.

## Outputs and Results

### Primary Data Files

- `corpus_with_results.csv`: Complete analysis results with sentiment scores, AI detection probabilities, and feature vectors
- `corpus_chunks.csv`: Input corpus with chunked text data
- `detection_summary.csv`: Aggregated detection performance metrics

### Statistical Evidence

Located in `stats/`:
- Model coefficients and summaries
- Chi-square test results
- Bayesian posterior probabilities
- Regression diagnostics (VIF, residuals)

### Visualizations

Located in `plots/`:
- Sentiment distribution plots
- AI detection probability histograms
- Feature correlation matrices
- Model diagnostic plots

## Configuration and Parameters

### Analysis Parameters

Key parameters in `anlysis.py`:
- `BATCH_SIZE`: Number of chunks processed simultaneously (default: 100)
- `MAX_LENGTH`: Maximum token length for AI detection (default: 512)
- Sentiment thresholds and scoring methods

### Model Selection

AI detection uses pre-trained transformer models. Modify the model loading in `anlysis.py` to use different architectures.

### Statistical Test Parameters

Chi-square tests use default significance levels (α = 0.05). Modify `statistical_tests_runner.py` for different thresholds.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all required packages are installed in the active Python environment.

2. **NLTK Data**: Run the NLTK download commands if sentiment analysis fails.

3. **Memory Issues**: Reduce `BATCH_SIZE` in `anlysis.py` for systems with limited RAM.

4. **CUDA Errors**: The analysis will fall back to CPU if CUDA is not available, but processing will be slower.

5. **File Path Issues**: Ensure relative paths are correct when running scripts from different directories.

### Performance Optimization

- Use GPU acceleration if available (CUDA-compatible GPU recommended)
- Process smaller batches for memory-constrained systems
- Use the checkpoint system to resume interrupted analyses

## Research Methodology

This project implements the methodology described in the research instrument. Key components:

1. **Corpus Selection**: PMC articles provide diverse scientific writing samples
2. **Text Chunking**: 8-sentence chunks balance context preservation with analysis granularity
3. **Sentiment Analysis**: Multi-method approach using VADER and TextBlob for robustness
4. **AI Detection**: Transformer-based classification models
5. **Statistical Validation**: Chi-square tests for independence, regression for relationships

## Code Reproducibility

All analysis scripts include detailed logging and checkpointing. The `appendices/CODE_MANIFEST.md` provides exact reproduction commands for each analysis component.

## License

This project is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Make changes with clear commit messages
4. Submit a pull request with detailed description

## Citation

If you use this code in your research, please cite:

[Include appropriate citation information here]

## Contact

For questions or issues, please open an issue on the project repository or contact the maintainers.
