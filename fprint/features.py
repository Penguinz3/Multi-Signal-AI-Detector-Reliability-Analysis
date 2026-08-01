from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from src.utils import count_punctuation, ngram_transition_entropy, split_sentences, tokenize_words, word_shannon_entropy

FEATURE_NAMES = (
    "word_count", "sentence_length_mean", "sentence_length_sd", "average_word_length",
    "type_token_ratio", "repetition_rate", "punctuation_density", "word_entropy",
    "bigram_transition_entropy", "vader_compound",
)


@lru_cache(maxsize=1)
def _sentiment_analyzer():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except (ImportError, LookupError) as error:
        raise RuntimeError("Install nltk and its vader_lexicon before feature extraction.") from error


def target_features(text: str) -> dict[str, float]:
    tokens = tokenize_words(text)
    sentences = split_sentences(text)
    lengths = [len(tokenize_words(sentence)) for sentence in sentences]
    count = len(tokens)
    mean = sum(lengths) / len(lengths) if lengths else 0.0
    variance = sum((value - mean) ** 2 for value in lengths) / len(lengths) if lengths else 0.0
    counts = Counter(tokens)
    return {
        "word_count": float(count),
        "sentence_length_mean": mean,
        "sentence_length_sd": math.sqrt(variance),
        "average_word_length": sum(map(len, tokens)) / count if count else 0.0,
        "type_token_ratio": len(counts) / count if count else 0.0,
        "repetition_rate": (count - len(counts)) / count if count else 0.0,
        "punctuation_density": count_punctuation(text) / len(text) if text else 0.0,
        "word_entropy": word_shannon_entropy(text),
        "bigram_transition_entropy": ngram_transition_entropy(tokens, 2),
        "vader_compound": float(_sentiment_analyzer().polarity_scores(text)["compound"]),
    }
