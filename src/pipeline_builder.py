"""
Builds the single, reusable sklearn Pipeline used for both training and
inference. Combining everything into one Pipeline object (instead of
saving a vectorizer and a model separately, as the original script did)
guarantees inference always applies the exact same transformations used
at training time, and removes the risk of loading a mismatched
vectorizer/model pair.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from src.features import StructuralFeatures
from src.preprocessing import clean_text


class TextCleaner(BaseEstimator, TransformerMixin):
    """Thin sklearn-compatible wrapper around clean_text so it can sit
    inside a Pipeline. Takes raw messages, returns cleaned strings."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(t) for t in X]


def build_pipeline(
    classifier,
    word_max_features: int = 5000,
    word_ngram_range=(1, 2),
    char_max_features: int = 3000,
    char_ngram_range=(3, 5),
) -> Pipeline:
    """Returns a Pipeline: raw text -> [word TF-IDF(cleaned text) +
    char n-gram TF-IDF(cleaned text) + structural features] -> classifier.

    The character n-gram branch is added specifically to catch obfuscated
    spam tokens (e.g. "FR33", "WlNNER", "c4ll now") that word-level TF-IDF
    treats as out-of-vocabulary noise but that share sub-word patterns
    with known spam vocabulary. Every step is fit only on training data
    when .fit() is called, so there is no leakage into the
    vectorizer/features.
    """

    word_branch = Pipeline(
        steps=[
            ("clean", TextCleaner()),
            ("tfidf_word", TfidfVectorizer(max_features=word_max_features, ngram_range=word_ngram_range)),
        ]
    )

    char_branch = Pipeline(
        steps=[
            ("clean", TextCleaner()),
            (
                "tfidf_char",
                TfidfVectorizer(
                    max_features=char_max_features,
                    ngram_range=char_ngram_range,
                    analyzer="char_wb",
                ),
            ),
        ]
    )

    structural_branch = StructuralFeatures()

    combined_features = FeatureUnion(
        transformer_list=[
            ("word", word_branch),
            ("char", char_branch),
            ("structural", structural_branch),
        ]
    )

    return Pipeline(
        steps=[
            ("features", combined_features),
            ("clf", classifier),
        ]
    )
