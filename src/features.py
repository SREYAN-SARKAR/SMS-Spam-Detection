"""
Feature engineering.

Adds structural spam-relevant features (message length, digit count,
capital-letter ratio, repeated-punctuation count) alongside TF-IDF text
features, combined via sklearn's FeatureUnion-style ColumnTransformer.

Each feature is justified for SMS spam specifically:
  - char_count / word_count : spam messages in this dataset average ~139
    chars vs ~71 for ham -- length alone carries signal.
  - digit_ratio             : spam messages often contain phone numbers,
    prize amounts, shortcodes.
  - has_url / has_currency  : promotional/phishing spam commonly includes
    links or money symbols (captured as tokens in clean_text, but also
    exposed as explicit binary features here so linear models can weight
    them directly).
  - upper_ratio             : spam frequently uses ALL CAPS for emphasis
    ("WINNER!!", "FREE").
  - exclaim_count           : promotional spam uses repeated punctuation
    ("!!!", "???") far more than ordinary ham messages.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"http\S+|www\S+")
CURRENCY_RE = re.compile(r"[£$€]")


class StructuralFeatures(BaseEstimator, TransformerMixin):
    """Extracts numeric structural features from raw (uncleaned) SMS text.
    Operates on the ORIGINAL message, not the cleaned tokens, since
    capitalization and punctuation are stripped during cleaning."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(X).fillna("")
        char_count = s.str.len()
        word_count = s.str.split().apply(len)
        digit_count = s.str.count(r"\d")
        digit_ratio = (digit_count / char_count.replace(0, 1)).fillna(0)
        upper_count = s.apply(lambda t: sum(1 for c in t if c.isupper()))
        upper_ratio = (upper_count / char_count.replace(0, 1)).fillna(0)
        has_url = s.str.contains(URL_RE).astype(int)
        has_currency = s.str.contains(CURRENCY_RE).astype(int)
        exclaim_count = s.str.count("!")

        feats = np.column_stack(
            [
                char_count.values,
                word_count.values,
                digit_ratio.values,
                upper_ratio.values,
                has_url.values,
                has_currency.values,
                exclaim_count.values,
            ]
        )
        return feats

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [
                "char_count",
                "word_count",
                "digit_ratio",
                "upper_ratio",
                "has_url",
                "has_currency",
                "exclaim_count",
            ]
        )
