"""
Loading and cleaning the SMS spam dataset, plus text normalization.

Fixes applied vs. the original script:
  - Deduplicates records BEFORE any train/test split (the raw dataset has
    403 exact duplicate rows out of 5,572 -- left in place, these can land
    in both train and test and leak information / inflate test scores).
  - Downloads the NLTK resource needed by newer NLTK versions
    ('punkt_tab'), not just the legacy 'punkt'.
  - Keeps structural spam signals (URLs, digits, currency symbols) as
    placeholder tokens instead of silently deleting them -- deleting them
    throws away information that's genuinely predictive of spam.
"""

import os
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config import RAW_DATA_PATH


def _ensure_nltk_resources():
    """Download NLTK resources only if missing. Handles both legacy and
    newer NLTK tokenizer resource names."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for find_path, download_name in resources:
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


_ensure_nltk_resources()
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"http\S+|www\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE = re.compile(r"\b\d{5,}\b")  # 5+ consecutive digits: SMS shortcodes/phone numbers
CURRENCY_RE = re.compile(r"[£$€]")


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the dataset, normalize columns to ['label', 'message'],
    normalize labels to {0, 1}, and drop duplicate messages."""
    df = pd.read_csv(path, encoding="latin-1")

    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    elif "label" in df.columns and "message" in df.columns:
        df = df[["label", "message"]]
    else:
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]

    df = df.dropna(subset=["label", "message"])

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: 1 if x in ("spam", "1", "s", "true", "yes") else 0)
    )

    before = len(df)
    df = df.drop_duplicates(subset="message").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[load_data] Removed {removed} duplicate message(s) ({removed / before:.1%} of raw rows).")

    return df


def clean_text(text: str) -> str:
    """Normalize SMS text while preserving spam-relevant structural signals
    as placeholder tokens (URL/PHONE/CURRENCY) rather than deleting them."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = EMAIL_RE.sub(" ", text)
    text = URL_RE.sub(" urltoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = CURRENCY_RE.sub(" currencytoken ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df["label"].value_counts())
    print(clean_text("WINNER!! Call 09061701461 now to claim your £900 prize! www.win.com"))
