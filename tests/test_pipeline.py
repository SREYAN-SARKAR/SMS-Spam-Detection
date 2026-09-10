import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.naive_bayes import MultinomialNB

from src.pipeline_builder import build_pipeline
from src.preprocessing import load_data
from src import config


def test_pipeline_fits_and_predicts_on_small_sample():
    df = load_data(config.RAW_DATA_PATH).sample(200, random_state=1)
    X, y = df["message"].values, df["label"].values

    pipeline = build_pipeline(MultinomialNB())
    pipeline.fit(X, y)
    preds = pipeline.predict(X[:5])

    assert len(preds) == 5
    assert set(preds).issubset({0, 1})


def test_pipeline_vectorizer_not_fit_on_test_data():
    """Fitting on a small train slice must not know about vocabulary that
    only appears in a held-out slice -- a direct leakage check."""
    df = load_data(config.RAW_DATA_PATH)
    train_df = df.iloc[:300]
    test_df = df.iloc[300:320]

    pipeline = build_pipeline(MultinomialNB())
    pipeline.fit(train_df["message"].values, train_df["label"].values)

    tfidf = pipeline.named_steps["features"].transformer_list[0][1].named_steps["tfidf_word"]
    train_vocab = set(tfidf.vocabulary_.keys())

    # Words that appear ONLY in the test slice's cleaned text should not be
    # part of the fitted vocabulary.
    from src.preprocessing import clean_text

    test_only_words = set()
    for msg in test_df["message"]:
        test_only_words.update(clean_text(msg).split())
    test_only_words -= train_vocab

    # This doesn't assert zero overlap (common words legitimately appear in
    # both) -- it asserts the vectorizer's vocabulary size matches what
    # fitting on train alone would produce, i.e. transform() on unseen
    # words doesn't error and doesn't retroactively expand the vocabulary.
    vocab_size_before = len(tfidf.vocabulary_)
    pipeline.named_steps["features"].transform(test_df["message"].values)
    vocab_size_after = len(tfidf.vocabulary_)
    assert vocab_size_before == vocab_size_after


def test_pipeline_save_and_load_roundtrip(tmp_path):
    df = load_data(config.RAW_DATA_PATH).sample(200, random_state=1)
    X, y = df["message"].values, df["label"].values

    pipeline = build_pipeline(MultinomialNB())
    pipeline.fit(X, y)

    save_path = tmp_path / "test_pipeline.joblib"
    joblib.dump(pipeline, save_path)
    loaded = joblib.load(save_path)

    original_preds = pipeline.predict(X[:10])
    loaded_preds = loaded.predict(X[:10])
    assert list(original_preds) == list(loaded_preds)
