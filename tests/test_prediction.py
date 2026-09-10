import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.predict import predict_text

MODEL_EXISTS = os.path.exists(config.PIPELINE_PATH)
skip_if_no_model = pytest.mark.skipif(
    not MODEL_EXISTS, reason="No trained pipeline found; run `python -m src.train` first."
)


def test_predict_empty_input_returns_error():
    result = predict_text("")
    assert result["error"] is not None
    assert result["label"] is None


def test_predict_none_input_returns_error():
    result = predict_text(None)
    assert result["error"] is not None


def test_predict_whitespace_only_input_returns_error():
    result = predict_text("   ")
    assert result["error"] is not None


@skip_if_no_model
def test_predict_obvious_spam():
    result = predict_text("WINNER!! You have been selected to receive a FREE prize. Call 09061701461 now!")
    assert result["label"] == 1
    assert result["label_name"] == "spam"


@skip_if_no_model
def test_predict_obvious_ham():
    result = predict_text("Hey, are we still meeting for lunch tomorrow at noon?")
    assert result["label"] == 0
    assert result["label_name"] == "ham"


@skip_if_no_model
def test_predict_returns_probability_between_0_and_1():
    result = predict_text("Are you free this weekend?")
    assert result["spam_probability"] is None or 0.0 <= result["spam_probability"] <= 1.0


@skip_if_no_model
def test_predict_handles_unicode_and_emoji():
    result = predict_text("Hey 😊 how's it going? café meetup later? 你好")
    assert result["error"] is None
    assert result["label"] in (0, 1)


@skip_if_no_model
def test_predict_handles_very_long_message():
    long_text = "call now free prize " * 500
    result = predict_text(long_text)
    assert result["error"] is None
