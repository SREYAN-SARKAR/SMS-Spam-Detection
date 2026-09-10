import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text, load_data
from src import config


def test_clean_text_lowercases():
    assert clean_text("HELLO WORLD") == "hello world"


def test_clean_text_removes_url_leaves_token():
    result = clean_text("Click here www.freegift.com now")
    assert "urltoken" in result
    assert "www" not in result


def test_clean_text_handles_phone_number():
    result = clean_text("Call 09061701461 to claim")
    assert "phonetoken" in result


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""


def test_clean_text_handles_non_string_input():
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_clean_text_removes_stopwords():
    result = clean_text("this is a free gift for you")
    assert "is" not in result.split()
    assert "for" not in result.split()


def test_load_data_deduplicates():
    df = load_data(config.RAW_DATA_PATH)
    assert df["message"].duplicated().sum() == 0


def test_load_data_binary_labels():
    df = load_data(config.RAW_DATA_PATH)
    assert set(df["label"].unique()) <= {0, 1}
