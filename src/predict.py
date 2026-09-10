"""
Inference. This is the ONLY place that loads the saved pipeline and runs
predictions -- app/app.py, app/streamlit_app.py, and the CLI all import
from here instead of each re-implementing loading + prediction logic
(the original script duplicated this logic three times).
"""

import os

import joblib

from src import config

_PIPELINE = None


def load_pipeline(path: str = config.PIPELINE_PATH):
    global _PIPELINE
    if _PIPELINE is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained pipeline found at {path}. Run `python -m src.train` first."
            )
        _PIPELINE = joblib.load(path)
    return _PIPELINE


def predict_text(text: str, path: str = config.PIPELINE_PATH) -> dict:
    """Predict spam/ham for a single raw SMS string. Handles empty/invalid
    input gracefully instead of raising or silently mis-predicting."""
    if text is None or not isinstance(text, str) or text.strip() == "":
        return {"label": None, "label_name": None, "spam_probability": None, "error": "Empty or invalid input."}

    pipeline = load_pipeline(path)
    pred = int(pipeline.predict([text])[0])
    prob = None
    if hasattr(pipeline, "predict_proba"):
        prob = float(pipeline.predict_proba([text])[0, 1])

    return {
        "label": pred,
        "label_name": "spam" if pred == 1 else "ham",
        "spam_probability": prob,
        "error": None,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.predict '<sms text>'")
    else:
        print(predict_text(sys.argv[1]))
