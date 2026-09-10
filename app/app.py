"""
Flask API
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, jsonify, request
from src.predict import predict_text
app = Flask(__name__)
MAX_MESSAGE_LENGTH = 5000  # basic robustness guard against extremely long input
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if text is None:
        return jsonify({"error": 'No text provided. Send JSON like {"text": "..."}'}), 400
    if not isinstance(text, str):
        return jsonify({"error": "`text` must be a string."}), 400
    if len(text) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"`text` exceeds max length of {MAX_MESSAGE_LENGTH} characters."}), 400
    try:
        result = predict_text(text)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    if result["error"]:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
