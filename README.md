# SMS Spam Detection

An end-to-end SMS spam classifier built on classical ML (TF-IDF + Logistic Regression), with a reproducible training pipeline, a Flask API, a Streamlit UI, and a pytest test suite.

## Problem Statement

Classify an SMS message as **spam** or **ham** (not spam), using the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) dataset.

## Dataset

- **Source file:** `data/raw/spam.csv`
- **Raw size:** 5,572 rows (2 usable columns: label, message; 3 extra empty columns dropped)
- **After deduplication:** 5,169 unique messages — 403 exact duplicate rows (7.2%) were removed before splitting, since leaving them in risks the same message appearing in both train and test sets
- **Class balance (post-dedup):** 4,516 ham (87.4%) / 653 spam (12.6%) — imbalanced, which is why accuracy alone is not used as the primary metric
- **Message length:** spam messages average ~139 characters (std 29), ham messages average ~71 characters (std 58) — spam is longer and more length-consistent, which is one reason a structural `char_count` feature was added

## Technology Stack

Python, pandas, scikit-learn, NLTK, joblib, Flask, Streamlit, pytest

## ML Pipeline (v2)

```
Raw SMS text
   │
   ├── Word branch:       clean_text() → TF-IDF word 1–2 grams (max 5000 features)
   ├── Char branch:       clean_text() → TF-IDF char 3–5 grams, char_wb (max 3000 features)
   └── Structural branch: char_count, word_count, digit_ratio, upper_ratio,
                          has_url, has_currency, exclaim_count
   │
   ▼ (combined via FeatureUnion)
Calibrated Linear SVM (tuned C, tuned decision threshold)
   │
   ▼
spam / ham + probability
```

**v2 changes from the first version:** added a character n-gram TF-IDF branch (catches obfuscated spam like "FR33" or "c4ll now" that word-level TF-IDF treats as unknown vocabulary), widened the model comparison to 4 candidates (Multinomial NB, Complement NB, Logistic Regression, calibrated Linear SVM) with a wider hyperparameter grid including `class_weight='balanced'`, and added cross-validated decision-threshold tuning instead of leaving the default 0.5 cutoff unexamined.

Everything — text cleaning, TF-IDF, structural features, and the classifier — is a single `sklearn.Pipeline` object (`src/pipeline_builder.py`), fit only on the training split and saved as one artifact. This removes the risk (present in earlier versions of this project) of loading a mismatched vectorizer/model pair, and guarantees inference uses the exact transformations used at training time.

### Text preprocessing

Lowercasing, stopword removal, lemmatization, and whitespace normalization are applied. URLs, phone-number-like digit sequences, and currency symbols are replaced with placeholder tokens (`urltoken`, `phonetoken`, `currencytoken`) rather than deleted, since their *presence* is itself predictive of spam.

### Why these features

- `char_count` / `word_count`: spam messages in this dataset are meaningfully longer on average (139 vs 71 chars).
- `digit_ratio`: spam frequently contains phone numbers, shortcodes, or prize amounts.
- `has_url` / `has_currency`: promotional/phishing spam commonly includes links or money symbols.
- `upper_ratio`: spam often uses ALL CAPS for emphasis ("WINNER!!", "FREE").
- `exclaim_count`: promotional spam uses repeated punctuation far more than ordinary messages.

## Model Comparison (v2, actual results, 5-fold stratified CV on the training split)

| Model | CV F1 (mean ± std) | Fit time |
|---|---|---|
| Calibrated Linear SVM | 0.9369 ± 0.0129 | 23.2s |
| Logistic Regression | 0.9170 ± 0.0129 | 5.3s |
| Multinomial Naive Bayes | 0.8721 ± 0.0187 | 7.3s |
| Complement Naive Bayes | 0.8642 ± 0.0151 | 4.4s |

Adding the character n-gram branch lifted every model's CV F1 relative to the word-only-TF-IDF v1 pipeline (e.g. Multinomial NB rose from 0.599 to 0.872 — the char n-grams give it cleaner, more uniformly-scaled count signal than the structural features alone did). Linear SVM was the strongest candidate and was tuned with `GridSearchCV` over `C ∈ {0.3, 1.0, 3.0}` × `class_weight ∈ {None, "balanced"}`. Best CV F1: **0.9399** at `C=3.0, class_weight=None`.

**Decision threshold:** rather than leaving the default 0.5 cutoff unexamined, the threshold was tuned via 5-fold out-of-fold predictions on the training set only, maximizing F1 (candidate 0.5 is included in the search, so this step can't make F1 worse than the untuned default). Selected threshold: **0.373**.

## Final Model — Held-out Test Set Results

(5,169 deduplicated records → 80/20 stratified split; test set touched exactly once, after model selection, tuning, and threshold selection were all complete)

| Metric | v1 (Logistic Regression) | v2 (tuned SVM + char n-grams + threshold) |
|---|---|---|
| Accuracy | 0.9855 | 0.9845 |
| Precision (spam) | 0.9833 | 0.9528 |
| Recall (spam) | 0.9008 | 0.9237 |
| F1 (spam) | 0.9402 | 0.9380 |
| ROC-AUC | 0.9947 | 0.9962 |

**Honest read of this trade-off:** v2 catches meaningfully more spam (recall 0.924 vs 0.901 — 4 fewer missed spam messages out of 10 test-set spam) and has a better ROC-AUC (a purer measure of ranking quality, independent of threshold). But it does this by accepting more false positives (6 vs 2 legitimate messages misflagged), so overall F1 is essentially flat (0.938 vs 0.940), not a strict win. **Which version is actually better depends on what you're optimizing for:**
- If missing spam is the bigger cost (e.g. this feeds a downstream filter with a human review step) → v2 is the better choice.
- If a false positive is the bigger cost (e.g. this silently blocks messages with no review) → v1 is the better choice, and `train_and_save(min_precision=0.97)` reproduces a threshold that recovers v1-like precision (0.975) while keeping v2's model family.

This is a real, expected trade-off of pushing recall on an already-strong model — not a case where one version is objectively superior. If you want a genuine, unambiguous F1 improvement over v1, the two levers with the most headroom left are (a) a broader hyperparameter sweep (`RandomizedSearchCV` across more `C` values and n-gram ranges) and (b) an ensemble/stacking of the SVM and Logistic Regression predictions, which weren't attempted here to keep the model a single, simple, fast-to-serve pipeline.

**Limitations:** false negatives on this split skew toward short, informally-worded, or obfuscated spam ("SMS. ac sun0819 posts HELLO...", "sms.shsex.net...") that doesn't closely match the more overt "WINNER/FREE/call now" vocabulary dominating the training data. False positives skew toward casual, emotionally warm ham messages that happen to contain promo-adjacent words ("deal", "$95/pax") — a realistic limitation of any TF-IDF-based model trained on this dataset's specific spam style and size.

## Project Structure

```
SMS-Spam-Detection/
├── data/
│   ├── raw/spam.csv
│   └── processed/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── pipeline_builder.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── spam_pipeline.joblib
│   ├── results_summary.csv
│   └── cv_results.csv
├── tests/
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   └── test_pipeline.py
├── app/
│   ├── app.py            # Flask API
│   └── streamlit_app.py  # Streamlit UI
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+. NLTK resources (`stopwords`, `wordnet`, `punkt`, `punkt_tab`) are downloaded automatically on first run if missing.

## Usage

**Train:**
```bash
python -m src.train
```
Runs cross-validated model selection, hyperparameter tuning, final held-out evaluation, and saves the pipeline to `models/spam_pipeline.joblib`.

**Predict from the command line:**
```bash
python -m src.predict "Congratulations! You've won a free prize, call now!"
```

**Run the Flask API:**
```bash
python app/app.py
# then:
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text": "Free entry! Call now!"}'
```

**Run the Streamlit UI:**
```bash
streamlit run app/streamlit_app.py
```

**Run tests:**
```bash
pytest tests/ -v
```

## Deployment Notes

For a small classical-ML project like this, the simplest reasonable deployment path is: containerize the Flask app with the saved `spam_pipeline.joblib` baked in (or mounted as a volume), and deploy to any container host (Render, Railway, a small EC2/VM, Cloud Run). A full CI/CD or model-registry setup is not warranted at this scale — retraining is fast (~10s) and can be re-run manually or via a simple scheduled job if the dataset changes.

## Future Improvements

- Calibrate predicted probabilities (`CalibratedClassifierCV`) if the raw probability is ever shown as a "confidence" score to end users.
- Add character n-gram TF-IDF as an additional feature branch — may help with obfuscated spam (e.g., "FR33" instead of "FREE").
- Expand the structural feature set (e.g., punctuation diversity, word-length distribution) if error analysis on a larger/newer dataset shows continued weakness on short informal spam.
- Track experiments (e.g., with MLflow) if this grows beyond a single-model project.

## License

MIT
