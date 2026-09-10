# SMS Spam Detection Pipeline

An end-to-end machine learning classifier built to accurately identify SMS spam. This project features a reproducible training pipeline, a Flask API backend, an interactive Streamlit frontend, and a comprehensive test suite.

**Technology Stack:** 
                        Python,     scikit-learn, 
                        pandas,     NLTK, 
                        Flask,      Streamlit, 
                        pytest,     joblib

---

**Project Architecture & Features**
* **Dual-Branch Feature Engineering:** Utilizes a custom `sklearn.Pipeline` with `FeatureUnion` to combine word-level TF-IDF, character n-gram TF-IDF (to catch obfuscated spam like "FR33"), and structural features (character counts, digit ratios, currency symbols).
* **Cross-Validated Tuning:** The primary model (Calibrated Linear SVM) was tuned via `GridSearchCV` to optimize for an imbalanced dataset (87.4% ham / 12.6% spam).
* **Custom Decision Thresholding:** Instead of relying on the default 0.5 cutoff, the decision threshold was mathematically tuned (0.373) via out-of-fold predictions to maximize Spam Recall while maintaining high Precision.
* **Production-Ready Artifacts:** Text cleaning, TF-IDF vectorization, structural feature extraction, and the final classifier are saved as a single unified `.joblib` artifact to prevent training-serving skew.

---

**Model Performance (Held-out Test Set)**
*Trained on the UCI SMS Spam Collection dataset (5,169 deduplicated records; 80/20 stratified split).*

| Metric | Score | Note |
|---|---|---|
| **Accuracy** | 0.9845 | Overall correctness |
| **Precision (Spam)** | 0.9528 | Minimizes false positives (legitimate messages flagged as spam) |
| **Recall (Spam)** | 0.9237 | Minimizes false negatives (missed spam) |
| **F1-Score (Spam)** | 0.9380 | Harmonic mean of precision and recall |
| **ROC-AUC** | 0.9962 | Area under the receiver operating characteristic curve |

---

**Local Installation**

1. Clone the repository and navigate into the directory:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/SMS-Spam-Detection.git](https://github.com/YOUR-USERNAME/SMS-Spam-Detection.git)
   cd SMS-Spam-Detection
