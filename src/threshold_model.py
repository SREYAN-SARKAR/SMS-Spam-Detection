"""
Wraps a fitted sklearn Pipeline with a tuned decision threshold.

By default, classifiers threshold predict_proba at 0.5. For spam
detection specifically, 0.5 is not necessarily optimal: this project
prioritizes spam recall (catching more spam) while keeping spam
precision high enough not to burden users with false positives. The
threshold below is chosen via cross-validation on the TRAINING set only
(see src/train.py) -- the held-out test set is never used to pick it.

This is saved as a single artifact (see src/train.py), so predict.py,
the Flask API, and the Streamlit app all get the tuned threshold
automatically -- there's no second file to keep in sync.
"""

from sklearn.base import BaseEstimator, ClassifierMixin


class ThresholdedPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, pipeline, threshold: float = 0.5):
        self.pipeline = pipeline
        self.threshold = threshold

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
