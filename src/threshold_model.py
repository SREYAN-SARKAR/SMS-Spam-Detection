"""
Wraps a fitted sklearn Pipeline with a tuned decision threshold.
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
