"""
Evaluation utilities. Reports the full metric set relevant to spam
detection specifically -- not just accuracy, which is misleading on an
imbalanced dataset (86.6% ham / 13.4% spam: a model that always predicts
"ham" would score 86.6% accuracy while being useless).

Recall on the spam class matters (missed spam = spam reaches the user);
precision on the spam class matters at least as much (a false positive
means a legitimate message gets blocked/flagged, which is often worse
for user trust than one missed spam message). F1 balances both, which is
why it's used as the model-selection criterion in train.py.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_pipeline(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_spam": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall_spam": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_spam": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    if hasattr(pipeline, "predict_proba") and len(np.unique(y_test)) == 2:
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def full_report(pipeline, X_test, y_test) -> str:
    """Human-readable classification report + confusion matrix, used for
    manual inspection / error analysis, not for automated model selection."""
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"])
    cm = confusion_matrix(y_test, y_pred)
    cm_str = (
        f"Confusion matrix:\n"
        f"                 predicted ham  predicted spam\n"
        f"actual ham       {cm[0][0]:<14} {cm[0][1]}\n"
        f"actual spam      {cm[1][0]:<14} {cm[1][1]}"
    )
    return f"{report}\n{cm_str}"
