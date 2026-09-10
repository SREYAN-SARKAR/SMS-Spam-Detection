"""
Training entry point (v2 -- pushed for higher spam recall while keeping
precision high).

Changes vs. v1:
  - Added char n-gram TF-IDF branch (via pipeline_builder) to catch
    obfuscated spam tokens.
  - Added ComplementNB and a calibrated Linear SVM to the model
    comparison (ComplementNB specifically handles class-imbalanced text
    counts better than MultinomialNB; LinearSVC is often the strongest
    linear model for sparse TF-IDF text).
  - Widened the Logistic Regression / SVM hyperparameter grids and added
    class_weight='balanced' as a tunable option, since the dataset is
    87/13 imbalanced.
  - Added a CV-based decision-threshold search (on the training set only)
    to explicitly trade a little precision for meaningfully higher spam
    recall, since 0.5 is an arbitrary default, not a tuned choice.

All model selection, hyperparameter tuning, and threshold tuning happen
via cross-validation on X_train only. The held-out test set is touched
exactly once, at the very end.
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

from src import config
from src.evaluate import evaluate_pipeline
from src.pipeline_builder import build_pipeline
from src.preprocessing import load_data
from src.threshold_model import ThresholdedPipeline

CANDIDATE_MODELS = {
    "multinomial_nb": MultinomialNB(),
    "complement_nb": ComplementNB(),
    "logistic_regression": LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
    "linear_svm": CalibratedClassifierCV(
        LinearSVC(random_state=config.RANDOM_STATE, max_iter=20000, dual=True), cv=2
    ),
}

PARAM_GRIDS = {
    "multinomial_nb": {"clf__alpha": [0.1, 0.5, 1.0, 2.0]},
    "complement_nb": {"clf__alpha": [0.1, 0.5, 1.0, 2.0]},
    "logistic_regression": {
        "clf__C": [1.0, 3.0, 10.0, 30.0],
        "clf__class_weight": [None, "balanced"],
    },
    "linear_svm": {
        "clf__estimator__C": [0.3, 1.0, 3.0],
        "clf__estimator__class_weight": [None, "balanced"],
    },
}


def select_best_model(X_train, y_train):
    """Cross-validated model selection on the training set only."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    cv_results = []
    for name, clf in CANDIDATE_MODELS.items():
        pipe = build_pipeline(clf)
        t0 = time.time()
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        elapsed = time.time() - t0
        cv_results.append(
            {
                "model": name,
                "cv_f1_mean": scores.mean(),
                "cv_f1_std": scores.std(),
                "fit_time_sec": round(elapsed, 1),
            }
        )
        print(f"[CV] {name}: f1 = {scores.mean():.4f} (+/- {scores.std():.4f})  [{elapsed:.1f}s]")

    cv_df = pd.DataFrame(cv_results).sort_values("cv_f1_mean", ascending=False)
    cv_df.to_csv(config.CV_RESULTS_PATH, index=False)
    best_name = cv_df.iloc[0]["model"]
    return best_name, cv_df


def tune_model(best_name, X_train, y_train):
    """GridSearchCV on the training set only, for the winning model family."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    pipe = build_pipeline(CANDIDATE_MODELS[best_name])
    grid = PARAM_GRIDS[best_name]
    search = GridSearchCV(pipe, param_grid=grid, scoring="f1", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"[GridSearch] Best params for {best_name}: {search.best_params_}")
    print(f"[GridSearch] Best CV f1: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


def select_threshold(fitted_pipeline, X_train, y_train, min_precision: float | None = None):
    """Chooses a decision threshold using out-of-fold predicted
    probabilities on the TRAINING set only (never the test set).

    Default strategy: pick the threshold that directly maximizes spam F1
    on out-of-fold predictions. This is a genuine tuning step (0.5 is an
    arbitrary default, not a chosen one) and, unlike a hard precision
    floor, it can't make F1 worse than the untuned default -- it searches
    over 0.5 as one of its candidates.

    If `min_precision` is set, a precision floor is enforced instead
    (useful if a deployment context truly cannot tolerate false
    positives below a certain rate) -- but this can trade away overall
    F1, so it isn't the default.
    """
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    oof_proba = cross_val_predict(fitted_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_proba)
    # precision_recall_curve returns arrays 1 longer than thresholds; align.
    precisions, recalls = precisions[:-1], recalls[:-1]

    if min_precision is not None:
        candidates = [(p, r, t) for p, r, t in zip(precisions, recalls, thresholds) if p >= min_precision]
        if candidates:
            best = max(candidates, key=lambda tup: tup[1])
            chosen_threshold = float(best[2])
            print(
                f"[Threshold] Selected {chosen_threshold:.3f} "
                f"(OOF precision={best[0]:.4f}, recall={best[1]:.4f}, meets min_precision={min_precision})"
            )
            return chosen_threshold
        print(f"[Threshold] No threshold met min_precision={min_precision}; falling back to F1-optimal.")

    f1s = [f1_score(y_train, (oof_proba >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    chosen_threshold = float(thresholds[best_idx])
    print(
        f"[Threshold] F1-optimal threshold: {chosen_threshold:.3f} "
        f"(OOF precision={precisions[best_idx]:.4f}, recall={recalls[best_idx]:.4f}, f1={f1s[best_idx]:.4f})"
    )
    return chosen_threshold


def train_and_save(dataset_path: str = config.RAW_DATA_PATH, min_precision: float | None = None):
    df = load_data(dataset_path)
    print(
        f"Loaded {len(df)} deduplicated records "
        f"({(df['label'] == 1).sum()} spam / {(df['label'] == 0).sum()} ham)"
    )

    X = df["message"].astype(object).to_numpy()
    y = df["label"].to_numpy()

    # Test set is set aside ONCE, before any model selection / tuning / thresholding.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    best_name, cv_df = select_best_model(X_train, y_train)
    print("\nCross-validation results:")
    print(cv_df.to_string(index=False))

    best_pipeline, best_params, best_cv_f1 = tune_model(best_name, X_train, y_train)

    # Threshold tuning uses out-of-fold probabilities on X_train only.
    threshold = select_threshold(best_pipeline, X_train, y_train, min_precision=min_precision)
    final_model = ThresholdedPipeline(best_pipeline, threshold=threshold)
    final_model.fit(X_train, y_train)  # refit on the full training set

    # Final, single evaluation on the untouched test set.
    metrics = evaluate_pipeline(final_model, X_test, y_test)
    print("\nFinal held-out test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    joblib.dump(final_model, config.PIPELINE_PATH)
    print(f"\nSaved pipeline ({best_name}, tuned, threshold={threshold:.3f}) to {config.PIPELINE_PATH}")

    summary = pd.DataFrame(
        [
            {
                "model": best_name,
                "tuned_params": best_params,
                "cv_f1": best_cv_f1,
                "decision_threshold": threshold,
                **metrics,
            }
        ]
    )
    summary.to_csv(config.RESULTS_PATH, index=False)
    print(f"Saved results summary to {config.RESULTS_PATH}")

    return config.PIPELINE_PATH, metrics


if __name__ == "__main__":
    train_and_save()
