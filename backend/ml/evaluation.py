#!/usr/bin/env python3
"""Model evaluation — the paper's *Model Evaluation* stage.

The paper names accuracy and F1-score explicitly. Accuracy alone is not enough
on an imbalanced problem (the paper itself says fake profiles are the rarer
class), so this reports the majority-class baseline next to it, plus precision,
recall, the confusion matrix and ROC-AUC.

`lift_over_baseline` is the number that actually matters: a 94% accuracy on a
94%-negative dataset is a model that has learned to say "Real" every time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate(y_test, y_pred, proba_positive, *, positive: str,
             classes: list[str] | None = None) -> dict[str, Any]:
    """Full metric set for a binary classifier's predictions.

    Deliberately takes arrays rather than a fitted estimator: XGBoost's
    classifier reports `classes_` as `[0, 1]` while the rest of the project
    speaks in `"Fake"`/`"Real"`, and threading label decoding through here would
    put two representations in one function. The caller decodes; this computes.

    Cross-validated accuracy is `cross_val_accuracy()`, kept separate because it
    refits the estimator and is the slow part.
    """
    y_test = pd.Series(y_test).astype(str).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(str).reset_index(drop=True)
    proba = np.asarray(proba_positive, dtype=float)
    classes = classes or sorted(set(y_test) | set(y_pred))
    y_true_bin = (y_test == positive).astype(int)

    baseline = float(y_test.value_counts(normalize=True).max())
    accuracy = float(accuracy_score(y_test, y_pred))

    # Ordered [negative, positive] so the four cells are unambiguous.
    labels = [c for c in classes if c != positive] + [positive]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    metrics: dict[str, Any] = {
        "accuracy": round(accuracy, 4),
        "baseline_accuracy": round(baseline, 4),
        "lift_over_baseline": round(accuracy - baseline, 4),
        "precision": round(float(precision_score(y_test, y_pred, pos_label=positive, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, pos_label=positive, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, pos_label=positive, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true_bin, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true_bin, proba)), 4),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
            "true_negative": tn, "false_positive": fp,
            "false_negative": fn, "true_positive": tp,
        },
        "test_samples": int(len(y_test)),
        "positive_class": positive,
    }

    # Specificity matters here: a false positive is a real user wrongly flagged.
    metrics["specificity"] = round(tn / (tn + fp), 4) if (tn + fp) else 0.0
    return metrics


def cross_val_accuracy(estimator, X, y, *, folds: int = 5,
                       random_state: int = 42) -> dict[str, Any]:
    """Stratified k-fold accuracy for the *unfitted* estimator.

    Accuracy is invariant under a relabelling, so this is run on whatever
    encoding the estimator was built for — no decoding needed.
    """
    scores = cross_val_score(
        estimator, X, y,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state),
        scoring="accuracy", n_jobs=1,
    )
    return {
        "cv_accuracy_mean": round(float(scores.mean()), 4),
        "cv_accuracy_std": round(float(scores.std()), 4),
        "cv_folds": folds,
    }


def quality_warnings(metrics: dict[str, Any]) -> list[str]:
    """Caveats that must travel with the model rather than sit in a report."""
    warnings: list[str] = []
    if metrics["lift_over_baseline"] <= 0.01:
        warnings.append(
            f"Accuracy ({metrics['accuracy']}) does not meaningfully exceed the "
            f"majority-class baseline ({metrics['baseline_accuracy']}). This model "
            "has learned nothing and must not be used to action real accounts."
        )
    if metrics["roc_auc"] < 0.6:
        warnings.append(
            f"ROC-AUC is {metrics['roc_auc']} — barely better than the 0.5 of a coin "
            "toss. The ranking this model produces is not informative."
        )
    if metrics["recall"] < 0.5 and metrics["lift_over_baseline"] > 0.01:
        warnings.append(
            f"Recall on the positive class is {metrics['recall']}: the model misses "
            "more than half of the fake profiles it is shown."
        )
    return warnings


def format_report(metrics: dict[str, Any]) -> str:
    """Human-readable block for stdout and the markdown report."""
    cm = metrics["confusion_matrix"]
    neg, pos = cm["labels"]
    lines = [
        f"  accuracy            {metrics['accuracy']:.4f}   (baseline {metrics['baseline_accuracy']:.4f},"
        f" lift {metrics['lift_over_baseline']:+.4f})",
        f"  precision           {metrics['precision']:.4f}",
        f"  recall              {metrics['recall']:.4f}",
        f"  f1                  {metrics['f1']:.4f}",
        f"  specificity         {metrics['specificity']:.4f}",
        f"  roc_auc             {metrics['roc_auc']:.4f}",
        f"  pr_auc              {metrics['pr_auc']:.4f}",
    ]
    if "cv_accuracy_mean" in metrics:
        lines.append(
            f"  cv accuracy         {metrics['cv_accuracy_mean']:.4f} "
            f"+/- {metrics['cv_accuracy_std']:.4f}  ({metrics['cv_folds']}-fold)"
        )
    lines += [
        "",
        f"  confusion matrix (rows = actual, columns = predicted)",
        f"                 pred {neg:<8s} pred {pos}",
        f"    actual {neg:<8s} {cm['true_negative']:<13,} {cm['false_positive']:,}",
        f"    actual {pos:<8s} {cm['false_negative']:<13,} {cm['true_positive']:,}",
    ]
    return "\n".join(lines)
