"""Medical classification metrics with pneumonia as the positive class."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    targets: list[int], predictions: list[int], probabilities: list[float]
) -> dict[str, Any]:
    """Calculate metrics with explicit false-negative and specificity reporting."""
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    if len(np.unique(targets)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(targets, probabilities))
        metrics["pr_auc"] = float(average_precision_score(targets, probabilities))
    return metrics
