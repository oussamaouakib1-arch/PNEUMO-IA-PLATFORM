"""Probability calibration and clinically oriented threshold selection."""

from itertools import pairwise
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, recall_score


def calibration_metrics(
    targets: list[int],
    probabilities: list[float],
    bins: int = 10,
) -> dict[str, float]:
    """Calculate Brier score and expected calibration error."""
    target_array = np.asarray(targets)
    probability_array = np.asarray(probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    expected_calibration_error = 0.0
    for lower, upper in pairwise(edges):
        inclusive_upper = upper == 1.0
        mask = (probability_array >= lower) & (
            probability_array <= upper if inclusive_upper else probability_array < upper
        )
        if mask.any():
            confidence = float(probability_array[mask].mean())
            observed = float(target_array[mask].mean())
            expected_calibration_error += float(mask.mean()) * abs(confidence - observed)
    return {
        "brier_score": float(brier_score_loss(targets, probabilities)),
        "expected_calibration_error": expected_calibration_error,
    }


def select_threshold(
    targets: list[int],
    probabilities: list[float],
    minimum_recall: float = 0.90,
) -> dict[str, Any]:
    """Choose the highest-F1 threshold satisfying a minimum pneumonia recall."""
    candidates = np.linspace(0.05, 0.95, 91)
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predictions = (np.asarray(probabilities) >= threshold).astype(int)
        recall = float(recall_score(targets, predictions, zero_division=0))
        f1 = float(f1_score(targets, predictions, zero_division=0))
        if recall >= minimum_recall:
            feasible.append((f1, float(threshold), recall))
    if not feasible:
        return {"threshold": 0.5, "recall": 0.0, "f1": 0.0, "constraint_met": False}
    f1, selected_threshold, recall = max(feasible, key=lambda item: (item[0], item[1]))
    return {
        "threshold": float(selected_threshold),
        "recall": recall,
        "f1": f1,
        "constraint_met": True,
    }
