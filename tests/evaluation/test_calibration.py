import pytest

from pneumonia.evaluation.calibration import calibration_metrics, select_threshold


def test_calibration_metrics_are_bounded() -> None:
    metrics = calibration_metrics([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9], bins=4)

    assert 0 <= metrics["brier_score"] <= 1
    assert 0 <= metrics["expected_calibration_error"] <= 1


def test_threshold_selection_respects_minimum_recall() -> None:
    result = select_threshold(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.6, 0.9],
        minimum_recall=1.0,
    )

    assert result["constraint_met"] is True
    assert result["recall"] == pytest.approx(1.0)
    assert result["threshold"] <= 0.6
