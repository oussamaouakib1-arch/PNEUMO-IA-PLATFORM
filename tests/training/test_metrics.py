import pytest

from pneumonia.training.metrics import binary_metrics


def test_binary_metrics_prioritize_pneumonia_as_positive_class() -> None:
    metrics = binary_metrics(
        targets=[0, 0, 1, 1],
        predictions=[0, 1, 1, 0],
        probabilities=[0.1, 0.7, 0.8, 0.4],
    )

    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 1
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert "roc_auc" in metrics
