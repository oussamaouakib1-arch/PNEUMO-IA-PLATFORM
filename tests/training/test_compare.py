from pathlib import Path
from typing import Any

from pneumonia.training import compare


def test_comparison_sorts_by_recall_and_writes_report(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    recalls = {"baseline": 0.8, "resnet50": 0.9}

    def fake_train(config: Any) -> dict[str, Any]:
        recall = recalls[config.model_name]
        return {
            "checkpoint": str(config.output_dir / "best_model.pt"),
            "test_metrics": {
                "recall": recall,
                "specificity": 0.7,
                "f1": recall - 0.1,
                "roc_auc": 0.9,
                "pr_auc": 0.9,
                "false_negative": 2,
            },
        }

    monkeypatch.setattr(compare, "train_baseline", fake_train)
    results = compare.compare_models(
        dataset_root=tmp_path,
        manifest_path=tmp_path / "manifest.csv",
        model_names=["baseline", "resnet50"],
        output_root=tmp_path / "models",
        epochs=1,
        batch_size=2,
        image_size=32,
        max_batches=1,
        pretrained=False,
        freeze_backbone=True,
        use_mlflow=False,
    )

    assert [result["model"] for result in results] == ["resnet50", "baseline"]
    assert (tmp_path / "models" / "comparison.json").exists()
