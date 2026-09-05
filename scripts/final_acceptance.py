"""Generate a machine-readable final acceptance report for the project."""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REQUIRED_PATHS = (
    Path("models/baseline/best_model.pt"),
    Path("artifacts/evaluation/baseline/evaluation.json"),
    Path("artifacts/phase3/data_quality_report.json"),
    Path("data/splits/manifest.csv"),
    Path("Dockerfile"),
    Path("compose.yaml"),
    Path("infra/k8s/base/kustomization.yaml"),
    Path("infra/monitoring/prometheus.yml"),
    Path("infra/monitoring/grafana/dashboards/pneumonia-api.json"),
)


def command_check(name: str, command: list[str]) -> dict[str, Any]:
    """Run one read-only validation command."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return {"name": name, "passed": False, "detail": str(exc)}
    output = (result.stdout or result.stderr).strip()
    passed = result.returncode == 0
    return {
        "name": name,
        "passed": passed,
        "detail": "ok" if passed else output[-500:],
    }


def build_report() -> dict[str, Any]:
    """Collect artifacts, quality gates and infrastructure checks."""
    files = {str(path): path.is_file() for path in REQUIRED_PATHS}
    evaluation_path = Path("artifacts/evaluation/baseline/evaluation.json")
    quality_path = Path("artifacts/phase3/data_quality_report.json")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    data_quality = json.loads(quality_path.read_text(encoding="utf-8"))
    metrics = evaluation["test_metrics"]
    gates = {
        "accuracy_at_least_0_85": metrics["accuracy"] >= 0.85,
        "recall_at_least_0_90": metrics["recall"] >= 0.90,
        "f1_at_least_0_85": metrics["f1"] >= 0.85,
        "roc_auc_at_least_0_90": metrics["roc_auc"] >= 0.90,
        "no_rejected_images": data_quality["images_rejected"] == 0,
        "all_required_files_present": all(files.values()),
    }
    infrastructure = [
        command_check(
            "docker_compose",
            [
                "docker",
                "compose",
                "--profile",
                "application",
                "--profile",
                "monitoring",
                "config",
                "--quiet",
            ],
        ),
        command_check("kubernetes_kustomize", ["kubectl", "kustomize", "infra/k8s/base"]),
    ]
    passed = all(gates.values()) and all(check["passed"] for check in infrastructure)
    return {
        "status": "accepted" if passed else "failed",
        "model": evaluation["model"],
        "decision_threshold": evaluation["threshold_selection"]["threshold"],
        "test_metrics": metrics,
        "data_quality": data_quality,
        "quality_gates": gates,
        "required_files": files,
        "infrastructure_checks": infrastructure,
        "declared_limitations": [
            "Academic prototype; not a certified medical device.",
            "No clinical tabular data or NLP because the supplied dataset contains images only.",
            "External clinical validation and prospective evaluation remain required.",
        ],
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/final/acceptance.json"),
    )
    args = parser.parse_args()

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "accepted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
