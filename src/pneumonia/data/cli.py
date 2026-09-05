"""Command-line entry point for the phase 3 data pipeline."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from pneumonia.data.catalog import build_catalog
from pneumonia.data.io import write_catalog, write_json, write_split_manifest
from pneumonia.data.report import build_report
from pneumonia.data.split import (
    assert_no_patient_leakage,
    assign_grouped_splits,
    duplicate_hashes,
)
from pneumonia.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the pipeline CLI parser."""
    parser = argparse.ArgumentParser(description="Validate and split chest X-ray data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/chest_xray"),
        help="Dataset root containing train/val/test and NORMAL/PNEUMONIA folders.",
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/validated/catalog.csv"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/manifest.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/phase3/data_quality_report.json"),
    )
    parser.add_argument(
        "--duplicates",
        type=Path,
        default=Path("artifacts/phase3/duplicates.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Execute cataloging, validation, deduplication, and grouped splitting."""
    records = build_catalog(args.input)
    if not records:
        raise RuntimeError(f"No supported images found under {args.input}")

    assignments = assign_grouped_splits(records, seed=args.seed)
    assert_no_patient_leakage(assignments)
    report = build_report(records, assignments)

    write_catalog(records, args.catalog)
    write_split_manifest(records, assignments, args.manifest)
    write_json(duplicate_hashes(records), args.duplicates)
    write_json(report, args.report)
    LOGGER.info("Phase 3 report: %s", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line pipeline."""
    configure_logging()
    args = build_parser().parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
