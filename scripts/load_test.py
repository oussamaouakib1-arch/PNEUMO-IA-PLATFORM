"""Run a small concurrent inference load test against a deployed API."""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import requests


def percentile(values: list[float], quantile: float) -> float:
    """Return a nearest-rank percentile for a non-empty list."""
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def run_request(api_url: str, image_name: str, content: bytes, timeout: float) -> float:
    """Send one prediction and return its latency in seconds."""
    started_at = perf_counter()
    response = requests.post(
        f"{api_url}/predict",
        params={"include_gradcam": "false"},
        files={"file": (image_name, content, "image/jpeg")},
        timeout=timeout,
    )
    response.raise_for_status()
    return perf_counter() - started_at


def load_test(
    api_url: str,
    image_path: Path,
    request_count: int,
    concurrency: int,
    timeout: float,
) -> dict[str, Any]:
    """Run requests concurrently and summarize reliability and latency."""
    content = image_path.read_bytes()
    latencies: list[float] = []
    errors: list[str] = []
    started_at = perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_request, api_url, image_path.name, content, timeout)
            for _ in range(request_count)
        ]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except requests.RequestException as exc:
                errors.append(str(exc))

    duration = perf_counter() - started_at
    return {
        "requests": request_count,
        "successes": len(latencies),
        "errors": len(errors),
        "concurrency": concurrency,
        "duration_seconds": duration,
        "throughput_requests_per_second": len(latencies) / duration,
        "latency_seconds": {
            "minimum": min(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50) if latencies else None,
            "p95": percentile(latencies, 0.95) if latencies else None,
            "maximum": max(latencies) if latencies else None,
        },
        "error_details": errors[:5],
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive.")

    result = load_test(
        args.api_url.rstrip("/"),
        args.image,
        args.requests,
        min(args.concurrency, args.requests),
        args.timeout,
    )
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
