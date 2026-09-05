"""Prometheus metrics isolated per FastAPI application instance."""

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


class ApiMetrics:
    """Collect bounded HTTP and inference metrics for one application."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "pneumonia_http_requests_total",
            "Total HTTP requests handled by the API.",
            ("method", "path", "status"),
            registry=self.registry,
        )
        self.latency = Histogram(
            "pneumonia_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            ("method", "path"),
            registry=self.registry,
        )
        self.predictions = Counter(
            "pneumonia_predictions_total",
            "Predictions returned by model and label.",
            ("model", "prediction"),
            registry=self.registry,
        )

    def render(self) -> bytes:
        """Render metrics using the Prometheus text exposition format."""
        return generate_latest(self.registry)
