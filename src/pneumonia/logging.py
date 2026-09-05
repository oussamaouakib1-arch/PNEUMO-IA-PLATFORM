"""Application logging helpers."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure a consistent log format for local services and jobs."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
