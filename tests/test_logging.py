import logging

from pneumonia.logging import configure_logging


def test_configure_logging_sets_requested_level() -> None:
    configure_logging("DEBUG")

    assert logging.getLogger().level == logging.DEBUG
