"""Logging configuration using Loguru.

Provides a centralized logger with both console and file sinks.
Every processing step logs record counts, timing, and errors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger as _logger


def setup_logger(
    level: str = "INFO",
    log_file: str | None = "outputs/pipeline.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure the global Loguru logger.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file. None disables file logging.
        rotation: Log rotation size or time.
        retention: How long to keep old log files.
    """
    _logger.remove()

    _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            str(log_path),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )


def get_logger(**kwargs: Any) -> Any:
    """Return the configured Loguru logger."""
    return _logger
