"""
Logging Utilities
=================
Provides a standardised logger that writes to both the console and a rotating
log file, with configurable verbosity levels.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def get_logger(
    name: str = "amsw",
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create or retrieve a named logger.

    If *log_dir* is specified, log records are also written to
    ``<log_dir>/<name>_<timestamp>.log``.

    Args:
        name:    Logger name (default "amsw").
        log_dir: Optional directory for the log file.
        level:   Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — return as-is
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            Path(log_dir) / f"{name}_{timestamp}.log",
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
