from __future__ import annotations

import logging
from logging import Logger


def configure_logging(level: int = logging.INFO) -> Logger:
    """
    Configure root logger with a concise formatter suitable for CLI output.
    """
    logger = logging.getLogger("ocr_local_cli")
    if logger.handlers:
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_logger(name: str) -> Logger:
    parent = configure_logging()
    return parent.getChild(name)

