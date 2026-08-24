import logging
import sys
from pathlib import Path

from src.config import REPORTS_LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger configured with both a StreamHandler (console) and
    a FileHandler (reports/logs/pipeline.log).

    The logger is set to INFO level. Format includes timestamp, logger name,
    and message.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers if already configured (prevent duplicate logs
    # when imported multiple times)
    if logger.handlers:
        return logger

    # Ensure the log directory exists
    REPORTS_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler
    log_file = REPORTS_LOGS_DIR / "pipeline.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Formatter with timestamp
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger