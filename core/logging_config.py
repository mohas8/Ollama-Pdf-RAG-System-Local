#!/usr/bin/env python3
"""
Centralized logging configuration for BanglaRAG system.
Provides structured logging with different levels and file outputs.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from core.constants import (
    LOGS_DIR,
    MAIN_LOG_FILE,
    ERROR_LOG_FILE,
    PERFORMANCE_LOG_FILE,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVELS,
)


class BanglaRAGLogger:
    """Centralized logger for the BanglaRAG system."""

    _loggers = {}
    _initialized = False

    @classmethod
    def setup_logging(cls, log_level: str = "INFO") -> None:
        """Set up logging configuration for the entire application."""
        if cls._initialized:
            return

        # Ensure logs directory exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

        console_formatter = logging.Formatter(
            fmt="%(levelname)s - %(name)s - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Main log file handler (all logs)
        main_file_handler = logging.handlers.RotatingFileHandler(
            filename=LOGS_DIR / MAIN_LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        main_file_handler.setLevel(logging.DEBUG)
        main_file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(main_file_handler)

        # Error log file handler (errors only)
        error_file_handler = logging.handlers.RotatingFileHandler(
            filename=LOGS_DIR / ERROR_LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8",
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_file_handler)

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger for a specific module."""
        if not cls._initialized:
            cls.setup_logging()

        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger

        return cls._loggers[name]

    @classmethod
    def get_performance_logger(cls) -> logging.Logger:
        """Get a specialized logger for performance metrics."""
        if not cls._initialized:
            cls.setup_logging()

        perf_logger = logging.getLogger("performance")

        # Add performance file handler if not already added
        if not any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            and PERFORMANCE_LOG_FILE in str(h.baseFilename)
            for h in perf_logger.handlers
        ):

            perf_handler = logging.handlers.RotatingFileHandler(
                filename=LOGS_DIR / PERFORMANCE_LOG_FILE,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3,
                encoding="utf-8",
            )

            perf_formatter = logging.Formatter(
                fmt="%(asctime)s - PERF - %(message)s", datefmt=LOG_DATE_FORMAT
            )
            perf_handler.setFormatter(perf_formatter)
            perf_handler.setLevel(logging.INFO)
            perf_logger.addHandler(perf_handler)

        return perf_logger


class PerformanceTracker:
    """Track and log performance metrics."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.logger = BanglaRAGLogger.get_performance_logger()

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Started: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if exc_type is None:
                self.logger.info(
                    f"Completed: {self.operation_name} - Duration: {duration:.2f}s"
                )
            else:
                self.logger.error(
                    f"Failed: {self.operation_name} - Duration: {duration:.2f}s - Error: {exc_val}"
                )


def log_system_info():
    """Log system information at startup."""
    logger = BanglaRAGLogger.get_logger("system")

    import platform
    import sys

    logger.info("=" * 50)
    logger.info("BanglaRAG System Starting")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()[0]}")
    logger.info("=" * 50)


def suppress_third_party_logs():
    """Suppress verbose logs from third-party libraries."""
    # Suppress transformers warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    # Suppress langchain verbose logs
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langchain_community").setLevel(logging.WARNING)

    # Suppress chromadb logs
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    # Suppress requests logs
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# Convenience functions for common logging patterns
logger = BanglaRAGLogger.get_logger("banglarag")


def log_info(message: str, component: str = "system") -> None:
    """Log an info message."""
    BanglaRAGLogger.get_logger(component).info(message)


def log_warning(message: str, component: str = "system") -> None:
    """Log a warning message."""
    BanglaRAGLogger.get_logger(component).warning(message)


def log_error(message: str, component: str = "system", exc_info: bool = False) -> None:
    """Log an error message."""
    BanglaRAGLogger.get_logger(component).error(message, exc_info=exc_info)


def log_debug(message: str, component: str = "system") -> None:
    """Log a debug message."""
    BanglaRAGLogger.get_logger(component).debug(message)


def log_critical(message: str, component: str = "system") -> None:
    """Log a critical message."""
    BanglaRAGLogger.get_logger(component).critical(message)


# Initialize logging on import
BanglaRAGLogger.setup_logging()
suppress_third_party_logs()
