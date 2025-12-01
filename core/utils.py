#!/usr/bin/env python3
"""
Utility functions for BanglaRAG system.
Common helper functions used across multiple modules.
"""

import os
import json
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import threading

from core.logging_config import BanglaRAGLogger, PerformanceTracker
from core.exceptions import ValidationException, FileProcessingException

logger = BanglaRAGLogger.get_logger("utils")


def validate_not_empty(value: Any, name: str) -> Any:
    """Validate that a value is not None or empty."""
    if value is None or (isinstance(value, (str, list, dict)) and len(value) == 0):
        raise ValidationException(f"{name} cannot be None or empty")
    return value


def validate_file_exists(file_path: Union[str, Path]) -> Path:
    """Validate that a file exists and return Path object."""
    path = Path(file_path)
    if not path.exists():
        raise FileProcessingException(f"File does not exist: {file_path}")
    if not path.is_file():
        raise FileProcessingException(f"Path is not a file: {file_path}")
    return path


def validate_directory_exists(dir_path: Union[str, Path]) -> Path:
    """Validate that a directory exists and return Path object."""
    path = Path(dir_path)
    if not path.exists():
        raise FileProcessingException(f"Directory does not exist: {dir_path}")
    if not path.is_dir():
        raise FileProcessingException(f"Path is not a directory: {dir_path}")
    return path


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_load(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Safely load JSON file with error handling."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON file {file_path}: {e}")
        return None


def safe_json_save(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """Safely save data to JSON file with error handling."""
    try:
        ensure_directory(Path(file_path).parent)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        return False


def get_file_hash(file_path: Union[str, Path]) -> str:
    """Get MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash file {file_path}: {e}")
        return ""


def get_text_hash(text: str) -> str:
    """Get MD5 hash of text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def clean_filename(filename: str) -> str:
    """Clean filename by removing/replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename.strip()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable string."""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f}{size_names[i]}"


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0):
    """Decorator for retrying functions with exponential backoff."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = backoff_factor * (2**attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {sleep_time:.1f}s..."
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}"
                        )

            raise last_exception

        return wrapper

    return decorator


def timeout_handler(timeout_seconds: int):
    """Decorator to add timeout to functions."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                logger.error(
                    f"Function {func.__name__} timed out after {timeout_seconds}s"
                )
                raise TimeoutError(f"Function {func.__name__} timed out")

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator


def measure_performance(func: Callable):
    """Decorator to measure and log function performance."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with PerformanceTracker(func.__name__):
            return func(*args, **kwargs)

    return wrapper


class SimpleCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None

            # Check if expired
            if time.time() - self._timestamps[key] > self.ttl_seconds:
                del self._cache[key]
                del self._timestamps[key]
                return None

            return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self._lock:
            # Remove oldest item if cache is full
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = min(self._timestamps.keys(), key=self._timestamps.get)
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]

            self._cache[key] = value
            self._timestamps[key] = time.time()

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


def create_temp_file(suffix: str = "", prefix: str = "banglarag_") -> str:
    """Create a temporary file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)  # Close the file descriptor
    return path


def find_pdf_files(directory: Union[str, Path]) -> List[Path]:
    """Find all PDF files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        return []

    pdf_files = []
    for file_path in directory.glob("*.pdf"):
        if file_path.is_file():
            pdf_files.append(file_path)

    return sorted(pdf_files)


def sanitize_text(text: str) -> str:
    """Sanitize text by removing extra whitespace and special characters."""
    if not text:
        return ""

    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove control characters but keep newlines
    sanitized = "".join(char for char in text if ord(char) >= 32 or char in "\n\r\t")

    return sanitized.strip()


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries, with later ones taking precedence."""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


class ProgressBar:
    """Simple console progress bar."""

    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.width = width
        self.current = 0

    def update(self, increment: int = 1) -> None:
        """Update progress bar."""
        self.current += increment
        self._display()

    def _display(self) -> None:
        """Display the progress bar."""
        if self.total == 0:
            return

        progress = self.current / self.total
        filled = int(self.width * progress)
        bar = "█" * filled + "░" * (self.width - filled)
        percent = progress * 100

        print(
            f"\r|{bar}| {percent:.1f}% ({self.current}/{self.total})",
            end="",
            flush=True,
        )

        if self.current >= self.total:
            print()  # New line when complete
