#!/usr/bin/env python3
"""
Core module for BanglaRAG system.
Contains shared utilities, constants, logging, and exceptions.
"""

from .constants import *
from .logging_config import (
    BanglaRAGLogger,
    PerformanceTracker,
    log_info,
    log_warning,
    log_error,
    log_debug,
)
from .exceptions import *
from .utils import *

__version__ = "2.0.0"
__all__ = [
    # Constants
    "APP_NAME",
    "APP_VERSION",
    "APP_DESCRIPTION",
    "PREFERRED_LLM_MODEL",
    "FALLBACK_LLM_MODELS",
    "ENGLISH_EMBEDDING_MODEL",
    "BANGLA_EMBEDDING_MODEL",
    "DEFAULT_WHISPER_MODEL",
    "BANGLA_STT_MODELS",
    "MAX_TOKENS",
    "TEMPERATURE",
    "TIMEOUT_SECONDS",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_RETRIEVAL_COUNT",
    "AUDIO_CHUNK_SIZE",
    "AUDIO_CHANNELS",
    "AUDIO_RATE",
    "DEFAULT_RECORDING_DURATION",
    "DATABASE_DIRECTORY",
    "CHROMA_DB_NAME",
    "ENABLE_CACHING",
    "OLLAMA_BASE_URL",
    "OLLAMA_API_TIMEOUT",
    "ENGLISH_CODE",
    "BANGLA_CODE",
    "SUPPORTED_LANGUAGES",
    "MIN_TEXT_LENGTH_FOR_DETECTION",
    "MAIN_MENU_OPTIONS",
    "BANNER_WIDTH",
    "ERROR_MESSAGES",
    "SUCCESS_MESSAGES",
    "get_project_root",
    "ensure_directory_exists",
    "get_available_models_config",
    # Logging
    "BanglaRAGLogger",
    "PerformanceTracker",
    "log_info",
    "log_warning",
    "log_error",
    "log_debug",
    "log_critical",
    # Exceptions
    "BanglaRAGException",
    "DatabaseException",
    "EmbeddingException",
    "TranslationException",
    "ModelException",
    "AudioException",
    "FileProcessingException",
    "NetworkException",
    "ConfigurationException",
    "ValidationException",
    # Utils
    "validate_not_empty",
    "validate_file_exists",
    "validate_directory_exists",
    "ensure_directory",
    "safe_json_load",
    "safe_json_save",
    "get_file_hash",
    "get_text_hash",
    "clean_filename",
    "truncate_text",
    "format_duration",
    "format_file_size",
    "retry_with_backoff",
    "timeout_handler",
    "measure_performance",
    "SimpleCache",
    "create_temp_file",
    "find_pdf_files",
    "sanitize_text",
    "chunk_list",
    "merge_dicts",
    "ProgressBar",
]
