#!/usr/bin/env python3
"""
Constants and configuration values for BanglaRAG system.
Centralized location for all magic numbers, URLs, and configuration constants.
"""

from typing import List, Dict, Any
from pathlib import Path
import os

# ============================================================================
# SYSTEM CONSTANTS
# ============================================================================

# Application Info
APP_NAME = "BanglaRAG System"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Mixed-Language RAG with Voice Input Support"

# File Extensions
PDF_EXTENSION = ".pdf"
JSON_EXTENSION = ".json"
PICKLE_EXTENSION = ".pickle"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Preferred Models
PREFERRED_LLM_MODEL = "mistral:7b"
FALLBACK_LLM_MODELS = [
    "mistral:7b",
    "llama3.2:latest",
    "qwen2:1.5b",
    "phi3:latest",
    "mistral",
    "llama2",
]

# Embedding Models
ENGLISH_EMBEDDING_MODEL = "nomic-embed-text:latest"
BANGLA_EMBEDDING_MODEL = "sagorsarker/bangla-bert-base"

# Whisper Models
DEFAULT_WHISPER_MODEL = "base"
AVAILABLE_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

# BanglaSpeech2Text Models
BANGLA_STT_MODELS = {
    "tiny": {"size": "100-200MB", "wer": 74},
    "base": {"size": "200-300MB", "wer": 46},
    "small": {"size": "1GB", "wer": 18},
    "large": {"size": "3-4GB", "wer": 11},
}

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# LLM Settings
MAX_TOKENS = 180
TEMPERATURE = 0.1
TIMEOUT_SECONDS = 25

# Chunk Settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
ENGLISH_CHUNK_SIZE = 1000
BANGLA_CHUNK_SIZE = 800

# Retrieval Settings
DEFAULT_RETRIEVAL_COUNT = 3
MAX_RETRIEVAL_COUNT = 10

# Audio Settings
AUDIO_CHUNK_SIZE = 1024
AUDIO_FORMAT = "paInt16"  # pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 44100
DEFAULT_RECORDING_DURATION = 5

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Database Paths
DATABASE_DIRECTORY = "db"
CHROMA_DB_NAME = "banglarag_db"

# Cache Settings
ENABLE_CACHING = True
CACHE_SIZE = 100
CACHE_TTL_SECONDS = 3600

# ============================================================================
# NETWORK CONFIGURATION
# ============================================================================

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_TIMEOUT = 30
OLLAMA_MAX_RETRIES = 3

# Translation Service
TRANSLATION_SERVICE_TIMEOUT = 10

# ============================================================================
# LANGUAGE SETTINGS
# ============================================================================

# Language Codes
ENGLISH_CODE = "en"
BANGLA_CODE = "bn"
SUPPORTED_LANGUAGES = [ENGLISH_CODE, BANGLA_CODE]

# Language Detection
MIN_TEXT_LENGTH_FOR_DETECTION = 50
LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD = 0.8

# Translation Settings
SKIP_TRANSLATION_FOR_ENGLISH = True
CACHE_TRANSLATIONS = True

# ============================================================================
# FILE PATHS AND DIRECTORIES
# ============================================================================

# Project Structure
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_DIR = ROOT_DIR / DATABASE_DIRECTORY
LOGS_DIR = ROOT_DIR / "logs"
TEMP_DIR = ROOT_DIR / "temp"
TEST_REPORTS_DIR = ROOT_DIR / "Test Reports"

# Log Files
MAIN_LOG_FILE = "banglarag.log"
ERROR_LOG_FILE = "banglarag_errors.log"
PERFORMANCE_LOG_FILE = "banglarag_performance.log"

# ============================================================================
# UI CONSTANTS
# ============================================================================

# Menu Options
MAIN_MENU_OPTIONS = {
    1: "Show System Status",
    2: "Process Documents (Create/Update Database)",
    3: "Test Database Queries",
    4: "Interactive Chat Session",
    5: "Voice Input Session",
    6: "Single Voice Query",
    7: "Mixed Language Demo",
    8: "Run System Tests",
    9: "Voice Input Tests",
    10: "Check Dependencies",
    11: "Exit",
}

# Display Settings
BANNER_WIDTH = 60
PROGRESS_BAR_WIDTH = 50

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    "database_not_found": "Vector database not found. Please create database first.",
    "pdf_not_found": "No PDF files found in current directory.",
    "ollama_connection_failed": "Failed to connect to Ollama service.",
    "embedding_failed": "Failed to generate embeddings.",
    "translation_failed": "Translation service unavailable.",
    "audio_device_error": "Audio device not available or permission denied.",
    "model_loading_failed": "Failed to load the specified model.",
    "invalid_input": "Invalid input provided.",
    "file_processing_error": "Error processing file.",
    "network_timeout": "Network request timed out.",
}

# Success Messages
SUCCESS_MESSAGES = {
    "database_created": "Vector database created successfully.",
    "document_processed": "Documents processed successfully.",
    "query_completed": "Query completed successfully.",
    "translation_completed": "Translation completed successfully.",
    "audio_recorded": "Audio recorded successfully.",
    "model_loaded": "Model loaded successfully.",
}

# ============================================================================
# REGEX PATTERNS
# ============================================================================

# Common patterns for text processing
PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "url": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
    "bangla": r"[\u0980-\u09FF]+",
    "english": r"[a-zA-Z]+",
}

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# Test Settings
TEST_TIMEOUT = 30
MAX_TEST_RETRIES = 3
TEST_DATA_SAMPLE_SIZE = 100

# Test Categories
TEST_CATEGORIES = [
    "core_functionality",
    "language_processing",
    "voice_input",
    "database_operations",
    "performance",
    "error_handling",
    "integration",
]

# ============================================================================
# GRADIO/UI CONFIGURATION
# ============================================================================

# Gradio Settings (if using web interface)
GRADIO_PORT = 7860
GRADIO_HOST = "localhost"
GRADIO_SHARE = False

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log Levels
LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

# Log Format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# PERFORMANCE THRESHOLDS
# ============================================================================

# Performance Targets
TARGET_RESPONSE_TIME = 10.0  # seconds
TARGET_SUCCESS_RATE = 0.80  # 80%
TARGET_CACHE_HIT_RATE = 0.30  # 30%

# Memory Limits
MAX_MEMORY_USAGE_MB = 2048
MAX_CHUNK_MEMORY_MB = 512

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_project_root() -> Path:
    """Get the project root directory."""
    return ROOT_DIR


def ensure_directory_exists(directory: Path) -> None:
    """Ensure a directory exists, create if it doesn't."""
    directory.mkdir(parents=True, exist_ok=True)


def get_available_models_config() -> Dict[str, Any]:
    """Get configuration for available models."""
    return {
        "llm": {"preferred": PREFERRED_LLM_MODEL, "fallback": FALLBACK_LLM_MODELS},
        "embedding": {
            "english": ENGLISH_EMBEDDING_MODEL,
            "bangla": BANGLA_EMBEDDING_MODEL,
        },
        "whisper": {
            "default": DEFAULT_WHISPER_MODEL,
            "available": AVAILABLE_WHISPER_MODELS,
        },
    }


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Create necessary directories on import
for directory in [LOGS_DIR, TEMP_DIR, DB_DIR]:
    ensure_directory_exists(directory)

# Export commonly used constants
__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "APP_DESCRIPTION",
    "PREFERRED_LLM_MODEL",
    "FALLBACK_LLM_MODELS",
    "MAX_TOKENS",
    "TEMPERATURE",
    "TIMEOUT_SECONDS",
    "DATABASE_DIRECTORY",
    "DEFAULT_RETRIEVAL_COUNT",
    "OLLAMA_BASE_URL",
    "ERROR_MESSAGES",
    "SUCCESS_MESSAGES",
    "get_project_root",
    "ensure_directory_exists",
    "get_available_models_config",
]
