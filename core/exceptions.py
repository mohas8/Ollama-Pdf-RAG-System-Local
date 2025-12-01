#!/usr/bin/env python3
"""
Exception classes for BanglaRAG system.
Defines custom exceptions for better error handling.
"""


class BanglaRAGException(Exception):
    """Base exception for BanglaRAG system."""

    pass


class DatabaseException(BanglaRAGException):
    """Exception raised for database-related errors."""

    pass


class EmbeddingException(BanglaRAGException):
    """Exception raised for embedding-related errors."""

    pass


class TranslationException(BanglaRAGException):
    """Exception raised for translation-related errors."""

    pass


class ModelException(BanglaRAGException):
    """Exception raised for model-related errors."""

    pass


class AudioException(BanglaRAGException):
    """Exception raised for audio processing errors."""

    pass


class FileProcessingException(BanglaRAGException):
    """Exception raised for file processing errors."""

    pass


class NetworkException(BanglaRAGException):
    """Exception raised for network-related errors."""

    pass


class ConfigurationException(BanglaRAGException):
    """Exception raised for configuration errors."""

    pass


class ValidationException(BanglaRAGException):
    """Exception raised for input validation errors."""

    pass
