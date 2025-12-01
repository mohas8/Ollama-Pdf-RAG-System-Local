#!/usr/bin/env python3
"""
Refactored embedding module for BanglaRAG system.
Provides factory pattern for embedding models with proper error handling and caching.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import numpy as np
import re
import warnings
from functools import lru_cache

from langchain_ollama import OllamaEmbeddings
from langdetect import detect
from transformers import AutoTokenizer, AutoModel
import torch

from core.logging_config import BanglaRAGLogger
from core.exceptions import EmbeddingException, ModelException
from core.utils import retry_with_backoff, measure_performance, SimpleCache
from core.constants import (
    ENGLISH_EMBEDDING_MODEL,
    BANGLA_EMBEDDING_MODEL,
    ENGLISH_CODE,
    BANGLA_CODE,
    MIN_TEXT_LENGTH_FOR_DETECTION,
)

logger = BanglaRAGLogger.get_logger("embedding")
warnings.filterwarnings("ignore")


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for given text."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass

    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """Check if model supports given language."""
        pass


class OllamaEmbeddingModel(EmbeddingModel):
    """Ollama-based embedding model for English text."""

    def __init__(self, model_name: str = ENGLISH_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: Optional[OllamaEmbeddings] = None
        self._dimension: Optional[int] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the Ollama embedding model."""
        try:
            self._model = OllamaEmbeddings(model=self.model_name)
            # Test the model with a sample query to ensure it works
            test_embedding = self._model.embed_query("test")
            self._dimension = len(test_embedding)
            logger.info(
                f"Initialized Ollama model: {self.model_name} (dim: {self._dimension})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama model {self.model_name}: {e}")
            raise ModelException(f"Failed to initialize Ollama model: {e}")

    @retry_with_backoff(max_retries=3)
    @measure_performance
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding using Ollama model."""
        if not self._model:
            raise EmbeddingException("Ollama model not initialized")

        try:
            # Preprocess for better technical term matching
            processed_text = self._preprocess_technical_query(text)
            embedding = self._model.embed_query(processed_text)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Failed to generate Ollama embedding: {e}")
            raise EmbeddingException(f"Ollama embedding failed: {e}")

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            raise EmbeddingException("Model not properly initialized")
        return self._dimension

    def supports_language(self, language: str) -> bool:
        """Check if model supports given language."""
        return language == ENGLISH_CODE

    def _preprocess_technical_query(self, text: str) -> str:
        """Preprocess English queries for better technical term matching."""
        # Expand common abbreviations
        abbreviations = {
            "BST": "Binary Search Tree",
            "DP": "Dynamic Programming",
            "DFS": "Depth First Search",
            "BFS": "Breadth First Search",
            "AVL": "Adelson-Velsky and Landis Tree",
            "MST": "Minimum Spanning Tree",
            "LCS": "Longest Common Subsequence",
        }

        for abbr, full in abbreviations.items():
            pattern = r"\b" + re.escape(abbr) + r"\b"
            text = re.sub(pattern, full, text, flags=re.IGNORECASE)

        # Normalize technical terms
        normalizations = {
            r"\balgos?\b": "algorithm",
            r"\bstruct\b": "structure",
            r"\bfuncs?\b": "function",
            r"\bcomplexity\b": "time complexity space complexity",
        }

        for pattern, replacement in normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text


class BanglaBERTEmbeddingModel(EmbeddingModel):
    """BanglaBERT-based embedding model for Bangla text."""

    def __init__(self, model_name: str = BANGLA_EMBEDDING_MODEL):
        self.model_name = model_name
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        self._dimension = 768  # BanglaBERT dimension
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the BanglaBERT model."""
        try:
            logger.info(f"Loading BanglaBERT model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            logger.info("BanglaBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BanglaBERT model: {e}")
            raise ModelException(f"Failed to initialize BanglaBERT model: {e}")

    @retry_with_backoff(max_retries=2)
    @measure_performance
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding using BanglaBERT."""
        if not self._tokenizer or not self._model:
            raise EmbeddingException("BanglaBERT model not initialized")

        try:
            # Tokenize input text
            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            )

            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use mean pooling of the last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()

            return embeddings.numpy()

        except Exception as e:
            logger.error(f"Failed to generate BanglaBERT embedding: {e}")
            raise EmbeddingException(f"BanglaBERT embedding failed: {e}")

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def supports_language(self, language: str) -> bool:
        """Check if model supports given language."""
        return language == BANGLA_CODE


class LanguageDetector:
    """Language detection utility with caching."""

    def __init__(self):
        self._cache = SimpleCache(max_size=1000, ttl_seconds=3600)

    @lru_cache(maxsize=100)
    def detect_language(self, text: str) -> str:
        """
        Detect language of given text with caching.

        Args:
            text: Input text to analyze

        Returns:
            Language code ('en' for English, 'bn' for Bangla)
        """
        # Check cache first
        cache_key = f"lang:{hash(text)}"
        cached_result = self._cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            # Clean and validate text
            cleaned_text = " ".join(text.split())

            if len(cleaned_text) < MIN_TEXT_LENGTH_FOR_DETECTION:
                return ENGLISH_CODE  # Default to English for short texts

            # Detect language
            detected_lang = detect(cleaned_text)

            # Map to supported languages
            if detected_lang == "bn":
                result = BANGLA_CODE
            else:
                result = ENGLISH_CODE

            # Cache the result
            self._cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return ENGLISH_CODE  # Default to English on error


class EmbeddingFactory:
    """Factory for creating and managing embedding models."""

    def __init__(self):
        self._models: Dict[str, EmbeddingModel] = {}
        self._language_detector = LanguageDetector()
        self._embedding_cache = SimpleCache(max_size=500, ttl_seconds=1800)

    def get_model(self, language: str) -> EmbeddingModel:
        """Get embedding model for specified language."""
        if language not in self._models:
            if language == ENGLISH_CODE:
                try:
                    self._models[language] = OllamaEmbeddingModel()
                except Exception as e:
                    logger.error(f"Failed to create English embedding model: {e}")
                    raise ModelException(
                        f"English embedding model creation failed: {e}"
                    )

            elif language == BANGLA_CODE:
                try:
                    self._models[language] = BanglaBERTEmbeddingModel()
                except Exception as e:
                    logger.warning(
                        f"BanglaBERT not available, falling back to English model: {e}"
                    )
                    # Fallback to English model for Bangla
                    if ENGLISH_CODE not in self._models:
                        self._models[ENGLISH_CODE] = OllamaEmbeddingModel()
                    self._models[language] = self._models[ENGLISH_CODE]

            else:
                logger.warning(f"Unsupported language: {language}, using English model")
                if ENGLISH_CODE not in self._models:
                    self._models[ENGLISH_CODE] = OllamaEmbeddingModel()
                self._models[language] = self._models[ENGLISH_CODE]

        return self._models[language]

    def get_mixed_language_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using appropriate language model.

        Args:
            text: Input text in any supported language

        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        cache_key = f"embed:{hash(text)}"
        cached_embedding = self._embedding_cache.get(cache_key)
        if cached_embedding is not None:
            logger.debug("Using cached embedding")
            return cached_embedding

        # Detect language
        language = self._language_detector.detect_language(text)
        logger.debug(f"Detected language: {language} for text: {text[:50]}...")

        # Get appropriate model and generate embedding
        model = self.get_model(language)
        embedding = model.embed_text(text)

        # Cache the result
        self._embedding_cache.set(cache_key, embedding)

        return embedding

    def get_embedding_function_with_fallback(self) -> OllamaEmbeddings:
        """
        Get Ollama embedding function with fallback models.
        This maintains compatibility with existing code.
        """
        fallback_models = [
            "nomic-embed-text",
            "mxbai-embed-large",
            "all-minilm",
            "llama2",
        ]

        for model_name in fallback_models:
            try:
                embedding_function = OllamaEmbeddings(model=model_name)
                # Test the model
                embedding_function.embed_query("test")
                logger.info(f"Using Ollama model: {model_name}")
                return embedding_function
            except Exception as e:
                logger.warning(f"Model {model_name} not available: {e}")
                continue

        raise ModelException("No working Ollama embedding models found")


# Global factory instance
_embedding_factory: Optional[EmbeddingFactory] = None


def get_embedding_factory() -> EmbeddingFactory:
    """Get global embedding factory instance."""
    global _embedding_factory
    if _embedding_factory is None:
        _embedding_factory = EmbeddingFactory()
    return _embedding_factory


# Compatibility functions for existing code
def get_embedding_function_with_fallback() -> OllamaEmbeddings:
    """Compatibility function for existing code."""
    return get_embedding_factory().get_embedding_function_with_fallback()


def get_mixed_language_embedding(text: str) -> np.ndarray:
    """Compatibility function for mixed language embedding."""
    return get_embedding_factory().get_mixed_language_embedding(text)


def detect_language(text: str) -> str:
    """Compatibility function for language detection."""
    return get_embedding_factory()._language_detector.detect_language(text)


def embed_english(text: str) -> np.ndarray:
    """Compatibility function for English embedding."""
    factory = get_embedding_factory()
    model = factory.get_model(ENGLISH_CODE)
    return model.embed_text(text)


def embed_bangla(text: str) -> np.ndarray:
    """Compatibility function for Bangla embedding."""
    factory = get_embedding_factory()
    model = factory.get_model(BANGLA_CODE)
    return model.embed_text(text)


# Test functions
def test_embeddings() -> bool:
    """Test the embedding functionality."""
    try:
        logger.info("Testing embedding system...")
        factory = get_embedding_factory()

        # Test English embedding
        english_text = "This is a test sentence about algorithms."
        english_embedding = factory.get_mixed_language_embedding(english_text)
        logger.info(f"English embedding: dimension={len(english_embedding)}")

        # Test Bangla embedding (may fallback to English)
        bangla_text = "এটি একটি পরীক্ষার বাক্য।"
        bangla_embedding = factory.get_mixed_language_embedding(bangla_text)
        logger.info(f"Bangla embedding: dimension={len(bangla_embedding)}")

        logger.info("Embedding system test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Embedding system test failed: {e}")
        return False


if __name__ == "__main__":
    test_embeddings()
