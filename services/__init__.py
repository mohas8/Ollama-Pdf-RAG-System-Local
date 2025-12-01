#!/usr/bin/env python3
"""
Services module for BanglaRAG system.
Contains business logic and service layer components.
"""

from .embedding_service import (
    EmbeddingFactory,
    get_embedding_factory,
    get_mixed_language_embedding,
    detect_language,
    embed_english,
    embed_bangla,
    get_embedding_function_with_fallback,
)
from .database_service import (
    DatabaseManager,
    get_database_manager,
    load_database,
    query_database,
    create_or_update_database,
)
from .llm_service import (
    ModelManager,
    RAGQueryProcessor,
    get_model_manager,
    get_rag_processor,
    query_ollama,
    get_available_models,
    test_ollama_connection,
)
from .voice_service import (
    VoiceInputService,
    get_voice_service,
    record_voice_input,
    transcribe_audio_file,
)

__all__ = [
    # Embedding service
    "EmbeddingFactory",
    "get_embedding_factory",
    "get_mixed_language_embedding",
    "detect_language",
    "embed_english",
    "embed_bangla",
    "get_embedding_function_with_fallback",
    # Database service
    "DatabaseManager",
    "get_database_manager",
    "load_database",
    "query_database",
    "create_or_update_database",
    # LLM service
    "ModelManager",
    "RAGQueryProcessor",
    "get_model_manager",
    "get_rag_processor",
    "query_ollama",
    "get_available_models",
    "test_ollama_connection",
    # Voice service
    "VoiceInputService",
    "get_voice_service",
    "record_voice_input",
    "transcribe_audio_file",
]
