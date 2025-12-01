#!/usr/bin/env python3
"""
Refactored LLM service for BanglaRAG system.
Provides clean interface for language model operations with proper prompt management.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import requests
import json
import threading
import time
from functools import lru_cache

from core.logging_config import BanglaRAGLogger
from core.exceptions import ModelException, NetworkException
from core.utils import (
    retry_with_backoff,
    measure_performance,
    SimpleCache,
    timeout_handler,
)
from core.constants import (
    PREFERRED_LLM_MODEL,
    FALLBACK_LLM_MODELS,
    MAX_TOKENS,
    TEMPERATURE,
    TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
    OLLAMA_API_TIMEOUT,
)

logger = BanglaRAGLogger.get_logger("llm")


class QueryType(Enum):
    """Types of queries for different prompt strategies."""

    DEFINITION = "definition"
    PROCESS = "process"
    COMPLEXITY = "complexity"
    PURPOSE = "purpose"
    GENERAL = "general"


class PromptTemplate:
    """Template for generating prompts based on query type."""

    TEMPLATES = {
        QueryType.DEFINITION: """
INSTRUCTIONS FOR DEFINITION:
- Provide a clear, concise definition (1-2 sentences)
- Include key characteristics or properties  
- Use precise technical terminology
- Avoid excessive detail unless specifically asked

ANSWER (concise definition):""",
        QueryType.PROCESS: """
INSTRUCTIONS FOR PROCESS:
- Explain the key steps or process briefly
- Focus on the main algorithm or procedure
- Include time/space complexity if relevant
- Keep explanation structured and clear

ANSWER (process explanation):""",
        QueryType.COMPLEXITY: """
INSTRUCTIONS FOR COMPLEXITY:
- State both average case and worst case if different
- Use Big O notation correctly
- Be specific about what operations are being measured
- Include space complexity if relevant

ANSWER (complexity analysis):""",
        QueryType.PURPOSE: """
INSTRUCTIONS FOR PURPOSE:
- Explain the main use case or application
- Include why it's preferred over alternatives
- Mention key advantages
- Keep explanation practical and clear

ANSWER (purpose explanation):""",
        QueryType.GENERAL: """
INSTRUCTIONS:
- Answer exactly what is asked - no more, no less
- Be precise and factually accurate
- Include relevant technical details
- Cite page numbers when available

ANSWER:""",
    }

    @classmethod
    def detect_query_type(cls, question: str) -> QueryType:
        """Detect query type from question content."""
        question_lower = question.lower()

        if any(
            phrase in question_lower
            for phrase in ["what is", "define", "definition of"]
        ):
            return QueryType.DEFINITION
        elif any(
            phrase in question_lower for phrase in ["how does", "how to", "explain how"]
        ):
            return QueryType.PROCESS
        elif any(
            phrase in question_lower
            for phrase in ["time complexity", "space complexity", "complexity"]
        ):
            return QueryType.COMPLEXITY
        elif any(
            phrase in question_lower for phrase in ["used for", "purpose of", "why use"]
        ):
            return QueryType.PURPOSE
        else:
            return QueryType.GENERAL

    @classmethod
    def generate_prompt(
        cls, question: str, context: str, query_type: Optional[QueryType] = None
    ) -> str:
        """Generate optimized prompt based on query type."""
        if query_type is None:
            query_type = cls.detect_query_type(question)

        instruction = cls.TEMPLATES[query_type]

        return f"""You are an expert computer science educator specializing in algorithms and data structures.
Answer the question based ONLY on the provided context. Be precise, accurate, and appropriately scoped.

CONTEXT:
{context}

QUESTION: {question}
{instruction}"""


class LLMModel(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is available."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class OllamaModel(LLMModel):
    """Ollama-based language model."""

    def __init__(self, model_name: str, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url
        self._session = requests.Session()
        self._session.timeout = OLLAMA_API_TIMEOUT
        self._model_info: Optional[Dict] = None
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if model is available."""
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                available_models = [m["name"] for m in models.get("models", [])]
                if self.model_name not in available_models:
                    logger.warning(
                        f"Model {self.model_name} not found in available models: {available_models}"
                    )
                else:
                    logger.info(f"Model {self.model_name} is available")
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")

    @retry_with_backoff(max_retries=3)
    @timeout_handler(TIMEOUT_SECONDS)
    @measure_performance
    def generate_response(
        self,
        prompt: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        **kwargs,
    ) -> str:
        """Generate response using Ollama API."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stop": ["Human:", "Assistant:", "User:"],
                },
            }

            response = self._session.post(
                f"{self.base_url}/api/generate", json=payload, timeout=TIMEOUT_SECONDS
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                raise NetworkException(f"Ollama API error: {response.status_code}")

        except requests.Timeout:
            logger.error(f"Ollama request timed out for model {self.model_name}")
            raise NetworkException("Ollama request timed out")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise ModelException(f"Response generation failed: {e}")

    def is_available(self) -> bool:
        """Check if model is available."""
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [m["name"] for m in models.get("models", [])]
                return self.model_name in available_models
        except Exception:
            pass
        return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self._model_info is None:
            try:
                response = self._session.post(
                    f"{self.base_url}/api/show",
                    json={"name": self.model_name},
                    timeout=10,
                )
                if response.status_code == 200:
                    self._model_info = response.json()
                else:
                    self._model_info = {
                        "name": self.model_name,
                        "status": "unavailable",
                    }
            except Exception as e:
                logger.error(f"Failed to get model info: {e}")
                self._model_info = {"name": self.model_name, "error": str(e)}

        return self._model_info


class ModelManager:
    """Manages multiple LLM models with fallback and caching."""

    def __init__(
        self,
        preferred_model: str = PREFERRED_LLM_MODEL,
        fallback_models: List[str] = None,
    ):
        self.preferred_model = preferred_model
        self.fallback_models = fallback_models or FALLBACK_LLM_MODELS
        self._models: Dict[str, OllamaModel] = {}
        self._response_cache = SimpleCache(
            max_size=200, ttl_seconds=1800
        )  # 30 min cache
        self._active_model: Optional[str] = None
        self._lock = threading.Lock()
        self._stats = {"requests": 0, "cache_hits": 0, "model_switches": 0, "errors": 0}
        self._warm_up_models()

    def _warm_up_models(self) -> None:
        """Warm up models in background."""

        def warm_up():
            try:
                # Initialize preferred model
                self._get_or_create_model(self.preferred_model)
                logger.info("Model warm-up completed")
            except Exception as e:
                logger.warning(f"Model warm-up failed: {e}")

        threading.Thread(target=warm_up, daemon=True).start()

    def _get_or_create_model(self, model_name: str) -> OllamaModel:
        """Get or create model instance."""
        if model_name not in self._models:
            self._models[model_name] = OllamaModel(model_name)
        return self._models[model_name]

    def _find_available_model(self) -> Optional[str]:
        """Find first available model from preferred and fallback list."""
        all_models = [self.preferred_model] + self.fallback_models

        for model_name in all_models:
            try:
                model = self._get_or_create_model(model_name)
                if model.is_available():
                    return model_name
            except Exception as e:
                logger.warning(f"Model {model_name} check failed: {e}")
                continue

        return None

    @measure_performance
    def generate_response(
        self,
        prompt: str,
        use_cache: bool = True,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """Generate response with fallback and caching."""
        # Check cache first
        if use_cache:
            cache_key = (
                f"response:{hash(prompt)}:{kwargs.get('max_tokens', MAX_TOKENS)}"
            )
            cached_response = self._response_cache.get(cache_key)
            if cached_response:
                self._stats["cache_hits"] += 1
                logger.debug("Using cached response")
                return cached_response

        with self._lock:
            self._stats["requests"] += 1

            # Determine which model to use
            target_model = model_name or self._active_model or self.preferred_model

            # Try to generate response
            for attempt_model in [target_model] + self.fallback_models:
                try:
                    model = self._get_or_create_model(attempt_model)
                    response = model.generate_response(prompt, **kwargs)

                    # Update active model if it changed
                    if self._active_model != attempt_model:
                        self._active_model = attempt_model
                        self._stats["model_switches"] += 1
                        logger.info(f"Switched to model: {attempt_model}")

                    # Cache the response
                    if use_cache:
                        cache_key = f"response:{hash(prompt)}:{kwargs.get('max_tokens', MAX_TOKENS)}"
                        self._response_cache.set(cache_key, response)

                    return response

                except Exception as e:
                    logger.warning(f"Model {attempt_model} failed: {e}")
                    self._stats["errors"] += 1
                    continue

            logger.error("All models failed to generate response")
            return None

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        available = []
        all_models = [self.preferred_model] + self.fallback_models

        for model_name in all_models:
            try:
                model = self._get_or_create_model(model_name)
                if model.is_available():
                    available.append(model_name)
            except Exception:
                continue

        return available

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get model manager statistics."""
        cache_hit_rate = (
            self._stats["cache_hits"] / max(self._stats["requests"], 1) * 100
            if self._stats["requests"] > 0
            else 0
        )

        return {
            "active_model": self._active_model,
            "available_models": self.get_available_models(),
            "total_requests": self._stats["requests"],
            "cache_hits": self._stats["cache_hits"],
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "model_switches": self._stats["model_switches"],
            "errors": self._stats["errors"],
            "cache_size": self._response_cache.size(),
        }

    def clear_cache(self) -> None:
        """Clear response cache."""
        self._response_cache.clear()
        logger.info("Model manager cache cleared")


class RAGQueryProcessor:
    """Processes RAG queries with context and prompt optimization."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.prompt_template = PromptTemplate()

    @measure_performance
    def process_rag_query(
        self,
        question: str,
        context_documents: List[Any],
        max_context_length: int = 2000,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process RAG query with context and generate response."""
        try:
            # Prepare context from documents
            context = self._prepare_context(context_documents, max_context_length)

            # Generate optimized prompt
            prompt = self.prompt_template.generate_prompt(question, context)

            # Generate response
            response = self.model_manager.generate_response(prompt, **kwargs)

            if response:
                # Extract citations from context documents
                citations = self._extract_citations(context_documents)

                return {
                    "response": response,
                    "question": question,
                    "context_length": len(context),
                    "citations": citations,
                    "success": True,
                    "model_used": self.model_manager._active_model,
                }
            else:
                return {
                    "response": "I apologize, but I couldn't generate a response at the moment.",
                    "question": question,
                    "success": False,
                    "error": "No response generated",
                }

        except Exception as e:
            logger.error(f"RAG query processing failed: {e}")
            return {
                "response": "An error occurred while processing your question.",
                "question": question,
                "success": False,
                "error": str(e),
            }

    def _prepare_context(self, documents: List[Any], max_length: int) -> str:
        """Prepare context string from documents."""
        context_parts = []
        current_length = 0

        for doc in documents:
            # Extract content and metadata
            if hasattr(doc, "page_content"):
                content = doc.page_content
                metadata = getattr(doc, "metadata", {})
            else:
                content = str(doc)
                metadata = {}

            # Add metadata info
            page_info = ""
            if "page_number" in metadata:
                page_info = f" (Page {metadata['page_number']})"
            elif "page" in metadata:
                page_info = f" (Page {metadata['page'] + 1})"

            formatted_content = f"{content}{page_info}"

            # Check length limit
            if current_length + len(formatted_content) > max_length:
                # Truncate if needed
                remaining_space = max_length - current_length
                if remaining_space > 100:  # Only add if there's reasonable space
                    formatted_content = formatted_content[: remaining_space - 3] + "..."
                    context_parts.append(formatted_content)
                break

            context_parts.append(formatted_content)
            current_length += len(formatted_content)

        return "\n\n".join(context_parts)

    def _extract_citations(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """Extract citation information from documents."""
        citations = []

        for doc in documents:
            if hasattr(doc, "metadata"):
                metadata = doc.metadata
                citation = {
                    "file_name": metadata.get("file_name", "Unknown Document"),
                    "page_number": metadata.get(
                        "page_number", metadata.get("page", 0) + 1
                    ),
                    "source": metadata.get("source", "Unknown Source"),
                }
                citations.append(citation)

        return citations


# Global instances
_model_manager: Optional[ModelManager] = None
_rag_processor: Optional[RAGQueryProcessor] = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def get_rag_processor() -> RAGQueryProcessor:
    """Get global RAG processor instance."""
    global _rag_processor
    if _rag_processor is None:
        _rag_processor = RAGQueryProcessor(get_model_manager())
    return _rag_processor


# Compatibility functions
def query_ollama(prompt: str, model: str = None, **kwargs) -> Optional[str]:
    """Compatibility function for ollama queries."""
    manager = get_model_manager()
    return manager.generate_response(prompt, model_name=model, **kwargs)


def get_available_models() -> List[str]:
    """Compatibility function to get available models."""
    manager = get_model_manager()
    return manager.get_available_models()


def test_ollama_connection(model_name: str = None) -> bool:
    """Test Ollama connection."""
    try:
        manager = get_model_manager()
        test_response = manager.generate_response(
            "Hello", model_name=model_name, max_tokens=5
        )
        return test_response is not None
    except Exception as e:
        logger.error(f"Ollama connection test failed: {e}")
        return False


def test_llm_service() -> bool:
    """Test LLM service functionality."""
    try:
        logger.info("Testing LLM service...")
        manager = get_model_manager()

        # Test basic response generation
        response = manager.generate_response("What is 2+2?", max_tokens=10)
        if response:
            logger.info(f"Basic test response: {response[:50]}...")
        else:
            logger.error("Failed to generate basic response")
            return False

        # Test RAG processor
        processor = get_rag_processor()
        test_docs = [{"page_content": "Test document content", "metadata": {"page": 0}}]
        rag_result = processor.process_rag_query("What is this about?", test_docs)

        if rag_result["success"]:
            logger.info("RAG processor test successful")
        else:
            logger.error(f"RAG processor test failed: {rag_result.get('error')}")
            return False

        # Get stats
        stats = manager.get_manager_stats()
        logger.info(f"Manager stats: {stats}")

        logger.info("LLM service test completed successfully")
        return True

    except Exception as e:
        logger.error(f"LLM service test failed: {e}")
        return False


if __name__ == "__main__":
    test_llm_service()
