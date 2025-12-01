#!/usr/bin/env python3
"""
Refactored database service for BanglaRAG system.
Provides clean interface for vector database operations with proper dependency injection.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import threading
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb

from core.logging_config import BanglaRAGLogger
from core.exceptions import DatabaseException
from core.utils import (
    retry_with_backoff,
    measure_performance,
    SimpleCache,
    ensure_directory,
)
from core.constants import DATABASE_DIRECTORY, DEFAULT_RETRIEVAL_COUNT
from services.embedding_service import get_embedding_factory

logger = BanglaRAGLogger.get_logger("database")


class VectorDatabase(ABC):
    """Abstract base class for vector databases."""

    @abstractmethod
    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to the database."""
        pass

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = DEFAULT_RETRIEVAL_COUNT
    ) -> List[Document]:
        """Perform similarity search."""
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """Get total number of documents."""
        pass

    @abstractmethod
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        pass

    @abstractmethod
    def get_existing_ids(self) -> List[str]:
        """Get list of existing document IDs."""
        pass


class ChromaVectorDatabase(VectorDatabase):
    """ChromaDB implementation of vector database."""

    def __init__(
        self,
        persist_directory: str = DATABASE_DIRECTORY,
        collection_name: str = "banglarag",
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._db: Optional[Chroma] = None
        self._client: Optional[chromadb.PersistentClient] = None
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize ChromaDB database."""
        try:
            # Ensure directory exists
            ensure_directory(self.persist_directory)

            # Get embedding function
            embedding_factory = get_embedding_factory()
            embedding_function = (
                embedding_factory.get_embedding_function_with_fallback()
            )

            # Create ChromaDB client
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))

            # Create Chroma instance
            self._db = Chroma(
                client=self._client,
                collection_name=self.collection_name,
                embedding_function=embedding_function,
            )

            logger.info(f"ChromaDB initialized at {self.persist_directory}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise DatabaseException(f"Database initialization failed: {e}")

    @retry_with_backoff(max_retries=3)
    @measure_performance
    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to ChromaDB."""
        if not self._db:
            raise DatabaseException("Database not initialized")

        if not documents:
            logger.warning("No documents to add")
            return

        try:
            # Extract IDs if provided in metadata
            if ids is None:
                ids = [
                    doc.metadata.get("id", f"doc_{i}")
                    for i, doc in enumerate(documents)
                ]

            # Add documents
            self._db.add_documents(documents, ids=ids)
            logger.info(f"Added {len(documents)} documents to database")

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise DatabaseException(f"Failed to add documents: {e}")

    @retry_with_backoff(max_retries=2)
    @measure_performance
    def similarity_search(
        self, query: str, k: int = DEFAULT_RETRIEVAL_COUNT
    ) -> List[Document]:
        """Perform similarity search in ChromaDB."""
        if not self._db:
            raise DatabaseException("Database not initialized")

        try:
            results = self._db.similarity_search(query, k=k)
            logger.debug(f"Found {len(results)} similar documents for query")
            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise DatabaseException(f"Similarity search failed: {e}")

    def get_document_count(self) -> int:
        """Get total number of documents in database."""
        if not self._db:
            return 0

        try:
            collection_data = self._db.get()
            return len(collection_data.get("ids", []))
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        if not self._db:
            raise DatabaseException("Database not initialized")

        try:
            self._db.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise DatabaseException(f"Failed to delete documents: {e}")

    def get_existing_ids(self) -> List[str]:
        """Get list of existing document IDs."""
        if not self._db:
            return []

        try:
            existing_data = self._db.get()
            return existing_data.get("ids", [])
        except Exception as e:
            logger.error(f"Failed to get existing IDs: {e}")
            return []


class DatabaseManager:
    """Manages database operations with caching and optimization."""

    def __init__(self, database: VectorDatabase):
        self.database = database
        self._query_cache = SimpleCache(max_size=100, ttl_seconds=600)  # 10 min cache
        self._metadata_cache = SimpleCache(
            max_size=50, ttl_seconds=1800
        )  # 30 min cache
        self._lock = threading.Lock()
        self._stats = {
            "queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "documents_added": 0,
            "last_query_time": None,
        }

    def add_documents_batch(
        self, documents: List[Document], batch_size: int = 100
    ) -> Dict[str, Any]:
        """Add documents in batches with duplicate detection."""
        if not documents:
            return {"added": 0, "skipped": 0, "errors": 0}

        with self._lock:
            try:
                # Get existing IDs to avoid duplicates
                existing_ids = set(self.database.get_existing_ids())

                # Filter out duplicates
                new_documents = []
                new_ids = []

                for doc in documents:
                    doc_id = doc.metadata.get("id")
                    if doc_id and doc_id not in existing_ids:
                        new_documents.append(doc)
                        new_ids.append(doc_id)

                if not new_documents:
                    logger.info("No new documents to add (all duplicates)")
                    return {"added": 0, "skipped": len(documents), "errors": 0}

                # Add documents in batches
                added_count = 0
                error_count = 0

                for i in range(0, len(new_documents), batch_size):
                    batch_docs = new_documents[i : i + batch_size]
                    batch_ids = new_ids[i : i + batch_size]

                    try:
                        self.database.add_documents(batch_docs, batch_ids)
                        added_count += len(batch_docs)
                        logger.info(
                            f"Added batch {i//batch_size + 1}: {len(batch_docs)} documents"
                        )
                    except Exception as e:
                        logger.error(f"Failed to add batch {i//batch_size + 1}: {e}")
                        error_count += len(batch_docs)

                self._stats["documents_added"] += added_count

                # Clear caches after adding documents
                self._clear_caches()

                return {
                    "added": added_count,
                    "skipped": len(documents) - len(new_documents),
                    "errors": error_count,
                }

            except Exception as e:
                logger.error(f"Batch document addition failed: {e}")
                raise DatabaseException(f"Batch addition failed: {e}")

    @measure_performance
    def search_with_cache(
        self, query: str, k: int = DEFAULT_RETRIEVAL_COUNT
    ) -> List[Document]:
        """Perform similarity search with caching."""
        # Create cache key
        cache_key = f"search:{hash(query)}:{k}"

        # Check cache
        cached_result = self._query_cache.get(cache_key)
        if cached_result is not None:
            self._stats["cache_hits"] += 1
            logger.debug("Using cached search result")
            return cached_result

        # Perform search
        try:
            results = self.database.similarity_search(query, k)

            # Cache the results
            self._query_cache.set(cache_key, results)

            # Update stats
            self._stats["queries"] += 1
            self._stats["cache_misses"] += 1
            self._stats["last_query_time"] = time.time()

            return results

        except Exception as e:
            logger.error(f"Search with cache failed: {e}")
            raise DatabaseException(f"Search failed: {e}")

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            doc_count = self.database.get_document_count()
            cache_hit_rate = (
                self._stats["cache_hits"] / max(self._stats["queries"], 1) * 100
                if self._stats["queries"] > 0
                else 0
            )

            return {
                "document_count": doc_count,
                "total_queries": self._stats["queries"],
                "cache_hits": self._stats["cache_hits"],
                "cache_misses": self._stats["cache_misses"],
                "cache_hit_rate": f"{cache_hit_rate:.1f}%",
                "documents_added": self._stats["documents_added"],
                "last_query_time": self._stats["last_query_time"],
                "cache_size": self._query_cache.size(),
            }

        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._clear_caches()
        logger.info("Database caches cleared")

    def _clear_caches(self) -> None:
        """Clear internal caches."""
        self._query_cache.clear()
        self._metadata_cache.clear()

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            self.database.get_document_count()
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False


class DatabaseFactory:
    """Factory for creating database instances."""

    @staticmethod
    def create_chroma_database(
        persist_directory: str = DATABASE_DIRECTORY, collection_name: str = "banglarag"
    ) -> DatabaseManager:
        """Create ChromaDB database manager."""
        try:
            chroma_db = ChromaVectorDatabase(persist_directory, collection_name)
            return DatabaseManager(chroma_db)
        except Exception as e:
            logger.error(f"Failed to create ChromaDB: {e}")
            raise DatabaseException(f"Database creation failed: {e}")


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseFactory.create_chroma_database()
    return _database_manager


# Compatibility functions for existing code
def load_database() -> Optional[Chroma]:
    """Compatibility function - returns ChromaDB instance."""
    try:
        manager = get_database_manager()
        return manager.database._db  # Access underlying ChromaDB instance
    except Exception as e:
        logger.error(f"Failed to load database: {e}")
        return None


def query_database(
    query: str, db: Optional[Any] = None, k: int = DEFAULT_RETRIEVAL_COUNT
) -> List[Document]:
    """Compatibility function for database queries."""
    try:
        if db is None:
            manager = get_database_manager()
            return manager.search_with_cache(query, k)
        else:
            # Use provided database instance
            return db.similarity_search(query, k)
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        return []


def create_or_update_database(
    chunks_with_ids: List[Document], persist_directory: str = DATABASE_DIRECTORY
) -> Optional[Any]:
    """Compatibility function for creating/updating database."""
    try:
        manager = DatabaseFactory.create_chroma_database(persist_directory)
        result = manager.add_documents_batch(chunks_with_ids)
        logger.info(f"Database update result: {result}")
        return manager.database._db  # Return ChromaDB instance for compatibility
    except Exception as e:
        logger.error(f"Database creation/update failed: {e}")
        return None


def test_database() -> bool:
    """Test database functionality."""
    try:
        logger.info("Testing database functionality...")
        manager = get_database_manager()

        # Test connection
        if not manager.test_connection():
            logger.error("Database connection test failed")
            return False

        # Test query
        results = manager.search_with_cache("test query", k=1)
        logger.info(f"Test query returned {len(results)} results")

        # Get database info
        info = manager.get_database_info()
        logger.info(f"Database info: {info}")

        logger.info("Database test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False


if __name__ == "__main__":
    test_database()
