#!/usr/bin/env python3
"""
Load course knowledge base into ChromaDB for chatbot queries.
This script processes the course content and creates embeddings.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logging_config import log_info, log_error
from services.embedding_service import get_embedding_factory
from services.database_service import get_database_manager
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re


def load_course_content(file_path: str) -> str:
    """Load course content from text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        log_info(f"Loaded course content from {file_path}", "loader")
        return content
    except Exception as e:
        log_error(f"Failed to load course content: {e}", "loader")
        raise


def parse_course_sections(content: str) -> list[Document]:
    """Parse course content into structured sections."""
    documents = []

    # Split by module sections
    sections = re.split(r"={50,}\n(MODULE \d+:.*?)\n={50,}", content)

    current_module = "Course Overview"

    for i, section in enumerate(sections):
        if section.startswith("MODULE"):
            current_module = section.strip()
        elif section.strip() and not section.strip().startswith("="):
            # Further split into subsections
            subsections = section.split("\n\n")

            for subsection in subsections:
                if len(subsection.strip()) > 50:  # Skip very short sections
                    # Include module title in content for better retrieval
                    content_with_module = f"{current_module}\n\n{subsection.strip()}"
                    doc = Document(
                        page_content=content_with_module,
                        metadata={
                            "source": "course_knowledge_base.txt",
                            "module": current_module,
                            "type": "course_content",
                        },
                    )
                    documents.append(doc)

    log_info(f"Parsed {len(documents)} course sections", "loader")
    return documents


def chunk_documents(
    documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 100
) -> list[Document]:
    """Split documents into smaller chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked_docs = []
    doc_counter = 0
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc_counter += 1
            chunked_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "id": f"course_chunk_{doc_counter}",  # Add unique ID
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            chunked_docs.append(chunked_doc)

    log_info(
        f"Created {len(chunked_docs)} chunks from {len(documents)} documents", "loader"
    )
    return chunked_docs


def create_course_database(
    knowledge_base_path: str,
    collection_name: str = "course_materials",
    chunk_size: int = 500,
):
    """Create ChromaDB collection from course knowledge base."""
    try:
        print("🚀 Starting Course Database Creation")
        print("=" * 60)

        # Load content
        print("\n📖 Loading course content...")
        content = load_course_content(knowledge_base_path)
        print(f"✅ Loaded {len(content)} characters")

        # Parse into sections
        print("\n📑 Parsing course sections...")
        documents = parse_course_sections(content)
        print(f"✅ Parsed {len(documents)} sections")

        # Chunk documents
        print(f"\n✂️  Chunking documents (size: {chunk_size})...")
        chunked_docs = chunk_documents(documents, chunk_size=chunk_size)
        print(f"✅ Created {len(chunked_docs)} chunks")

        # Initialize embedding service
        print("\n🔤 Initializing embedding service...")
        embedding_factory = get_embedding_factory()
        embedding_function = embedding_factory.get_embedding_function_with_fallback()
        print("✅ Embedding service ready")

        # Create database
        print("\n💾 Creating ChromaDB collection...")
        from services.database_service import DatabaseFactory

        db_manager = DatabaseFactory.create_chroma_database(
            collection_name=collection_name
        )

        # Add documents using batch method
        db_manager.add_documents_batch(chunked_docs)

        print(f"✅ Added {len(chunked_docs)} chunks to database")

        # Test the database
        print("\n🔍 Testing database with sample query...")
        test_query = "What is an array?"
        results = db_manager.search_with_cache(test_query, k=3)

        print(f"\n📊 Test Results:")
        print(f"Query: '{test_query}'")
        print(f"Found {len(results)} relevant documents")

        if results:
            print(f"\nTop Result:")
            print(f"Content: {results[0].page_content[:200]}...")
            print(f"Module: {results[0].metadata.get('module', 'Unknown')}")

        print("\n" + "=" * 60)
        print("✅ Course database created successfully!")
        print(f"📍 Database location: db/")
        print(f"📊 Total chunks: {len(chunked_docs)}")
        print(f"🎯 Collection: {collection_name}")

        return True

    except Exception as e:
        log_error(f"Failed to create course database: {e}", "loader", exc_info=True)
        print(f"\n❌ Error: {e}")
        return False


def main():
    """Main function to create course database."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load course knowledge base into ChromaDB"
    )
    parser.add_argument(
        "--input",
        default="web/course_knowledge_base.txt",
        help="Path to course knowledge base file",
    )
    parser.add_argument(
        "--collection", default="course_materials", help="Name of ChromaDB collection"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Size of text chunks"
    )

    args = parser.parse_args()

    # Convert to absolute path
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / args.input

    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        sys.exit(1)

    success = create_course_database(
        str(input_path), collection_name=args.collection, chunk_size=args.chunk_size
    )

    if success:
        print("\n🎉 You can now start the chatbot API:")
        print("   python web/chatbot_api.py")
        print("\n🌐 Then open the course page:")
        print("   web/course-example.html")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
