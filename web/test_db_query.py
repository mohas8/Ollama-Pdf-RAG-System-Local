#!/usr/bin/env python3
"""
Quick test to see what's actually in the database
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.database_service import DatabaseFactory

# Connect to course_materials collection
db_manager = DatabaseFactory.create_chroma_database(collection_name="course_materials")

# Test query
query = "What is module 1 of the course"
results = db_manager.search_with_cache(query, k=3)

print(f"Query: {query}")
print(f"Found {len(results)} documents\n")
print("=" * 80)

for i, doc in enumerate(results, 1):
    print(f"\n📄 Document {i}:")
    print(f"Content: {doc.page_content[:500]}")
    print(f"Module: {doc.metadata.get('module', 'Unknown')}")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"ID: {doc.metadata.get('id', 'Unknown')}")
    print("-" * 80)
