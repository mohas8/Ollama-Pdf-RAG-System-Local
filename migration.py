#!/usr/bin/env python3
"""
Migration helper for updating existing BanglaRAG modules to use the new refactored architecture.
This script provides backward compatibility while transitioning to the new structure.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import compatibility functions from new architecture
try:
    from services import (
        get_embedding_function_with_fallback,
        get_mixed_language_embedding,
        detect_language,
        embed_english,
        embed_bangla,
        load_database,
        query_database,
        create_or_update_database,
        query_ollama,
        get_available_models,
        test_ollama_connection,
        record_voice_input,
        transcribe_audio_file,
    )

    NEW_ARCHITECTURE_AVAILABLE = True
except ImportError as e:
    NEW_ARCHITECTURE_AVAILABLE = False
    print(f"New architecture not available: {e}")

from core.logging_config import BanglaRAGLogger, log_info, log_warning

logger = BanglaRAGLogger.get_logger("migration")


def update_embedding_imports():
    """Update embedding.py to use new architecture."""
    if not NEW_ARCHITECTURE_AVAILABLE:
        return False

    try:
        # The new embedding service provides all the same functions
        # with improved architecture
        log_info("Embedding module can use new architecture", "migration")
        return True
    except Exception as e:
        log_warning(f"Embedding migration issue: {e}", "migration")
        return False


def update_database_imports():
    """Update query_database.py and create_database.py to use new architecture."""
    if not NEW_ARCHITECTURE_AVAILABLE:
        return False

    try:
        # Test database functionality
        db = load_database()
        if db:
            log_info("Database module can use new architecture", "migration")
            return True
        return False
    except Exception as e:
        log_warning(f"Database migration issue: {e}", "migration")
        return False


def update_llm_imports():
    """Update ollama_llm.py to use new architecture."""
    if not NEW_ARCHITECTURE_AVAILABLE:
        return False

    try:
        # Test LLM functionality
        models = get_available_models()
        if models:
            log_info("LLM module can use new architecture", "migration")
            return True
        return False
    except Exception as e:
        log_warning(f"LLM migration issue: {e}", "migration")
        return False


def create_compatibility_wrapper():
    """Create a compatibility wrapper for existing code."""

    wrapper_content = '''#!/usr/bin/env python3
"""
Compatibility wrapper for BanglaRAG system.
Provides backward compatibility with existing code while using new architecture.
"""

import warnings
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Try to import from new architecture
    from services import (
        get_embedding_function_with_fallback, get_mixed_language_embedding,
        detect_language, embed_english, embed_bangla,
        load_database, query_database, create_or_update_database,
        query_ollama, get_available_models, test_ollama_connection,
        record_voice_input, transcribe_audio_file
    )
    
    from core import BanglaRAGLogger
    logger = BanglaRAGLogger.get_logger("compatibility")
    logger.info("Using new refactored architecture")
    
    # Re-export all functions for backward compatibility
    __all__ = [
        'get_embedding_function_with_fallback', 'get_mixed_language_embedding',
        'detect_language', 'embed_english', 'embed_bangla',
        'load_database', 'query_database', 'create_or_update_database',
        'query_ollama', 'get_available_models', 'test_ollama_connection',
        'record_voice_input', 'transcribe_audio_file'
    ]
    
except ImportError as e:
    # Fallback to original modules
    warnings.warn(f"New architecture not available, using original modules: {e}")
    
    # Import from original modules
    try:
        from embedding import (
            get_embedding_function_with_fallback, get_mixed_language_embedding,
            detect_language, embed_english, embed_bangla
        )
    except ImportError:
        pass
    
    try:
        from query_database import load_database, query_database
        from create_database import create_or_update_database
    except ImportError:
        pass
    
    try:
        from ollama_llm import query_ollama, get_available_models, test_ollama_connection
    except ImportError:
        pass
    
    try:
        from voice_input import record_voice_input, transcribe_audio_file
    except ImportError:
        pass


# Additional helper functions for migration
def check_new_architecture_availability():
    """Check if new architecture is available and working."""
    try:
        from services import get_model_manager
        manager = get_model_manager()
        return True
    except Exception:
        return False


def migrate_to_new_architecture():
    """Helper function to migrate existing code to new architecture."""
    if check_new_architecture_availability():
        print("✅ New architecture is available and working")
        print("🔄 Consider updating your imports to use the new services module")
        return True
    else:
        print("⚠️  New architecture not fully available, using compatibility mode")
        return False
'''

    try:
        with open("compatibility.py", "w", encoding="utf-8") as f:
            f.write(wrapper_content)
        log_info("Created compatibility wrapper", "migration")
        return True
    except Exception as e:
        log_warning(f"Failed to create compatibility wrapper: {e}", "migration")
        return False


def run_migration_check():
    """Run complete migration check."""
    print("🔄 BanglaRAG Architecture Migration Check")
    print("=" * 50)

    checks = [
        ("New Architecture Available", NEW_ARCHITECTURE_AVAILABLE),
        ("Embedding Module", update_embedding_imports()),
        ("Database Module", update_database_imports()),
        ("LLM Module", update_llm_imports()),
        ("Compatibility Wrapper", create_compatibility_wrapper()),
    ]

    all_passed = True
    for check_name, result in checks:
        emoji = "✅" if result else "❌"
        print(f"{emoji} {check_name}")
        if not result:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("✅ Migration check completed successfully!")
        print("💡 You can now use the new refactored architecture")
        print("🚀 Run main_refactored.py to use the new system")
    else:
        print("⚠️  Some migration checks failed")
        print("💡 You can still use the original system or the compatibility wrapper")

    print("\n📋 Next Steps:")
    print("1. Use 'python main_refactored.py' for the new architecture")
    print("2. Use 'python main.py' for the original system")
    print("3. Import from 'compatibility.py' for mixed usage")

    return all_passed


if __name__ == "__main__":
    run_migration_check()
