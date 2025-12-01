#!/usr/bin/env python3
"""
Refactored main application for BanglaRAG system.
Provides a clean, modular interface using the new service architecture.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.logging_config import BanglaRAGLogger, log_info, log_error, log_warning
from core.constants import (
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    MAIN_MENU_OPTIONS,
    BANNER_WIDTH,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
)
from core.utils import ensure_directory, find_pdf_files

from services import (
    get_embedding_factory,
    get_database_manager,
    get_model_manager,
    get_rag_processor,
    get_voice_service,
)

# Initialize logging
logger = BanglaRAGLogger.get_logger("main")


class BanglaRAGApplication:
    """Main application class for BanglaRAG system."""

    def __init__(self):
        self.embedding_factory = None
        self.database_manager = None
        self.model_manager = None
        self.rag_processor = None
        self.voice_service = None
        self._initialize_services()

    def _initialize_services(self) -> None:
        """Initialize all services."""
        try:
            log_info("Initializing BanglaRAG services...", "main")

            # Initialize services lazily - they will be created when first accessed
            log_info("Services initialization completed", "main")

        except Exception as e:
            log_error(f"Failed to initialize services: {e}", "main", exc_info=True)
            raise

    def print_banner(self) -> None:
        """Print application banner."""
        print("=" * BANNER_WIDTH)
        print(f"🎤 {APP_NAME}")
        print(f"📚 {APP_DESCRIPTION}")
        print("🌐 Supporting English & Bangla | 🎙️ Voice & Text Input")
        print(f"🔧 Version {APP_VERSION} - Refactored Architecture")
        print("=" * BANNER_WIDTH)

    def check_dependencies(self) -> Dict[str, Any]:
        """Check system dependencies and return status."""
        status = {
            "pdf_files": [],
            "database": False,
            "embedding": False,
            "llm": False,
            "voice": False,
            "issues": [],
        }

        try:
            # Check PDF files
            pdf_files = find_pdf_files(Path.cwd())
            status["pdf_files"] = [str(f) for f in pdf_files]
            if not pdf_files:
                status["issues"].append(ERROR_MESSAGES["pdf_not_found"])

            # Check database
            try:
                db_manager = get_database_manager()
                if db_manager.test_connection():
                    status["database"] = True
                    log_info("✅ Database connection successful", "main")
                else:
                    status["issues"].append(ERROR_MESSAGES["database_not_found"])
            except Exception as e:
                status["issues"].append(f"Database error: {e}")

            # Check embedding
            try:
                embedding_factory = get_embedding_factory()
                # Test with a simple query
                embedding_factory.get_mixed_language_embedding("test")
                status["embedding"] = True
                log_info("✅ Embedding system working", "main")
            except Exception as e:
                status["issues"].append(f"Embedding error: {e}")

            # Check LLM
            try:
                model_manager = get_model_manager()
                available_models = model_manager.get_available_models()
                if available_models:
                    status["llm"] = True
                    log_info(f"✅ LLM models available: {available_models}", "main")
                else:
                    status["issues"].append(ERROR_MESSAGES["ollama_connection_failed"])
            except Exception as e:
                status["issues"].append(f"LLM error: {e}")

            # Check voice input
            try:
                voice_service = get_voice_service()
                info = voice_service.get_service_info()
                if info["models"]:
                    status["voice"] = True
                    log_info("✅ Voice input available", "main")
                else:
                    status["issues"].append("Voice input not available")
            except Exception as e:
                status["issues"].append(f"Voice error: {e}")

        except Exception as e:
            log_error(f"Dependency check failed: {e}", "main")
            status["issues"].append(f"System error: {e}")

        return status

    def show_system_status(self) -> None:
        """Show detailed system status."""
        print("\n🔍 SYSTEM STATUS CHECK")
        print("=" * 50)

        status = self.check_dependencies()

        # PDF Files
        print(f"\n📄 PDF Files: {len(status['pdf_files'])} found")
        for pdf in status["pdf_files"][:3]:  # Show first 3
            print(f"   • {Path(pdf).name}")
        if len(status["pdf_files"]) > 3:
            print(f"   ... and {len(status['pdf_files']) - 3} more")

        # Services Status
        services = [
            ("Database", status["database"]),
            ("Embedding", status["embedding"]),
            ("LLM Models", status["llm"]),
            ("Voice Input", status["voice"]),
        ]

        print("\n🔧 Services Status:")
        for service, is_available in services:
            emoji = "✅" if is_available else "❌"
            print(f"   {emoji} {service}")

        # Issues
        if status["issues"]:
            print("\n⚠️  Issues Found:")
            for issue in status["issues"]:
                print(f"   • {issue}")
        else:
            print("\n✅ All systems operational!")

        # Get detailed stats if services are available
        try:
            if status["database"]:
                db_manager = get_database_manager()
                db_info = db_manager.get_database_info()
                print(f"\n📊 Database Info:")
                print(f"   Documents: {db_info.get('document_count', 0)}")
                print(f"   Cache Hit Rate: {db_info.get('cache_hit_rate', '0%')}")

            if status["llm"]:
                model_manager = get_model_manager()
                llm_stats = model_manager.get_manager_stats()
                print(f"\n🤖 LLM Stats:")
                print(f"   Active Model: {llm_stats.get('active_model', 'None')}")
                print(f"   Cache Hit Rate: {llm_stats.get('cache_hit_rate', '0%')}")

        except Exception as e:
            log_warning(f"Failed to get detailed stats: {e}", "main")

    def process_documents(self) -> None:
        """Process documents and create/update database."""
        print("\n🔄 DOCUMENT PROCESSING")
        print("=" * 50)

        try:
            # Find PDF files
            pdf_files = find_pdf_files(Path.cwd())
            if not pdf_files:
                print("❌ No PDF files found in current directory")
                return

            print(f"📄 Found {len(pdf_files)} PDF files")

            # Import document processing modules
            from loader import documents
            from split import chunks
            from assign_ids import assign_unique_ids

            print(f"✅ Loaded {len(documents)} pages from PDFs")
            print(f"✅ Created {len(chunks)} chunks")

            # Assign IDs
            chunks_with_ids = assign_unique_ids(chunks)
            print(f"✅ Assigned unique IDs to chunks")

            # Update database
            db_manager = get_database_manager()
            result = db_manager.add_documents_batch(chunks_with_ids)

            print(f"\n📊 Database Update Results:")
            print(f"   Added: {result['added']} documents")
            print(f"   Skipped: {result['skipped']} duplicates")
            print(f"   Errors: {result['errors']} failed")

            if result["added"] > 0:
                print(f"\n✅ {SUCCESS_MESSAGES['document_processed']}")
            else:
                print("\n💡 Database is already up to date")

        except Exception as e:
            log_error(f"Document processing failed: {e}", "main", exc_info=True)
            print(f"❌ Document processing failed: {e}")

    def test_database_queries(self) -> None:
        """Test database with sample queries."""
        print("\n🔍 DATABASE QUERY TESTS")
        print("=" * 50)

        test_queries = [
            "What is an algorithm?",
            "How does binary search work?",
            "Time complexity of sorting algorithms",
            "What is a data structure?",
        ]

        try:
            db_manager = get_database_manager()

            for i, query in enumerate(test_queries, 1):
                print(f"\n🧪 Test {i}: {query}")

                try:
                    results = db_manager.search_with_cache(query, k=2)

                    if results:
                        print(f"   ✅ Found {len(results)} relevant documents")
                        for j, doc in enumerate(results):
                            preview = doc.page_content[:100].replace("\n", " ")
                            page_info = doc.metadata.get("page_number", "Unknown")
                            print(f"   [{j+1}] Page {page_info}: {preview}...")
                    else:
                        print("   ⚠️  No results found")

                except Exception as e:
                    print(f"   ❌ Query failed: {e}")

            # Show database info
            db_info = db_manager.get_database_info()
            print(f"\n📊 Database Statistics:")
            print(f"   Total Documents: {db_info.get('document_count', 0)}")
            print(f"   Total Queries: {db_info.get('total_queries', 0)}")
            print(f"   Cache Hit Rate: {db_info.get('cache_hit_rate', '0%')}")

        except Exception as e:
            log_error(f"Database testing failed: {e}", "main", exc_info=True)
            print(f"❌ Database testing failed: {e}")

    def interactive_chat_session(self) -> None:
        """Start interactive chat session."""
        print("\n💬 INTERACTIVE CHAT SESSION")
        print("=" * 50)
        print("Ask questions about your documents. Type 'quit' to exit.")
        print("Examples: 'What is binary search?', 'Explain sorting algorithms'")

        try:
            db_manager = get_database_manager()
            rag_processor = get_rag_processor()

            while True:
                print("\n" + "-" * 30)
                question = input("🤔 Your question: ").strip()

                if question.lower() in ["quit", "exit", "q"]:
                    print("👋 Chat session ended!")
                    break

                if not question:
                    continue

                try:
                    print("🔍 Searching for relevant information...")

                    # Get relevant documents
                    relevant_docs = db_manager.search_with_cache(question, k=3)

                    if not relevant_docs:
                        print("⚠️  No relevant information found in the database.")
                        continue

                    print("🤖 Generating response...")

                    # Generate response
                    result = rag_processor.process_rag_query(question, relevant_docs)

                    if result["success"]:
                        print(f"\n💡 Answer:")
                        print(f"{result['response']}")

                        # Show citations
                        if result.get("citations"):
                            print(f"\n📚 Sources:")
                            for i, citation in enumerate(result["citations"], 1):
                                file_name = citation.get("file_name", "Unknown")
                                page_num = citation.get("page_number", "Unknown")
                                print(f"   [{i}] {file_name}, Page {page_num}")
                    else:
                        print(
                            f"❌ Failed to generate response: {result.get('error', 'Unknown error')}"
                        )

                except KeyboardInterrupt:
                    print("\n👋 Chat session interrupted!")
                    break
                except Exception as e:
                    log_error(f"Chat query failed: {e}", "main")
                    print(f"❌ Error processing question: {e}")

        except Exception as e:
            log_error(f"Chat session failed: {e}", "main", exc_info=True)
            print(f"❌ Chat session failed: {e}")

    def voice_input_session(self) -> None:
        """Start voice input session with options for regular or continuous mode."""
        print("\n🎤 VOICE INPUT SESSION")
        print("=" * 50)

        try:
            voice_service = get_voice_service()
            info = voice_service.get_service_info()

            if not info["recording_available"]:
                print("❌ Voice recording not available")
                print("Please install required audio dependencies")
                return

            if not info["models"]:
                print("❌ No voice recognition models available")
                return

            print("🎙️  Voice input ready!")
            print(f"Available models: {list(info['models'].keys())}")
            print()
            print("Choose voice mode:")
            print("1. 🎤 Regular mode (press Enter to record 5 seconds)")
            print("2. 🔄 Continuous mode (automatic pause detection)")
            print("3. 🚪 Back to main menu")

            while True:
                choice = input("\nSelect mode (1-3): ").strip()

                if choice == "1":
                    self._regular_voice_session()
                    break
                elif choice == "2":
                    self._continuous_voice_session()
                    break
                elif choice == "3":
                    return
                else:
                    print("❌ Invalid choice. Please select 1-3.")

        except Exception as e:
            log_error(f"Voice session failed: {e}", "main", exc_info=True)
            print(f"❌ Voice session failed: {e}")

    def _regular_voice_session(self) -> None:
        """Regular voice session with manual recording."""
        voice_service = get_voice_service()
        db_manager = get_database_manager()
        rag_processor = get_rag_processor()

        print("\n🎤 REGULAR VOICE MODE")
        print("Press Enter to start recording, or 'quit' to exit")

        while True:
            print("\n" + "-" * 30)
            user_input = input("Press Enter to record (or 'quit' to exit): ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Voice session ended!")
                break

            try:
                # Record and transcribe
                print("🎤 Recording... (5 seconds)")
                result = voice_service.record_and_transcribe(duration=5.0)

                if not result["success"]:
                    print(
                        f"❌ Voice recognition failed: {result.get('error', 'Unknown error')}"
                    )
                    continue

                question = result["text"].strip()
                if not question:
                    print("⚠️  No speech detected. Please try again.")
                    continue

                print(f'🎯 Recognized: "{question}"')

                # Process as RAG query
                print("🔍 Searching for relevant information...")
                relevant_docs = db_manager.search_with_cache(question, k=3)

                if not relevant_docs:
                    print("⚠️  No relevant information found.")
                    continue

                print("🤖 Generating response...")
                rag_result = rag_processor.process_rag_query(question, relevant_docs)

                if rag_result["success"]:
                    print(f"\n💡 Answer:")
                    print(f"{rag_result['response']}")
                else:
                    print(f"❌ Failed to generate response: {rag_result.get('error')}")

            except KeyboardInterrupt:
                print("\n👋 Voice session interrupted!")
                break
            except Exception as e:
                log_error(f"Voice query failed: {e}", "main")
                print(f"❌ Error processing voice input: {e}")

    def _continuous_voice_session(self) -> None:
        """Continuous voice session with automatic pause detection."""
        try:
            from services.continuous_voice_service import ContinuousVoiceInterface

            print("\n🔄 CONTINUOUS VOICE MODE")
            print("=" * 50)
            print("🗣️  Natural conversation mode:")
            print("   • Speak your question naturally")
            print("   • Pause for 2 seconds when finished")
            print("   • Get automatic response")
            print("   • Continue with next question")
            print("   • Press Ctrl+C to exit")
            print()

            # Configuration options
            print("⚙️  Settings:")
            try:
                silence_input = input("Silence threshold in seconds [2.0]: ").strip()
                silence_threshold = float(silence_input) if silence_input else 2.0
            except ValueError:
                silence_threshold = 2.0

            try:
                min_speech_input = input("Minimum speech duration [0.5]: ").strip()
                min_speech_duration = (
                    float(min_speech_input) if min_speech_input else 0.5
                )
            except ValueError:
                min_speech_duration = 0.5

            print(
                f"✅ Using {silence_threshold}s silence threshold and {min_speech_duration}s minimum speech"
            )
            print("\nStarting continuous voice session...")

            interface = ContinuousVoiceInterface()
            success = interface.start_continuous_session(
                silence_threshold=silence_threshold,
                min_speech_duration=min_speech_duration,
            )

            if success:
                print("\n📊 Session completed!")
                interface.show_conversation_history()

        except ImportError:
            print("❌ Continuous voice service not available")
            print("💡 Install additional dependencies: pip install webrtcvad numpy")
            print("🔄 Falling back to regular voice mode...")
            self._regular_voice_session()
        except Exception as e:
            log_error(f"Continuous voice session failed: {e}", "main")
            print(f"❌ Continuous voice session failed: {e}")
            print("🔄 Falling back to regular voice mode...")
            self._regular_voice_session()

    def run_system_tests(self) -> None:
        """Run comprehensive system tests."""
        print("\n🧪 SYSTEM TESTS")
        print("=" * 50)

        tests = [
            ("Embedding System", self._test_embedding),
            ("Database Operations", self._test_database),
            ("LLM Integration", self._test_llm),
            ("Voice Input", self._test_voice),
            ("End-to-End RAG", self._test_rag_pipeline),
        ]

        results = {}

        for test_name, test_func in tests:
            print(f"\n🔬 Testing {test_name}...")
            try:
                result = test_func()
                results[test_name] = result
                emoji = "✅" if result else "❌"
                print(f"   {emoji} {test_name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                results[test_name] = False
                log_error(f"Test {test_name} failed: {e}", "main")
                print(f"   ❌ {test_name}: ERROR - {e}")

        # Summary
        print(f"\n📊 Test Summary:")
        passed = sum(results.values())
        total = len(results)
        print(f"   Passed: {passed}/{total} ({passed/total*100:.1f}%)")

        if passed == total:
            print("   🎉 All tests passed!")
        else:
            print("   ⚠️  Some tests failed - check logs for details")

    def _test_embedding(self) -> bool:
        """Test embedding system."""
        try:
            embedding_factory = get_embedding_factory()
            # Test English
            eng_emb = embedding_factory.get_mixed_language_embedding(
                "test english text"
            )
            # Test Bangla (may fallback to English)
            bn_emb = embedding_factory.get_mixed_language_embedding("পরীক্ষা")
            return len(eng_emb) > 0 and len(bn_emb) > 0
        except Exception:
            return False

    def _test_database(self) -> bool:
        """Test database operations."""
        try:
            db_manager = get_database_manager()
            return db_manager.test_connection()
        except Exception:
            return False

    def _test_llm(self) -> bool:
        """Test LLM integration."""
        try:
            model_manager = get_model_manager()
            response = model_manager.generate_response("Hello", max_tokens=5)
            return response is not None and len(response.strip()) > 0
        except Exception:
            return False

    def _test_voice(self) -> bool:
        """Test voice input system."""
        try:
            voice_service = get_voice_service()
            info = voice_service.get_service_info()
            return len(info["models"]) > 0
        except Exception:
            return False

    def _test_rag_pipeline(self) -> bool:
        """Test end-to-end RAG pipeline."""
        try:
            db_manager = get_database_manager()
            rag_processor = get_rag_processor()

            # Get some documents
            docs = db_manager.search_with_cache("test", k=1)
            if not docs:
                return False

            # Process RAG query
            result = rag_processor.process_rag_query("What is this about?", docs)
            return result["success"]
        except Exception:
            return False

    def show_main_menu(self) -> None:
        """Display main menu options."""
        print("\n📋 MAIN MENU")
        for option, description in MAIN_MENU_OPTIONS.items():
            print(f"{option}. {description}")

    def run(self) -> None:
        """Run the main application loop."""
        try:
            log_info(
                f"Starting {APP_NAME} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "main",
            )

            self.print_banner()

            while True:
                self.show_main_menu()

                try:
                    choice = input("\nSelect option (1-11): ").strip()

                    if choice == "1":
                        self.show_system_status()
                    elif choice == "2":
                        self.process_documents()
                    elif choice == "3":
                        self.test_database_queries()
                    elif choice == "4":
                        self.interactive_chat_session()
                    elif choice == "5":
                        self.voice_input_session()
                    elif choice == "6":
                        print(
                            "🎤 Single voice query feature - Use option 5 for voice session"
                        )
                    elif choice == "7":
                        print(
                            "🌐 Mixed language demo - Use option 4 for interactive chat"
                        )
                    elif choice == "8":
                        self.run_system_tests()
                    elif choice == "9":
                        print("🔬 Voice tests included in option 8 (System Tests)")
                    elif choice == "10":
                        self.check_dependencies()
                    elif choice == "11":
                        print("👋 Goodbye!")
                        break
                    else:
                        print("❌ Invalid option. Please select 1-11.")

                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    log_error(f"Menu option error: {e}", "main")
                    print(f"❌ Error: {e}")

        except Exception as e:
            log_error(f"Application error: {e}", "main", exc_info=True)
            print(f"❌ Application error: {e}")

        finally:
            log_info("BanglaRAG application shutting down", "main")


def main():
    """Main entry point."""
    try:
        app = BanglaRAGApplication()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
