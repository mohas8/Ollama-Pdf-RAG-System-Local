#!/usr/bin/env python3
"""
Flask API for embeddable chatbot with streaming support.
Integrates with BanglaRAG system for course-based Q&A.
Includes authentication, web search, and video search features.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import (
    Flask,
    request,
    jsonify,
    Response,
    stream_with_context,
    send_from_directory,
    make_response,
)
from flask_cors import CORS
import json
import time
from typing import Generator, Dict, Any
import os

from core.logging_config import BanglaRAGLogger, log_info, log_error
from services.database_service import get_database_manager
from services.llm_service import get_model_manager, get_rag_processor
from services.embedding_service import get_embedding_factory

# Import auth module
from auth import (
    register_user,
    login_user,
    logout_user,
    verify_token,
    require_auth,
    require_teacher,
)

# Import Gemini service
from gemini_service import (
    search_web,
    search_videos,
    generate_mcq_questions,
    explain_topic,
)

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Enable CORS with credentials

logger = BanglaRAGLogger()

# Initialize services
db_manager = None
db_manager_pdf = None  # Second database for PDF book
model_manager = None
rag_processor = None


def initialize_services():
    """Initialize BanglaRAG services with PDF database ONLY."""
    global db_manager, db_manager_pdf, model_manager, rag_processor

    try:
        log_info("Initializing chatbot API services...", "api")
        from services.database_service import DatabaseFactory

        # ONLY use PDF book (Cormen algorithms) - banglarag collection
        try:
            db_manager = DatabaseFactory.create_chroma_database(
                collection_name="banglarag"
            )
            log_info("✅ PDF database (algorithm book) loaded successfully", "api")
            log_info("📚 Using ONLY the algorithm book database", "api")
        except Exception as e:
            log_error(f"❌ Failed to load PDF database: {e}", "api")
            return False

        # Set db_manager_pdf to None (not used)
        db_manager_pdf = None

        model_manager = get_model_manager()
        rag_processor = get_rag_processor()

        log_info(
            "Chatbot API services initialized with PDF database ONLY (algorithm book)",
            "api",
        )
        return True
    except Exception as e:
        log_error(f"Failed to initialize services: {e}", "api", exc_info=True)
        return False


def is_query_relevant(query: str, relevant_docs: list, language: str = "english") -> tuple:
    """
    Check if the query is relevant to the knowledge base (algorithms, data structures, CS topics).
    
    Returns:
        tuple: (is_relevant: bool, decline_message: str or None)
    """
    query_lower = query.lower()
    
    # Keywords that indicate relevant topics (algorithms, data structures, CS, course content)
    relevant_keywords_en = [
        # Algorithms
        "algorithm", "sort", "search", "binary", "linear", "merge", "quick", "heap",
        "bubble", "insertion", "selection", "radix", "counting", "bucket",
        "dijkstra", "bellman", "floyd", "kruskal", "prim", "bfs", "dfs",
        "dynamic programming", "greedy", "divide and conquer", "backtracking",
        "recursion", "iteration", "complexity", "big o", "time complexity", "space complexity",
        "asymptotic", "worst case", "best case", "average case",
        # Data Structures
        "data structure", "array", "linked list", "stack", "queue", "tree", "graph",
        "hash", "heap", "trie", "b-tree", "red-black", "avl", "binary tree",
        "binary search tree", "bst", "priority queue", "deque", "set", "map",
        "dictionary", "matrix", "adjacency", "vertex", "edge", "node",
        # CS fundamentals
        "pointer", "memory", "allocation", "traversal", "inorder", "preorder", "postorder",
        "level order", "depth first", "breadth first", "hashing", "collision",
        "load factor", "amortized", "recurrence", "master theorem",
        # Course-related
        "cormen", "introduction to algorithms", "clrs", "chapter", "page",
        "exercise", "problem", "pseudocode", "proof", "theorem", "lemma",
    ]
    
    relevant_keywords_bn = [
        # Bangla keywords for algorithms and data structures
        "অ্যালগরিদম", "সার্চ", "সর্ট", "ডাটা স্ট্রাকচার", "অ্যারে", "লিংকড লিস্ট",
        "স্ট্যাক", "কিউ", "ট্রি", "গ্রাফ", "হ্যাশ", "হিপ", "জটিলতা", "সময় জটিলতা",
        "বাইনারি", "রিকার্সন", "ডায়নামিক প্রোগ্রামিং", "গ্রিডি", "ডিভাইড অ্যান্ড কনকার",
    ]
    
    # Combine keywords based on language
    all_keywords = relevant_keywords_en + relevant_keywords_bn
    
    # Check if query contains any relevant keywords
    has_relevant_keyword = any(keyword in query_lower for keyword in all_keywords)
    
    # Check if we found good quality results from the knowledge base
    has_good_results = False
    if relevant_docs:
        # Check if at least one document has reasonable content overlap with the query
        query_words = set(query_lower.split())
        for doc in relevant_docs:
            doc_content = doc.page_content.lower() if hasattr(doc, 'page_content') else str(doc).lower()
            doc_words = set(doc_content.split())
            # Check for meaningful overlap (at least 2 query words appear in document)
            overlap = query_words & doc_words
            if len(overlap) >= 2 or has_relevant_keyword:
                has_good_results = True
                break
    
    # Query is relevant if it has relevant keywords OR has good results from knowledge base
    is_relevant = has_relevant_keyword or has_good_results
    
    if not is_relevant:
        if language == "bangla":
            decline_message = "দুঃখিত, আমি শুধুমাত্র অ্যালগরিদম, ডাটা স্ট্রাকচার এবং কম্পিউটার বিজ্ঞান সম্পর্কিত প্রশ্নের উত্তর দিতে পারি। অনুগ্রহ করে এই বিষয়গুলির সাথে সম্পর্কিত একটি প্রশ্ন করুন।"
        else:
            decline_message = "I'm sorry, but I can only answer questions related to algorithms, data structures, and computer science topics covered in the course materials. Please ask a question related to these subjects."
        return False, decline_message
    
    return True, None


def search_dual_databases(query: str, k: int = 3):
    """
    Search both course materials and PDF database with smart prioritization.

    ALWAYS prioritizes PDF database (algorithm book) first.
    Only uses course materials for course-specific questions (syllabus, schedule, etc.)
    or as fallback if PDF has no results.
    """
    all_results = []

    # Search ONLY the PDF database (algorithm book)
    try:
        all_results = db_manager.search_with_cache(query, k=k)
        log_info(
            f"✅ Found {len(all_results)} results from PDF database (algorithm book)",
            "api",
        )

        # Add source tag to mark as PDF
        for doc in all_results:
            if hasattr(doc, "metadata"):
                doc.metadata["search_source"] = "pdf"
                # Ensure source shows as the algorithm book
                if (
                    "source" not in doc.metadata
                    or "course_knowledge" in doc.metadata.get("source", "").lower()
                ):
                    doc.metadata["source"] = "Cormen - Introduction to Algorithms.pdf"

    except Exception as e:
        log_error(f"❌ Error searching PDF database: {e}", "api")
        all_results = []

    if not all_results:
        log_info("⚠️ No relevant results found in PDF database", "api")

    return all_results


@app.route("/")
def index():
    """Serve the course example page."""
    return send_from_directory(os.path.dirname(__file__), "course-example.html")


@app.route("/login")
def login_page():
    """Serve the login page."""
    return send_from_directory(os.path.dirname(__file__), "login.html")


@app.route("/teachers")
def teachers():
    """Serve the teachers dashboard page (legacy)."""
    return send_from_directory(os.path.dirname(__file__), "teachers.html")


@app.route("/teachers-dashboard")
def teachers_dashboard():
    """Serve the new teachers dashboard page."""
    return send_from_directory(os.path.dirname(__file__), "teachers-dashboard.html")


@app.route("/student-dashboard")
def student_dashboard():
    """Serve the student dashboard page."""
    return send_from_directory(os.path.dirname(__file__), "student-dashboard.html")


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static files - no restrictions."""
    try:
        return send_from_directory(os.path.dirname(__file__), filename)
    except:
        return jsonify({"error": "File not found"}), 404


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {"status": "ok", "service": "BanglaRAG Chatbot API", "version": "2.0.0"}
    )


@app.route("/api/models", methods=["GET"])
def get_models():
    """Get available models."""
    try:
        if not model_manager:
            return jsonify({"error": "Service not initialized"}), 503

        models = model_manager.get_available_models()
        current_model = model_manager._active_model or model_manager.preferred_model

        return jsonify(
            {"models": models, "current_model": current_model, "success": True}
        )
    except Exception as e:
        log_error(f"Error getting models: {e}", "api")
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/set-model", methods=["POST"])
def set_model():
    """Set the active model."""
    try:
        data = request.json
        model_name = data.get("model")

        if not model_name:
            return jsonify({"error": "Model name required", "success": False}), 400

        if not model_manager:
            return jsonify({"error": "Service not initialized", "success": False}), 503

        # Set the active model directly
        available_models = model_manager.get_available_models()
        if model_name in available_models:
            model_manager._active_model = model_name
            success = True
        else:
            success = False

        return jsonify(
            {
                "success": success,
                "current_model": model_manager._active_model
                or model_manager.preferred_model,
            }
        )
    except Exception as e:
        log_error(f"Error setting model: {e}", "api")
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """Non-streaming chat endpoint."""
    try:
        data = request.json
        query = data.get("query", "").strip()
        k = data.get("k", 3)  # Number of relevant documents to retrieve
        language = data.get("language", "english")  # Get selected language

        if not query:
            return jsonify({"error": "Query is required", "success": False}), 400

        if not db_manager or not rag_processor:
            return jsonify({"error": "Service not initialized", "success": False}), 503

        # Search for relevant documents from both databases
        log_info(f"Processing query: {query}", "api")
        relevant_docs = search_dual_databases(query, k=k)

        # Check if query is relevant to the knowledge base
        is_relevant, decline_message = is_query_relevant(query, relevant_docs, language)
        if not is_relevant:
            log_info(f"Query declined as irrelevant: {query}", "api")
            return jsonify(
                {
                    "response": decline_message,
                    "sources": [],
                    "success": True,
                    "declined": True,
                }
            )

        if not relevant_docs:
            no_results_msg = "I couldn't find relevant information in the course materials to answer your question." if language == "english" else "আমি আপনার প্রশ্নের উত্তর দিতে কোর্স উপাদানে প্রাসঙ্গিক তথ্য খুঁজে পাইনি।"
            return jsonify(
                {
                    "response": no_results_msg,
                    "sources": [],
                    "success": True,
                }
            )

        # Generate response using RAG with language parameter
        rag_result = rag_processor.process_rag_query(query, relevant_docs, language=language)

        if rag_result["success"]:
            sources = [
                {
                    "content": (
                        doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content
                    ),
                    "metadata": doc.metadata,
                }
                for doc in relevant_docs[:3]
            ]

            return jsonify(
                {
                    "response": rag_result["response"],
                    "sources": sources,
                    "model": (
                        model_manager.get_current_model()
                        if model_manager
                        else "unknown"
                    ),
                    "success": True,
                }
            )
        else:
            return (
                jsonify(
                    {
                        "error": rag_result.get("error", "Failed to generate response"),
                        "success": False,
                    }
                ),
                500,
            )

    except Exception as e:
        log_error(f"Chat error: {e}", "api", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Streaming chat endpoint with real-time response."""
    try:
        data = request.json
        query = data.get("query", "").strip()
        k = data.get("k", 3)
        language = data.get("language", "english")  # Get selected language

        if not query:
            return jsonify({"error": "Query is required"}), 400

        if not db_manager or not model_manager:
            return jsonify({"error": "Service not initialized"}), 503

        def generate() -> Generator[str, None, None]:
            """Generate streaming response."""
            try:
                # Send initial status
                yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"

                # Search for relevant documents from both databases
                relevant_docs = search_dual_databases(query, k=k)

                # Check if query is relevant to the knowledge base
                is_relevant, decline_message = is_query_relevant(query, relevant_docs, language)
                if not is_relevant:
                    log_info(f"Query declined as irrelevant: {query}", "api")
                    yield f"data: {json.dumps({'type': 'declined', 'message': decline_message})}\n\n"
                    return

                if not relevant_docs:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant information found'})}\n\n"
                    return

                # Send sources
                sources = [
                    {
                        "content": (
                            doc.page_content[:200] + "..."
                            if len(doc.page_content) > 200
                            else doc.page_content
                        ),
                        "metadata": doc.metadata,
                    }
                    for doc in relevant_docs[:3]
                ]
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

                # Send generation status
                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"

                # Build context from relevant documents
                context = "\n\n".join(
                    [
                        f"Document {i+1}:\n{doc.page_content}"
                        for i, doc in enumerate(relevant_docs)
                    ]
                )

                # Create prompt based on selected language
                if language == "bangla":
                    prompt = f"""নিম্নলিখিত কোর্স উপাদানের উপর ভিত্তি করে প্রশ্নের উত্তর দিন। সুনির্দিষ্ট হোন এবং প্রদত্ত তথ্য উদ্ধৃত করুন। বাংলায় উত্তর দিন।

কোর্স উপাদান:
{context}

প্রশ্ন: {query}

উত্তর:"""
                else:
                    prompt = f"""Based on the following course materials, answer the question. Be specific and cite the information provided.

Course Materials:
{context}

Question: {query}

Answer:"""

                # Stream response from model
                current_model = (
                    model_manager._active_model or model_manager.preferred_model
                )

                # Use Ollama's streaming API
                import requests

                response = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={
                        "model": current_model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {"temperature": 0.7, "num_predict": 2048},
                    },
                    stream=True,
                )

                # Stream tokens as they arrive
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                token = chunk["response"]
                                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

                            if chunk.get("done", False):
                                yield f"data: {json.dumps({'type': 'done', 'model': current_model})}\n\n"
                                break
                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                log_error(f"Streaming error: {e}", "api", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except Exception as e:
        log_error(f"Stream setup error: {e}", "api", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/collections", methods=["GET"])
def get_collections():
    """Get available collections."""
    try:
        if not db_manager:
            return jsonify({"error": "Service not initialized"}), 503

        # For now, return default collection
        return jsonify(
            {
                "collections": ["course_materials"],
                "current": "course_materials",
                "success": True,
            }
        )
    except Exception as e:
        log_error(f"Error getting collections: {e}", "api")
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/teachers/generate-questions", methods=["POST"])
def generate_questions():
    """
    Generate questions automatically from course content.

    Request body:
    {
        "module": "MODULE 1" or "all",
        "difficulty": "easy" | "medium" | "hard" | "mixed",
        "num_questions": 5,
        "question_types": ["multiple-choice", "true-false", "short-answer", "explain"]
    }
    """
    try:
        data = request.json
        module = data.get("module", "all")
        difficulty = data.get("difficulty", "mixed")
        num_questions = data.get("num_questions", 5)
        question_types = data.get("question_types", ["multiple-choice"])

        log_info(
            f"Generating {num_questions} questions for module: {module}, "
            f"difficulty: {difficulty}, types: {question_types}",
            "api",
        )

        # Retrieve relevant course content
        if module == "all":
            query = "data structures algorithms course content"
        else:
            query = f"{module} content topics concepts"

        # Get more context for question generation
        course_chunks = db_manager.search_with_cache(
            query, k=min(num_questions * 2, 10)
        )

        if not course_chunks:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No course content found for question generation",
                    }
                ),
                404,
            )

        # Prepare context from course materials
        context = "\n\n".join(
            [
                f"SECTION: {chunk.metadata.get('module', 'Unknown')}\n{chunk.page_content}"
                for chunk in course_chunks
            ]
        )

        # Build prompt for question generation
        type_instructions = {
            "multiple-choice": "multiple choice questions with 4 options (A, B, C, D)",
            "true-false": "true/false questions",
            "short-answer": "short answer questions requiring 1-2 sentence responses",
            "explain": "explanation questions asking to describe or explain concepts",
        }

        selected_types = [
            type_instructions[t] for t in question_types if t in type_instructions
        ]
        types_text = ", ".join(selected_types)

        difficulty_guidance = {
            "easy": "Basic recall and understanding level questions",
            "medium": "Application and analysis level questions",
            "hard": "Advanced synthesis and evaluation level questions",
            "mixed": "Mix of easy, medium, and hard questions",
        }

        prompt = f"""You are an expert educator creating assessment questions for a Data Structures course.

COURSE CONTENT:
{context}

TASK: Generate {num_questions} high-quality assessment questions.

REQUIREMENTS:
- Question Types: {types_text}
- Difficulty Level: {difficulty_guidance[difficulty]}
- Module Focus: {module if module != 'all' else 'All course modules'}
- Questions must be clear, unambiguous, and directly related to the course content above
- For multiple choice: provide exactly 4 options with clear correct answer
- For true/false: provide clear reasoning for the answer
- Questions should test conceptual understanding, not just memorization

CRITICAL: You MUST respond with ONLY valid JSON - no markdown, no explanation, no code blocks.
Start your response with [ and end with ]

FORMAT (JSON only):
[
  {{
    "question": "What is the time complexity of...",
    "type": "multiple-choice",
    "options": ["O(1)", "O(n)", "O(log n)", "O(n^2)"],
    "answer": "O(n)",
    "difficulty": "medium",
    "module": "{module if module != 'all' else 'General'}"
  }}
]

Generate exactly {num_questions} questions. JSON ONLY - no other text:"""

        # Generate questions using LLM
        log_info("Sending prompt to LLM for question generation", "api")
        response = model_manager.generate_response(prompt)

        if not response:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No response from LLM. Please check if Ollama is running.",
                    }
                ),
                500,
            )

        log_info(f"LLM Response (first 300 chars): {response[:300]}", "api")

        # Parse JSON response with multiple strategies
        questions = []
        import re

        # Strategy 1: Clean markdown and parse as JSON array
        try:
            response_clean = response.strip()
            response_clean = re.sub(
                r"^```json\s*", "", response_clean, flags=re.IGNORECASE
            )
            response_clean = re.sub(r"^```\s*", "", response_clean)
            response_clean = re.sub(r"\s*```$", "", response_clean)

            json_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", response_clean)
            if json_match:
                json_str = json_match.group()
                questions = json.loads(json_str)

                if not isinstance(questions, list):
                    questions = []
                else:
                    log_info(
                        f"Strategy 1 SUCCESS: Parsed {len(questions)} questions", "api"
                    )
        except Exception as e:
            log_error(f"Strategy 1 failed: {e}", "api")

        # Strategy 2: Try parsing entire cleaned response
        if not questions:
            try:
                questions = json.loads(response_clean)
                if isinstance(questions, list):
                    log_info(
                        f"Strategy 2 SUCCESS: Parsed {len(questions)} questions", "api"
                    )
                else:
                    questions = []
            except Exception as e:
                log_error(f"Strategy 2 failed: {e}", "api")

        # Strategy 3: Extract individual question objects
        if not questions:
            try:
                question_pattern = r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}'
                matches = re.finditer(question_pattern, response, re.DOTALL)

                for match in matches:
                    try:
                        q = json.loads(match.group())
                        if "question" in q and "answer" in q:
                            questions.append(q)
                    except:
                        continue

                if questions:
                    log_info(
                        f"Strategy 3 SUCCESS: Extracted {len(questions)} questions",
                        "api",
                    )
            except Exception as e:
                log_error(f"Strategy 3 failed: {e}", "api")

        # If we successfully parsed questions
        if questions:
            log_info(f"Successfully generated {len(questions)} questions", "api")

            return jsonify(
                {"success": True, "questions": questions, "count": len(questions)}
            )

        # All parsing strategies failed
        log_error(f"All parsing strategies failed", "api")
        log_error(f"Raw response (first 1000 chars): {response[:1000]}", "api")

        return (
            jsonify(
                {
                    "success": False,
                    "error": "Failed to parse generated questions from LLM",
                    "raw_response": response[:1000],
                    "hint": "The LLM may have generated an invalid format. Try again with different settings.",
                    "strategies_tried": [
                        "JSON array extraction",
                        "Direct JSON parse",
                        "Individual object extraction",
                    ],
                }
            ),
            500,
        )

    except Exception as e:
        log_error(f"Error generating questions: {e}", "api", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/teachers/generate-questions/stream", methods=["POST"])
def generate_questions_stream():
    """
    Stream questions as they're generated for real-time display.
    Uses Server-Sent Events (SSE) for streaming.
    """
    try:
        data = request.json
        module = data.get("module", "all")
        difficulty = data.get("difficulty", "mixed")
        num_questions = data.get("num_questions", 5)
        question_types = data.get("question_types", ["multiple-choice"])

        log_info(
            f"Streaming {num_questions} questions for module: {module}",
            "api",
        )

        def generate_stream():
            """Generator function for SSE streaming."""
            try:
                # Retrieve relevant course content
                if module == "all":
                    query = "data structures algorithms course content"
                else:
                    query = f"{module} content topics concepts"

                # Send initial status
                yield f"data: {json.dumps({'type': 'status', 'message': 'Searching course materials...'})}\n\n"

                course_chunks = db_manager.search_with_cache(
                    query, k=min(num_questions * 2, 10)
                )

                if not course_chunks:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'No course content found'})}\n\n"
                    return

                # Prepare context
                context = "\n\n".join(
                    [
                        f"SECTION: {chunk.metadata.get('module', 'Unknown')}\n{chunk.page_content}"
                        for chunk in course_chunks
                    ]
                )

                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating questions...'})}\n\n"

                # Build prompt for streaming
                type_instructions = {
                    "multiple-choice": "multiple choice questions with 4 options",
                    "true-false": "true/false questions",
                    "short-answer": "short answer questions",
                    "explain": "explanation questions",
                }

                selected_types = [
                    type_instructions[t]
                    for t in question_types
                    if t in type_instructions
                ]
                types_text = ", ".join(selected_types)

                difficulty_guidance = {
                    "easy": "Basic recall level",
                    "medium": "Application level",
                    "hard": "Advanced synthesis level",
                    "mixed": "Mix of easy, medium, and hard",
                }

                # Use non-streaming API but send questions as they're generated
                # Generate all questions at once, then stream them to client
                prompt = f"""You are an expert educator creating assessment questions for a Data Structures course.

COURSE CONTENT:
{context[:2000]}

TASK: Generate {num_questions} high-quality assessment questions.

REQUIREMENTS:
- Question Types: {types_text}
- Difficulty Level: {difficulty_guidance[difficulty]}
- Module Focus: {module if module != 'all' else 'All course modules'}
- Questions must be clear, unambiguous, and directly related to the course content above
- For multiple choice: provide exactly 4 options with clear correct answer
- For true/false: provide clear reasoning for the answer
- Questions should test conceptual understanding, not just memorization

CRITICAL: You MUST respond with ONLY valid JSON - no markdown, no explanation, no code blocks.
Start your response with [ and end with ]

FORMAT (JSON only):
[
  {{
    "question": "What is...",
    "type": "multiple-choice",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Option A",
    "difficulty": "medium",
    "module": "{module if module != 'all' else 'General'}"
  }}
]

Generate exactly {num_questions} questions. JSON ONLY - no other text:"""

                # Generate questions (this happens in one call)
                response = model_manager.generate_response(prompt)

                if not response:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'No response from LLM'})}\n\n"
                    return

                log_info(f"LLM Response (first 500 chars): {response[:500]}", "api")

                # Parse and stream questions one by one
                import re

                questions = []

                # Strategy 1: Clean and parse as JSON array
                try:
                    response_clean = response.strip()
                    # Remove markdown code blocks
                    response_clean = re.sub(
                        r"^```json\s*", "", response_clean, flags=re.IGNORECASE
                    )
                    response_clean = re.sub(r"^```\s*", "", response_clean)
                    response_clean = re.sub(r"\s*```$", "", response_clean)

                    # Try to extract JSON array
                    json_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", response_clean)
                    if json_match:
                        json_str = json_match.group()
                        questions = json.loads(json_str)
                        log_info(
                            f"Strategy 1 SUCCESS: Parsed {len(questions)} questions",
                            "api",
                        )
                except Exception as e:
                    log_error(f"Strategy 1 failed: {e}", "api")

                # Strategy 2: Try parsing entire cleaned response
                if not questions:
                    try:
                        questions = json.loads(response_clean)
                        if isinstance(questions, list):
                            log_info(
                                f"Strategy 2 SUCCESS: Parsed {len(questions)} questions",
                                "api",
                            )
                        else:
                            questions = []
                    except Exception as e:
                        log_error(f"Strategy 2 failed: {e}", "api")

                # Strategy 3: Extract individual question objects
                if not questions:
                    try:
                        # Find all JSON objects that look like questions
                        question_pattern = r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}'
                        matches = re.finditer(question_pattern, response, re.DOTALL)

                        for match in matches:
                            try:
                                q = json.loads(match.group())
                                if "question" in q and "answer" in q:
                                    questions.append(q)
                            except:
                                continue

                        if questions:
                            log_info(
                                f"Strategy 3 SUCCESS: Extracted {len(questions)} questions",
                                "api",
                            )
                    except Exception as e:
                        log_error(f"Strategy 3 failed: {e}", "api")

                # If we got questions, stream them
                if questions:
                    for idx, question in enumerate(questions, 1):
                        # Ensure required fields
                        if "question" not in question or "answer" not in question:
                            continue

                        # Set defaults for missing fields
                        question.setdefault("type", "short-answer")
                        question.setdefault("difficulty", difficulty)
                        question.setdefault(
                            "module", module if module != "all" else "General"
                        )

                        yield f"data: {json.dumps({'type': 'question', 'data': question, 'index': idx})}\n\n"
                        time.sleep(0.3)  # Small delay for visual effect

                    # Send completion
                    yield f"data: {json.dumps({'type': 'complete', 'total': len(questions)})}\n\n"
                else:
                    # All strategies failed
                    log_error(
                        f"All parsing strategies failed. Raw response: {response[:1000]}",
                        "api",
                    )
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to parse questions. LLM may have generated invalid format. Please try again.', 'raw': response[:500]})}\n\n"

            except Exception as e:
                log_error(f"Error in streaming generation: {e}", "api", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return Response(
            stream_with_context(generate_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        log_error(f"Error setting up stream: {e}", "api", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== AUTHENTICATION ENDPOINTS ====================

@app.route("/api/auth/register", methods=["POST"])
def api_register():
    """Register a new user (student or teacher)."""
    try:
        data = request.json
        username = data.get("username", "").strip()
        email = data.get("email", "").strip()
        password = data.get("password", "")
        role = data.get("role", "student")
        full_name = data.get("full_name", "").strip()

        if not username or not email or not password:
            return jsonify({"success": False, "error": "All fields are required"}), 400

        result = register_user(username, email, password, role, full_name)
        
        if result["success"]:
            log_info(f"New user registered: {username} ({role})", "api")
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        log_error(f"Registration error: {e}", "api", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    """Login user and return token."""
    try:
        data = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "")

        if not username or not password:
            return jsonify({"success": False, "error": "Username and password required"}), 400

        result = login_user(username, password)
        
        if result["success"]:
            log_info(f"User logged in: {username}", "api")
            response = make_response(jsonify(result))
            # Set cookie for convenience
            response.set_cookie(
                "auth_token",
                result["token"],
                max_age=7*24*60*60,  # 7 days
                httponly=True,
                samesite="Lax"
            )
            return response
        else:
            return jsonify(result), 401

    except Exception as e:
        log_error(f"Login error: {e}", "api", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    """Logout user."""
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            token = request.cookies.get("auth_token")
        
        if token:
            result = logout_user(token)
            response = make_response(jsonify(result))
            response.delete_cookie("auth_token")
            return response
        
        return jsonify({"success": True, "message": "Logged out"})

    except Exception as e:
        log_error(f"Logout error: {e}", "api", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/auth/verify", methods=["GET"])
def api_verify_token():
    """Verify token and return user info."""
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            token = request.cookies.get("auth_token")
        
        if not token:
            return jsonify({"valid": False, "error": "No token provided"}), 401
        
        result = verify_token(token)
        return jsonify(result)

    except Exception as e:
        log_error(f"Token verification error: {e}", "api", exc_info=True)
        return jsonify({"valid": False, "error": str(e)}), 500


# ==================== SEARCH ENDPOINTS (Gemini) ====================

@app.route("/api/search/web", methods=["POST"])
def api_search_web():
    """Search web for information using Gemini."""
    try:
        data = request.json
        query = data.get("query", "").strip()
        num_results = data.get("num_results", 5)

        if not query:
            return jsonify({"success": False, "error": "Query is required"}), 400

        log_info(f"Web search: {query}", "api")
        result = search_web(query, num_results)
        
        return jsonify(result)

    except Exception as e:
        log_error(f"Web search error: {e}", "api", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/search/videos", methods=["POST"])
def api_search_videos():
    """Find educational videos using Gemini."""
    try:
        data = request.json
        topic = data.get("topic", "").strip()
        num_videos = data.get("num_videos", 5)

        if not topic:
            return jsonify({"success": False, "error": "Topic is required"}), 400

        log_info(f"Video search: {topic}", "api")
        result = search_videos(topic, num_videos)
        
        return jsonify(result)

    except Exception as e:
        log_error(f"Video search error: {e}", "api", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/generate/mcq", methods=["POST"])
def api_generate_mcq():
    """Generate MCQ questions using Gemini."""
    try:
        data = request.json
        topic = data.get("topic", "").strip()
        num_questions = data.get("num_questions", 5)
        difficulty = data.get("difficulty", "mixed")

        if not topic:
            return jsonify({"success": False, "error": "Topic is required"}), 400

        log_info(f"MCQ generation: {topic}, {num_questions} questions", "api")
        result = generate_mcq_questions(topic, num_questions, difficulty)
        
        return jsonify(result)

    except Exception as e:
        log_error(f"MCQ generation error: {e}", "api", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/explain", methods=["POST"])
def api_explain_topic():
    """Get detailed explanation of a topic using Gemini."""
    try:
        data = request.json
        topic = data.get("topic", "").strip()
        detail_level = data.get("detail_level", "comprehensive")

        if not topic:
            return jsonify({"success": False, "error": "Topic is required"}), 400

        log_info(f"Topic explanation: {topic}", "api")
        result = explain_topic(topic, detail_level)
        
        return jsonify(result)

    except Exception as e:
        log_error(f"Explain error: {e}", "api", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting BanglaRAG Chatbot API...")

    if initialize_services():
        print("✅ Services initialized successfully")
        print("🌐 API running at http://localhost:5000")
        print("📡 Endpoints:")
        print("   - GET  /api/health")
        print("   - GET  /api/models")
        print("   - POST /api/set-model")
        print("   - POST /api/chat")
        print("   - POST /api/chat/stream (SSE)")
        print("   - GET  /api/collections")
        print("   - POST /api/teachers/generate-questions")
        print("   - POST /api/teachers/generate-questions/stream (SSE)")
        print("\n🔐 Auth Endpoints:")
        print("   - POST /api/auth/register")
        print("   - POST /api/auth/login")
        print("   - POST /api/auth/logout")
        print("   - GET  /api/auth/verify")
        print("\n🔍 Search Endpoints (Gemini):")
        print("   - POST /api/search/web")
        print("   - POST /api/search/videos")
        print("   - POST /api/generate/mcq")
        print("   - POST /api/explain")
        print("\n📄 Pages:")
        print("   - GET  /                  (Student chatbot)")
        print("   - GET  /login             (Login/Register)")
        print("   - GET  /student-dashboard (Student dashboard)")
        print("   - GET  /teachers-dashboard (Teachers dashboard)")
        print("\n🎤 Ready to receive requests!")

        app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
    else:
        print("❌ Failed to initialize services")
        sys.exit(1)
