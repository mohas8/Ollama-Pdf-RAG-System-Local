# 🎯 MASTER PROMPT: BanglaRAG Chat Web Interface

## 📋 Project Overview

Build a **complete, production-ready educational chatbot web interface** with dual-database RAG (Retrieval-Augmented Generation), real-time streaming, voice input, multi-language support, and smart source prioritization. The system should be embeddable as a floating widget on course pages.

**Target Use Case:** University course assistant that answers questions using both course materials (TXT) and reference textbooks (PDF), with intelligent source selection based on question type.

---

## 🏗️ System Architecture

### Backend: Flask REST API
- **Framework:** Flask 2.x with CORS enabled
- **Streaming:** Server-Sent Events (SSE) for real-time token streaming
- **Dual Database:** ChromaDB with two collections:
  - `course_materials` - Course syllabus, modules, assignments (TXT file)
  - `banglarag` - Reference textbook (PDF documents)
- **LLM:** Ollama with model management (llama3.2, qwen2, phi3)
- **Embeddings:** nomic-embed-text via Ollama API

### Frontend: Embeddable Widget
- **Technology:** Vanilla JavaScript (ES6+), CSS3, HTML5
- **Design:** Floating chatbot widget with slide-up animation
- **Responsive:** Mobile-first, works on all screen sizes
- **Components:** Chat window, message display, streaming indicator, voice input, model/language selectors

---

## 🎨 Core Features (REQUIRED)

### 1. ✅ Smart Dual-Database Search with Source Prioritization

**Logic:**
```python
def search_dual_databases(query: str, k: int = 3):
    """
    Intelligently search both databases with smart prioritization:
    - Course-specific questions → ONLY course materials
    - Theory questions → Prefer PDF, then course materials
    - Mixed queries → Balanced results
    """
    
    # Course-specific keywords
    course_keywords = [
        "module", "syllabus", "prerequisite", "instructor",
        "course", "assignment", "exam", "lecture", "lab",
        "schedule", "grading", "deadline", "section"
    ]
    
    # Check if query is course-specific
    is_course_query = any(kw in query.lower() for kw in course_keywords)
    
    if is_course_query:
        # ONLY search course materials
        return course_db.search(query, k=k)
    else:
        # Theory question - prefer PDF
        pdf_results = pdf_db.search(query, k=k)
        course_results = course_db.search(query, k=k)
        
        # Return 60% PDF, 40% course materials
        return pdf_results[:k//2 + 1] + course_results[:k//2]
```

**Key Points:**
- Add `search_source` metadata tag to each document
- Log which database was prioritized for each query
- Display source information in citations

---

### 2. 🔄 Real-Time Streaming Responses

**Backend Endpoint:**
```python
@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """
    Stream LLM responses token-by-token using SSE.
    
    Request Body:
    {
        "query": "What is an array?",
        "k": 3,  # number of context documents
        "language": "english"  # or "bangla"
    }
    
    SSE Events:
    - status: "Searching knowledge base..."
    - sources: [{"content": "...", "metadata": {...}}]
    - token: {"token": "word"} (repeated for each word)
    - done: {"model": "llama3.2", "tokens": 150}
    - error: {"message": "Error text"}
    """
    
    def generate():
        # 1. Search databases
        yield f"data: {json.dumps({'type': 'status', 'message': 'Searching...'})}\n\n"
        
        docs = search_dual_databases(query, k)
        
        # 2. Send sources
        yield f"data: {json.dumps({'type': 'sources', 'sources': format_sources(docs)})}\n\n"
        
        # 3. Build prompt with language instruction
        language_instruction = (
            "Please answer in Bengali (বাংলা ভাষায়)." 
            if language == "bangla" 
            else "Please answer in English."
        )
        
        prompt = f"""Based on the following course materials, answer the question. 
Be specific and cite the information provided. {language_instruction}

Context:
{format_context(docs)}

Question: {query}

Answer:"""
        
        # 4. Stream tokens
        yield f"data: {json.dumps({'type': 'status', 'message': 'Generating...'})}\n\n"
        
        for token in model_manager.stream_tokens(prompt):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        
        # 5. Done
        yield f"data: {json.dumps({'type': 'done', 'model': current_model})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')
```

**Frontend Streaming:**
```javascript
async sendMessage() {
    const query = this.input.value.trim();
    
    const response = await fetch(`${API_URL}/api/chat/stream`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            query: query,
            k: 3,
            language: this.currentLanguage
        })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullResponse = '';
    
    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                
                switch(data.type) {
                    case 'status':
                        this.showStatus(data.message);
                        break;
                    case 'sources':
                        this.storeSources(data.sources);
                        break;
                    case 'token':
                        fullResponse += data.token;
                        this.updateMessage(messageId, fullResponse);
                        break;
                    case 'done':
                        this.appendSources(messageId);
                        break;
                }
            }
        }
    }
}
```

---

### 3. 🎤 Voice Input with Speech Recognition

**Implementation:**
```javascript
class VoiceInput {
    constructor() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (!SpeechRecognition) {
            console.warn('Speech recognition not supported');
            return;
        }
        
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
    }
    
    start(language) {
        // Set language: 'en-US' or 'bn-BD'
        this.recognition.lang = language === 'bangla' ? 'bn-BD' : 'en-US';
        
        this.recognition.onstart = () => {
            this.voiceButton.classList.add('recording');
            this.showStatus('🎤 Listening...');
        };
        
        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            this.inputField.value = transcript;
            this.sendMessage();
        };
        
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.showStatus('❌ Voice input failed');
        };
        
        this.recognition.onend = () => {
            this.voiceButton.classList.remove('recording');
        };
        
        this.recognition.start();
    }
}
```

**UI Button:**
```html
<button class="voice-button" onclick="voiceInput.start(currentLanguage)">
    <svg><!-- Microphone icon --></svg>
</button>
```

**Styling:**
```css
.voice-button {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--primary-color);
    border: none;
    cursor: pointer;
    transition: all 0.3s;
}

.voice-button.recording {
    background: var(--danger-color);
    animation: pulse 1s infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}
```

---

### 4. 🌍 Multi-Language Support (English & Bangla)

**UI Language Toggle:**
```html
<div class="language-toggle">
    <button id="lang-en" class="active" onclick="setLanguage('english')">
        English
    </button>
    <button id="lang-bn" onclick="setLanguage('bangla')">
        বাংলা
    </button>
</div>
```

**Language Switch Logic:**
```javascript
function setLanguage(language) {
    this.currentLanguage = language;
    
    // Update UI
    document.getElementById('lang-en').classList.toggle('active', language === 'english');
    document.getElementById('lang-bn').classList.toggle('active', language === 'bangla');
    
    // Update voice recognition language
    if (this.recognition) {
        this.recognition.lang = language === 'bangla' ? 'bn-BD' : 'en-US';
    }
    
    // Add notification message
    const message = language === 'english' 
        ? '🌐 Switched to English. I will respond in English.'
        : '🌐 বাংলায় পরিবর্তন করা হয়েছে। আমি বাংলায় উত্তর দেব।';
    
    this.addMessage('bot', message);
}
```

**Backend Language Handling:**
- Add language instruction to LLM prompt
- Pass language parameter in API requests
- Store language preference in browser localStorage

---

### 5. 🤖 Dynamic Model Selection

**Backend Endpoint:**
```python
@app.route("/api/models", methods=["GET"])
def get_models():
    """Get available Ollama models."""
    models = model_manager.get_available_models()
    current = model_manager.get_current_model()
    
    return jsonify({
        "models": models,
        "current_model": current,
        "success": True
    })

@app.route("/api/set-model", methods=["POST"])
def set_model():
    """Switch active model."""
    data = request.json
    model_name = data.get("model")
    
    if model_name in available_models:
        model_manager._active_model = model_name
        return jsonify({"success": True, "current_model": model_name})
    
    return jsonify({"success": False, "error": "Model not found"}), 404
```

**Frontend Model Selector:**
```javascript
async loadModels() {
    const response = await fetch(`${API_URL}/api/models`);
    const data = await response.json();
    
    if (data.success) {
        this.modelSelect.innerHTML = data.models.map(model => 
            `<option value="${model}" ${model === data.current_model ? 'selected' : ''}>
                ${model}
            </option>`
        ).join('');
    }
}

async setModel(modelName) {
    const response = await fetch(`${API_URL}/api/set-model`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model: modelName})
    });
    
    const data = await response.json();
    if (data.success) {
        this.showStatus(`Switched to ${modelName}`);
    }
}
```

---

### 6. 📚 Source Citations Display

**Format Sources:**
```javascript
function formatSources(sources) {
    if (!sources || sources.length === 0) return '';
    
    const sourcesHtml = sources.map((source, index) => `
        <div class="source-item">
            <div class="source-number">${index + 1}</div>
            <div class="source-content">
                <div class="source-text">${source.content.substring(0, 200)}...</div>
                <div class="source-meta">
                    📄 ${source.metadata.source || 'Unknown'}
                    ${source.metadata.search_source ? 
                        `<span class="badge">${source.metadata.search_source}</span>` : ''
                    }
                    ${source.metadata.page ? `• Page ${source.metadata.page}` : ''}
                    ${source.metadata.module ? `• Module ${source.metadata.module}` : ''}
                </div>
            </div>
        </div>
    `).join('');
    
    return `
        <div class="sources-container">
            <div class="sources-header">📚 Sources Used:</div>
            ${sourcesHtml}
        </div>
    `;
}
```

**Styling:**
```css
.sources-container {
    margin-top: 20px;
    padding: 15px;
    background: #f8f9fa;
    border-radius: 8px;
    border-left: 4px solid var(--primary-color);
}

.sources-header {
    font-weight: 600;
    margin-bottom: 10px;
    color: var(--primary-color);
}

.source-item {
    display: flex;
    gap: 10px;
    margin-bottom: 10px;
    padding: 10px;
    background: white;
    border-radius: 6px;
}

.source-number {
    width: 24px;
    height: 24px;
    background: var(--primary-color);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: bold;
    flex-shrink: 0;
}

.source-meta {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 5px;
}

.badge {
    background: var(--secondary-color);
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 10px;
    text-transform: uppercase;
}
```

---

### 7. 🎨 Beautiful UI/UX Design

**Floating Chat Button:**
```css
.chat-button {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4F46E5, #10B981);
    border: none;
    cursor: pointer;
    box-shadow: 0 10px 25px rgba(79, 70, 229, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
    z-index: 9999;
}

.chat-button:hover {
    transform: scale(1.1) rotate(5deg);
    box-shadow: 0 15px 35px rgba(79, 70, 229, 0.4);
}

.chat-button svg {
    width: 28px;
    height: 28px;
    fill: white;
}
```

**Chat Window:**
```css
.chat-window {
    position: fixed;
    bottom: 90px;
    right: 20px;
    width: 400px;
    height: 600px;
    background: white;
    border-radius: 16px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.15);
    display: none;
    flex-direction: column;
    overflow: hidden;
    z-index: 9998;
    animation: slideUp 0.3s ease;
}

.chat-window.active {
    display: flex;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
```

**Chat Header:**
```css
.chat-header {
    background: linear-gradient(135deg, #4F46E5, #764BA2);
    color: white;
    padding: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.chat-header-content {
    display: flex;
    align-items: center;
    gap: 12px;
}

.chat-avatar {
    width: 40px;
    height: 40px;
    background: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.chat-title h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
}

.chat-title p {
    margin: 0;
    font-size: 12px;
    opacity: 0.8;
}
```

**Messages Area:**
```css
.chat-messages {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    background: #f9fafb;
}

.message {
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.message.user {
    flex-direction: row-reverse;
}

.message-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}

.message.user .message-avatar {
    background: var(--primary-color);
    color: white;
}

.message.bot .message-avatar {
    background: var(--secondary-color);
    color: white;
}

.message-content {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.5;
}

.message.user .message-content {
    background: var(--primary-color);
    color: white;
    border-bottom-right-radius: 4px;
}

.message.bot .message-content {
    background: white;
    color: var(--text-primary);
    border-bottom-left-radius: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
```

**Input Area:**
```css
.chat-input-container {
    padding: 15px;
    background: white;
    border-top: 1px solid #e5e7eb;
    display: flex;
    gap: 10px;
    align-items: flex-end;
}

.chat-input {
    flex: 1;
    padding: 10px 15px;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    resize: none;
    font-family: inherit;
    font-size: 14px;
    max-height: 100px;
    transition: border-color 0.2s;
}

.chat-input:focus {
    outline: none;
    border-color: var(--primary-color);
}

.send-button {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--primary-color);
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
}

.send-button:hover {
    background: var(--primary-hover);
    transform: scale(1.05);
}

.send-button svg {
    width: 20px;
    height: 20px;
    fill: white;
}
```

**Streaming Indicator:**
```css
.streaming-indicator {
    display: inline-flex;
    gap: 4px;
    padding: 8px;
}

.streaming-indicator span {
    width: 8px;
    height: 8px;
    background: var(--primary-color);
    border-radius: 50%;
    animation: bounce 1.4s infinite;
}

.streaming-indicator span:nth-child(2) {
    animation-delay: 0.2s;
}

.streaming-indicator span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes bounce {
    0%, 80%, 100% {
        transform: scale(0);
        opacity: 0.5;
    }
    40% {
        transform: scale(1);
        opacity: 1;
    }
}
```

---

## 📦 Project Structure

```
project/
├── web/
│   ├── chatbot_api.py              # Flask API server
│   ├── chatbot-widget.js           # Frontend chatbot class
│   ├── chatbot-widget.css          # Widget styling
│   ├── course-example.html         # Demo course page
│   ├── course_knowledge_base.txt   # Course materials
│   └── load_course_database.py     # Database loader
│
├── services/
│   ├── database_service.py         # ChromaDB manager
│   ├── llm_service.py              # Ollama LLM interface
│   ├── embedding_service.py        # Embedding generator
│   └── voice_service.py            # Voice input handler
│
├── core/
│   ├── logging_config.py           # Logging setup
│   ├── constants.py                # App constants
│   ├── exceptions.py               # Custom exceptions
│   └── utils.py                    # Utility functions
│
├── db/                             # ChromaDB storage
│   ├── chroma.sqlite3
│   └── collections/
│
├── requirements.txt                # Python dependencies
└── main.py                         # Entry point
```

---

## 🔧 Technical Implementation Details

### 1. Database Service (services/database_service.py)

```python
from chromadb import Client, Settings
from chromadb.config import Settings as ChromaSettings
import chromadb

class DatabaseFactory:
    """Factory for creating database instances."""
    
    @staticmethod
    def create_chroma_database(collection_name: str, persist_directory: str = "./db"):
        """
        Create or load ChromaDB collection.
        
        Args:
            collection_name: Name of the collection ('course_materials' or 'banglarag')
            persist_directory: Path to database directory
        """
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        return ChromaDatabaseManager(collection)

class ChromaDatabaseManager:
    """Manages ChromaDB operations."""
    
    def __init__(self, collection):
        self.collection = collection
    
    def search(self, query: str, k: int = 3):
        """
        Search for relevant documents.
        
        Returns: List of Document objects with .page_content and .metadata
        """
        # Get embeddings from embedding service
        embedding = get_embedding_service().embed_query(query)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )
        
        # Convert to Document objects
        documents = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                page_content=results['documents'][0][i],
                metadata=results['metadatas'][0][i]
            )
            documents.append(doc)
        
        return documents
    
    def add_documents(self, texts: List[str], metadatas: List[dict], ids: List[str]):
        """Add documents to collection."""
        embeddings = [get_embedding_service().embed_query(text) for text in texts]
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
```

---

### 2. LLM Service (services/llm_service.py)

```python
import requests
import json

class ModelManager:
    """Manages Ollama LLM interactions."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.preferred_model = "llama3.2:latest"
        self._active_model = None
        self.session = requests.Session()
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except:
            return [self.preferred_model]
    
    def get_current_model(self) -> str:
        """Get currently active model."""
        return self._active_model or self.preferred_model
    
    def stream_tokens(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream LLM response token by token.
        
        Yields: Individual tokens as they are generated
        """
        model = self.get_current_model()
        
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'response' in data:
                    yield data['response']
                
                if data.get('done', False):
                    break
    
    def generate_response(self, prompt: str) -> str:
        """Generate complete response (non-streaming)."""
        tokens = self.stream_tokens(prompt)
        return ''.join(tokens)

class RAGProcessor:
    """Processes RAG queries."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def process_rag_query(self, query: str, documents: List[Document]) -> dict:
        """
        Process query with retrieved documents.
        
        Returns: {"success": bool, "response": str, "error": str}
        """
        try:
            # Build context
            context = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(documents)
            ])
            
            # Build prompt
            prompt = f"""Based on the following course materials, answer the question. 
Be specific and cite the information provided.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response
            response = self.model_manager.generate_response(prompt)
            
            return {
                "success": True,
                "response": response
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

---

### 3. Embedding Service (services/embedding_service.py)

```python
import requests
import json

class EmbeddingFactory:
    """Factory for creating embedding generators."""
    
    @staticmethod
    def create_ollama_embeddings(model_name: str = "nomic-embed-text"):
        return OllamaEmbeddings(model_name)

class OllamaEmbeddings:
    """Generates embeddings using Ollama API."""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Returns: List of floats (embedding vector)
        """
        response = self.session.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model_name,
                "prompt": text
            }
        )
        
        data = response.json()
        return data['embedding']
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_query(text) for text in texts]
```

---

### 4. Flask API Server (web/chatbot_api.py)

**Complete implementation provided in earlier sections. Key endpoints:**

- `GET /` - Serve course page
- `GET /api/health` - Health check
- `GET /api/models` - List available models
- `POST /api/set-model` - Switch active model
- `POST /api/chat` - Non-streaming chat
- `POST /api/chat/stream` - Streaming chat (SSE)

---

### 5. Frontend Widget (web/chatbot-widget.js)

**Complete implementation provided in earlier sections. Key class:**

```javascript
class BanglaRAGChatbot {
    constructor(config) {
        // Initialize with config
        this.apiUrl = config.apiUrl || 'http://localhost:5000';
        this.init();
    }
    
    init() {
        this.injectStyles();
        this.createWidget();
        this.attachEventListeners();
        this.loadModels();
        this.initVoiceRecognition();
    }
    
    // Methods:
    // - sendMessage()
    // - streamResponse()
    // - toggleVoiceInput()
    // - setLanguage()
    // - formatSources()
    // - addMessage()
    // - updateMessage()
}
```

---

## 🚀 Setup Instructions

### Prerequisites

```bash
# 1. Install Python 3.10+
python --version

# 2. Install Ollama
# Download from: https://ollama.ai
ollama --version

# 3. Pull required models
ollama pull llama3.2:latest
ollama pull qwen2:1.5b
ollama pull phi3:latest
ollama pull nomic-embed-text
```

### Installation

```bash
# 1. Clone/create project directory
mkdir banglarag-chatbot
cd banglarag-chatbot

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install flask flask-cors chromadb requests

# Optional: Voice and PDF support
pip install SpeechRecognition pyaudio PyPDF2 pypdf
```

### Database Setup

```bash
# Run database loader
python web/load_course_database.py

# Expected output:
# ✅ Loaded course materials: 68 chunks
# ✅ Database saved to: ./db/
```

### Running the Server

```bash
# Start Flask server
python web/chatbot_api.py

# Or use:
python -m web.chatbot_api

# Server runs at: http://localhost:5000
```

### Embedding in Course Page

```html
<!DOCTYPE html>
<html>
<head>
    <title>My Course</title>
    <link rel="stylesheet" href="http://localhost:5000/chatbot-widget.css">
</head>
<body>
    <h1>Course Content</h1>
    <!-- Your course content here -->
    
    <!-- Chatbot Widget -->
    <script>
        window.BanglaRAGConfig = {
            apiUrl: 'http://localhost:5000',
            primaryColor: '#4F46E5',
            botName: 'Course Assistant',
            botAvatar: '🤖',
            welcomeMessage: 'Hi! Ask me anything about the course!'
        };
    </script>
    <script src="http://localhost:5000/chatbot-widget.js"></script>
</body>
</html>
```

---

## 🧪 Testing Checklist

### Backend Tests
- [ ] `/api/health` returns 200 OK
- [ ] `/api/models` returns list of models
- [ ] `/api/set-model` switches model successfully
- [ ] `/api/chat` returns complete response
- [ ] `/api/chat/stream` sends SSE events
- [ ] Dual database search prioritizes correctly
- [ ] Course keywords trigger course-only search
- [ ] Theory questions prefer PDF sources

### Frontend Tests
- [ ] Chat button appears and toggles window
- [ ] Model selector loads and switches models
- [ ] Language toggle changes between EN/BN
- [ ] Voice button starts speech recognition
- [ ] Message sending works (Enter key)
- [ ] Streaming displays tokens in real-time
- [ ] Sources display correctly with metadata
- [ ] Mobile responsive design works
- [ ] Auto-scroll to bottom on new messages

### Integration Tests
- [ ] English question → English response
- [ ] Bangla question → Bangla response
- [ ] Course question → cites course materials
- [ ] Theory question → cites PDF book
- [ ] Voice input → transcribes to text → sends
- [ ] Model switch → next response uses new model

---

## 📊 Performance Optimization

### Backend Optimizations
```python
# 1. Cache embeddings
from functools import lru_cache

@lru_cache(maxsize=100)
def get_embedding_cached(text: str):
    return embedding_service.embed_query(text)

# 2. Connection pooling
session = requests.Session()
adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20)
session.mount('http://', adapter)

# 3. Async database queries
import asyncio

async def search_databases_parallel(query):
    course_task = asyncio.create_task(search_course_db(query))
    pdf_task = asyncio.create_task(search_pdf_db(query))
    
    course_results, pdf_results = await asyncio.gather(course_task, pdf_task)
    return combine_results(course_results, pdf_results)
```

### Frontend Optimizations
```javascript
// 1. Debounce input
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// 2. Lazy load messages
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            loadMoreMessages();
        }
    });
});

// 3. Virtual scrolling for long chats
// Only render visible messages
```

---

## 🔒 Security Considerations

### Backend Security
```python
# 1. Input sanitization
from flask import escape
query = escape(request.json.get('query', ''))

# 2. Rate limiting
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route("/api/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    pass

# 3. CORS restrictions (production)
CORS(app, origins=["https://yourdomain.com"])

# 4. API key authentication (optional)
@app.before_request
def check_api_key():
    if request.endpoint != 'health_check':
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Unauthorized'}), 401
```

### Frontend Security
```javascript
// 1. XSS prevention
function sanitizeHTML(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 2. Content Security Policy
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; script-src 'self' 'unsafe-inline';">

// 3. HTTPS only (production)
if (location.protocol !== 'https:') {
    location.replace(`https:${location.href.substring(location.protocol.length)}`);
}
```

---

## 🎯 Success Criteria

### Functional Requirements ✅
- [x] Real-time streaming responses
- [x] Dual database search with smart prioritization
- [x] Voice input in English and Bangla
- [x] Multi-language support (EN/BN)
- [x] Dynamic model selection
- [x] Source citations with metadata
- [x] Embeddable floating widget
- [x] Mobile responsive design

### Performance Requirements ✅
- Response time: < 3 seconds for search
- Streaming latency: < 100ms per token
- UI responsiveness: 60fps animations
- Database queries: < 500ms
- Widget load time: < 2 seconds

### User Experience Requirements ✅
- Intuitive chat interface
- Clear visual feedback (loading states)
- Smooth animations and transitions
- Accessible keyboard navigation
- Error messages are user-friendly
- Sources are easy to read

---

## 📝 Additional Features (Optional Enhancements)

### 1. Conversation History
```javascript
// Save to localStorage
saveConversation() {
    localStorage.setItem('chat_history', JSON.stringify(this.messages));
}

// Load on init
loadConversation() {
    const history = localStorage.getItem('chat_history');
    if (history) {
        this.messages = JSON.parse(history);
        this.renderMessages();
    }
}
```

### 2. Export Chat
```javascript
exportChat() {
    const text = this.messages.map(msg => 
        `[${msg.type}] ${msg.content}`
    ).join('\n\n');
    
    const blob = new Blob([text], {type: 'text/plain'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'chat-export.txt';
    a.click();
}
```

### 3. Feedback System
```python
@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    data = request.json
    message_id = data.get('message_id')
    rating = data.get('rating')  # 👍 or 👎
    
    # Save to database
    save_feedback(message_id, rating)
    
    return jsonify({'success': True})
```

### 4. Typing Indicator
```javascript
showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    this.messagesContainer.appendChild(indicator);
}
```

### 5. Code Syntax Highlighting
```javascript
formatMessage(text) {
    // Detect code blocks
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang || 'text'}">${escapeHTML(code)}</code></pre>`;
    });
    
    // Apply syntax highlighting
    document.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
    });
    
    return text;
}
```

---

## 🎓 Learning Resources

### Flask & Python
- Flask Documentation: https://flask.palletsprojects.com/
- Python Async: https://docs.python.org/3/library/asyncio.html
- Server-Sent Events: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events

### LangChain & RAG
- LangChain Docs: https://python.langchain.com/docs/get_started
- ChromaDB Guide: https://docs.trychroma.com/
- RAG Tutorial: https://www.pinecone.io/learn/retrieval-augmented-generation/

### Frontend
- Web Speech API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API
- CSS Animations: https://animate.style/
- Fetch Streaming: https://developer.mozilla.org/en-US/docs/Web/API/Streams_API

---

## 🏁 Conclusion

This master prompt provides a **complete blueprint** for building a production-ready educational chatbot with:

✅ **Dual-database RAG** with smart source prioritization  
✅ **Real-time streaming** for responsive UX  
✅ **Voice input** for accessibility  
✅ **Multi-language** support (English & Bangla)  
✅ **Beautiful UI** with smooth animations  
✅ **Embeddable widget** for any course page  
✅ **Dynamic model switching** for flexibility  
✅ **Source citations** for transparency  

Follow the implementation details, test thoroughly, and you'll have a chatbot that students love to use! 🚀

---

**Version:** 2.0.0  
**Last Updated:** October 14, 2025  
**Status:** ✅ Production Ready
