# 🎓 BanglaRAG LMS Chatbot - Embeddable Course Assistant

A floating chatbot widget that can be embedded in any HTML page with just **3 lines of code**! Features real-time streaming responses, model selection, and RAG-powered course Q&A.

## ✨ Features

- 🎤 **Floating Chat Interface**: Elegant circular button that opens a chat window
- 🔄 **Real-Time Streaming**: Watch responses generate token-by-token
- 🤖 **Model Selection**: Switch between different Ollama models on-the-fly
- 📚 **RAG-Powered**: Answers based on your course knowledge base
- 🎨 **Fully Customizable**: Colors, avatars, welcome messages
- 📱 **Responsive Design**: Works on desktop and mobile
- ⚡ **Easy Embed**: Just 3 lines of code to add to any page!

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Flask and CORS support
pip install flask flask-cors requests

# Make sure you have the main BanglaRAG dependencies
pip install -r requirements.txt
```

### 2. Load Course Knowledge Base

```bash
# From the main project directory
python web/load_course_database.py
```

This will:

- Parse the course content from `web/course_knowledge_base.txt`
- Create embeddings using your configured model
- Store in ChromaDB for fast retrieval

### 3. Start the API Server

```bash
# Start the Flask API backend
python web/chatbot_api.py
```

The API will start at `http://localhost:5000` with endpoints:

- `GET /api/health` - Health check
- `GET /api/models` - Get available models
- `POST /api/set-model` - Switch model
- `POST /api/chat` - Regular chat (returns complete response)
- `POST /api/chat/stream` - Streaming chat (Server-Sent Events)

### 4. Open the Course Page

Simply open `web/course-example.html` in your browser!

## 📝 Embedding in Your Own HTML

### Method 1: Simple Embed (Recommended)

Add these 3 lines before closing `</body>` tag:

```html
<!-- Load the chatbot styles -->
<link rel="stylesheet" href="chatbot-widget.css" />

<!-- Configure and initialize chatbot -->
<script>
  window.BanglaRAGConfig = {
    apiUrl: "http://localhost:5000",
    botName: "Course Assistant",
    botAvatar: "🎓",
    userAvatar: "👨‍🎓",
    welcomeMessage: "Hi! Ask me anything about the course!",
    primaryColor: "#667eea",
  };
</script>
<script src="chatbot-widget.js"></script>
```

That's it! The chatbot will appear automatically.

### Method 2: Manual Initialization

```html
<link rel="stylesheet" href="chatbot-widget.css" />
<script src="chatbot-widget.js"></script>

<script>
  // Initialize chatbot when ready
  const chatbot = new BanglaRAGChatbot({
    apiUrl: "http://localhost:5000",
    botName: "My Course Bot",
    botAvatar: "🤖",
    primaryColor: "#4F46E5",
    welcomeMessage: "How can I help you today?",
  });
</script>
```

## ⚙️ Configuration Options

```javascript
window.BanglaRAGConfig = {
  // Required
  apiUrl: "http://localhost:5000", // Your API endpoint

  // Appearance
  botName: "Course Assistant", // Chatbot name in header
  botAvatar: "🎓", // Emoji or image URL
  userAvatar: "👨‍🎓", // User's avatar
  primaryColor: "#667eea", // Main theme color

  // Behavior
  welcomeMessage: "Hi! ...", // Initial greeting
  position: "bottom-right", // Widget position (future)
};
```

## 🎨 Customization

### Change Colors

Edit `chatbot-widget.css`:

```css
:root {
  --primary-color: #4f46e5; /* Main brand color */
  --primary-hover: #4338ca; /* Hover state */
  --secondary-color: #10b981; /* Accent color */
}
```

### Modify Appearance

The chatbot uses CSS classes prefixed with `banglarag-`:

- `.banglarag-chat-button` - The floating button
- `.banglarag-chat-window` - The chat container
- `.banglarag-chat-header` - Header section
- `.banglarag-message` - Individual messages

## 📂 Project Structure

```
web/
├── chatbot_api.py              # Flask API backend
├── chatbot-widget.css          # Chatbot styling
├── chatbot-widget.js           # Chatbot functionality
├── course-example.html         # Sample LMS page
├── course_knowledge_base.txt   # Course content
└── load_course_database.py     # Database loader script
```

## 🔧 API Endpoints

### GET /api/health

Health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "service": "BanglaRAG Chatbot API",
  "version": "2.0.0"
}
```

### GET /api/models

Get list of available Ollama models.

**Response:**

```json
{
  "models": ["qwen2:1.5b", "llama2", "mistral"],
  "current_model": "qwen2:1.5b",
  "success": true
}
```

### POST /api/set-model

Change the active model.

**Request:**

```json
{
  "model": "llama2"
}
```

**Response:**

```json
{
  "success": true,
  "current_model": "llama2"
}
```

### POST /api/chat

Get a complete response (non-streaming).

**Request:**

```json
{
  "query": "What is an array?",
  "k": 3
}
```

**Response:**

```json
{
    "response": "An array is...",
    "sources": [...],
    "model": "qwen2:1.5b",
    "success": true
}
```

### POST /api/chat/stream

Get streaming response with Server-Sent Events (SSE).

**Request:**

```json
{
  "query": "What is an array?",
  "k": 3
}
```

**Response (SSE):**

```
data: {"type": "status", "message": "Searching knowledge base..."}

data: {"type": "sources", "sources": [...]}

data: {"type": "token", "token": "An"}

data: {"type": "token", "token": " array"}

data: {"type": "done", "model": "qwen2:1.5b"}
```

## 📚 Adding Your Own Course Content

### 1. Edit the Knowledge Base

Edit `web/course_knowledge_base.txt` with your course content. Use clear section headers:

```text
=======================================================
MODULE 1: YOUR TOPIC
=======================================================

Content goes here...

Key Concepts:
- Point 1
- Point 2

Examples:
Provide examples here...
```

### 2. Reload the Database

```bash
python web/load_course_database.py --input web/course_knowledge_base.txt
```

### 3. Restart the API

```bash
# Stop the current API (Ctrl+C)
# Start again
python web/chatbot_api.py
```

## 🔍 How It Works

### Architecture

```
┌─────────────┐         ┌──────────────┐         ┌────────────┐
│   HTML      │────────▶│  JavaScript  │────────▶│  Flask API │
│   Page      │         │   Widget     │  HTTP   │  Backend   │
└─────────────┘         └──────────────┘         └────────────┘
                                                         │
                                                         ▼
                                                  ┌────────────┐
                                                  │  ChromaDB  │
                                                  │  +         │
                                                  │  Ollama    │
                                                  └────────────┘
```

### Flow

1. **User asks question** → Widget captures input
2. **Widget sends to API** → POST to `/api/chat/stream`
3. **API searches knowledge base** → ChromaDB similarity search
4. **API generates response** → Ollama LLM with context
5. **Streams back tokens** → Server-Sent Events (SSE)
6. **Widget displays real-time** → Updates UI as tokens arrive

## 🐛 Troubleshooting

### Chatbot button doesn't appear

1. Check that CSS and JS files are loaded:

```html
<link rel="stylesheet" href="chatbot-widget.css" />
<script src="chatbot-widget.js"></script>
```

2. Verify paths are correct relative to your HTML file

3. Check browser console for errors (F12)

### "Service not initialized" error

1. Make sure Flask API is running:

```bash
python web/chatbot_api.py
```

2. Check API is accessible at `http://localhost:5000/api/health`

3. Verify CORS is enabled (should be by default)

### No responses from chatbot

1. Check if course database is loaded:

```bash
python web/load_course_database.py
```

2. Verify Ollama is running:

```bash
ollama list
```

3. Test API directly:

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is an array?"}'
```

### CORS errors in browser

If you see CORS errors, make sure:

1. Flask-CORS is installed: `pip install flask-cors`
2. API has `CORS(app)` enabled (already in code)
3. You're accessing from same origin or `http://localhost`

## 🎯 Production Deployment

### Security Considerations

1. **Change API URL**: Update `apiUrl` in config to your production server
2. **Add Authentication**: Implement API key or JWT tokens
3. **Rate Limiting**: Add request throttling
4. **HTTPS**: Use SSL certificates for production
5. **CORS Policy**: Restrict to specific domains

### Example Production Config

```javascript
window.BanglaRAGConfig = {
  apiUrl: "https://api.yourdomain.com", // Your production API
  // ... other config
};
```

## 📈 Performance Tips

1. **Adjust chunk size**: Smaller chunks = more precise, larger = more context

```bash
python web/load_course_database.py --chunk-size 300
```

2. **Use faster model**: Switch to smaller, faster Ollama model

```javascript
// In chatbot or API config
model: "qwen2:1.5b"; // Faster than larger models
```

3. **Enable caching**: Database manager includes built-in LRU cache

4. **Batch loading**: Load knowledge base once, query many times

## 🤝 Contributing

Feel free to extend and customize:

- Add more API endpoints
- Enhance UI/UX
- Add voice input support
- Implement conversation history
- Add multi-language support

## 📄 License

Part of the BanglaRAG project - Educational use

## 🆘 Support

- Check main BanglaRAG documentation
- Review API logs for debugging
- Test endpoints with curl/Postman
- Use browser DevTools to inspect network requests

---

**Ready to use!** 🎉 Just follow the Quick Start guide and you'll have a working LMS chatbot in minutes!
