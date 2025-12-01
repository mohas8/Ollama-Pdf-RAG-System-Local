# 🎤 Voice Mode & Model Configuration Guide

## ✅ COMPLETED FEATURES

### 1. 🎤 Voice Input Mode

The chatbot now has **voice input** capability! Look for the microphone button in the chat interface.

#### How to Use Voice Mode:

1. **Click the 🎤 microphone button** (left of the text input)
2. **Allow microphone access** when browser asks
3. **Speak your question** clearly
4. **Watch it transcribe** into the text box
5. **Click send** or press Enter

#### Voice Features:

- ✅ **Auto-language detection** - Uses English or বাংলা based on your language setting
- ✅ **Visual feedback** - Red pulsing animation while recording
- ✅ **Error handling** - Clear messages if mic access denied
- ✅ **Browser compatibility** - Works in Chrome, Edge, Safari

#### Supported Languages:

- **English Mode**: `en-US` - For English voice input
- **বাংলা Mode**: `bn-BD` - For Bengali voice input

---

## 🤖 Model Configuration

### Where to Change the Model List

**File:** `core/constants.py` (Lines 47-51)

```python
# Preferred Models
PREFERRED_LLM_MODEL = "llama3.2:latest"  # 👈 CHANGE THIS to set default model
FALLBACK_LLM_MODELS = [
    "llama3.2:latest",
    "qwen2:1.5b",
    "phi3:latest",
]  # 👈 CHANGE THIS to set model dropdown list (only models installed on your system)
```

### Your Installed Models:

Based on `ollama list`, you have:

- ✅ **llama3.2:latest** (2.0 GB) - Default model
- ✅ **qwen2:1.5b** (934 MB) - Lightweight & fast
- ✅ **phi3:latest** (2.2 GB) - Balanced performance

### How to Add/Remove Models:

#### To Add a New Model:

1. **Install via Ollama:**

   ```bash
   ollama pull mistral:latest
   ```

2. **Update constants.py:**

   ```python
   FALLBACK_LLM_MODELS = [
       "llama3.2:latest",
       "qwen2:1.5b",
       "phi3:latest",
       "mistral:latest",  # ← Add here
   ]
   ```

3. **Restart the server** - The new model appears in dropdown!

#### To Remove a Model:

Simply delete it from the `FALLBACK_LLM_MODELS` list in `constants.py`

---

## 🎨 Technical Details

### Voice Input Implementation:

- **API:** Web Speech API (`SpeechRecognition`)
- **Files Modified:**
  - `web/chatbot-widget.js` - Voice recognition logic
  - `web/chatbot-widget.css` - Microphone button styling

### Key Functions:

- `initVoiceRecognition()` - Initialize speech recognition
- `toggleVoiceInput()` - Start/stop recording
- Language auto-switches with UI language selector

### Model Selector Implementation:

- **API Endpoint:** `GET /api/models` - Returns available models
- **API Endpoint:** `POST /api/set-model` - Change active model
- **Files Modified:**
  - `web/chatbot_api.py` - Fixed `get_current_model()` issue
  - `core/constants.py` - Model list configuration

---

## 🚀 Quick Start

1. **Start the server:**

   ```bash
   python web\chatbot_api.py
   ```

2. **Open browser:**

   ```
   http://localhost:5000/
   ```

3. **Test voice mode:**

   - Click 🎤 microphone button
   - Allow microphone access
   - Say: "What is an array?"
   - Watch it transcribe and respond!

4. **Switch models:**

   - Click model dropdown (top of chat)
   - Select different model
   - Continue chatting with new model

5. **Switch languages:**
   - Click "বাংলা" button
   - Click 🎤 and speak in Bengali
   - Get responses in Bengali!

---

## ⚠️ Troubleshooting

### Voice Not Working?

- **Check browser**: Use Chrome, Edge, or Safari (Firefox limited support)
- **Check permissions**: Allow microphone access in browser settings
- **HTTPS required**: For non-localhost, voice needs HTTPS

### Model Not Appearing?

- **Verify installation**: Run `ollama list` to confirm model exists
- **Check spelling**: Model names must match exactly (case-sensitive)
- **Restart server**: After changing `constants.py`

### "Loading models..." Forever?

- **API issue**: Server may not be running
- **Check console**: Open browser DevTools (F12) → Console tab
- **Check network**: Look for failed `/api/models` request

---

## 📝 Summary

### ✅ Completed:

- [x] Voice input button added to chatbot UI
- [x] Web Speech API integration (English & Bengali)
- [x] Recording animation and status indicators
- [x] Model list configuration in `constants.py`
- [x] Model dropdown showing 3 installed models
- [x] Fixed API model management issues

### 🎯 Features:

- 🎤 **Voice Mode** - Click mic, speak, get transcription
- 🤖 **Model Selector** - Switch between 3 AI models
- 🌍 **Bilingual Voice** - English & Bengali speech recognition
- ⚡ **Real-time Streaming** - Watch responses generate live
- 📚 **Course Knowledge** - 6 data structures modules

---

## 🎉 You're Ready!

Open **http://localhost:5000/** and enjoy your **voice-enabled bilingual chatbot**! 🚀

Try saying:

- 🇺🇸 "What is a binary tree?"
- 🇧🇩 "অ্যারে কি?"
