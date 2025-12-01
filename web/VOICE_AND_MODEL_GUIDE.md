# 🎤 Voice Mode & Model Configuration Guide

## 🎙️ **NEW: Voice Input Feature**

### **How to Use Voice Mode:**

1. **Open the chatbot** (click the floating chat button)
2. **Click the microphone button** 🎤 (left side of input box)
3. **Speak your question** in English or বাংলা
4. **Stop speaking** - text appears automatically
5. **Click send** to submit your question

### **Voice Features:**

✅ **Auto language detection** - Switches between English/Bangla based on your language setting  
✅ **Visual feedback** - Red pulsing animation while recording  
✅ **Browser-based** - No server processing needed  
✅ **Works offline** - Uses your device's speech recognition

### **Supported Browsers:**

- ✅ Google Chrome (Recommended)
- ✅ Microsoft Edge
- ✅ Safari (Mac/iOS)
- ❌ Firefox (not supported yet)

### **Troubleshooting Voice Input:**

**"Microphone access denied"**

- Click the 🔒 icon in browser address bar
- Allow microphone access
- Refresh the page

**"No speech detected"**

- Check your microphone is connected
- Speak louder or closer to microphone
- Try clicking the button again

**"Voice input not supported"**

- Use Chrome, Edge, or Safari browser
- Update your browser to latest version

---

## 🤖 **Model Configuration**

### **Method 1: Change Default Model in Code** ⭐ Recommended

**File:** `core/constants.py`  
**Lines:** 30-31

```python
# 👇 Change this line to set your default model
PREFERRED_LLM_MODEL = "llama3.2"

# 👇 Change this list to show models in the dropdown
FALLBACK_LLM_MODELS = ["llama3.2", "qwen2:1.5b", "phi3", "mistral", "llama2"]
```

**Steps:**

1. Open `core/constants.py`
2. Find line 30-31 (look for `PREFERRED_LLM_MODEL`)
3. Change `"llama3.2"` to your preferred model
4. Update `FALLBACK_LLM_MODELS` list to add/remove models
5. Save file
6. Restart the server

---

### **Method 2: Change Model via UI** (Easiest for Users)

1. Open chatbot
2. Look at top section
3. Click the **Model** dropdown
4. Select your preferred model
5. Model changes immediately!

---

### **Method 3: Environment Variable** (Quick Testing)

**Windows PowerShell:**

```powershell
$env:PREFERRED_LLM_MODEL = "qwen2:1.5b"
python web\chatbot_api.py
```

**Windows CMD:**

```cmd
set PREFERRED_LLM_MODEL=qwen2:1.5b
python web\chatbot_api.py
```

**Linux/Mac:**

```bash
export PREFERRED_LLM_MODEL=qwen2:1.5b
python web/chatbot_api.py
```

---

## 📋 **Available Models on Your System**

**Check installed models:**

```bash
ollama list
```

**Install new models:**

```bash
ollama pull llama3.2:latest
ollama pull qwen2:1.5b
ollama pull phi3:latest
ollama pull mistral:latest
```

---

## 🎯 **Recommended Models**

| Model               | Size  | Speed    | Quality    | Best For             |
| ------------------- | ----- | -------- | ---------- | -------------------- |
| **qwen2:1.5b**      | 1.5GB | ⚡⚡⚡⚡ | ⭐⭐⭐     | Quick answers        |
| **phi3:latest**     | 2.3GB | ⚡⚡⚡   | ⭐⭐⭐⭐   | Balanced performance |
| **llama3.2:latest** | 2GB   | ⚡⚡⚡   | ⭐⭐⭐⭐   | General purpose      |
| **mistral:latest**  | 4.1GB | ⚡⚡     | ⭐⭐⭐⭐⭐ | Complex reasoning    |

---

## 🔧 **Model List Not Showing?**

If the model dropdown shows "Loading models..." forever:

1. **Check Ollama is running:**

   ```bash
   ollama list
   ```

2. **Verify models are available:**

   - Must see models in the list
   - Models should not show errors

3. **Check browser console:**

   - Press F12
   - Look for errors in Console tab
   - Share errors if you need help

4. **Fix the model names:**
   - In `core/constants.py`, use exact names from `ollama list`
   - Example: Use `"llama3.2:latest"` not `"llama3.2"` if that's what ollama shows

---

## 🎉 **Quick Start**

1. **Set your preferred model** in `core/constants.py` (line 30)
2. **Start the server:** `python web\chatbot_api.py`
3. **Open browser:** http://localhost:5000/
4. **Try voice mode:** Click 🎤 and speak
5. **Switch models:** Use the dropdown if needed

---

## 💡 **Tips**

- **Voice works in both languages** - Switch language first, then use voice
- **Model changes persist** - Selected model stays active until you change it
- **Fallback models** - If preferred model fails, system tries backup models automatically
- **Cache enabled** - Repeated questions get instant responses

---

## 🆘 **Need Help?**

**Voice not working?** Use supported browser (Chrome/Edge/Safari)  
**Models not loading?** Check `ollama list` and `core/constants.py`  
**Slow responses?** Try lighter model like `qwen2:1.5b`  
**Connection errors?** Restart Ollama service

---

Enjoy your enhanced chatbot with voice input! 🚀
