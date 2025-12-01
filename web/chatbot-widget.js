/**
 * BanglaRAG Chatbot Widget
 * Embeddable floating chat interface with streaming support
 */

class BanglaRAGChatbot {
  constructor(config = {}) {
    this.config = {
      apiUrl: config.apiUrl || "http://localhost:5000",
      position: config.position || "bottom-right",
      primaryColor: config.primaryColor || "#4F46E5",
      botName: config.botName || "Course Assistant",
      botAvatar: config.botAvatar || "🤖",
      userAvatar: config.userAvatar || "👤",
      welcomeMessage:
        config.welcomeMessage ||
        "Hi! I'm your course assistant. Ask me anything about the course materials!",
      ...config,
    };

    this.isOpen = false;
    this.currentModel = null;
    this.availableModels = [];
    this.messages = [];
    this.isStreaming = false;
    this.currentLanguage = "english"; // Default language
    this.recognition = null; // Voice recognition instance
    this.isRecording = false; // Voice recording state

    this.init();
  }

  init() {
    this.injectStyles();
    this.createWidget();
    this.attachEventListeners();
    this.loadModels();

    // Add welcome message
    this.addMessage("bot", this.config.welcomeMessage);
  }

  injectStyles() {
    // The CSS is loaded externally, but we can inject custom colors
    const style = document.createElement("style");
    style.textContent = `
            :root {
                --primary-color: ${this.config.primaryColor};
            }
        `;
    document.head.appendChild(style);
  }

  createWidget() {
    const container = document.createElement("div");
    container.className = "banglarag-chatbot-container";
    container.innerHTML = `
            <!-- Chat Button -->
            <button class="banglarag-chat-button" id="banglarag-chat-toggle">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
                    <path d="M7 9h10v2H7zm0-3h10v2H7zm0 6h7v2H7z"/>
                </svg>
            </button>
            
            <!-- Chat Window -->
            <div class="banglarag-chat-window" id="banglarag-chat-window">
                <!-- Header -->
                <div class="banglarag-chat-header">
                    <div class="banglarag-chat-header-content">
                        <div class="banglarag-chat-avatar">${this.config.botAvatar}</div>
                        <div class="banglarag-chat-title">
                            <h3>${this.config.botName}</h3>
                            <p id="banglarag-status">Online</p>
                        </div>
                    </div>
                    <button class="banglarag-close-button" id="banglarag-close-button">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                        </svg>
                    </button>
                </div>
                
                <!-- Model & Language Selector -->
                <div class="banglarag-model-selector">
                    <div class="banglarag-selector-group">
                        <label for="banglarag-model-select">Model:</label>
                        <select id="banglarag-model-select">
                            <option value="">Loading models...</option>
                        </select>
                    </div>
                    <div class="banglarag-selector-group">
                        <label>Language:</label>
                        <div class="banglarag-language-toggle">
                            <button id="banglarag-lang-en" class="active" data-lang="english">English</button>
                            <button id="banglarag-lang-bn" data-lang="bangla">বাংলা</button>
                        </div>
                    </div>
                </div>
                
                <!-- Messages -->
                <div class="banglarag-chat-messages" id="banglarag-messages"></div>
                
                <!-- Input -->
                <div class="banglarag-chat-input-container">
                    <button class="banglarag-voice-button" id="banglarag-voice-button" title="Voice input">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" fill="currentColor"/>
                            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" fill="currentColor"/>
                        </svg>
                    </button>
                    <textarea 
                        class="banglarag-chat-input" 
                        id="banglarag-input" 
                        placeholder="Type your question or use voice..."
                        rows="1"
                    ></textarea>
                    <button class="banglarag-send-button" id="banglarag-send-button">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor"/>
                        </svg>
                    </button>
                </div>
            </div>
        `;

    document.body.appendChild(container);

    // Cache DOM elements
    this.elements = {
      button: document.getElementById("banglarag-chat-toggle"),
      window: document.getElementById("banglarag-chat-window"),
      closeButton: document.getElementById("banglarag-close-button"),
      messages: document.getElementById("banglarag-messages"),
      input: document.getElementById("banglarag-input"),
      sendButton: document.getElementById("banglarag-send-button"),
      voiceButton: document.getElementById("banglarag-voice-button"),
      modelSelect: document.getElementById("banglarag-model-select"),
      status: document.getElementById("banglarag-status"),
    };
  }

  attachEventListeners() {
    // Toggle chat window
    this.elements.button.addEventListener("click", () => this.toggleChat());
    this.elements.closeButton.addEventListener("click", () =>
      this.toggleChat()
    );

    // Send message
    this.elements.sendButton.addEventListener("click", () =>
      this.sendMessage()
    );
    this.elements.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Auto-resize textarea
    this.elements.input.addEventListener("input", (e) => {
      e.target.style.height = "auto";
      e.target.style.height = Math.min(e.target.scrollHeight, 100) + "px";
    });

    // Model selection
    this.elements.modelSelect.addEventListener("change", (e) => {
      this.setModel(e.target.value);
    });

    // Language selection
    document
      .getElementById("banglarag-lang-en")
      .addEventListener("click", () => {
        this.setLanguage("english");
      });
    document
      .getElementById("banglarag-lang-bn")
      .addEventListener("click", () => {
        this.setLanguage("bangla");
      });

    // Voice input
    this.elements.voiceButton.addEventListener("click", () => {
      this.toggleVoiceInput();
    });
  }

  toggleChat() {
    this.isOpen = !this.isOpen;
    this.elements.window.classList.toggle("active");
    this.elements.button.classList.toggle("active");

    if (this.isOpen) {
      this.elements.input.focus();
      this.scrollToBottom();
    }
  }

  async loadModels() {
    try {
      const response = await fetch(`${this.config.apiUrl}/api/models`);
      const data = await response.json();

      if (data.success) {
        this.availableModels = data.models;
        this.currentModel = data.current_model;

        // Update model selector
        this.elements.modelSelect.innerHTML = this.availableModels
          .map(
            (model) =>
              `<option value="${model}" ${
                model === this.currentModel ? "selected" : ""
              }>${model}</option>`
          )
          .join("");
      }
    } catch (error) {
      console.error("Failed to load models:", error);
      this.elements.modelSelect.innerHTML =
        '<option value="">Error loading models</option>';
    }
  }

  async setModel(modelName) {
    try {
      const response = await fetch(`${this.config.apiUrl}/api/set-model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelName }),
      });

      const data = await response.json();
      if (data.success) {
        this.currentModel = data.current_model;
        this.showStatus(`Switched to ${modelName}`);
      }
    } catch (error) {
      console.error("Failed to set model:", error);
    }
  }

  setLanguage(language) {
    this.currentLanguage = language;

    // Update button states
    document
      .getElementById("banglarag-lang-en")
      .classList.toggle("active", language === "english");
    document
      .getElementById("banglarag-lang-bn")
      .classList.toggle("active", language === "bangla");

    // Show status
    const langName = language === "english" ? "English" : "বাংলা";
    this.showStatus(`Language: ${langName}`);

    // Add language indicator message
    const langMessage =
      language === "english"
        ? "🌐 Switched to English. I will respond in English."
        : "🌐 বাংলায় পরিবর্তন করা হয়েছে। আমি বাংলায় উত্তর দেব।";
    this.addMessage("bot", langMessage);
  }

  showStatus(message, duration = 3000) {
    this.elements.status.textContent = message;
    setTimeout(() => {
      this.elements.status.textContent = "Online";
    }, duration);
  }

  initVoiceRecognition() {
    // Check if browser supports Web Speech API
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.error("Speech recognition not supported");
      return false;
    }

    this.recognition = new SpeechRecognition();
    this.recognition.continuous = false;
    this.recognition.interimResults = false;

    // Set language based on current language setting
    this.recognition.lang =
      this.currentLanguage === "bangla" ? "bn-BD" : "en-US";

    this.recognition.onstart = () => {
      this.isRecording = true;
      this.elements.voiceButton.classList.add("recording");
      this.showStatus("🎤 Listening...");
    };

    this.recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      this.elements.input.value = transcript;
      this.elements.input.style.height = "auto";
      this.elements.input.style.height =
        Math.min(this.elements.input.scrollHeight, 100) + "px";
      this.showStatus("✅ Voice captured");
    };

    this.recognition.onerror = (event) => {
      this.isRecording = false;
      this.elements.voiceButton.classList.remove("recording");

      let errorMessage = "Voice input error";
      if (event.error === "not-allowed") {
        errorMessage = "⚠️ Microphone access denied";
      } else if (event.error === "no-speech") {
        errorMessage = "⚠️ No speech detected";
      }
      this.showStatus(errorMessage);
    };

    this.recognition.onend = () => {
      this.isRecording = false;
      this.elements.voiceButton.classList.remove("recording");
    };

    return true;
  }

  toggleVoiceInput() {
    // Initialize if not already done
    if (!this.recognition && !this.initVoiceRecognition()) {
      alert(
        "Voice input is not supported in your browser. Please use Chrome, Edge, or Safari."
      );
      return;
    }

    if (this.isRecording) {
      // Stop recording
      this.recognition.stop();
    } else {
      // Update language before starting
      this.recognition.lang =
        this.currentLanguage === "bangla" ? "bn-BD" : "en-US";

      // Start recording
      try {
        this.recognition.start();
      } catch (error) {
        console.error("Recognition start error:", error);
      }
    }
  }

  async sendMessage() {
    const query = this.elements.input.value.trim();
    if (!query || this.isStreaming) return;

    // Add user message
    this.addMessage("user", query);
    this.elements.input.value = "";
    this.elements.input.style.height = "auto";

    // Disable input during streaming
    this.setInputState(false);
    this.isStreaming = true;

    // Create bot message container for streaming
    const botMessageId = "bot-msg-" + Date.now();
    this.addMessage("bot", "", botMessageId);

    try {
      await this.streamResponse(query, botMessageId);
    } catch (error) {
      console.error("Chat error:", error);
      this.updateMessage(botMessageId, `❌ Error: ${error.message}`);
    } finally {
      this.isStreaming = false;
      this.setInputState(true);
    }
  }

  async streamResponse(query, messageId) {
    const response = await fetch(`${this.config.apiUrl}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        k: 3,
        language: this.currentLanguage, // Send selected language
      }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullResponse = "";
    let sources = [];

    // Show streaming indicator
    this.updateMessage(
      messageId,
      '<div class="banglarag-streaming-indicator"><span></span><span></span><span></span></div>'
    );

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = JSON.parse(line.slice(6));

          switch (data.type) {
            case "status":
              this.showStatus(data.message, 1000);
              break;

            case "sources":
              sources = data.sources;
              break;

            case "token":
              fullResponse += data.token;
              this.updateMessage(messageId, this.formatMessage(fullResponse));
              this.scrollToBottom();
              break;

            case "done":
              // Add sources if available
              if (sources.length > 0) {
                const sourcesHtml = this.formatSources(sources);
                this.updateMessage(
                  messageId,
                  this.formatMessage(fullResponse) + sourcesHtml
                );
              }
              this.showStatus(`Response generated with ${data.model}`);
              break;

            case "error":
              this.updateMessage(messageId, `❌ Error: ${data.message}`);
              break;
          }
        }
      }
    }
  }

  formatMessage(text) {
    // Basic markdown-like formatting
    return text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/\n/g, "<br>");
  }

  formatSources(sources) {
    if (!sources || sources.length === 0) return "";

    const sourcesHtml = sources
      .map(
        (source, index) => `
            <div class="banglarag-source-item">
                📄 Source ${index + 1}: ${source.metadata?.source || "Unknown"} 
                ${source.metadata?.page ? `(Page ${source.metadata.page})` : ""}
            </div>
        `
      )
      .join("");

    return `
            <div class="banglarag-sources">
                <div class="banglarag-sources-title">📚 Sources:</div>
                ${sourcesHtml}
            </div>
        `;
  }

  addMessage(type, content, id = null) {
    const messageId = id || `msg-${Date.now()}`;
    const avatar =
      type === "user" ? this.config.userAvatar : this.config.botAvatar;

    const messageDiv = document.createElement("div");
    messageDiv.className = `banglarag-message ${type}`;
    messageDiv.id = messageId;
    messageDiv.innerHTML = `
            <div class="banglarag-message-avatar">${avatar}</div>
            <div class="banglarag-message-content">${content}</div>
        `;

    this.elements.messages.appendChild(messageDiv);
    this.scrollToBottom();

    return messageId;
  }

  updateMessage(messageId, content) {
    const message = document.getElementById(messageId);
    if (message) {
      const contentDiv = message.querySelector(".banglarag-message-content");
      if (contentDiv) {
        contentDiv.innerHTML = content;
      }
    }
  }

  setInputState(enabled) {
    this.elements.input.disabled = !enabled;
    this.elements.sendButton.disabled = !enabled;
  }

  scrollToBottom() {
    setTimeout(() => {
      this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
    }, 100);
  }
}

// Auto-initialize if config is provided
if (typeof window.BanglaRAGConfig !== "undefined") {
  window.BanglaRAGChatbot = new BanglaRAGChatbot(window.BanglaRAGConfig);
}
