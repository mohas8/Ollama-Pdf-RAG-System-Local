# Ollama-Pdf-RAG-System-Local

Local, private Retrieval-Augmented Generation (RAG) over PDFs. Builds embeddings, stores them in ChromaDB, and answers questions from your documents. Optional voice input (Whisper) and English/Bangla translation.

## Features
- PDF parsing (PyPDF2/pypdf) and embedding (sentence-transformers)
- Local vector store with ChromaDB and retrieval via LangChain
- Torch/Transformers-backed text generation pipeline (local-first)
- Optional speech-to-text with Whisper + VAD (webrtcvad) and mic input (pyaudio)
- Optional translation (googletrans) for EN/Bn workflows
- Simple .env-based configuration

## Quick Start (Windows)
1. Prerequisites:
   - Python 3.10+ recommended
   - FFmpeg (needed by Whisper). Example: `choco install ffmpeg` or download from ffmpeg.org
   - Microphone (optional, for voice)
2. Setup:
   - Create venv: `python -m venv .venv`
   - Activate: `.venv\Scripts\activate`
   - Install deps: `pip install -r requirements.txt`
   - If pyaudio fails: `pip install pipwin && pipwin install pyaudio`
3. Run:
   - Use the repo’s scripts to index PDFs into ChromaDB and start a chat/QA session (e.g., an ingest script followed by a chat script). 
   - Configure paths/models via `.env` if provided by the code.

## Tech Stack
- LangChain, ChromaDB, sentence-transformers
- Torch, Transformers
- PyPDF2/pypdf
- Whisper, SpeechRecognition, pyaudio, webrtcvad
- googletrans, numpy, pandas, python-dotenv

Notes:
- Replace run commands above with the actual script names in this repo.
- Whisper requires FFmpeg in PATH.


