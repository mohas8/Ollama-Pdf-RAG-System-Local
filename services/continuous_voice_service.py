#!/usr/bin/env python3
"""
Enhanced voice service with continuous listening and automatic pause detection.
Implements natural conversation flow with automatic question detection and response.
"""

import time
import threading
import queue
import numpy as np
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import webrtcvad
import collections

try:
    import pyaudio
    import wave

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    pyaudio = None

from core.logging_config import BanglaRAGLogger
from core.exceptions import AudioException
from core.utils import create_temp_file
from core.constants import AUDIO_RATE, AUDIO_CHANNELS, AUDIO_CHUNK_SIZE
from services.voice_service import get_voice_service
from services.database_service import get_database_manager
from services.llm_service import get_rag_processor

logger = BanglaRAGLogger.get_logger("continuous_voice")


class VoiceActivityDetector:
    """Voice Activity Detection using WebRTC VAD."""

    def __init__(self, sample_rate=16000, frame_duration_ms=30):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        try:
            # WebRTC VAD - more accurate for speech detection
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(2)  # 0=least aggressive, 3=most aggressive
            self.webrtc_available = True
            logger.info("Using WebRTC VAD for voice activity detection")
        except ImportError:
            self.webrtc_available = False
            logger.warning(
                "WebRTC VAD not available, using basic energy-based detection"
            )

    def is_speech(self, audio_frame: bytes) -> bool:
        """Detect if audio frame contains speech."""
        if self.webrtc_available:
            try:
                # WebRTC VAD expects 16kHz, 16-bit audio
                return self.vad.is_speech(audio_frame, self.sample_rate)
            except Exception as e:
                logger.debug(f"WebRTC VAD error, falling back to energy detection: {e}")

        # Fallback: Simple energy-based detection
        return self._energy_based_detection(audio_frame)

    def _energy_based_detection(
        self, audio_frame: bytes, threshold: float = 0.01
    ) -> bool:
        """Simple energy-based voice activity detection."""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_frame, dtype=np.int16)
            # Calculate RMS energy
            energy = np.sqrt(np.mean(audio_data**2)) / 32768.0  # Normalize to 0-1
            return energy > threshold
        except Exception:
            return False


class ContinuousVoiceSession:
    """Manages continuous voice conversation with automatic pause detection."""

    def __init__(
        self,
        silence_threshold: float = 2.0,  # 2 seconds of silence to trigger processing
        min_speech_duration: float = 0.5,  # Minimum speech duration to consider
        max_question_duration: float = 30.0,  # Maximum question duration
        sample_rate: int = 16000,
    ):  # 16kHz for better VAD performance

        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.max_question_duration = max_question_duration
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * 0.03)  # 30ms chunks for VAD

        # Audio components
        self.audio = None
        self.stream = None
        self.vad = VoiceActivityDetector(sample_rate)

        # Session state
        self.is_active = False
        self.is_listening = False
        self.audio_buffer = collections.deque(
            maxlen=int(sample_rate * max_question_duration / self.chunk_size)
        )
        self.speech_frames = []
        self.silence_start = None
        self.speech_start = None

        # Services
        self.voice_service = get_voice_service()
        self.db_manager = get_database_manager()
        self.rag_processor = get_rag_processor()

        # Threading
        self.audio_thread = None
        self.processing_lock = threading.Lock()

        # Callbacks
        self.on_question_detected: Optional[Callable[[str], None]] = None
        self.on_response_ready: Optional[Callable[[str, str], None]] = None
        self.on_session_status: Optional[Callable[[str], None]] = None

    def start_session(self) -> bool:
        """Start continuous voice session."""
        if not AUDIO_AVAILABLE:
            logger.error("Audio libraries not available")
            return False

        try:
            # Initialize audio
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=AUDIO_CHANNELS,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=None,
            )

            self.is_active = True
            self.is_listening = True

            # Start audio processing thread
            self.audio_thread = threading.Thread(
                target=self._audio_processing_loop, daemon=True
            )
            self.audio_thread.start()

            logger.info("Continuous voice session started")
            self._notify_status(
                "🎤 Listening... Ask your question and pause for 2 seconds"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to start voice session: {e}")
            self._cleanup()
            return False

    def stop_session(self):
        """Stop continuous voice session."""
        self.is_active = False
        self.is_listening = False

        # Wait for processing to complete
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)

        self._cleanup()
        logger.info("Continuous voice session stopped")
        self._notify_status("👋 Voice session ended")

    def _audio_processing_loop(self):
        """Main audio processing loop."""
        logger.info("Audio processing loop started")

        while self.is_active:
            try:
                # Read audio chunk
                audio_chunk = self.stream.read(
                    self.chunk_size, exception_on_overflow=False
                )
                current_time = time.time()

                # Add to circular buffer
                self.audio_buffer.append((audio_chunk, current_time))

                # Voice activity detection
                is_speech = self.vad.is_speech(audio_chunk)

                if is_speech:
                    self._handle_speech_detected(audio_chunk, current_time)
                else:
                    self._handle_silence_detected(current_time)

            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                if not self.is_active:
                    break
                time.sleep(0.1)  # Brief pause before retry

    def _handle_speech_detected(self, audio_chunk: bytes, current_time: float):
        """Handle when speech is detected."""
        # Reset silence timer
        self.silence_start = None

        # Start speech if not already started
        if self.speech_start is None:
            self.speech_start = current_time
            self.speech_frames = []
            logger.debug("Speech started")
            self._notify_status("🎙️ Recording question...")

        # Add to speech buffer
        self.speech_frames.append(audio_chunk)

        # Check for maximum question duration
        if current_time - self.speech_start > self.max_question_duration:
            logger.warning(
                "Maximum question duration reached, processing current audio"
            )
            self._process_speech_buffer()

    def _handle_silence_detected(self, current_time: float):
        """Handle when silence is detected."""
        if self.speech_start is not None:
            # We were recording speech, now there's silence
            if self.silence_start is None:
                self.silence_start = current_time
                logger.debug("Silence started after speech")

            # Check if silence duration exceeds threshold
            silence_duration = current_time - self.silence_start
            if silence_duration >= self.silence_threshold:
                # Check if we have enough speech
                speech_duration = self.silence_start - self.speech_start
                if speech_duration >= self.min_speech_duration:
                    logger.info(
                        f"Question detected: {speech_duration:.1f}s speech + {silence_duration:.1f}s silence"
                    )
                    self._process_speech_buffer()
                else:
                    logger.debug("Speech too short, ignoring")
                    self._reset_speech_detection()

    def _process_speech_buffer(self):
        """Process accumulated speech audio."""
        if not self.speech_frames:
            return

        with self.processing_lock:
            try:
                self._notify_status("🔄 Processing your question...")

                # Save audio to temporary file
                temp_file = self._save_speech_to_file(self.speech_frames)

                # Transcribe audio
                transcription_result = self.voice_service.transcribe_file(temp_file)

                if not transcription_result.get("success", False):
                    logger.error("Transcription failed")
                    self._notify_status(
                        "❌ Sorry, I couldn't understand your question. Please try again."
                    )
                    self._reset_speech_detection()
                    return

                question = transcription_result.get("text", "").strip()
                if not question:
                    logger.warning("No text transcribed")
                    self._notify_status("⚠️ No speech detected. Please speak clearly.")
                    self._reset_speech_detection()
                    return

                logger.info(f"Question transcribed: {question}")
                if self.on_question_detected:
                    self.on_question_detected(question)

                # Process RAG query
                self._notify_status("🤖 Generating response...")
                response = self._process_rag_query(question)

                if self.on_response_ready:
                    self.on_response_ready(question, response)

                self._reset_speech_detection()
                self._notify_status("🎤 Ready for your next question...")

            except Exception as e:
                logger.error(f"Error processing speech: {e}")
                self._notify_status("❌ Error processing question. Please try again.")
                self._reset_speech_detection()

    def _save_speech_to_file(self, speech_frames: list) -> str:
        """Save speech frames to temporary WAV file."""
        temp_file = create_temp_file(suffix=".wav", prefix="continuous_voice_")

        with wave.open(temp_file, "wb") as wf:
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(speech_frames))

        return temp_file

    def _process_rag_query(self, question: str) -> str:
        """Process question through RAG system."""
        try:
            # Search for relevant documents
            relevant_docs = self.db_manager.search_with_cache(question, k=3)

            if not relevant_docs:
                return "I couldn't find relevant information in the documents for your question. Please try rephrasing or ask about topics covered in the loaded documents."

            # Generate response
            rag_result = self.rag_processor.process_rag_query(question, relevant_docs)

            if rag_result.get("success", False):
                return rag_result.get(
                    "response", "Sorry, I couldn't generate a response."
                )
            else:
                return f"I encountered an error while processing your question: {rag_result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"RAG query processing failed: {e}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."

    def _reset_speech_detection(self):
        """Reset speech detection state."""
        self.speech_start = None
        self.silence_start = None
        self.speech_frames = []

    def _notify_status(self, message: str):
        """Notify status change."""
        logger.info(f"Status: {message}")
        if self.on_session_status:
            self.on_session_status(message)

    def _cleanup(self):
        """Clean up audio resources."""
        try:
            if self.stream:
                self.stream.close()
                self.stream = None

            if self.audio:
                self.audio.terminate()
                self.audio = None

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class ContinuousVoiceInterface:
    """User interface for continuous voice interaction."""

    def __init__(self):
        self.session: Optional[ContinuousVoiceSession] = None
        self.conversation_history = []

    def get_service_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            "available": AUDIO_AVAILABLE,
            "webrtcvad_available": True,  # Since we're importing it successfully
            "features": [
                "continuous_listening",
                "automatic_pause_detection",
                "natural_conversation",
            ],
            "silence_threshold_default": 2.0,
            "min_speech_duration_default": 0.5,
        }

    def start_continuous_session(
        self, silence_threshold: float = 2.0, min_speech_duration: float = 0.5
    ) -> bool:
        """Start continuous voice conversation."""
        print("\n🎤 CONTINUOUS VOICE SESSION")
        print("=" * 50)
        print("🗣️  Speak naturally - pause for 2 seconds to get response")
        print("⏸️  The system will automatically detect when you finish speaking")
        print("🔄 After each response, it will start listening again")
        print("❌ Press Ctrl+C to exit")
        print()

        self.session = ContinuousVoiceSession(
            silence_threshold=silence_threshold, min_speech_duration=min_speech_duration
        )

        # Set up callbacks
        self.session.on_question_detected = self._on_question_detected
        self.session.on_response_ready = self._on_response_ready
        self.session.on_session_status = self._on_status_update

        # Start session
        if self.session.start_session():
            try:
                # Keep main thread alive
                while self.session.is_active:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n👋 Stopping voice session...")
                self.session.stop_session()
            return True
        else:
            print("❌ Failed to start continuous voice session")
            return False

    def _on_question_detected(self, question: str):
        """Handle detected question."""
        print(f'\n🎯 You asked: "{question}"')
        self.conversation_history.append(
            {"type": "question", "content": question, "time": datetime.now()}
        )

    def _on_response_ready(self, question: str, response: str):
        """Handle response ready."""
        print(f"\n💡 Answer:")
        print(f"{response}")
        print("\n" + "-" * 50)
        self.conversation_history.append(
            {"type": "answer", "content": response, "time": datetime.now()}
        )

    def _on_status_update(self, status: str):
        """Handle status updates."""
        print(f"\r{status}", end="", flush=True)

    def show_conversation_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("No conversation history")
            return

        print("\n📜 CONVERSATION HISTORY")
        print("=" * 50)

        for i, entry in enumerate(self.conversation_history, 1):
            timestamp = entry["time"].strftime("%H:%M:%S")
            if entry["type"] == "question":
                print(f"\n[{timestamp}] 🤔 Q{i//2 + 1}: {entry['content']}")
            else:
                print(f"[{timestamp}] 💡 A{i//2}: {entry['content'][:100]}...")


# Integration with main application
def add_continuous_voice_to_main():
    """Add continuous voice option to main menu."""
    return """
    
def continuous_voice_session(self) -> None:
    \"\"\"Start continuous voice session with automatic pause detection.\"\"\"
    try:
        interface = ContinuousVoiceInterface()
        
        print("\\n🎤 CONTINUOUS VOICE SESSION")
        print("=" * 50)
        print("This mode allows natural conversation:")
        print("• Speak your question naturally")
        print("• Pause for 2 seconds when done")
        print("• Get automatic response")
        print("• Continue with next question")
        print()
        
        choice = input("Start continuous session? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            success = interface.start_continuous_session()
            if success:
                print("\\n📊 Session completed!")
                interface.show_conversation_history()
        else:
            print("Continuous voice session cancelled")
            
    except Exception as e:
        log_error(f"Continuous voice session failed: {e}", "main", exc_info=True)
        print(f"❌ Continuous voice session failed: {e}")
"""


if __name__ == "__main__":
    # Test continuous voice session
    interface = ContinuousVoiceInterface()
    interface.start_continuous_session()
