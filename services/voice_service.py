#!/usr/bin/env python3
"""
Refactored voice input service for BanglaRAG system.
Provides clean interface for voice recognition with proper resource management.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import os
import threading
import time
from contextlib import contextmanager

try:
    import whisper
    import pyaudio
    import wave

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    whisper = None
    pyaudio = None
    wave = None

try:
    from banglaspeech2text import Speech2Text
    import speech_recognition as sr

    BANGLA_STT_AVAILABLE = True
except ImportError:
    BANGLA_STT_AVAILABLE = False
    Speech2Text = None
    sr = None

from core.logging_config import BanglaRAGLogger
from core.exceptions import AudioException, ModelException, FileProcessingException
from core.utils import retry_with_backoff, measure_performance, create_temp_file
from core.constants import (
    AUDIO_CHUNK_SIZE,
    AUDIO_CHANNELS,
    AUDIO_RATE,
    DEFAULT_RECORDING_DURATION,
    DEFAULT_WHISPER_MODEL,
    BANGLA_STT_MODELS,
)

logger = BanglaRAGLogger.get_logger("voice")


class AudioRecorder:
    """Handles audio recording with proper resource management."""

    def __init__(self):
        if not AUDIO_AVAILABLE:
            raise AudioException(
                "Audio libraries not available. Install pyaudio and other dependencies."
            )

        self._audio: Optional[pyaudio.PyAudio] = None
        self._recording = False
        self._lock = threading.Lock()

    @contextmanager
    def _audio_context(self):
        """Context manager for PyAudio instance."""
        try:
            self._audio = pyaudio.PyAudio()
            yield self._audio
        finally:
            if self._audio:
                self._audio.terminate()
                self._audio = None

    @measure_performance
    def record_audio(
        self,
        duration: float = DEFAULT_RECORDING_DURATION,
        sample_rate: int = AUDIO_RATE,
        channels: int = AUDIO_CHANNELS,
        chunk_size: int = AUDIO_CHUNK_SIZE,
    ) -> str:
        """
        Record audio and save to temporary file.

        Returns:
            Path to recorded audio file
        """
        with self._lock:
            if self._recording:
                raise AudioException("Already recording")

            self._recording = True

        try:
            frames = []
            temp_file = create_temp_file(suffix=".wav", prefix="voice_")

            with self._audio_context() as audio:
                # Check for available audio devices
                if audio.get_device_count() == 0:
                    raise AudioException("No audio devices found")

                # Open audio stream
                try:
                    stream = audio.open(
                        format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                    )
                except Exception as e:
                    raise AudioException(f"Failed to open audio stream: {e}")

                logger.info(f"Recording for {duration} seconds...")

                # Record audio
                for _ in range(int(sample_rate / chunk_size * duration)):
                    try:
                        data = stream.read(chunk_size, exception_on_overflow=False)
                        frames.append(data)
                    except Exception as e:
                        logger.error(f"Error reading audio data: {e}")
                        break

                stream.stop_stream()
                stream.close()

            # Save to WAV file
            try:
                with wave.open(temp_file, "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(sample_rate)
                    wf.writeframes(b"".join(frames))

                logger.info(f"Audio saved to {temp_file}")
                return temp_file

            except Exception as e:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                raise FileProcessingException(f"Failed to save audio file: {e}")

        except Exception as e:
            logger.error(f"Audio recording failed: {e}")
            raise AudioException(f"Recording failed: {e}")

        finally:
            with self._lock:
                self._recording = False

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording


class VoiceRecognitionModel(ABC):
    """Abstract base class for voice recognition models."""

    @abstractmethod
    def transcribe(self, audio_file: str) -> Dict[str, Any]:
        """Transcribe audio file to text."""
        pass

    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """Check if model supports given language."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class WhisperModel(VoiceRecognitionModel):
    """Whisper-based voice recognition model."""

    def __init__(self, model_size: str = DEFAULT_WHISPER_MODEL):
        if not AUDIO_AVAILABLE or not whisper:
            raise ModelException("Whisper not available")

        self.model_size = model_size
        self._model: Optional[whisper.Whisper] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self._model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise ModelException(f"Whisper model loading failed: {e}")

    @retry_with_backoff(max_retries=2)
    @measure_performance
    def transcribe(self, audio_file: str) -> Dict[str, Any]:
        """Transcribe audio using Whisper."""
        if not self._model:
            raise ModelException("Whisper model not loaded")

        try:
            # Validate audio file
            if not os.path.exists(audio_file):
                raise FileProcessingException(f"Audio file not found: {audio_file}")

            # Transcribe
            result = self._model.transcribe(audio_file)

            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "unknown"),
                "confidence": None,  # Whisper doesn't provide confidence scores
                "model": f"whisper-{self.model_size}",
                "success": True,
            }

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {
                "text": "",
                "language": "unknown",
                "confidence": None,
                "model": f"whisper-{self.model_size}",
                "success": False,
                "error": str(e),
            }

    def supports_language(self, language: str) -> bool:
        """Whisper supports many languages."""
        return True  # Whisper is multilingual

    def get_model_info(self) -> Dict[str, Any]:
        """Get Whisper model information."""
        return {
            "name": "whisper",
            "size": self.model_size,
            "multilingual": True,
            "available": self._model is not None,
        }


class BanglaSpeechModel(VoiceRecognitionModel):
    """BanglaSpeech2Text-based voice recognition model."""

    def __init__(self, model_size: str = "base"):
        if not BANGLA_STT_AVAILABLE:
            raise ModelException("BanglaSpeech2Text not available")

        self.model_size = model_size
        self._model: Optional[Speech2Text] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load BanglaSpeech2Text model."""
        try:
            logger.info(f"Loading BanglaSpeech2Text model: {self.model_size}")
            self._model = Speech2Text(model_name=self.model_size)
            logger.info("BanglaSpeech2Text model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BanglaSpeech2Text model: {e}")
            raise ModelException(f"BanglaSpeech2Text model loading failed: {e}")

    @retry_with_backoff(max_retries=2)
    @measure_performance
    def transcribe(self, audio_file: str) -> Dict[str, Any]:
        """Transcribe audio using BanglaSpeech2Text."""
        if not self._model:
            raise ModelException("BanglaSpeech2Text model not loaded")

        try:
            # Validate audio file
            if not os.path.exists(audio_file):
                raise FileProcessingException(f"Audio file not found: {audio_file}")

            # Transcribe
            result = self._model.transcribe(audio_file)

            # Extract text from result
            if isinstance(result, dict):
                text = result.get("text", str(result))
            else:
                text = str(result)

            return {
                "text": text.strip(),
                "language": "bn",  # BanglaSpeech2Text is for Bangla
                "confidence": None,
                "model": f"banglaspeech2text-{self.model_size}",
                "success": True,
            }

        except Exception as e:
            logger.error(f"BanglaSpeech2Text transcription failed: {e}")
            return {
                "text": "",
                "language": "bn",
                "confidence": None,
                "model": f"banglaspeech2text-{self.model_size}",
                "success": False,
                "error": str(e),
            }

    def supports_language(self, language: str) -> bool:
        """BanglaSpeech2Text supports Bangla."""
        return language.lower() in ["bn", "bangla", "bengali"]

    def get_model_info(self) -> Dict[str, Any]:
        """Get BanglaSpeech2Text model information."""
        model_info = BANGLA_STT_MODELS.get(self.model_size, {})
        return {
            "name": "banglaspeech2text",
            "size": self.model_size,
            "multilingual": False,
            "language": "bangla",
            "wer": model_info.get("wer"),
            "size_info": model_info.get("size"),
            "available": self._model is not None,
        }


class VoiceInputService:
    """Main service for handling voice input operations."""

    def __init__(self):
        self.recorder = AudioRecorder() if AUDIO_AVAILABLE else None
        self._models: Dict[str, VoiceRecognitionModel] = {}
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize available voice recognition models."""
        # Initialize Whisper
        try:
            if AUDIO_AVAILABLE and whisper:
                self._models["whisper"] = WhisperModel()
                logger.info("Whisper model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Whisper: {e}")

        # Initialize BanglaSpeech2Text
        try:
            if BANGLA_STT_AVAILABLE:
                self._models["bangla_stt"] = BanglaSpeechModel()
                logger.info("BanglaSpeech2Text model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize BanglaSpeech2Text: {e}")

        if not self._models:
            logger.warning("No voice recognition models available")

    def get_available_models(self) -> List[str]:
        """Get list of available voice recognition models."""
        return list(self._models.keys())

    @measure_performance
    def record_and_transcribe(
        self,
        duration: float = DEFAULT_RECORDING_DURATION,
        model_preference: Optional[str] = None,
        language_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record audio and transcribe to text."""
        if not self.recorder:
            raise AudioException("Audio recording not available")

        if not self._models:
            raise ModelException("No voice recognition models available")

        temp_file = None
        try:
            # Record audio
            logger.info(f"Starting voice recording for {duration} seconds...")
            temp_file = self.recorder.record_audio(duration)

            # Choose model
            model = self._choose_model(model_preference, language_hint)
            if not model:
                raise ModelException("No suitable model found")

            # Transcribe
            logger.info(f"Transcribing with {model.get_model_info()['name']}...")
            result = model.transcribe(temp_file)

            # Add recording info
            result.update({"recording_duration": duration, "audio_file": temp_file})

            return result

        except Exception as e:
            logger.error(f"Voice input processing failed: {e}")
            return {
                "text": "",
                "success": False,
                "error": str(e),
                "recording_duration": duration,
            }

        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")

    def transcribe_file(
        self,
        audio_file: str,
        model_preference: Optional[str] = None,
        language_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe existing audio file."""
        if not self._models:
            raise ModelException("No voice recognition models available")

        try:
            # Choose model
            model = self._choose_model(model_preference, language_hint)
            if not model:
                raise ModelException("No suitable model found")

            # Transcribe
            logger.info(f"Transcribing file {audio_file}...")
            result = model.transcribe(audio_file)

            return result

        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return {"text": "", "success": False, "error": str(e)}

    def _choose_model(
        self,
        model_preference: Optional[str] = None,
        language_hint: Optional[str] = None,
    ) -> Optional[VoiceRecognitionModel]:
        """Choose appropriate model based on preferences and language hint."""
        # If specific model requested
        if model_preference and model_preference in self._models:
            return self._models[model_preference]

        # If language hint provided
        if language_hint:
            # For Bangla, prefer BanglaSpeech2Text if available
            if language_hint.lower() in ["bn", "bangla", "bengali"]:
                if "bangla_stt" in self._models:
                    return self._models["bangla_stt"]

            # For other languages, use Whisper
            if "whisper" in self._models:
                return self._models["whisper"]

        # Default: prefer Whisper for general use
        if "whisper" in self._models:
            return self._models["whisper"]

        # Fallback to any available model
        if self._models:
            return next(iter(self._models.values()))

        return None

    def get_service_info(self) -> Dict[str, Any]:
        """Get voice input service information."""
        model_info = {}
        for name, model in self._models.items():
            model_info[name] = model.get_model_info()

        return {
            "audio_available": AUDIO_AVAILABLE,
            "bangla_stt_available": BANGLA_STT_AVAILABLE,
            "recording_available": self.recorder is not None,
            "models": model_info,
            "default_duration": DEFAULT_RECORDING_DURATION,
        }


# Global service instance
_voice_service: Optional[VoiceInputService] = None


def get_voice_service() -> VoiceInputService:
    """Get global voice input service instance."""
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceInputService()
    return _voice_service


# Compatibility functions
def record_voice_input(duration: float = DEFAULT_RECORDING_DURATION) -> Dict[str, Any]:
    """Compatibility function for voice recording."""
    service = get_voice_service()
    return service.record_and_transcribe(duration)


def transcribe_audio_file(file_path: str) -> Dict[str, Any]:
    """Compatibility function for file transcription."""
    service = get_voice_service()
    return service.transcribe_file(file_path)


def test_voice_service() -> bool:
    """Test voice input service functionality."""
    try:
        logger.info("Testing voice input service...")
        service = get_voice_service()

        # Get service info
        info = service.get_service_info()
        logger.info(f"Voice service info: {info}")

        # Check if audio is available
        if not info["audio_available"]:
            logger.warning("Audio libraries not available - skipping audio tests")
            return False

        if not info["models"]:
            logger.warning("No voice recognition models available")
            return False

        logger.info("Voice input service test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Voice service test failed: {e}")
        return False


if __name__ == "__main__":
    test_voice_service()
