"""Speech-to-Text Transcription Provider Interface.

Defines the contract for future speech-to-text systems (e.g. Whisper)
without coupling heavy PyTorch/CUDA dependencies to the NLP layer.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class TranscriptTurn(BaseModel):
    """Transcription turn output from a speech recognition engine."""
    text: str
    start: float
    end: float

    model_config = ConfigDict(extra="ignore")


class TranscriptionProvider(ABC):
    """Abstract base class for speech-to-text providers."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> List[TranscriptTurn]:
        """Transcribe speech in an audio file or stream into timestamped text turns.
        
        Args:
            audio_path: Path to target audio file or stream buffer.
            
        Returns:
            List of detected TranscriptTurns.
        """
        raise NotImplementedError("Transcription backends (e.g. Whisper) are plugged in at the audio layer.")
