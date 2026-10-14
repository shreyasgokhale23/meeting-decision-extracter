"""Whisper transcription provider implementation.

Provides a concrete `TranscriptionProvider` using the `whisper` library (OpenAI's Whisper).

The provider loads a model (default: 'base') and offers a `transcribe` method that accepts a path to an audio file and returns a list of
`TranscriptTurn` objects with start/end timestamps and the recognized text.

Errors are raised for missing files or transcription failures, with clear messages.
"""

import os
from typing import List

from pydantic import ValidationError
from whisper import load_model  # type: ignore

from transcription.interface import TranscriptTurn, TranscriptionProvider


class WhisperProvider(TranscriptionProvider):
    """Concrete transcription provider backed by OpenAI Whisper.

    Parameters
    ----------
    model_name: str, optional
        Name of the Whisper model to load. Typical values are ``'tiny'``, ``'base'``, ``'small'``, ``'medium'`` or ``'large'``.
        The default ``'base'`` balances speed and accuracy for development laptops.
    """

    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        try:
            # Lazily load the model; this may take a few seconds on first call.
            self.model = load_model(self.model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Whisper model '{self.model_name}': {exc}") from exc

    def transcribe(self, audio_path: str) -> List[TranscriptTurn]:
        """Transcribe an audio file into timestamped ``TranscriptTurn`` objects.

        Parameters
        ----------
        audio_path: str
            Path to the audio file. Must exist and be a format supported by Whisper/FFmpeg.

        Returns
        -------
        List[TranscriptTurn]
            Ordered list of transcript turns with ``start``, ``end`` (seconds) and ``text``.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Whisper's `transcribe` returns a dict with a ``segments`` key.
            result = self.model.transcribe(audio_path, word_timestamps=False)
        except Exception as exc:
            raise RuntimeError(f"Whisper transcription failed for '{audio_path}': {exc}") from exc

        if "segments" not in result:
            raise RuntimeError("Unexpected Whisper output: missing 'segments' key")

        turns: List[TranscriptTurn] = []
        for seg in result["segments"]:
            try:
                turn = TranscriptTurn(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=str(seg.get("text", "")).strip(),
                )
                # Skip empty/whitespace-only turns.
                if turn.text:
                    turns.append(turn)
            except ValidationError:
                # Skip invalid segments.
                continue
        return turns
