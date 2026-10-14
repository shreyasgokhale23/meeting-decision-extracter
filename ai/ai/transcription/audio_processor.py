"""End-to-End Audio Processing Orchestrator.

Coordinates:
Audio File (WAV)
     ↓
Diart Speaker Diarization (Who spoke when)
     ↓
Whisper Speech-to-Text (What was said when)
     ↓
Speaker-Aware Alignment (Temporal overlap matching)
     ↓
Existing NLP Pipeline (Actions, Decisions, Deadlines, Assignees, Statuses)
     ↓
Structured AudioMeetingResult

This module exposes a high-level API suitable for backend integration,
decoupling the caller from raw Diart, Whisper, and PyTorch details.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from diarization.diart_provider import DiartProvider
from diarization.interface import DiarizationProvider
from models.schemas import AudioMeetingResult, SpeakerTranscriptTurn
from nlp.adapters import from_speaker_transcript_turn
from nlp.pipeline import process_transcript
from transcription.interface import TranscriptionProvider
from transcription.speaker_aware import align_transcript_with_speakers
from transcription.whisper_provider import WhisperProvider

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Orchestrates end-to-end diarization, transcription, alignment, and NLP intelligence extraction.

    Supports dependency injection for transcription and diarization providers,
    enabling lightweight unit testing without heavy model downloads.
    """

    def __init__(
        self,
        transcription_provider: Optional[TranscriptionProvider] = None,
        diarization_provider: Optional[DiarizationProvider] = None,
    ) -> None:
        self._transcription_provider = transcription_provider
        self._diarization_provider = diarization_provider

    @property
    def transcription_provider(self) -> TranscriptionProvider:
        if self._transcription_provider is None:
            logger.info("Instantiating default WhisperProvider('base')...")
            self._transcription_provider = WhisperProvider("base")
        return self._transcription_provider

    @property
    def diarization_provider(self) -> DiarizationProvider:
        if self._diarization_provider is None:
            logger.info("Instantiating default DiartProvider()...")
            self._diarization_provider = DiartProvider()
        return self._diarization_provider

    def process_audio(
        self,
        audio_path: Union[str, Path],
        meeting_id: Optional[str] = None,
        reference_datetime: Optional[datetime] = None,
    ) -> AudioMeetingResult:
        """Process an audio file through diarization, STT, alignment, and NLP.

        Args:
            audio_path: Path to the target audio file (WAV).
            meeting_id: Optional session ID identifier.
            reference_datetime: Optional base date for relative deadline resolution.

        Returns:
            Structured AudioMeetingResult containing speaker-aware transcript and
            extracted meeting intelligence.
        """
        path_str = str(audio_path)
        p = Path(path_str)
        if not p.is_file():
            raise FileNotFoundError(f"Audio file not found: {path_str}")

        m_id = meeting_id or p.stem
        logger.info(f"Starting end-to-end audio processing for: {path_str} (meeting_id={m_id})")

        # Step 1: Speaker Diarization (Who spoke when)
        logger.info("[Step 1/4] Running speaker diarization via Diart...")
        speaker_segments = self.diarization_provider.diarize(path_str)
        unique_speakers = {s.speaker for s in speaker_segments}
        logger.info(f"Diarization produced {len(speaker_segments)} segments across {len(unique_speakers)} speakers: {sorted(unique_speakers)}")

        # Step 2: Speech-to-Text Transcription (What was said when)
        logger.info("[Step 2/4] Running speech-to-text via Whisper...")
        whisper_turns = self.transcription_provider.transcribe(path_str)
        logger.info(f"Transcription produced {len(whisper_turns)} transcript turns.")

        # Step 3: Speaker-Aware Transcript Alignment (Temporal overlap)
        logger.info("[Step 3/4] Aligning Whisper turns with Diart speaker segments...")
        aligned_turns: List[SpeakerTranscriptTurn] = align_transcript_with_speakers(
            whisper_turns=whisper_turns,
            diarization_segments=speaker_segments,
        )
        logger.info(f"Successfully aligned {len(aligned_turns)} speaker-aware transcript turns.")

        # Step 4: NLP Pipeline Processing
        logger.info("[Step 4/4] Passing speaker-aware turns through NLP extraction pipeline...")
        nlp_input_segments = [
            from_speaker_transcript_turn(turn, meeting_id=m_id)
            for turn in aligned_turns
        ]

        meeting_summary = process_transcript(
            segments=nlp_input_segments,
            meeting_id=m_id,
            reference_datetime=reference_datetime,
        )

        logger.info(
            f"Extraction complete | Actions: {len(meeting_summary.action_items)} | "
            f"Decisions: {len(meeting_summary.decisions)} | "
            f"Participants: {len(meeting_summary.participants)}"
        )

        return AudioMeetingResult(
            meeting_id=m_id,
            speaker_transcript=aligned_turns,
            meeting_summary=meeting_summary,
            action_items=meeting_summary.action_items,
            decisions=meeting_summary.decisions,
            participants=meeting_summary.participants,
        )


def process_audio(
    audio_path: Union[str, Path],
    meeting_id: Optional[str] = None,
    reference_datetime: Optional[datetime] = None,
    transcription_provider: Optional[TranscriptionProvider] = None,
    diarization_provider: Optional[DiarizationProvider] = None,
) -> AudioMeetingResult:
    """Convenience function to run end-to-end audio processing.

    Args:
        audio_path: Path to target audio file (WAV).
        meeting_id: Optional meeting ID.
        reference_datetime: Optional reference datetime for relative deadlines.
        transcription_provider: Optional custom transcription provider.
        diarization_provider: Optional custom diarization provider.

    Returns:
        Structured AudioMeetingResult.
    """
    processor = AudioProcessor(
        transcription_provider=transcription_provider,
        diarization_provider=diarization_provider,
    )
    return processor.process_audio(
        audio_path=audio_path,
        meeting_id=meeting_id,
        reference_datetime=reference_datetime,
    )
