"""Unit tests for AudioProcessor orchestration layer (ai/transcription/audio_processor.py).

Verifies the high-level audio processing pipeline with mocked DiarizationProvider
and TranscriptionProvider so tests run offline without GPU/model overhead.
"""

from datetime import datetime
from unittest.mock import MagicMock
import pytest

from diarization.interface import DiarizationProvider, SpeakerSegment
from models.schemas import AudioMeetingResult
from transcription.audio_processor import AudioProcessor, process_audio
from transcription.interface import TranscriptionProvider, TranscriptTurn


class MockDiarization(DiarizationProvider):
    def diarize(self, audio_path: str):
        return [
            SpeakerSegment(start=0.0, end=8.0, speaker="speaker0"),
            SpeakerSegment(start=10.0, end=20.0, speaker="speaker1"),
            SpeakerSegment(start=42.0, end=48.0, speaker="speaker2"),
        ]


class MockTranscription(TranscriptionProvider):
    def transcribe(self, audio_path: str):
        return [
            TranscriptTurn(start=0.5, end=7.5, text="Welcome everyone. Let's start the meeting."),
            TranscriptTurn(start=10.5, end=19.5, text="I will finish the frontend dashboard by Friday."),
            TranscriptTurn(start=42.5, end=47.5, text="We decided to use PostgreSQL for data persistence."),
        ]


@pytest.fixture
def dummy_wav(tmp_path):
    f = tmp_path / "mock_meeting.wav"
    f.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    return str(f)


def test_missing_audio_file_raises():
    processor = AudioProcessor(
        transcription_provider=MockTranscription(),
        diarization_provider=MockDiarization(),
    )
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        processor.process_audio("non_existent_audio.wav")


def test_audio_processor_orchestration(dummy_wav):
    processor = AudioProcessor(
        transcription_provider=MockTranscription(),
        diarization_provider=MockDiarization(),
    )

    ref_date = datetime(2026, 10, 5, 10, 0, 0)
    result = processor.process_audio(
        audio_path=dummy_wav,
        meeting_id="test_meeting_001",
        reference_datetime=ref_date,
    )

    assert isinstance(result, AudioMeetingResult)
    assert result.meeting_id == "test_meeting_001"

    # Verify speaker-aware transcript has 3 turns and 3 distinct speakers
    assert len(result.speaker_transcript) == 3
    speakers = [t.speaker for t in result.speaker_transcript]
    assert speakers == ["speaker0", "speaker1", "speaker2"]

    # Verify action items extracted
    assert len(result.action_items) >= 1
    action = result.action_items[0]
    assert "frontend dashboard" in action.action.lower()
    assert action.responsible_person == "speaker1"
    assert action.deadline == "2026-10-09"

    # Verify decisions extracted
    assert len(result.decisions) >= 1
    decision = result.decisions[0]
    assert "postgresql" in decision.decision.lower()
    assert decision.speaker == "speaker2"

    # Verify participants
    assert "speaker0" in result.participants
    assert "speaker1" in result.participants
    assert "speaker2" in result.participants


def test_process_audio_convenience_function(dummy_wav):
    result = process_audio(
        audio_path=dummy_wav,
        meeting_id="conv_test_001",
        transcription_provider=MockTranscription(),
        diarization_provider=MockDiarization(),
    )
    assert isinstance(result, AudioMeetingResult)
    assert len(result.speaker_transcript) == 3
