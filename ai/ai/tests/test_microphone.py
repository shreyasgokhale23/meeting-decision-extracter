"""Unit tests for MicrophoneRecorder and recording orchestration.

All unit tests run completely offline and mock the underlying sounddevice hardware
streams and speech/NLP models.
"""

import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from audio.microphone import (
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    EmptyRecordingError,
    MicrophoneRecorder,
    MicrophoneRecordingError,
    MicrophoneUnavailableError,
    NotRecordingError,
    RecordingInProgressError,
    check_microphone_available,
    get_wav_info,
    record_and_process_meeting,
)
from diarization.interface import DiarizationProvider, SpeakerSegment
from models.schemas import AudioMeetingResult
from transcription.interface import TranscriptionProvider, TranscriptTurn


class MockDiarization(DiarizationProvider):
    """Offline mock for DiarizationProvider."""

    def diarize(self, audio_path: str):
        return [
            SpeakerSegment(start=0.0, end=4.0, speaker="speaker0"),
            SpeakerSegment(start=5.0, end=10.0, speaker="speaker1"),
        ]


class MockTranscription(TranscriptionProvider):
    """Offline mock for TranscriptionProvider."""

    def transcribe(self, audio_path: str):
        return [
            TranscriptTurn(start=0.5, end=3.5, text="Welcome everyone to the sprint review."),
            TranscriptTurn(start=5.5, end=9.5, text="I will finish the backend API by Friday."),
        ]


def create_synthetic_wav(path: Path, sample_rate: int = 16000, duration_seconds: float = 1.0) -> Path:
    """Create a minimal synthetic 16-bit mono PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(sample_rate * duration_seconds)
    # Generate 440 Hz test tone
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    return path


# -----------------------------------------------------------------------------
# Configuration and Validation Tests
# -----------------------------------------------------------------------------

def test_recorder_initialization_defaults():
    rec = MicrophoneRecorder()
    assert rec.sample_rate == DEFAULT_SAMPLE_RATE
    assert rec.channels == DEFAULT_CHANNELS
    assert rec.dtype == "int16"
    assert rec.is_recording is False
    assert rec.recordings_dir == Path("data/recordings")


def test_recorder_initialization_invalid_arguments():
    with pytest.raises(ValueError, match="sample_rate must be a positive integer"):
        MicrophoneRecorder(sample_rate=0)

    with pytest.raises(ValueError, match="channels must be a positive integer"):
        MicrophoneRecorder(channels=-1)

    with pytest.raises(ValueError, match="Unsupported dtype"):
        MicrophoneRecorder(dtype="invalid_dtype")


# -----------------------------------------------------------------------------
# Device Availability Tests
# -----------------------------------------------------------------------------

@patch("sounddevice.query_devices")
def test_check_microphone_available_true(mock_query):
    mock_query.return_value = [
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Microphone", "max_input_channels": 2, "max_output_channels": 0},
    ]
    assert check_microphone_available() is True


@patch("sounddevice.query_devices")
def test_check_microphone_available_false(mock_query):
    mock_query.return_value = [
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
    ]
    assert check_microphone_available() is False


@patch("sounddevice.query_devices", side_effect=Exception("Driver fault"))
def test_check_microphone_available_on_exception(mock_query):
    assert check_microphone_available() is False


@patch("audio.microphone.check_microphone_available", return_value=False)
def test_start_raises_when_no_microphone_available(mock_avail):
    rec = MicrophoneRecorder()
    with pytest.raises(MicrophoneUnavailableError, match="No microphone or audio input device was found"):
        rec.start()


# -----------------------------------------------------------------------------
# Recording Lifecycle & WAV Output Tests
# -----------------------------------------------------------------------------

@patch("audio.microphone.check_microphone_available", return_value=True)
@patch("sounddevice.InputStream")
def test_recorder_start_and_stop_lifecycle(mock_stream_cls, mock_avail, tmp_path):
    mock_stream = MagicMock()
    mock_stream_cls.return_value = mock_stream

    rec = MicrophoneRecorder(sample_rate=16000, channels=1, recordings_dir=tmp_path)
    assert not rec.is_recording

    rec.start(meeting_id="meeting_lifecycle_test")
    assert rec.is_recording
    mock_stream.start.assert_called_once()

    # Verify attempting to start while active raises
    with pytest.raises(RecordingInProgressError, match="already active"):
        rec.start()

    # Simulate callback delivering audio samples (0.5s of audio)
    fake_chunk = np.zeros((8000, 1), dtype=np.int16)
    rec._audio_callback(fake_chunk, 8000, None, None)

    out_file = tmp_path / "lifecycle_output.wav"
    saved_path = rec.stop(output_path=out_file)

    assert not rec.is_recording
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()
    assert saved_path.is_file()

    # Inspect the written WAV file
    info = get_wav_info(saved_path)
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert info["bit_depth"] == 16
    assert info["duration_seconds"] == 0.5


def test_stop_when_not_recording_raises():
    rec = MicrophoneRecorder()
    with pytest.raises(NotRecordingError, match="No recording is currently active"):
        rec.stop()


@patch("audio.microphone.check_microphone_available", return_value=True)
@patch("sounddevice.InputStream")
def test_stop_empty_recording_raises(mock_stream_cls, mock_avail):
    mock_stream_cls.return_value = MagicMock()
    rec = MicrophoneRecorder()
    rec.start()
    # No audio callback invoked -> frames list is empty
    with pytest.raises(EmptyRecordingError, match="0 audio frames captured"):
        rec.stop()


@patch("audio.microphone.check_microphone_available", return_value=True)
@patch("sounddevice.InputStream")
def test_default_timestamped_filename_generation(mock_stream_cls, mock_avail, tmp_path):
    mock_stream_cls.return_value = MagicMock()
    rec = MicrophoneRecorder(recordings_dir=tmp_path)
    rec.start(meeting_id="sprint_42")

    fake_chunk = np.zeros((4000, 1), dtype=np.int16)
    rec._audio_callback(fake_chunk, 4000, None, None)

    saved_path = rec.stop()
    assert saved_path.parent == tmp_path
    assert saved_path.name.startswith("meeting_sprint_42_")
    assert saved_path.name.endswith(".wav")
    assert saved_path.is_file()


# -----------------------------------------------------------------------------
# Resource Cleanup & Context Manager Tests
# -----------------------------------------------------------------------------

@patch("audio.microphone.check_microphone_available", return_value=True)
@patch("sounddevice.InputStream")
def test_cleanup_releases_stream_safely(mock_stream_cls, mock_avail):
    mock_stream = MagicMock()
    mock_stream_cls.return_value = mock_stream

    rec = MicrophoneRecorder()
    rec.start()
    assert rec.is_recording

    rec.cleanup()
    assert not rec.is_recording
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()


def test_context_manager_cleanup():
    with MicrophoneRecorder() as rec:
        assert rec.is_recording is False
    assert rec.is_recording is False


# -----------------------------------------------------------------------------
# WAV Inspection Utility Tests
# -----------------------------------------------------------------------------

def test_get_wav_info_valid(tmp_path):
    wav_path = tmp_path / "valid_test.wav"
    create_synthetic_wav(wav_path, sample_rate=16000, duration_seconds=1.5)

    info = get_wav_info(wav_path)
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert info["bit_depth"] == 16
    assert info["duration_seconds"] == 1.5
    assert info["pcm_format"] == "PCM_16bit_mono"


def test_get_wav_info_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="WAV file not found"):
        get_wav_info("non_existent_audio_file.wav")


# -----------------------------------------------------------------------------
# End-to-End Orchestration Integration Tests (Mocked Providers)
# -----------------------------------------------------------------------------

def test_record_and_process_meeting_mocked(tmp_path):
    """Test full integration: record_and_process_meeting -> process_audio."""
    # Create synthetic recorded audio
    sample_wav = tmp_path / "recorded_meeting.wav"
    create_synthetic_wav(sample_wav, sample_rate=16000, duration_seconds=2.0)

    # Mock recorder that simulates recording by returning sample_wav
    mock_rec = MagicMock(spec=MicrophoneRecorder)
    mock_rec.record_until_stopped.return_value = sample_wav

    ref_date = datetime(2026, 10, 5, 10, 0, 0)
    result = record_and_process_meeting(
        meeting_id="meeting_e2e_unit",
        recorder=mock_rec,
        transcription_provider=MockTranscription(),
        diarization_provider=MockDiarization(),
        reference_datetime=ref_date,
    )

    mock_rec.record_until_stopped.assert_called_once()
    assert isinstance(result, AudioMeetingResult)
    assert result.meeting_id == "meeting_e2e_unit"
    assert len(result.speaker_transcript) == 2
    assert "speaker0" in result.participants
    assert "speaker1" in result.participants
    assert len(result.action_items) >= 1
    assert "backend api" in result.action_items[0].action.lower()


def test_record_and_process_meeting_cleans_up_on_failure(tmp_path):
    """Verify recorder.cleanup() is called if recording raises."""
    mock_rec = MagicMock(spec=MicrophoneRecorder)
    mock_rec.record_until_stopped.side_effect = RuntimeError("Microphone disconnected")

    with pytest.raises(RuntimeError, match="Microphone disconnected"):
        record_and_process_meeting(
            meeting_id="fail_meeting",
            recorder=mock_rec,
            transcription_provider=MockTranscription(),
            diarization_provider=MockDiarization(),
        )

    mock_rec.cleanup.assert_called_once()
