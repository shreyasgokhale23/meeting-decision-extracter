"""Unit tests for DiartProvider.

These tests mock only the external diart inference layer so that no model
downloads occur during CI. The mocks replicate the actual diart 0.9.x
call-chain:

    FileAudioSource(path, sample_rate)
    StreamingInference(pipeline, source, ...)
    inference()  → pyannote.core.Annotation

Real pyannote.core.Annotation objects are used so that the
Annotation → SpeakerSegment conversion code is exercised.
"""

import builtins
import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pyannote.core import Annotation, Segment  # real pyannote objects

from diarization.diart_provider import DiartProvider
from diarization.interface import SpeakerSegment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_annotation(*turns) -> Annotation:
    """Build a real pyannote.core.Annotation from (start, end, speaker) triples."""
    ann = Annotation()
    for start, end, speaker in turns:
        ann[Segment(start, end)] = speaker
    return ann


def _mock_diart(monkeypatch, annotation: Annotation):
    """Patch diart so that DiartProvider can be instantiated and run without
    downloading any models, and StreamingInference().__call__() returns
    *annotation*.
    """
    mock_pipeline = MagicMock()
    # pipeline.config.sample_rate
    mock_pipeline.config.sample_rate = 16000

    mock_source = MagicMock()

    mock_inference_instance = MagicMock()
    mock_inference_instance.return_value = annotation  # inference() → annotation

    mock_inference_cls = MagicMock(return_value=mock_inference_instance)
    mock_source_cls = MagicMock(return_value=mock_source)
    mock_pipeline_cls = MagicMock(return_value=mock_pipeline)

    monkeypatch.setattr("diart.SpeakerDiarization", mock_pipeline_cls)
    monkeypatch.setattr("diart.sources.FileAudioSource", mock_source_cls)
    monkeypatch.setattr("diart.inference.StreamingInference", mock_inference_cls)

    return mock_pipeline_cls, mock_source_cls, mock_inference_cls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_audio(tmp_path) -> str:
    """Create a minimal dummy WAV file for path-existence checks."""
    audio_file = tmp_path / "dummy.wav"
    # Minimal valid RIFF/WAV header (16-byte stub – enough for os.path.isfile)
    audio_file.write_bytes(
        b"RIFF\x24\x00\x00\x00WAVEfmt "
        b"\x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x80\x3e\x00\x00\x00\x7d\x00\x00"
        b"\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    return str(audio_file)


# ---------------------------------------------------------------------------
# Test 1 – Provider initialisation
# ---------------------------------------------------------------------------

def test_provider_initialises(monkeypatch):
    """DiartProvider can be constructed when diart is importable."""
    _mock_diart(monkeypatch, _make_annotation())
    provider = DiartProvider()
    assert provider is not None
    assert hasattr(provider, "_pipeline")
    assert hasattr(provider, "_FileAudioSource")
    assert hasattr(provider, "_StreamingInference")


# ---------------------------------------------------------------------------
# Test 2 – Missing audio file
# ---------------------------------------------------------------------------

def test_missing_audio_file_raises(monkeypatch):
    """diarize() raises FileNotFoundError for a non-existent path."""
    _mock_diart(monkeypatch, _make_annotation())
    provider = DiartProvider()
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        provider.diarize("definitely_does_not_exist.wav")


# ---------------------------------------------------------------------------
# Test 3 – Successful Annotation → SpeakerSegment conversion
# ---------------------------------------------------------------------------

def test_successful_conversion(dummy_audio, monkeypatch):
    """Segments from a multi-speaker annotation are converted correctly."""
    annotation = _make_annotation(
        (0.0, 4.8, "SPEAKER_00"),
        (4.8, 9.3, "SPEAKER_01"),
    )
    _mock_diart(monkeypatch, annotation)
    provider = DiartProvider()
    segments = provider.diarize(dummy_audio)

    assert len(segments) == 2
    assert all(isinstance(s, SpeakerSegment) for s in segments)

    spk0 = segments[0]
    assert spk0.speaker == "SPEAKER_00"
    assert spk0.start == pytest.approx(0.0)
    assert spk0.end == pytest.approx(4.8)

    spk1 = segments[1]
    assert spk1.speaker == "SPEAKER_01"
    assert spk1.start == pytest.approx(4.8)
    assert spk1.end == pytest.approx(9.3)


# ---------------------------------------------------------------------------
# Test 4 – Chronological sorting
# ---------------------------------------------------------------------------

def test_chronological_sorting(dummy_audio, monkeypatch):
    """Segments are returned sorted by start time regardless of annotation order."""
    # pyannote.core.Annotation stores segments sorted internally, but we
    # verify the provider's sort contract explicitly.
    annotation = _make_annotation(
        (4.8, 9.3, "SPEAKER_01"),  # second speaker first in source data
        (0.0, 4.8, "SPEAKER_00"),
        (9.3, 13.0, "SPEAKER_00"),
    )
    _mock_diart(monkeypatch, annotation)
    provider = DiartProvider()
    segments = provider.diarize(dummy_audio)

    starts = [s.start for s in segments]
    assert starts == sorted(starts), f"Segments not sorted: {starts}"


# ---------------------------------------------------------------------------
# Test 5 – Empty annotation
# ---------------------------------------------------------------------------

def test_empty_annotation(dummy_audio, monkeypatch):
    """An empty Annotation produces an empty list (no crash)."""
    _mock_diart(monkeypatch, _make_annotation())
    provider = DiartProvider()
    result = provider.diarize(dummy_audio)
    assert result == []


# ---------------------------------------------------------------------------
# Test 6 – Initialisation / import failure
# ---------------------------------------------------------------------------

def test_import_error_raises_import_error(monkeypatch):
    """DiartProvider raises ImportError if diart cannot be imported."""
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "diart":
            raise ImportError("no module named diart")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="diart is not installed"):
        DiartProvider()


def test_pipeline_init_failure_raises_runtime_error(monkeypatch):
    """DiartProvider raises RuntimeError if SpeakerDiarization() crashes."""
    mock_source_cls = MagicMock()
    mock_inference_cls = MagicMock()
    mock_pipeline_cls = MagicMock(side_effect=RuntimeError("model not found"))

    monkeypatch.setattr("diart.SpeakerDiarization", mock_pipeline_cls)
    monkeypatch.setattr("diart.sources.FileAudioSource", mock_source_cls)
    monkeypatch.setattr("diart.inference.StreamingInference", mock_inference_cls)

    with pytest.raises(RuntimeError, match="Failed to initialise"):
        DiartProvider()


# ---------------------------------------------------------------------------
# Test 7 – Inference / runtime failure
# ---------------------------------------------------------------------------

def test_inference_failure_raises_runtime_error(dummy_audio, monkeypatch):
    """diarize() wraps inference exceptions in RuntimeError."""
    mock_pipeline = MagicMock()
    mock_pipeline.config.sample_rate = 16000

    mock_inference_instance = MagicMock(side_effect=RuntimeError("CUDA OOM"))
    mock_inference_cls = MagicMock(return_value=mock_inference_instance)
    mock_source_cls = MagicMock()
    mock_pipeline_cls = MagicMock(return_value=mock_pipeline)

    monkeypatch.setattr("diart.SpeakerDiarization", mock_pipeline_cls)
    monkeypatch.setattr("diart.sources.FileAudioSource", mock_source_cls)
    monkeypatch.setattr("diart.inference.StreamingInference", mock_inference_cls)

    provider = DiartProvider()
    with pytest.raises(RuntimeError, match="Diart diarization failed"):
        provider.diarize(dummy_audio)


# ---------------------------------------------------------------------------
# Test 8 – Correct speaker labels
# ---------------------------------------------------------------------------

def test_correct_speaker_labels(dummy_audio, monkeypatch):
    """Speaker labels from the annotation are preserved verbatim."""
    annotation = _make_annotation(
        (0.0, 3.0, "SPEAKER_00"),
        (3.0, 6.0, "SPEAKER_01"),
        (6.0, 9.0, "SPEAKER_02"),
    )
    _mock_diart(monkeypatch, annotation)
    provider = DiartProvider()
    segments = provider.diarize(dummy_audio)

    labels = {s.speaker for s in segments}
    assert labels == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}


# ---------------------------------------------------------------------------
# Test 9 – Correct timestamps
# ---------------------------------------------------------------------------

def test_correct_timestamps(dummy_audio, monkeypatch):
    """Segment start/end times match the pyannote Annotation exactly."""
    turns = [
        (0.0, 2.5, "SPEAKER_00"),
        (2.5, 5.0, "SPEAKER_01"),
        (5.0, 8.75, "SPEAKER_00"),
    ]
    annotation = _make_annotation(*turns)
    _mock_diart(monkeypatch, annotation)
    provider = DiartProvider()
    segments = provider.diarize(dummy_audio)

    assert len(segments) == len(turns)
    for seg, (start, end, speaker) in zip(segments, sorted(turns)):
        assert seg.speaker == speaker
        assert seg.start == pytest.approx(start)
        assert seg.end == pytest.approx(end)


# ---------------------------------------------------------------------------
# Test 10 – Three-speaker regression (delta_new=0.8 config)
# ---------------------------------------------------------------------------

def test_three_speaker_regression(dummy_audio, monkeypatch):
    """Regression: DiartProvider must correctly propagate 3 unique speaker
    labels from the underlying annotation.  This guards against the original
    delta_new=1.0 default that merged the third speaker into an existing
    cluster for acoustically-similar voices.
    """
    annotation = _make_annotation(
        (0.0, 8.0, "speaker0"),
        (10.0, 20.0, "speaker1"),
        (20.0, 23.0, "speaker1"),
        (23.0, 25.0, "speaker0"),
        (42.0, 48.5, "speaker2"),
        (53.5, 57.5, "speaker2"),
        (73.0, 78.0, "speaker2"),
        (98.0, 103.0, "speaker2"),
        (113.6, 116.5, "speaker2"),
    )
    _mock_diart(monkeypatch, annotation)
    provider = DiartProvider()
    segments = provider.diarize(dummy_audio)

    unique_speakers = {s.speaker for s in segments}
    assert len(unique_speakers) == 3, (
        f"Expected 3 unique speakers, got {len(unique_speakers)}: {unique_speakers}"
    )
    assert unique_speakers == {"speaker0", "speaker1", "speaker2"}


def test_config_uses_lower_delta_new(monkeypatch):
    """Verify that DiartProvider passes a config with delta_new < 1.0 to
    SpeakerDiarization so that similar voices are not merged.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.config.sample_rate = 16000

    captured_config = {}

    def capture_pipeline(config=None):
        captured_config["config"] = config
        return mock_pipeline

    monkeypatch.setattr("diart.SpeakerDiarization", capture_pipeline)
    monkeypatch.setattr("diart.sources.FileAudioSource", MagicMock())
    monkeypatch.setattr("diart.inference.StreamingInference", MagicMock())

    DiartProvider()

    assert "config" in captured_config, "SpeakerDiarization was not called with a config"
    config = captured_config["config"]
    assert config is not None, "Config should not be None"
    assert hasattr(config, "delta_new"), "Config missing delta_new attribute"
    assert config.delta_new < 1.0, (
        f"delta_new should be < 1.0 for proper speaker separation, got {config.delta_new}"
    )
    assert config.delta_new == pytest.approx(0.8), (
        f"Expected delta_new=0.8, got {config.delta_new}"
    )

