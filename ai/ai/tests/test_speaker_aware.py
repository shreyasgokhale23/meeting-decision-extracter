"""Unit tests for Speaker-Aware Transcript Alignment (ai/transcription/speaker_aware.py).

Verifies temporal overlap calculation, multi-segment accumulation, fallback logic,
edge cases, and chronological ordering without downloading or calling real models.
"""

import pytest

from diarization.interface import SpeakerSegment
from models.schemas import SpeakerTranscriptTurn
from nlp.adapters import from_speaker_transcript_turn
from transcription.interface import TranscriptTurn
from transcription.speaker_aware import align_transcript_with_speakers, calculate_overlap


# ---------------------------------------------------------------------------
# Test 1 – Single turn matching single speaker
# ---------------------------------------------------------------------------

def test_one_whisper_turn_matching_one_speaker():
    """A single Whisper turn inside a single Diart segment is assigned that speaker."""
    turns = [TranscriptTurn(start=1.0, end=4.0, text="Hello world")]
    segments = [SpeakerSegment(start=0.0, end=5.0, speaker="speaker0")]

    result = align_transcript_with_speakers(turns, segments)

    assert len(result) == 1
    assert result[0].speaker == "speaker0"
    assert result[0].text == "Hello world"
    assert result[0].start == 1.0
    assert result[0].end == 4.0


# ---------------------------------------------------------------------------
# Test 2 – Multiple turns matching different speakers
# ---------------------------------------------------------------------------

def test_multiple_whisper_turns_matching_different_speakers():
    """Sequential turns matching distinct speaker intervals are assigned correctly."""
    turns = [
        TranscriptTurn(start=0.5, end=4.5, text="First utterance"),
        TranscriptTurn(start=5.5, end=9.5, text="Second utterance"),
    ]
    segments = [
        SpeakerSegment(start=0.0, end=5.0, speaker="speaker0"),
        SpeakerSegment(start=5.0, end=10.0, speaker="speaker1"),
    ]

    result = align_transcript_with_speakers(turns, segments)

    assert len(result) == 2
    assert result[0].speaker == "speaker0"
    assert result[0].text == "First utterance"
    assert result[1].speaker == "speaker1"
    assert result[1].text == "Second utterance"


# ---------------------------------------------------------------------------
# Test 3 – Turn spanning multiple speaker segments (Maximum overlap wins)
# ---------------------------------------------------------------------------

def test_turn_spanning_multiple_diart_segments():
    """When a turn spans multiple speakers, the speaker with max overlap is selected."""
    # Turn: 0.0 to 10.0 (duration 10s)
    # Speaker 0: 0.0 to 3.0 -> 3.0s overlap
    # Speaker 1: 3.0 to 10.0 -> 7.0s overlap (winner)
    turns = [TranscriptTurn(start=0.0, end=10.0, text="Spanning turn")]
    segments = [
        SpeakerSegment(start=0.0, end=3.0, speaker="speaker0"),
        SpeakerSegment(start=3.0, end=10.0, speaker="speaker1"),
    ]

    result = align_transcript_with_speakers(turns, segments)

    assert len(result) == 1
    assert result[0].speaker == "speaker1"


# ---------------------------------------------------------------------------
# Test 4 – Multiple segments from the same speaker accumulated
# ---------------------------------------------------------------------------

def test_multiple_diart_segments_from_same_speaker():
    """Overlaps from multiple segments belonging to the same speaker are summed."""
    # Turn: 0.0 to 10.0 (duration 10s)
    # Speaker 0: [0, 2] (2s) and [5, 10] (5s) -> total 7s
    # Speaker 1: [2, 5] (3s) -> total 3s
    turns = [TranscriptTurn(start=0.0, end=10.0, text="Interleaved segments")]
    segments = [
        SpeakerSegment(start=0.0, end=2.0, speaker="speaker0"),
        SpeakerSegment(start=2.0, end=5.0, speaker="speaker1"),
        SpeakerSegment(start=5.0, end=10.0, speaker="speaker0"),
    ]

    result = align_transcript_with_speakers(turns, segments)

    assert len(result) == 1
    assert result[0].speaker == "speaker0"


# ---------------------------------------------------------------------------
# Test 5 – Diart segments arriving out of order
# ---------------------------------------------------------------------------

def test_diart_segments_out_of_order():
    """Out-of-order Diart segments are handled and output remains chronologically sorted."""
    turns = [
        TranscriptTurn(start=10.0, end=15.0, text="Later turn"),
        TranscriptTurn(start=1.0, end=5.0, text="Earlier turn"),
    ]
    # Diart segments deliberately shuffled
    segments = [
        SpeakerSegment(start=10.0, end=20.0, speaker="speaker1"),
        SpeakerSegment(start=0.0, end=10.0, speaker="speaker0"),
    ]

    result = align_transcript_with_speakers(turns, segments)

    assert len(result) == 2
    # Result must be sorted chronologically by start
    assert result[0].start == 1.0
    assert result[0].speaker == "speaker0"
    assert result[0].text == "Earlier turn"

    assert result[1].start == 10.0
    assert result[1].speaker == "speaker1"
    assert result[1].text == "Later turn"


# ---------------------------------------------------------------------------
# Test 6 – Exact timestamp boundary
# ---------------------------------------------------------------------------

def test_exact_timestamp_boundary():
    """Boundary at exactly 5.0s: Turn [5.0, 10.0] matches segment [5.0, 10.0], not [0.0, 5.0]."""
    turns = [TranscriptTurn(start=5.0, end=10.0, text="Boundary turn")]
    segments = [
        SpeakerSegment(start=0.0, end=5.0, speaker="speaker0"),
        SpeakerSegment(start=5.0, end=10.0, speaker="speaker1"),
    ]

    result = align_transcript_with_speakers(turns, segments)

    assert len(result) == 1
    assert result[0].speaker == "speaker1"


# ---------------------------------------------------------------------------
# Test 7 – No speaker overlap (fallback to UNKNOWN)
# ---------------------------------------------------------------------------

def test_no_speaker_overlap():
    """When a turn occurs during silence with no speaker segment, returns 'UNKNOWN'."""
    turns = [TranscriptTurn(start=20.0, end=25.0, text="Late silent turn")]
    segments = [
        SpeakerSegment(start=0.0, end=10.0, speaker="speaker0"),
    ]

    result = align_transcript_with_speakers(turns, segments)

    assert len(result) == 1
    assert result[0].speaker == "UNKNOWN"
    assert result[0].text == "Late silent turn"


# ---------------------------------------------------------------------------
# Test 8 – Empty Diart output
# ---------------------------------------------------------------------------

def test_empty_diart_output():
    """When Diart produces no segments, all Whisper turns fallback to UNKNOWN without crashing."""
    turns = [
        TranscriptTurn(start=0.0, end=3.0, text="First utterance"),
        TranscriptTurn(start=4.0, end=7.0, text="Second utterance"),
    ]

    result = align_transcript_with_speakers(turns, [])

    assert len(result) == 2
    assert result[0].speaker == "UNKNOWN"
    assert result[1].speaker == "UNKNOWN"


# ---------------------------------------------------------------------------
# Test 9 – Empty Whisper output
# ---------------------------------------------------------------------------

def test_empty_whisper_output():
    """When Whisper produces no turns, alignment cleanly returns an empty list."""
    segments = [SpeakerSegment(start=0.0, end=10.0, speaker="speaker0")]

    result = align_transcript_with_speakers([], segments)

    assert result == []


# ---------------------------------------------------------------------------
# Test 10 – Three distinct speakers alignment
# ---------------------------------------------------------------------------

def test_three_distinct_speakers_alignment():
    """Aligns turns across 3 distinct speakers matching dummy_meeting setup."""
    turns = [
        TranscriptTurn(start=0.5, end=8.0, text="Introduction by host"),
        TranscriptTurn(start=10.5, end=20.0, text="Frontend update"),
        TranscriptTurn(start=42.5, end=48.0, text="Backend architecture"),
    ]
    segments = [
        SpeakerSegment(start=0.0, end=8.12, speaker="speaker0"),
        SpeakerSegment(start=10.0, end=20.5, speaker="speaker1"),
        SpeakerSegment(start=42.0, end=48.5, speaker="speaker2"),
    ]

    result = align_transcript_with_speakers(turns, segments)

    assert len(result) == 3
    assert [r.speaker for r in result] == ["speaker0", "speaker1", "speaker2"]
    assert result[0].text == "Introduction by host"
    assert result[1].text == "Frontend update"
    assert result[2].text == "Backend architecture"


# ---------------------------------------------------------------------------
# Test 11 – Malformed timestamps handled gracefully
# ---------------------------------------------------------------------------

def test_malformed_timestamps_handled():
    """Turns with end <= start or invalid values do not crash the aligner."""
    turns = [
        TranscriptTurn(start=5.0, end=2.0, text="Inverted timestamps"),
        TranscriptTurn(start=0.0, end=4.0, text="Valid turn"),
    ]
    segments = [
        SpeakerSegment(start=0.0, end=5.0, speaker="speaker0"),
    ]

    result = align_transcript_with_speakers(turns, segments)

    assert len(result) == 2
    # Inverted turn gets 0 overlap -> fallback
    assert result[0].text == "Valid turn"
    assert result[0].speaker == "speaker0"
    assert result[1].text == "Inverted timestamps"
    assert result[1].speaker == "UNKNOWN"


# ---------------------------------------------------------------------------
# Test 12 – Adapter converts SpeakerTranscriptTurn to TranscriptSegmentInput
# ---------------------------------------------------------------------------

def test_from_speaker_transcript_turn_adapter():
    """Verifies adapter from_speaker_transcript_turn produces valid TranscriptSegmentInput."""
    turn = SpeakerTranscriptTurn(
        speaker="speaker1",
        start=65.5,
        end=72.0,
        text="I will finish the frontend by Friday.",
    )

    inp = from_speaker_transcript_turn(turn, meeting_id="meet_101")

    assert inp.meeting_id == "meet_101"
    assert inp.speaker == "speaker1"
    assert inp.speaker_id == "speaker1"
    assert inp.timestamp == "00:01:05"  # 65 seconds
    assert inp.text == "I will finish the frontend by Friday."
