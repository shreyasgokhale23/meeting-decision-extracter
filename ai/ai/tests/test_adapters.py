"""Tests for universal input adapters."""

import pytest
from models.schemas import TranscriptSegmentInput
from nlp.adapters import (
    from_csv_row,
    from_diart_whisper,
    from_dict,
    from_text_line,
    from_whisper_turn,
)


def test_from_dict():
    payload = {
        "meeting_id": "m_100",
        "speaker": "Rahul",
        "timestamp": "10:00:00",
        "text": "I'll complete the backend by Friday.",
    }
    inp = from_dict(payload)
    assert isinstance(inp, TranscriptSegmentInput)
    assert inp.meeting_id == "m_100"
    assert inp.speaker == "Rahul"
    assert inp.text == "I'll complete the backend by Friday."


def test_from_csv_row():
    row = {
        "meeting_id": "meeting_001",
        "speaker_id": "SPEAKER_01",
        "speaker": "Sneha",
        "timestamp": "10:15:00",
        "text": "Sneha will prepare the dashboard UI tomorrow.",
    }
    inp = from_csv_row(row)
    assert isinstance(inp, TranscriptSegmentInput)
    assert inp.speaker_id == "SPEAKER_01"
    assert inp.speaker == "Sneha"


def test_from_text_line():
    # Line with timestamp and named speaker
    line1 = "[10:15:23] Rahul: I'll complete the backend by Friday."
    inp1 = from_text_line(line1, default_meeting_id="meeting_001")
    assert inp1.timestamp == "10:15:23"
    assert inp1.speaker == "Rahul"
    assert inp1.text == "I'll complete the backend by Friday."
    assert inp1.meeting_id == "meeting_001"

    # Line with diarized speaker tag
    line2 = "SPEAKER_00: Let's use Redis for caching."
    inp2 = from_text_line(line2)
    assert inp2.speaker_id == "SPEAKER_00"
    assert inp2.speaker == "SPEAKER_00"
    assert inp2.text == "Let's use Redis for caching."

    # Line with delegation prefix
    line3 = "Manager: Rahul should finish the API."
    inp3 = from_text_line(line3)
    assert inp3.speaker == "Manager"
    assert inp3.text == "Rahul should finish the API."


def test_from_whisper_turn():
    whisper_turn = {
        "text": " I'll complete the backend by Friday.",
        "start": 615.5,
        "end": 620.0,
    }
    inp = from_whisper_turn(whisper_turn, speaker="Rahul", meeting_id="m_200")
    assert inp.meeting_id == "m_200"
    assert inp.speaker == "Rahul"
    assert inp.timestamp == "00:10:15"
    assert inp.text == "I'll complete the backend by Friday."


def test_from_diart_whisper():
    diar_segment = {"speaker": "SPEAKER_02", "start": 125.0, "end": 130.0}
    whisper_text = "We decided to use MongoDB."
    inp = from_diart_whisper(diar_segment, whisper_text, meeting_id="m_300")
    assert inp.speaker_id == "SPEAKER_02"
    assert inp.timestamp == "00:02:05"
    assert inp.text == "We decided to use MongoDB."
