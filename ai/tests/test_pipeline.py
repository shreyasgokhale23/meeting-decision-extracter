"""End-to-end integration tests for NLP pipeline."""

from datetime import datetime
import pytest
from models.schemas import SentenceType, TaskStatus, TranscriptSegmentInput
from nlp.pipeline import process_segment, process_transcript


@pytest.fixture
def ref_date():
    return datetime(2026, 10, 5, 10, 0, 0)


def test_process_action_segment(ref_date):
    seg = {
        "meeting_id": "meeting_001",
        "speaker": "Rahul",
        "timestamp": "10:15:23",
        "text": "I'll complete the backend by Friday.",
    }
    res = process_segment(seg, reference_datetime=ref_date)
    assert res.meeting_id == "meeting_001"
    assert res.speaker == "Rahul"
    assert res.type == SentenceType.ACTION
    assert res.action == "Complete the backend"
    assert res.responsible_person == "Rahul"
    assert res.deadline_text == "by Friday"
    assert res.deadline_normalized == "2026-10-09"
    assert res.decision is None
    assert res.status == TaskStatus.PENDING


def test_process_decision_segment():
    seg = TranscriptSegmentInput(
        meeting_id="meeting_001",
        speaker="Amit",
        timestamp="10:17:12",
        text="Let's use MongoDB for the database.",
    )
    res = process_segment(seg)
    assert res.meeting_id == "meeting_001"
    assert res.speaker == "Amit"
    assert res.type == SentenceType.DECISION
    assert res.action is None
    assert res.responsible_person is None
    assert res.deadline_text is None
    assert res.decision == "Use MongoDB for the database"
    assert res.status == TaskStatus.UNKNOWN


def test_process_unknown_segment():
    seg = {
        "meeting_id": "meeting_001",
        "speaker": "Sneha",
        "timestamp": "10:20:00",
        "text": "Can everyone hear me?",
    }
    res = process_segment(seg)
    assert res.type == SentenceType.UNKNOWN
    assert res.action is None
    assert res.responsible_person is None
    assert res.deadline_text is None
    assert res.deadline_normalized is None
    assert res.decision is None
    assert res.status == TaskStatus.UNKNOWN


def test_process_transcript_batch(ref_date):
    segments = [
        {"speaker": "Rahul", "text": "I'll complete the backend by Friday."},
        {"speaker": "Amit", "text": "Let's use Redis for caching."},
        {"speaker": "Sneha", "text": "We discussed the project timeline."},
        {"speaker": "Priya", "text": "For reference, the current version is 2.1."},
        {"speaker": "Rahul", "text": "Can everyone hear me?"},
    ]
    output = process_transcript(segments, meeting_id="meeting_101", reference_datetime=ref_date)
    assert output.meeting_id == "meeting_101"
    assert len(output.action_items) == 1
    assert output.action_items[0].action == "Complete the backend"
    assert output.action_items[0].responsible_person == "Rahul"
    assert output.action_items[0].deadline == "2026-10-09"

    assert len(output.decisions) == 1
    assert output.decisions[0].decision == "Use Redis for caching"
    assert output.decisions[0].speaker == "Amit"

    assert len(output.transcript) == 5
    assert set(output.participants) == {"Rahul", "Amit", "Sneha", "Priya"}


def test_edge_cases():
    # Empty text
    empty_res = process_segment({"text": ""})
    assert empty_res.type == SentenceType.UNKNOWN

    # Missing speaker
    no_speaker = process_segment({"text": "Sneha will prepare the dashboard UI tomorrow."})
    assert no_speaker.responsible_person == "Sneha"
    assert no_speaker.type == SentenceType.ACTION

    # Malformed dictionary (should not crash)
    malformed = process_segment({"invalid_field": 1234})
    assert malformed.type == SentenceType.UNKNOWN
