"""Real AI Integration Test.

Runs the complete end-to-end integration:
ai/data/dummy_meeting.wav
    ↓
Backend service / endpoint
    ↓
ai_service.process_meeting_audio()
    ↓
existing process_audio() (Diart or Reference Diarization + Real Whisper STT + Alignment + NLP)
    ↓
AudioMeetingResult
    ↓
MongoDB persistence (transcripts, action_items, decisions, meetings)
    ↓
REST verification

Executes actual OpenAI Whisper 'base' model transcription on the actual dummy_meeting.wav audio file.
If HF_TOKEN is configured, also executes live Diart model weights from Hugging Face;
otherwise uses reference diarization segments to complete speaker alignment without network failure.
"""

import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import List

import pytest
from mongomock_motor import AsyncMongoMockClient

# Setup module paths
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
repo_root = backend_dir.parent
ai_dir = repo_root / "ai"

for p in (str(backend_dir), str(ai_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.meeting_service import process_meeting_background
from diarization.interface import DiarizationProvider, SpeakerSegment
from transcription.whisper_provider import WhisperProvider


class ReferenceDummyDiarizationProvider(DiarizationProvider):
    """Reference 3-speaker diarization provider for dummy_meeting.wav when HF_TOKEN is not provided."""

    def diarize(self, audio_path: str) -> List[SpeakerSegment]:
        return [
            SpeakerSegment(speaker="speaker0", start=0.0, end=8.4),
            SpeakerSegment(speaker="speaker1", start=8.4, end=20.5),
            SpeakerSegment(speaker="speaker0", start=20.5, end=24.6),
            SpeakerSegment(speaker="speaker1", start=24.6, end=33.6),
            SpeakerSegment(speaker="speaker0", start=33.6, end=37.5),
            SpeakerSegment(speaker="speaker1", start=37.5, end=40.8),
            SpeakerSegment(speaker="speaker2", start=40.8, end=48.3),
            SpeakerSegment(speaker="speaker0", start=48.3, end=53.4),
            SpeakerSegment(speaker="speaker2", start=53.4, end=57.2),
            SpeakerSegment(speaker="speaker0", start=57.2, end=59.2),
            SpeakerSegment(speaker="speaker1", start=59.2, end=68.2),
            SpeakerSegment(speaker="speaker0", start=68.2, end=72.8),
            SpeakerSegment(speaker="speaker2", start=72.8, end=78.3),
            SpeakerSegment(speaker="speaker1", start=78.3, end=84.8),
            SpeakerSegment(speaker="speaker0", start=84.8, end=89.7),
            SpeakerSegment(speaker="speaker1", start=89.7, end=100.8),
            SpeakerSegment(speaker="speaker0", start=100.8, end=110.9),
            SpeakerSegment(speaker="speaker1", start=110.9, end=116.5),
            SpeakerSegment(speaker="speaker2", start=116.5, end=121.4),
            SpeakerSegment(speaker="speaker1", start=121.4, end=126.1),
            SpeakerSegment(speaker="speaker0", start=126.1, end=132.2),
        ]


@pytest.mark.real_ai
@pytest.mark.asyncio
async def test_real_audio_pipeline_backend_integration():
    """Execute real dummy_meeting.wav through backend storage flow."""
    audio_wav_path = repo_root / "ai" / "data" / "dummy_meeting.wav"
    assert audio_wav_path.is_file(), f"dummy_meeting.wav not found at {audio_wav_path}"

    meeting_id = "meeting_real_dummy_001"
    ref_date = datetime(2026, 10, 5, 10, 0, 0)

    # Use mock MongoDB client for standalone testing
    mock_mongo = AsyncMongoMockClient()
    db = mock_mongo["test_real_meetings_db"]

    # 1. Create meeting in db
    now = datetime.now(UTC)
    await db.meetings.insert_one(
        {
            "meeting_id": meeting_id,
            "title": "Real Dummy Meeting Verification",
            "status": "created",
            "started_at": now,
            "created_at": now,
            "ended_at": None,
            "audio_path": None,
            "participants": [],
            "error": None,
        }
    )

    print("\n" + "=" * 70)
    print("STARTING REAL BACKEND INTEGRATION TEST WITH DUMMY WAV")
    print(f"Audio Path : {audio_wav_path}")
    print(f"Meeting ID : {meeting_id}")
    print("=" * 70)

    # Check if HF_TOKEN is present to decide diarization provider
    hf_token = os.getenv("HF_TOKEN")
    diar_provider = None
    if not hf_token:
        print("[INFO] HF_TOKEN not set in environment; using reference 3-speaker diarizer for Diart.")
        print("[INFO] Executing REAL OpenAI Whisper 'base' model transcription on dummy_meeting.wav...")
        diar_provider = ReferenceDummyDiarizationProvider()
    else:
        print("[INFO] HF_TOKEN detected; attempting live DiartProvider initialization...")

    whisper_provider = WhisperProvider("base")

    # Define custom processor wrapping existing process_audio
    from transcription.audio_processor import process_audio

    def real_ai_processor(audio_path, meeting_id, reference_datetime=None):
        return process_audio(
            audio_path=audio_path,
            meeting_id=meeting_id,
            reference_datetime=reference_datetime,
            transcription_provider=whisper_provider,
            diarization_provider=diar_provider,
        )

    # 2. Run background processing pipeline with real AI
    await process_meeting_background(
        db=db,
        meeting_id=meeting_id,
        audio_path=audio_wav_path,
        reference_datetime=ref_date,
        ai_processor=real_ai_processor,
    )

    # 3. Verify Meeting status
    meeting = await db.meetings.find_one({"meeting_id": meeting_id})
    assert meeting is not None, "Meeting record not found"
    assert meeting["status"] == "completed", f"Meeting failed: {meeting.get('error')}"
    assert meeting["ended_at"] is not None
    assert meeting["audio_path"] == str(audio_wav_path)
    assert len(meeting["participants"]) >= 2, f"Insufficient participants: {meeting['participants']}"

    # 4. Verify Transcripts
    transcripts = [t async for t in db.transcripts.find({"meeting_id": meeting_id}).sort("start_time", 1)]
    assert len(transcripts) > 0, "No transcripts persisted"
    unique_speakers = {t["speaker"] for t in transcripts}

    print("\n--- STORED TRANSCRIPTS ---")
    for t in transcripts[:6]:
        print(f"[{t['start_time']:5.1f}s - {t['end_time']:5.1f}s] {t['speaker']:<10} ({t['type']:<10}): {t['text']}")
    if len(transcripts) > 6:
        print(f"... and {len(transcripts) - 6} more turns (total: {len(transcripts)}).")

    # 5. Verify Action Items
    actions = [a async for a in db.action_items.find({"meeting_id": meeting_id})]
    assert len(actions) > 0, "No action items persisted"
    print("\n--- STORED ACTION ITEMS ---")
    for a in actions:
        print(f"* Action: {a['action']}")
        print(f"  Responsible: {a.get('responsible_person') or '-'} | Speaker: {a.get('speaker') or '-'}")
        print(f"  Deadline: {a.get('deadline') or '-'} | Status: {a.get('status')}")

    # 6. Verify Decisions
    decisions = [d async for d in db.decisions.find({"meeting_id": meeting_id})]
    assert len(decisions) > 0, "No decisions persisted"
    print("\n--- STORED DECISIONS ---")
    for d in decisions:
        print(f"* Decision: {d['decision']} (Speaker: {d.get('speaker') or '-'})")

    print("\n" + "=" * 70)
    print("REAL INTEGRATION METRICS")
    print("=" * 70)
    print(f"Status           : {meeting['status']}")
    print(f"Transcript Turns : {len(transcripts)}")
    print(f"Unique Speakers  : {sorted(list(unique_speakers))}")
    print(f"Participants     : {meeting['participants']}")
    print(f"Action Items     : {len(actions)}")
    print(f"Decisions        : {len(decisions)}")
    print("REAL BACKEND INTEGRATION TEST PASSED [OK]")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_real_audio_pipeline_backend_integration())
