"""Unit and integration tests for the Meeting Decision Extractor Backend API.

Mocks the AI service to ensure zero model downloads or GPU execution during backend testing.
Tests complete meeting lifecycle:
- Meeting creation (status: created)
- WAV upload & format validation
- Background AI processing trigger
- Speaker-aware transcript storage
- Action items & decisions storage with separate speaker & responsible_person
- Meeting completion (status: completed)
- AI failure handling (status: failed)
- WebSocket status broadcasts
- Duplicate processing idempotency
- Full REST endpoints
"""

import io
import os
import sys
import wave
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from mongomock_motor import AsyncMongoMockClient

# Ensure backend and ai modules are on path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
repo_root = backend_dir.parent
ai_dir = repo_root / "ai"

for p in (str(backend_dir), str(ai_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.main import app
from app.realtime import sockets
from models.schemas import (
    ActionItem,
    AudioMeetingResult,
    DecisionItem,
    MeetingOutput,
    SentenceType,
    SpeakerTranscriptTurn,
    TaskStatus,
    TranscriptSegmentOutput,
)


def create_dummy_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a minimal valid 16-bit mono PCM WAV byte stream for testing."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        num_frames = int(duration_sec * sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


def build_mock_ai_result(meeting_id: str) -> AudioMeetingResult:
    """Build a realistic AudioMeetingResult with separate speaker and responsible_person."""
    turns = [
        SpeakerTranscriptTurn(
            speaker="Manager",
            start=1.2,
            end=5.4,
            text="Rahul should complete the backend by Friday.",
        ),
        SpeakerTranscriptTurn(
            speaker="Architect",
            start=6.0,
            end=9.5,
            text="We decided to use MongoDB for the database.",
        ),
        SpeakerTranscriptTurn(
            speaker="Rahul",
            start=10.0,
            end=12.5,
            text="Understood, I am on it.",
        ),
    ]

    segments = [
        TranscriptSegmentOutput(
            meeting_id=meeting_id,
            speaker_id="SPEAKER_00",
            speaker="Manager",
            timestamp="00:00:01",
            text="Rahul should complete the backend by Friday.",
            type=SentenceType.ACTION,
            action="Complete the backend",
            responsible_person="Rahul",
            deadline_text="by Friday",
            deadline_normalized="2026-10-16",
            status=TaskStatus.PENDING,
        ),
        TranscriptSegmentOutput(
            meeting_id=meeting_id,
            speaker_id="SPEAKER_01",
            speaker="Architect",
            timestamp="00:00:06",
            text="We decided to use MongoDB for the database.",
            type=SentenceType.DECISION,
            decision="Use MongoDB for the database",
            status=TaskStatus.UNKNOWN,
        ),
        TranscriptSegmentOutput(
            meeting_id=meeting_id,
            speaker_id="SPEAKER_02",
            speaker="Rahul",
            timestamp="00:00:10",
            text="Understood, I am on it.",
            type=SentenceType.DISCUSSION,
            status=TaskStatus.UNKNOWN,
        ),
    ]

    summary = MeetingOutput(
        meeting_id=meeting_id,
        action_items=[
            ActionItem(
                action="Complete the backend",
                responsible_person="Rahul",
                deadline="2026-10-16",
                status=TaskStatus.PENDING,
            )
        ],
        decisions=[
            DecisionItem(
                decision="Use MongoDB for the database",
                speaker="Architect",
            )
        ],
        transcript=segments,
        participants=["Architect", "Manager", "Rahul"],
    )

    return AudioMeetingResult(
        meeting_id=meeting_id,
        speaker_transcript=turns,
        meeting_summary=summary,
        action_items=summary.action_items,
        decisions=summary.decisions,
        participants=summary.participants,
    )


@pytest_asyncio.fixture
async def test_client():
    """Configure FastAPI app with an isolated mock MongoDB and yield AsyncClient."""
    mock_mongo = AsyncMongoMockClient()
    db = mock_mongo["test_meetings_db"]
    await db.meetings.create_index([("meeting_id", 1)], unique=True)
    await db.transcripts.create_index([("meeting_id", 1), ("segment_key", 1)], unique=True)
    await db.action_items.create_index([("meeting_id", 1), ("segment_key", 1)], unique=True)
    await db.decisions.create_index([("meeting_id", 1), ("segment_key", 1)], unique=True)

    app.state.mongo = mock_mongo
    app.state.db = db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ============================================================================
# 1. Meeting Lifecycle Tests (Create -> Upload -> Background Processing)
# ============================================================================

@pytest.mark.asyncio
async def test_create_meeting(test_client: AsyncClient):
    """POST /api/meetings creates record with status 'created'."""
    res = await test_client.post("/api/meetings", json={"meeting_id": "mtg_001", "title": "Sprint Planning"})
    assert res.status_code == 201
    data = res.json()
    assert data["meeting_id"] == "mtg_001"
    assert data["title"] == "Sprint Planning"
    assert data["status"] == "created"
    assert "created_at" in data

    # Duplicate creation returns 409
    res_dup = await test_client.post("/api/meetings", json={"meeting_id": "mtg_001", "title": "Dup"})
    assert res_dup.status_code == 409


@pytest.mark.asyncio
async def test_audio_upload_validations(test_client: AsyncClient):
    """POST /api/meetings/{id}/audio validates meeting existence, extension, and headers."""
    # 1. Non-existent meeting
    wav_bytes = create_dummy_wav_bytes()
    res = await test_client.post(
        "/api/meetings/non_existent/audio",
        files={"file": ("meeting.wav", wav_bytes, "audio/wav")},
    )
    assert res.status_code == 404
    assert "meeting not found" in res.json()["detail"].lower()

    # Create meeting first
    await test_client.post("/api/meetings", json={"meeting_id": "mtg_val", "title": "Validation Test"})

    # 2. Non-WAV file extension
    res_ext = await test_client.post(
        "/api/meetings/mtg_val/audio",
        files={"file": ("recording.mp3", b"ID3\x03fake-audio", "audio/mpeg")},
    )
    assert res_ext.status_code == 400
    assert "only wav" in res_ext.json()["detail"].lower()

    # 3. Invalid WAV header content
    res_hdr = await test_client.post(
        "/api/meetings/mtg_val/audio",
        files={"file": ("recording.wav", b"NOT_A_VALID_WAV_HEADER", "audio/wav")},
    )
    assert res_hdr.status_code == 400
    assert "not a valid riff/wave" in res_hdr.json()["detail"].lower()


@pytest.mark.asyncio
async def test_audio_upload_and_successful_background_processing(test_client: AsyncClient):
    """Upload valid WAV -> processing status -> AI mock runs -> DB populated -> completed."""
    meeting_id = "mtg_e2e_01"
    await test_client.post("/api/meetings", json={"meeting_id": meeting_id, "title": "Full Cycle Test"})

    mock_result = build_mock_ai_result(meeting_id)
    ws_events = []

    async def record_event(mid: str, event: dict):
        if mid == meeting_id:
            ws_events.append(event)

    wav_bytes = create_dummy_wav_bytes(duration_sec=2.0)

    with patch("app.services.ai_service.process_meeting_audio", return_value=mock_result) as mock_ai, \
         patch.object(sockets, "publish", side_effect=record_event):

        # Upload audio
        res = await test_client.post(
            f"/api/meetings/{meeting_id}/audio",
            files={"file": ("meeting_audio.wav", wav_bytes, "audio/wav")},
        )
        assert res.status_code == 202
        upload_resp = res.json()
        assert upload_resp["meeting_id"] == meeting_id
        assert upload_resp["status"] == "processing"

        # Verify AI service was invoked with meeting_id and audio path
        mock_ai.assert_called_once()
        called_args, called_kwargs = mock_ai.call_args
        assert called_kwargs["meeting_id"] == meeting_id
        assert "audio_path" in called_kwargs or len(called_args) > 0

    # 1. Verify meeting is completed
    res_mtg = await test_client.get(f"/api/meetings/{meeting_id}")
    assert res_mtg.status_code == 200
    mtg_data = res_mtg.json()
    assert mtg_data["status"] == "completed"
    assert mtg_data["participants"] == ["Architect", "Manager", "Rahul"]
    assert mtg_data["audio_path"] is not None
    assert mtg_data["ended_at"] is not None

    # 2. Verify transcripts collection
    res_trans = await test_client.get(f"/api/meetings/{meeting_id}/transcript")
    assert res_trans.status_code == 200
    transcripts = res_trans.json()
    assert len(transcripts) == 3
    assert transcripts[0]["speaker"] == "Manager"
    assert transcripts[0]["type"] == "ACTION"
    assert transcripts[1]["speaker"] == "Architect"
    assert transcripts[1]["type"] == "DECISION"

    # 3. Verify action items collection (speaker vs responsible_person independence)
    res_actions = await test_client.get(f"/api/meetings/{meeting_id}/actions")
    assert res_actions.status_code == 200
    actions = res_actions.json()
    assert len(actions) == 1
    action = actions[0]
    assert action["action"] == "Complete the backend"
    assert action["responsible_person"] == "Rahul"
    assert action["speaker"] == "Manager"  # Distinct from responsible_person!
    assert action["deadline"] == "2026-10-16"
    assert action["status"] == "Pending"

    # 4. Verify decisions collection
    res_decisions = await test_client.get(f"/api/meetings/{meeting_id}/decisions")
    assert res_decisions.status_code == 200
    decisions = res_decisions.json()
    assert len(decisions) == 1
    assert decisions[0]["decision"] == "Use MongoDB for the database"
    assert decisions[0]["speaker"] == "Architect"

    # 5. Verify WebSocket events emitted
    event_names = [e["event"] for e in ws_events]
    assert "meeting_processing_started" in event_names
    assert "meeting_processing_completed" in event_names


@pytest.mark.asyncio
async def test_ai_failure_sets_meeting_status_to_failed(test_client: AsyncClient):
    """When AI pipeline raises an error, meeting status becomes 'failed' and error is logged."""
    meeting_id = "mtg_fail_01"
    await test_client.post("/api/meetings", json={"meeting_id": meeting_id, "title": "Failure Test"})

    ws_events = []
    async def record_event(mid: str, event: dict):
        if mid == meeting_id:
            ws_events.append(event)

    wav_bytes = create_dummy_wav_bytes(duration_sec=1.0)

    # Force AI service to raise an exception
    with patch("app.services.ai_service.process_meeting_audio", side_effect=RuntimeError("Diart model error")) as mock_ai, \
         patch.object(sockets, "publish", side_effect=record_event):

        res = await test_client.post(
            f"/api/meetings/{meeting_id}/audio",
            files={"file": ("meeting_fail.wav", wav_bytes, "audio/wav")},
        )
        assert res.status_code == 202

    # Check meeting status in database
    res_mtg = await test_client.get(f"/api/meetings/{meeting_id}")
    assert res_mtg.status_code == 200
    mtg_data = res_mtg.json()
    assert mtg_data["status"] == "failed"
    assert "Diart model error" in mtg_data["error"]

    # Verify WebSocket failure broadcast
    event_names = [e["event"] for e in ws_events]
    assert "meeting_processing_started" in event_names
    assert "meeting_processing_failed" in event_names


@pytest.mark.asyncio
async def test_duplicate_processing_idempotency(test_client: AsyncClient):
    """Retrying AI audio processing does not produce duplicate transcript, action, or decision records."""
    meeting_id = "mtg_idempotent_01"
    await test_client.post("/api/meetings", json={"meeting_id": meeting_id, "title": "Idempotency Test"})

    mock_result = build_mock_ai_result(meeting_id)
    wav_bytes = create_dummy_wav_bytes(duration_sec=1.5)

    with patch("app.services.ai_service.process_meeting_audio", return_value=mock_result):
        # First processing
        await test_client.post(
            f"/api/meetings/{meeting_id}/audio",
            files={"file": ("meeting.wav", wav_bytes, "audio/wav")},
        )

        # Re-trigger processing on same meeting
        await test_client.post(
            f"/api/meetings/{meeting_id}/audio",
            files={"file": ("meeting.wav", wav_bytes, "audio/wav")},
        )

    # Count records in database
    res_trans = await test_client.get(f"/api/meetings/{meeting_id}/transcript")
    assert len(res_trans.json()) == 3

    res_acts = await test_client.get(f"/api/meetings/{meeting_id}/actions")
    assert len(res_acts.json()) == 1

    res_decs = await test_client.get(f"/api/meetings/{meeting_id}/decisions")
    assert len(res_decs.json()) == 1


# ============================================================================
# 2. REST API & Result Aggregate Endpoint Tests
# ============================================================================

@pytest.mark.asyncio
async def test_get_meeting_result_endpoint(test_client: AsyncClient):
    """GET /api/meetings/{id}/result returns combined meeting summary in one call."""
    meeting_id = "mtg_res_01"
    await test_client.post("/api/meetings", json={"meeting_id": meeting_id, "title": "Aggregated View"})

    mock_result = build_mock_ai_result(meeting_id)
    wav_bytes = create_dummy_wav_bytes()

    with patch("app.services.ai_service.process_meeting_audio", return_value=mock_result):
        await test_client.post(
            f"/api/meetings/{meeting_id}/audio",
            files={"file": ("meeting.wav", wav_bytes, "audio/wav")},
        )

    res = await test_client.get(f"/api/meetings/{meeting_id}/result")
    assert res.status_code == 200
    payload = res.json()
    assert payload["meeting"]["meeting_id"] == meeting_id
    assert payload["meeting"]["status"] == "completed"
    assert len(payload["transcript"]) == 3
    assert len(payload["action_items"]) == 1
    assert len(payload["decisions"]) == 1
    assert payload["participants"] == ["Architect", "Manager", "Rahul"]


@pytest.mark.asyncio
async def test_patch_action_item_endpoint(test_client: AsyncClient):
    """PATCH /api/meetings/{id}/actions/{action_id} updates task status."""
    meeting_id = "mtg_patch_01"
    await test_client.post("/api/meetings", json={"meeting_id": meeting_id, "title": "Patch Test"})

    mock_result = build_mock_ai_result(meeting_id)
    wav_bytes = create_dummy_wav_bytes()

    with patch("app.services.ai_service.process_meeting_audio", return_value=mock_result):
        await test_client.post(
            f"/api/meetings/{meeting_id}/audio",
            files={"file": ("meeting.wav", wav_bytes, "audio/wav")},
        )

    # Get action ID
    res_acts = await test_client.get(f"/api/meetings/{meeting_id}/actions")
    actions = res_acts.json()
    action_id = actions[0]["id"]

    # Patch status to Completed
    res_patch = await test_client.patch(
        f"/api/meetings/{meeting_id}/actions/{action_id}",
        json={"status": "Completed"},
    )
    assert res_patch.status_code == 200
    assert res_patch.json()["status"] == "Completed"

    # Filter by status
    res_pending = await test_client.get(f"/api/meetings/{meeting_id}/actions?status=Pending")
    assert len(res_pending.json()) == 0
    res_completed = await test_client.get(f"/api/meetings/{meeting_id}/actions?status=Completed")
    assert len(res_completed.json()) == 1
