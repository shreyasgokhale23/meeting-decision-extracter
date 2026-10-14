"""Meeting Service layer.

Coordinates meeting creation, audio upload validation, background AI execution,
idempotent MongoDB persistence, and realtime WebSocket notifications.
"""

import hashlib
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from fastapi import HTTPException, UploadFile
from pymongo import ReturnDocument

from ..realtime import sockets
from ..services import ai_service

logger = logging.getLogger(__name__)

# Default recordings directory
DEFAULT_RECORDINGS_DIR = Path("data/recordings")


def get_recordings_dir() -> Path:
    env_dir = os.getenv("RECORDINGS_DIR")
    if env_dir:
        target = Path(env_dir)
    else:
        # Resolve relative to backend/ directory
        backend_dir = Path(__file__).resolve().parents[2]
        target = backend_dir / DEFAULT_RECORDINGS_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def compute_segment_key(
    meeting_id: str,
    speaker_id: str,
    start_time: float,
    end_time: float,
    text: str,
) -> str:
    """Compute a deterministic hash for a transcript turn to ensure idempotent retries."""
    raw = f"{meeting_id}\x1f{speaker_id}\x1f{start_time:.3f}\x1f{end_time:.3f}\x1f{text.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_action_key(meeting_id: str, action: str, responsible_person: Optional[str], deadline: Optional[str]) -> str:
    raw = f"{meeting_id}\x1f{action.strip()}\x1f{responsible_person or ''}\x1f{deadline or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_decision_key(meeting_id: str, decision: str, speaker: Optional[str]) -> str:
    raw = f"{meeting_id}\x1f{decision.strip()}\x1f{speaker or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def save_uploaded_audio(meeting_id: str, upload_file: UploadFile) -> Path:
    """Validate uploaded audio and stream it to a deterministic recording location.

    Raises:
        HTTPException(400): If the file is not a supported WAV file.
    """
    filename = upload_file.filename or "meeting.wav"
    if not filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{filename}'. Only WAV audio files (.wav) are supported.",
        )

    # Check initial bytes for RIFF header to confirm valid WAV container
    header_chunk = await upload_file.read(12)
    if len(header_chunk) < 12 or not (header_chunk.startswith(b"RIFF") and b"WAVE" in header_chunk):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid RIFF/WAVE formatted audio file.",
        )

    # Reset position to start before writing full file
    await upload_file.seek(0)

    recordings_dir = get_recordings_dir()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:8]
    sanitized_id = "".join(c for c in meeting_id if c.isalnum() or c in ("-", "_")) or "meeting"
    target_path = recordings_dir / f"meeting_{sanitized_id}_{timestamp}_{unique_suffix}.wav"

    try:
        with open(target_path, "wb") as out_f:
            while chunk := await upload_file.read(1024 * 1024):  # 1MB buffer
                out_f.write(chunk)
    except Exception as exc:
        logger.error("Failed to write uploaded audio to disk: %s", exc, exc_info=True)
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to persist audio file on server.") from exc

    logger.info("Saved meeting audio to: %s (size: %d bytes)", target_path, target_path.stat().st_size)
    return target_path


async def store_audio_meeting_result(
    db: Any,
    meeting_id: str,
    audio_path: Union[str, Path],
    result: Any,
) -> Dict[str, Any]:
    """Idempotently persist AudioMeetingResult into MongoDB collections.

    Collections updated:
    - transcripts
    - action_items
    - decisions
    - meetings (status, ended_at, participants, audio_path)
    """
    now = datetime.now(UTC)

    # 1. Store speaker-aware transcript segments
    turns = getattr(result, "speaker_transcript", [])
    meeting_summary = getattr(result, "meeting_summary", None)
    nlp_segments = getattr(meeting_summary, "transcript", []) if meeting_summary else []

    num_turns = max(len(turns), len(nlp_segments))
    stored_transcript_count = 0
    stored_action_count = 0
    stored_decision_count = 0
    seen_actions: set[str] = set()
    seen_decisions: set[str] = set()

    for i in range(num_turns):
        turn = turns[i] if i < len(turns) else None
        seg = nlp_segments[i] if i < len(nlp_segments) else None

        speaker = (
            getattr(seg, "speaker", None)
            or getattr(turn, "speaker", None)
            or "UNKNOWN"
        )
        speaker_id = (
            getattr(seg, "speaker_id", None)
            or getattr(turn, "speaker", None)
            or "UNKNOWN"
        )
        start_time = float(getattr(turn, "start", 0.0)) if turn else 0.0
        end_time = float(getattr(turn, "end", 0.0)) if turn else 0.0
        text = str(getattr(seg, "text", "") or getattr(turn, "text", "")).strip()

        if not text:
            continue

        timestamp = getattr(seg, "timestamp", None)
        if not timestamp:
            hrs = int(start_time // 3600)
            mins = int((start_time % 3600) // 60)
            secs = int(start_time % 60)
            timestamp = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        seg_type = str(getattr(seg, "type", "UNKNOWN"))
        seg_key = compute_segment_key(meeting_id, str(speaker_id), start_time, end_time, text)

        transcript_doc = {
            "meeting_id": meeting_id,
            "speaker_id": str(speaker_id),
            "speaker": str(speaker),
            "timestamp": timestamp,
            "start_time": start_time,
            "end_time": end_time,
            "text": text,
            "type": seg_type,
            "segment_key": seg_key,
            "received_at": now,
        }

        await db.transcripts.update_one(
            {"meeting_id": meeting_id, "segment_key": seg_key},
            {"$set": transcript_doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        stored_transcript_count += 1

        # 2. Extract and persist Action Items
        if seg and seg_type == "ACTION" and getattr(seg, "action", None):
            action_doc = {
                "meeting_id": meeting_id,
                "segment_key": seg_key,
                "speaker_id": str(speaker_id),
                "speaker": str(speaker),
                "action": seg.action,
                "responsible_person": seg.responsible_person,
                "deadline_text": seg.deadline_text,
                "deadline": seg.deadline_normalized or seg.deadline_text,
                "deadline_normalized": seg.deadline_normalized,
                "status": getattr(seg, "status", "Pending") or "Pending",
                "timestamp": timestamp,
                "updated_at": now,
            }
            await db.action_items.update_one(
                {"meeting_id": meeting_id, "segment_key": seg_key},
                {"$set": action_doc, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            stored_action_count += 1
            seen_actions.add(seg.action)

        # 3. Extract and persist Decisions
        elif seg and seg_type == "DECISION" and getattr(seg, "decision", None):
            decision_doc = {
                "meeting_id": meeting_id,
                "segment_key": seg_key,
                "speaker_id": str(speaker_id),
                "speaker": str(speaker),
                "decision": seg.decision,
                "timestamp": timestamp,
                "updated_at": now,
            }
            await db.decisions.update_one(
                {"meeting_id": meeting_id, "segment_key": seg_key},
                {"$set": decision_doc, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            stored_decision_count += 1
            seen_decisions.add(seg.decision)

    # Ensure any meeting-level aggregated action items not mapped per turn are also persisted
    aggregated_actions = getattr(result, "action_items", [])
    for act in aggregated_actions:
        action_text = getattr(act, "action", None)
        if not action_text or action_text in seen_actions:
            continue
        resp_person = getattr(act, "responsible_person", None)
        deadline = getattr(act, "deadline", None)
        act_key = compute_action_key(meeting_id, action_text, resp_person, deadline)

        doc = {
            "meeting_id": meeting_id,
            "segment_key": act_key,
            "action": action_text,
            "responsible_person": resp_person,
            "deadline": deadline,
            "status": str(getattr(act, "status", "Pending") or "Pending"),
            "updated_at": now,
        }
        await db.action_items.update_one(
            {"meeting_id": meeting_id, "segment_key": act_key},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        stored_action_count += 1
        seen_actions.add(action_text)

    # Ensure any meeting-level aggregated decisions not mapped per turn are also persisted
    aggregated_decisions = getattr(result, "decisions", [])
    for dec in aggregated_decisions:
        decision_text = getattr(dec, "decision", None)
        if not decision_text or decision_text in seen_decisions:
            continue
        speaker = getattr(dec, "speaker", None)
        dec_key = compute_decision_key(meeting_id, decision_text, speaker)

        doc = {
            "meeting_id": meeting_id,
            "segment_key": dec_key,
            "decision": decision_text,
            "speaker": speaker,
            "updated_at": now,
        }
        await db.decisions.update_one(
            {"meeting_id": meeting_id, "segment_key": dec_key},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        stored_decision_count += 1
        seen_decisions.add(decision_text)

    # 4. Update meeting document metadata
    participants = sorted(list(set(getattr(result, "participants", []))))
    await db.meetings.update_one(
        {"meeting_id": meeting_id},
        {
            "$set": {
                "status": "completed",
                "ended_at": now,
                "participants": participants,
                "audio_path": str(audio_path),
                "error": None,
                "updated_at": now,
            }
        },
    )

    logger.info(
        "Persisted meeting '%s' results: %d transcripts, %d actions, %d decisions, %d participants",
        meeting_id,
        stored_transcript_count,
        stored_action_count,
        stored_decision_count,
        len(participants),
    )

    return {
        "meeting_id": meeting_id,
        "transcripts": stored_transcript_count,
        "action_items": stored_action_count,
        "decisions": stored_decision_count,
        "participants": participants,
    }


async def process_meeting_background(
    db: Any,
    meeting_id: str,
    audio_path: Union[str, Path],
    reference_datetime: Optional[datetime] = None,
    ai_processor: Optional[Callable[..., Any]] = None,
) -> None:
    """Execute end-to-end AI audio processing as a background task.

    Steps:
    1. Set meeting status to "processing".
    2. Notify WebSocket clients: meeting_processing_started.
    3. Call process_audio() via AI service.
    4. Persist transcripts, action items, decisions, participants.
    5. Update meeting status to "completed".
    6. Notify WebSocket clients: meeting_processing_completed.
    On failure:
    - Update meeting status to "failed" with error message.
    - Notify WebSocket clients: meeting_processing_failed.
    - Log error details without crashing application.
    """
    logger.info("Background AI processing started for meeting: %s", meeting_id)

    # 1. Update meeting status to "processing"
    now = datetime.now(UTC)
    await db.meetings.update_one(
        {"meeting_id": meeting_id},
        {
            "$set": {
                "status": "processing",
                "audio_path": str(audio_path),
                "updated_at": now,
            }
        },
    )

    # 2. Emit meeting_processing_started event
    await sockets.publish(
        meeting_id,
        {
            "event": "meeting_processing_started",
            "data": {
                "meeting_id": meeting_id,
                "status": "processing",
            },
        },
    )

    try:
        # 3. Call AI processing
        processor = ai_processor or ai_service.process_meeting_audio
        result = processor(
            audio_path=audio_path,
            meeting_id=meeting_id,
            reference_datetime=reference_datetime,
        )

        # 4. Store returned results in MongoDB
        await store_audio_meeting_result(
            db=db,
            meeting_id=meeting_id,
            audio_path=audio_path,
            result=result,
        )

        # 5. Notify WebSocket clients: meeting_processing_completed
        await sockets.publish(
            meeting_id,
            {
                "event": "meeting_processing_completed",
                "data": {
                    "meeting_id": meeting_id,
                    "status": "completed",
                },
            },
        )
        logger.info("Background AI processing successfully completed for meeting: %s", meeting_id)

    except Exception as exc:
        logger.error(
            "Background AI processing failed for meeting '%s': %s",
            meeting_id,
            exc,
            exc_info=True,
        )
        error_msg = str(exc)
        fail_time = datetime.now(UTC)

        # Mark meeting as failed in database
        await db.meetings.update_one(
            {"meeting_id": meeting_id},
            {
                "$set": {
                    "status": "failed",
                    "error": error_msg,
                    "updated_at": fail_time,
                }
            },
        )

        # Notify WebSocket clients: meeting_processing_failed
        await sockets.publish(
            meeting_id,
            {
                "event": "meeting_processing_failed",
                "data": {
                    "meeting_id": meeting_id,
                    "status": "failed",
                    "error": error_msg,
                },
            },
        )
