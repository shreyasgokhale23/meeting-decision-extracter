import hashlib
import os
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from typing import Any

from bson import ObjectId
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pymongo import ASCENDING, DESCENDING, AsyncMongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

from .models import AIResult, MeetingCreate
from .realtime import sockets
from .services.meeting_service import process_meeting_background, save_uploaded_audio


MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "meeting_backend")
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]


def public_document(document: dict[str, Any]) -> dict[str, Any]:
    result = dict(document)
    if "_id" in result:
        result["id"] = str(result.pop("_id"))
    return jsonable_encoder(result)


def segment_key(result: AIResult) -> str:
    # Stable key makes retries safe without requiring a field outside the agreed contract.
    raw = "\x1f".join((result.meeting_id, result.speaker_id, result.timestamp.isoformat(), result.text))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncMongoClient(MONGODB_URI, tz_aware=True, serverSelectionTimeoutMS=5000)
    app.state.mongo = client
    app.state.db = client[MONGODB_DATABASE]
    await client.admin.command("ping")
    await app.state.db.meetings.create_index([("meeting_id", ASCENDING)], unique=True)
    await app.state.db.transcripts.create_index([("meeting_id", ASCENDING), ("start_time", ASCENDING)])
    await app.state.db.transcripts.create_index([("meeting_id", ASCENDING), ("segment_key", ASCENDING)], unique=True)
    await app.state.db.action_items.create_index([("meeting_id", ASCENDING), ("segment_key", ASCENDING)], unique=True)
    await app.state.db.action_items.create_index([("meeting_id", ASCENDING), ("status", ASCENDING)])
    await app.state.db.decisions.create_index([("meeting_id", ASCENDING), ("segment_key", ASCENDING)], unique=True)
    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="Meeting Backend API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    await app.state.db.command("ping")
    return {"status": "ok", "database": "connected"}


@app.post("/api/meetings", status_code=201)
async def create_meeting(payload: MeetingCreate):
    now = datetime.now(UTC)
    meeting = {
        "meeting_id": payload.meeting_id,
        "title": payload.title,
        "status": "created",
        "started_at": payload.started_at or now,
        "created_at": now,
        "ended_at": None,
        "audio_path": None,
        "participants": [],
        "error": None,
    }
    try:
        await app.state.db.meetings.insert_one(meeting)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="meeting_id already exists") from None
    return public_document(meeting)


@app.get("/api/meetings")
async def list_meetings(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    cursor = app.state.db.meetings.find().sort("created_at", DESCENDING).skip(offset).limit(limit)
    return [public_document(item) async for item in cursor]


@app.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    meeting = await app.state.db.meetings.find_one({"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="meeting not found")
    return public_document(meeting)


@app.post("/api/meetings/{meeting_id}/audio", status_code=202)
async def upload_meeting_audio(
    meeting_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Receive a complete meeting WAV recording and trigger asynchronous AI processing."""
    db = app.state.db
    meeting = await db.meetings.find_one({"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="meeting not found; create it first")

    # Validate and persist uploaded WAV
    saved_path = await save_uploaded_audio(meeting_id=meeting_id, upload_file=file)

    # Immediately mark meeting as processing
    now = datetime.now(UTC)
    await db.meetings.update_one(
        {"meeting_id": meeting_id},
        {
            "$set": {
                "status": "processing",
                "audio_path": str(saved_path),
                "error": None,
                "updated_at": now,
            }
        },
    )

    # Start AI processing in background
    background_tasks.add_task(
        process_meeting_background,
        db=db,
        meeting_id=meeting_id,
        audio_path=saved_path,
    )

    return {
        "meeting_id": meeting_id,
        "status": "processing",
    }


async def fetch_transcript_segments(meeting_id: str, limit: int, offset: int):
    if not await app.state.db.meetings.find_one({"meeting_id": meeting_id}, {"_id": 1}):
        raise HTTPException(status_code=404, detail="meeting not found")
    cursor = (
        app.state.db.transcripts.find({"meeting_id": meeting_id}, {"segment_key": 0})
        .sort("start_time", ASCENDING)
        .skip(offset)
        .limit(limit)
    )
    return [public_document(item) async for item in cursor]


@app.get("/api/meetings/{meeting_id}/transcript")
async def get_meeting_transcript(
    meeting_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Retrieve speaker-aware transcript turns for a meeting."""
    return await fetch_transcript_segments(meeting_id, limit, offset)


@app.get("/api/meetings/{meeting_id}/transcripts")
async def get_meeting_transcripts_alias(
    meeting_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Alias for backwards compatibility with previous transcript endpoint."""
    return await fetch_transcript_segments(meeting_id, limit, offset)


@app.get("/api/meetings/{meeting_id}/actions")
async def get_actions(
    meeting_id: str,
    status: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    if not await app.state.db.meetings.find_one({"meeting_id": meeting_id}, {"_id": 1}):
        raise HTTPException(status_code=404, detail="meeting not found")
    query = {"meeting_id": meeting_id}
    if status:
        query["status"] = status
    cursor = app.state.db.action_items.find(query, {"segment_key": 0}).sort("created_at", ASCENDING).skip(offset).limit(limit)
    return [public_document(item) async for item in cursor]


@app.patch("/api/meetings/{meeting_id}/actions/{action_id}")
async def update_action(meeting_id: str, action_id: str, changes: dict[str, Any]):
    allowed = {"status", "responsible_person", "deadline_text", "deadline_normalized", "deadline", "action"}
    if not changes or set(changes) - allowed:
        raise HTTPException(status_code=422, detail=f"allowed fields: {', '.join(sorted(allowed))}")
    if "status" in changes and (not isinstance(changes["status"], str) or not changes["status"].strip()):
        raise HTTPException(status_code=422, detail="status must be a non-empty string")
    if "deadline_normalized" in changes and changes["deadline_normalized"] is not None:
        try:
            changes["deadline_normalized"] = date.fromisoformat(changes["deadline_normalized"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="deadline_normalized must be YYYY-MM-DD") from None
    changes["updated_at"] = datetime.now(UTC)
    try:
        object_id = ObjectId(action_id)
    except Exception:
        raise HTTPException(status_code=404, detail="action item not found") from None
    updated = await app.state.db.action_items.find_one_and_update(
        {"_id": object_id, "meeting_id": meeting_id}, {"$set": changes}, return_document=ReturnDocument.AFTER
    )
    if not updated:
        raise HTTPException(status_code=404, detail="action item not found")
    return public_document(updated)


@app.get("/api/meetings/{meeting_id}/decisions")
async def get_decisions(
    meeting_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    if not await app.state.db.meetings.find_one({"meeting_id": meeting_id}, {"_id": 1}):
        raise HTTPException(status_code=404, detail="meeting not found")
    cursor = app.state.db.decisions.find({"meeting_id": meeting_id}, {"segment_key": 0}).sort("created_at", ASCENDING).skip(offset).limit(limit)
    return [public_document(item) async for item in cursor]


@app.get("/api/meetings/{meeting_id}/result")
async def get_meeting_result(meeting_id: str):
    """Retrieve full meeting details, participants, transcript, actions, and decisions in one payload."""
    db = app.state.db
    meeting = await db.meetings.find_one({"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="meeting not found")

    transcript_cursor = db.transcripts.find({"meeting_id": meeting_id}, {"segment_key": 0}).sort("start_time", ASCENDING)
    transcripts = [public_document(t) async for t in transcript_cursor]

    actions_cursor = db.action_items.find({"meeting_id": meeting_id}, {"segment_key": 0}).sort("created_at", ASCENDING)
    actions = [public_document(a) async for a in actions_cursor]

    decisions_cursor = db.decisions.find({"meeting_id": meeting_id}, {"segment_key": 0}).sort("created_at", ASCENDING)
    decisions = [public_document(d) async for d in decisions_cursor]

    return {
        "meeting": public_document(meeting),
        "participants": meeting.get("participants", []),
        "transcript": transcripts,
        "action_items": actions,
        "decisions": decisions,
    }


@app.post("/api/meetings/{meeting_id}/segments", status_code=202)
async def process_segment(meeting_id: str, payload: AIResult):
    """Legacy/Single-turn segment ingestion endpoint (retained for backward compatibility)."""
    if payload.meeting_id != meeting_id:
        raise HTTPException(status_code=422, detail="path meeting_id must match payload meeting_id")
    db = app.state.db
    if not await db.meetings.find_one({"meeting_id": meeting_id}, {"_id": 1}):
        raise HTTPException(status_code=404, detail="meeting not found; create it first")

    key = segment_key(payload)
    now = datetime.now(UTC)
    transcript = payload.model_dump(mode="python")
    transcript.update({"segment_key": key, "received_at": now})
    # Persist the original validated AI result for every classification, including retries.
    await db.transcripts.update_one({"segment_key": key}, {"$setOnInsert": transcript}, upsert=True)

    result = payload.model_dump(mode="json")
    if payload.type == "ACTION":
        action_doc = {
            "meeting_id": meeting_id,
            "segment_key": key,
            "speaker": payload.speaker,
            "speaker_id": payload.speaker_id,
            "responsible_person": payload.responsible_person,
            "action": payload.action,
            "deadline_text": payload.deadline_text,
            "deadline_normalized": payload.deadline_normalized,
            "deadline": str(payload.deadline_normalized) if payload.deadline_normalized else payload.deadline_text,
            "status": payload.status or "Pending",
            "timestamp": payload.timestamp.isoformat() if hasattr(payload.timestamp, "isoformat") else str(payload.timestamp),
            "updated_at": now,
        }
        await db.action_items.find_one_and_update(
            {"meeting_id": meeting_id, "segment_key": key},
            {"$set": action_doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
    elif payload.type == "DECISION":
        decision_doc = {
            "meeting_id": meeting_id,
            "segment_key": key,
            "speaker": payload.speaker,
            "speaker_id": payload.speaker_id,
            "decision": payload.decision,
            "timestamp": payload.timestamp.isoformat() if hasattr(payload.timestamp, "isoformat") else str(payload.timestamp),
            "updated_at": now,
        }
        await db.decisions.update_one(
            {"meeting_id": meeting_id, "segment_key": key},
            {"$set": decision_doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    await sockets.publish(meeting_id, {"event": "transcript_processed", "data": result})
    return {"accepted": True, "segment_key": key, "data": result}


@app.websocket("/ws/meetings/{meeting_id}")
async def meeting_stream(socket: WebSocket, meeting_id: str):
    if not await app.state.db.meetings.find_one({"meeting_id": meeting_id}, {"_id": 1}):
        await socket.close(code=4404, reason="meeting not found")
        return
    await sockets.connect(meeting_id, socket)
    try:
        while True:
            # Client messages are ignored; receiving keeps the connection lifecycle observable.
            await socket.receive_text()
    except WebSocketDisconnect:
        await sockets.disconnect(meeting_id, socket)
