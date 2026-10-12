# Meeting Backend

FastAPI + MongoDB backend for storing AI-processed meeting transcripts and broadcasting them live. It consumes the agreed AI result schema; it does not run transcription, NLP, or classification.

## Run with Docker

From this directory:

```sh
docker compose up --build
```

API: `http://localhost:8000`  
Interactive API docs: `http://localhost:8000/docs`  
WebSocket: `ws://localhost:8000/ws/meetings/{meeting_id}`

## Run locally

Use Python 3.12+ and MongoDB 7+. Create a virtual environment, install requirements, set `MONGODB_URI` and `MONGODB_DATABASE` (see `.env.example`), then run:

```sh
uvicorn app.main:app --reload
```

The MongoDB database and required indexes are initialized at API startup. `/health` checks database connectivity.

## API

1. `POST /api/meetings`

   ```json
   {"meeting_id":"meeting_001","title":"Backend planning"}
   ```

2. `POST /api/meetings/{meeting_id}/segments` with the complete AI result:

   ```json
   {
     "meeting_id":"meeting_001",
     "speaker_id":"SPEAKER_00",
     "speaker":"Manager",
     "timestamp":"2026-10-05T10:15:23+05:30",
     "text":"Rahul should complete the backend by Friday.",
     "type":"ACTION",
     "action":"Complete the backend",
     "responsible_person":"Rahul",
     "deadline_text":"by Friday",
     "deadline_normalized":"2026-10-09",
     "decision":null,
     "status":"Pending"
   }
   ```

Timezone-aware timestamps are preserved; timestamps without a timezone (as in the supplied example) are interpreted as UTC. The example date above reflects the supplied specification's date context; callers should send the actual meeting date. The backend validates the classification-specific fields and rejects unexpected schema fields. Each accepted segment is durably stored in `transcripts`; ACTION results also upsert `action_items`, DECISION results upsert `decisions`, and DISCUSSION/INFORMATION results create no derived records. A stable hash of meeting, speaker, timestamp, and transcript text makes retries idempotent. `speaker` and `responsible_person` stay separate.

### Read and update endpoints

- `GET /api/meetings` and `GET /api/meetings/{meeting_id}`
- `GET /api/meetings/{meeting_id}/transcripts?limit=100&offset=0`
- `GET /api/meetings/{meeting_id}/actions?status=Pending&limit=100&offset=0`
- `PATCH /api/meetings/{meeting_id}/actions/{action_id}` with e.g. `{"status":"Completed"}`
- `GET /api/meetings/{meeting_id}/decisions?limit=100&offset=0`

List endpoints are paginated; limits are capped at 500.

## Live events

Connect to `ws://localhost:8000/ws/meetings/meeting_001`. On each accepted segment, connected clients receive:

```json
{"event":"transcript_processed","data":{"meeting_id":"meeting_001","type":"ACTION"}}
```

The `data` object is the full validated AI result. WebSocket fanout is in-process; for multi-replica deployment, replace the in-memory fanout with a shared pub/sub layer (such as Redis). MongoDB remains the durable source of truth.

## Collections

- `meetings`: meeting metadata
- `transcripts`: original AI results plus receive time
- `action_items`: action, assignee, deadline, status, and speaker fields
- `decisions`: decision text and speaker fields

This starter has no authentication. Add the application's identity/access-control mechanism before exposing meeting data outside a trusted network.
