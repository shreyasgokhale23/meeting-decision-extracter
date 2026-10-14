# Meeting Decision Extractor — Backend Service

FastAPI + MongoDB backend for the **Meeting Decision Extractor** project. Orchestrates meeting lifecycles, receives completed audio recordings, triggers asynchronous AI processing (via the integrated AI/NLP pipeline), persists structured meeting intelligence in MongoDB, and broadcasts real-time status events over WebSockets.

---

## 1. Backend Architecture

The backend serves as the orchestration and persistence layer connecting client applications (e.g. Next.js dashboard) with the AI/NLP engine:

```
                  ┌──────────────────────────────────────────────┐
                  │          Microphone Recording                │
                  │   (Arbitrary duration meeting audio)         │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Completed Meeting WAV (16kHz PCM)         │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                      POST /api/meetings/{id}/audio
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │               FastAPI Backend                │
                  │  (Saves WAV to disk, sets status=processing) │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                             FastAPI BackgroundTasks
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │             Backend AI Service               │
                  │          (app/services/ai_service.py)        │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │          ai/transcription/audio_processor    │
                  │           process_audio(audio_path, id)      │
                  │   ┌──────────────────────────────────────┐   │
                  │   │ 1. Diart (Speaker Diarization)       │   │
                  │   │ 2. Whisper (Speech-to-Text STT)      │   │
                  │   │ 3. Speaker-Aware Alignment           │   │
                  │   │ 4. NLP Intelligence Pipeline         │   │
                  │   └──────────────────────────────────────┘   │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │             AudioMeetingResult               │
                  │  (speaker_transcript, actions, decisions,    │
                  │   participants, deadlines, statuses)         │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         MongoDB Persistence Layer            │
                  │   - meetings     - action_items              │
                  │   - transcripts  - decisions                 │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │            Realtime WebSocket Fanout         │
                  │        (ws/meetings/{meeting_id})            │
                  │       meeting_processing_completed           │
                  └──────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Post-Meeting Processing Model**: AI processing is executed **after the complete meeting audio is finished and uploaded**. There is NO streaming chunk-by-chunk AI processing. The microphone captures the full session, writes a 16 kHz 16-bit mono PCM WAV file, and posts it to the backend for background processing.

---

## 2. AI/NLP Integration Boundary

The backend maintains strict architectural separation:
- **No duplicated AI/NLP code**: The backend does NOT implement transcription, diarization, or classification algorithms.
- **Single Source of Truth**: Backend invokes `ai/transcription/audio_processor.py::process_audio()` directly through `app/services/ai_service.py`.
- **Preserves Full Meaning**:
  - `speaker`: The entity uttering the speech segment.
  - `responsible_person`: The entity assigned the action item.
  - Both fields are stored independently and never overwritten.

---

## 3. Meeting Lifecycle States

Every meeting progresses through four deterministic states:

```
[created]  ───(Upload WAV)───►  [processing]  ───(AI Complete)───►  [completed]
                                       │
                                (AI Failure)
                                       │
                                       ▼
                                   [failed]
```

1. **`created`**: Meeting record provisioned via `POST /api/meetings`.
2. **`processing`**: Audio WAV uploaded via `POST /api/meetings/{id}/audio`; file saved to `data/recordings/`; background processing started; `meeting_processing_started` broadcast to WebSockets.
3. **`completed`**: AI extraction finished; transcript turns, action items, decisions, and participants stored in MongoDB; `ended_at` timestamp recorded; `meeting_processing_completed` broadcast.
4. **`failed`**: Any processing or audio exception is caught safely; `status="failed"` and detailed `error` recorded in meeting document; `meeting_processing_failed` broadcast without crashing FastAPI.

---

## 4. REST API Endpoints

### Meeting Management
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/meetings` | Create a new meeting session (`status: "created"`). |
| `GET` | `/api/meetings` | List recent meetings with pagination. |
| `GET` | `/api/meetings/{meeting_id}` | Retrieve meeting metadata, status, audio path, and participants. |
| `POST` | `/api/meetings/{meeting_id}/audio` | Upload completed meeting WAV and begin background processing (`202 Accepted`). |

### Intelligence Retrieval
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/meetings/{meeting_id}/transcript` | Retrieve chronologically sorted speaker-aware transcript turns. |
| `GET` | `/api/meetings/{meeting_id}/transcripts` | Alias for backwards compatibility. |
| `GET` | `/api/meetings/{meeting_id}/actions` | Retrieve extracted action items (optional filter: `?status=Pending`). |
| `PATCH` | `/api/meetings/{meeting_id}/actions/{action_id}` | Update action item fields (e.g. `{"status": "Completed"}`). |
| `GET` | `/api/meetings/{meeting_id}/decisions` | Retrieve meeting decisions with speakers and timestamps. |
| `GET` | `/api/meetings/{meeting_id}/result` | Retrieve aggregated payload (meeting, participants, transcript, actions, decisions) in one request. |

### Legacy Single-Turn Ingestion (Retained for Backward Compatibility)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/meetings/{meeting_id}/segments` | Ingest and store an individual `AIResult` segment turn. |

---

## 5. WebSocket Real-Time Events

Clients (e.g. Next.js dashboard) connect to:
```
ws://localhost:8000/ws/meetings/{meeting_id}
```

The backend broadcasts state transitions automatically:

### 1. `meeting_processing_started`
```json
{
  "event": "meeting_processing_started",
  "data": {
    "meeting_id": "meeting_001",
    "status": "processing"
  }
}
```

### 2. `meeting_processing_completed`
```json
{
  "event": "meeting_processing_completed",
  "data": {
    "meeting_id": "meeting_001",
    "status": "completed"
  }
}
```

### 3. `meeting_processing_failed`
```json
{
  "event": "meeting_processing_failed",
  "data": {
    "meeting_id": "meeting_001",
    "status": "failed",
    "error": "Detailed error explanation"
  }
}
```

---

## 6. MongoDB Collections

- **`meetings`**: Session metadata, status (`created`/`processing`/`completed`/`failed`), timestamps, participants, `audio_path`, and optional `error`.
- **`transcripts`**: Speaker-aware transcript turns (`meeting_id`, `speaker_id`, `speaker`, `start_time`, `end_time`, `text`, `type`, `segment_key`).
- **`action_items`**: Extracted actions (`meeting_id`, `action`, `responsible_person`, `speaker`, `deadline`, `deadline_text`, `status`, `segment_key`).
- **`decisions`**: Recorded architectural/business decisions (`meeting_id`, `decision`, `speaker`, `timestamp`, `segment_key`).

Deterministic `segment_key` hashes and unique compound indexes ensure retrying processing or re-uploading audio is **completely idempotent** with zero duplicate records.

---

## 7. How to Run Locally

### Prerequisites
- Python 3.12+ (Python 3.14 compatible)
- MongoDB 7.0+ (or local container)
- FFmpeg installed / in PATH (automatically provided via `imageio-ffmpeg`)

### Step-by-Step
1. Create and activate a virtual environment:
   ```bash
   cd backend
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables (see `.env.example`):
   ```ini
   MONGODB_URI=mongodb://localhost:27017
   MONGODB_DATABASE=meeting_backend
   CORS_ORIGINS=http://localhost:3000
   RECORDINGS_DIR=data/recordings
   ```

4. Run the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

- API Base: `http://localhost:8000`
- Interactive API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

---

## 8. Running Tests

### Fast Mocked Unit & Integration Tests (No Model Downloads)
Runs in under 2 seconds:
```bash
python -m pytest backend/tests/test_backend.py -v
```
Verifies:
- Meeting creation and duplicate detection
- Audio WAV validation and header verification
- Asynchronous AI trigger and background execution
- MongoDB document persistence and speaker/assignee independence
- Meeting failure handling and error logging
- WebSocket status broadcasts
- Duplicate processing idempotency
- All REST endpoints (`/transcript`, `/actions`, `/decisions`, `/result`)

### Real AI Audio Integration Test
Runs actual OpenAI Whisper model transcription against `ai/data/dummy_meeting.wav`:
```bash
python backend/tests/test_real_ai_integration.py
```
> [!NOTE]
> If `HF_TOKEN` is exported in the environment, this test will also execute online Pyannote Diart model weights from Hugging Face. Otherwise, it uses reference diarization segments to test the full Whisper STT, speaker alignment, NLP extraction, and MongoDB storage flow.
