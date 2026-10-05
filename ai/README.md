# Meeting Decision Extractor — AI/NLP Module

Production-ready, modular AI/NLP processing engine for the **Meeting Decision Extractor** project. This module processes spoken or textual meeting transcripts in real time or batch mode, extracting actionable intelligence including action items, assignees, deadlines, decisions, and task statuses.

---

## 1. Purpose

During fast-paced team meetings, critical action items and architectural decisions are often lost in conversational transcripts. The AI/NLP module provides an automated, decoupled extraction pipeline that parses meeting utterances and produces structured, schema-validated JSON outputs ready for persistence in MongoDB and real-time dissemination to a Next.js dashboard via FastAPI and WebSockets.

---

## 2. Architecture & Design Principles

```
  Transcript Sources (CSV / Text / Whisper / Diart / FastAPI / Live Mic)
                                ↓
                 Universal Input Adapters (nlp/adapters.py)
                                ↓
                    TranscriptSegmentInput Schema
                                ↓
                 Text Normalization & Segmentation
                                ↓
              Hybrid Sentence Classifier (Rule/NLP/ML)
        ┌───────────────────────┼───────────────────────┐
        ↓                       ↓                       ↓
     [ACTION]               [DECISION]        [INFO / DISC / UNKNOWN]
        ↓                       ↓                       ↓
  Action Extractor       Decision Extractor     Pass-through & Clear
  Person Extractor       Decision Normalizer    Status = Unknown
  Deadline Extractor            ↓                       ↓
  Status Extractor              │                       │
        └───────────────────────┼───────────────────────┘
                                ↓
                   TranscriptSegmentOutput Schema
                                ↓
                   Meeting-Level Aggregation
               (ActionItems, Decisions, Participants)
```

### Key Principles
- **Decoupled Audio Layer**: The NLP core operates on clean Pydantic text models (`TranscriptSegmentInput`). Heavy STT (Whisper) and Diarization (Diart) dependencies are strictly abstracted behind interface contracts in `transcription/` and `diarization/`.
- **Universal Input Adapters**: Any data source (CSV rows, raw text lines with timestamps/speakers, Whisper turns, Diart segments, or FastAPI payloads) is adapted into `TranscriptSegmentInput` *before* hitting the NLP pipeline. You will never need to rewrite NLP code when moving from offline benchmarks to live meetings.
- **Strict Distinction between Speaker and Assignee**:
  - `speaker`: The person currently uttering the words.
  - `responsible_person`: The person accountable for completing the task (resolves first-person `"I'll..."` to speaker, third-person assignments `"Sneha will..."` to Sneha, and delegation `"Manager: Rahul should..."` to Rahul).
- **Graceful Degradation**: If an input is malformed, the pipeline catches exceptions, logs them with standard Python logging, and returns a clean `UNKNOWN` segment without terminating the stream.

---

## 3. Dependencies & Installation

### Prerequisites
- Python 3.11+ (Python 3.12 verified)
- pip

### Step-by-Step Installation
1. Clone the repository and navigate to the `ai/` directory:
   ```bash
   cd ai
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. Install minimal requirements:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Dependencies (`requirements.txt`)

Kept lightweight and free of heavy PyTorch/CUDA wheels:
- `pydantic>=2.0.0`: High-performance data validation and serialization.
- `dateparser>=1.2.0`: Relative and absolute date/time parsing.
- `pytest>=8.0.0`: Test runner and benchmark verification.
- `python-dateutil>=2.9.0`: Date arithmetic support.

---

## 5. Folder Structure

```text
ai/
│
├── data/
│   ├── dataset.csv                  # 12-column balanced benchmark dataset (150 rows)
│   ├── generate_dataset.py          # Script to generate benchmark dataset
│   ├── sample_meeting.txt           # Realistic raw multi-speaker meeting transcript
│   └── sample_transcript.json       # Diarized JSON transcript array
│
├── models/
│   ├── __init__.py                  # Model exports
│   └── schemas.py                   # Pydantic models (Input, Output, MeetingOutput, Enums)
│
├── nlp/
│   ├── __init__.py                  # NLP package public API
│   ├── adapters.py                  # Universal input adapters (CSV, Text, Whisper, Diart, FastAPI)
│   ├── normalizer.py                # Text normalization and sentence segmentation
│   ├── classifier.py                # Hybrid Rule/NLP sentence classifier (5 classes)
│   ├── action_extractor.py          # Imperative action clause extractor & normalizer
│   ├── person_extractor.py          # Speaker vs responsible_person resolution
│   ├── deadline_extractor.py        # Relative/absolute/context-dependent deadline extractor
│   ├── decision_extractor.py        # Meeting decision extractor & normalizer
│   ├── status_extractor.py          # Task status extractor (6 statuses)
│   └── pipeline.py                  # Main entrypoints (process_segment, process_transcript)
│
├── diarization/
│   ├── __init__.py
│   └── interface.py                 # Abstract base class DiarizationProvider
│
├── transcription/
│   ├── __init__.py
│   └── interface.py                 # Abstract base class TranscriptionProvider
│
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py           # Unit tests for 5-class sentence classification
│   ├── test_action_extractor.py     # Unit tests for action extraction
│   ├── test_person_extractor.py     # Unit tests for speaker & assignee resolution
│   ├── test_deadline_extractor.py   # Unit tests for relative/absolute deadlines
│   ├── test_decision_extractor.py   # Unit tests for decision statements
│   ├── test_status_extractor.py     # Unit tests for 6 task statuses
│   ├── test_adapters.py             # Unit tests for all universal input adapters
│   ├── test_pipeline.py             # End-to-end integration & edge-case tests
│   └── test_dataset.py              # 11-metric benchmark evaluation on dataset.csv
│
├── main.py                          # CLI demonstration entrypoint (Streaming + Batch)
├── requirements.txt                 # Pinned minimal dependencies
└── README.md                        # Documentation
```

---

## 6. How to Run

### Run Demo CLI
Executes both real-time streaming simulation and full-meeting batch processing:
```bash
python main.py
```

### Run Python Programmatically
```python
from datetime import datetime
from nlp import process_segment, process_transcript, from_dict

# 1. Process a single incoming live segment
segment_data = {
    "meeting_id": "meeting_001",
    "speaker": "Rahul",
    "timestamp": "10:15:23",
    "text": "I'll complete the backend by Friday."
}

result = process_segment(segment_data, reference_datetime=datetime(2026, 10, 5))
print(result.model_dump_json(indent=2))

# 2. Batch process an entire meeting
turns = [
    {"speaker": "Rahul", "text": "I'll complete the backend by Friday."},
    {"speaker": "Amit", "text": "Let's use Redis for caching."},
    {"speaker": "Sneha", "text": "The team discussed the API response time."}
]
meeting_summary = process_transcript(turns, meeting_id="meeting_001")
print(meeting_summary.model_dump_json(indent=2))
```

---

## 7. Input Formats

All inputs are converted into `TranscriptSegmentInput`:

```json
{
  "meeting_id": "meeting_001",
  "speaker_id": "SPEAKER_00",
  "speaker": "Rahul",
  "timestamp": "10:15:23",
  "text": "I'll complete the backend by Friday."
}
```

Universal input adapters support:
- Plain text line: `"[10:15:23] Rahul: I'll complete the backend by Friday."` (`from_text_line`)
- Raw delegation: `"Manager: Rahul should complete the backend."` (`from_text_line`)
- CSV row dictionary (`from_csv_row`)
- Whisper turn: `{"text": "...", "start": 0.0, "end": 4.2}` (`from_whisper_turn`)
- Diart diarization turn: `{"speaker": "SPEAKER_00", "start": 0.0, "end": 4.2}` (`from_diart_whisper`)

---

## 8. Output Formats

### Single Segment Output (`TranscriptSegmentOutput`)
```json
{
  "meeting_id": "meeting_001",
  "speaker_id": "SPEAKER_00",
  "speaker": "Rahul",
  "timestamp": "10:15:23",
  "text": "I'll complete the backend by Friday.",
  "type": "ACTION",
  "action": "Complete the backend",
  "responsible_person": "Rahul",
  "deadline_text": "by Friday",
  "deadline_normalized": "2026-10-09",
  "decision": null,
  "status": "Pending"
}
```

### Full Meeting Summary (`MeetingOutput`)
```json
{
  "meeting_id": "meeting_001",
  "action_items": [
    {
      "action": "Complete the backend",
      "responsible_person": "Rahul",
      "deadline": "2026-10-09",
      "status": "Pending"
    }
  ],
  "decisions": [
    {
      "decision": "Use Redis for caching",
      "speaker": "Amit"
    }
  ],
  "transcript": [ ... ],
  "participants": ["Amit", "Rahul", "Sneha"]
}
```

---

## 9. Testing & Quality Benchmark

Run all unit and integration tests:
```bash
python -m pytest tests/ -v
```

Run the benchmark evaluation script on `data/dataset.csv`:
```bash
python -m pytest tests/test_dataset.py -s
```
Or run directly:
```bash
python tests/test_dataset.py
```

### Benchmark Metrics Evaluated
1. Sentence Classification Accuracy
2. ACTION Precision / Recall / F1
3. DECISION Precision / Recall / F1
4. DISCUSSION Precision / Recall / F1
5. INFORMATION Precision / Recall / F1
6. UNKNOWN Precision / Recall / F1
7. Responsible-Person Extraction Accuracy
8. Deadline Extraction Accuracy
9. Deadline Normalization Accuracy
10. Decision Extraction Accuracy
11. Status Extraction Accuracy

---

## 10. Future Integration with Whisper (Speech-to-Text)

The `transcription/interface.py` file defines the `TranscriptionProvider` interface. When ready to add Whisper, implement the provider without changing any NLP code:

```python
# transcription/whisper_provider.py
from transcription.interface import TranscriptionProvider, TranscriptTurn
from nlp.adapters import from_whisper_turn
from nlp import process_segment
import whisper

class WhisperProvider(TranscriptionProvider):
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str):
        result = self.model.transcribe(audio_path)
        turns = []
        for seg in result["segments"]:
            turns.append(TranscriptTurn(text=seg["text"], start=seg["start"], end=seg["end"]))
        return turns

# Usage in pipeline:
# turn = whisper_provider.transcribe(audio_file)[0]
# segment_input = from_whisper_turn(turn.model_dump(), speaker="Rahul")
# result = process_segment(segment_input)
```

---

## 11. Future Integration with Diart (Speaker Diarization)

The `diarization/interface.py` file defines the `DiarizationProvider` interface. For live streaming diarization:

```python
# diarization/diart_provider.py
from diarization.interface import DiarizationProvider, SpeakerSegment
from nlp.adapters import from_diart_whisper
from nlp import process_segment

class DiartProvider(DiarizationProvider):
    def diarize(self, audio_path: str):
        # Initialize streaming pipeline from diart
        # yield or return list of SpeakerSegment(speaker=spk, start=t0, end=t1)
        pass

# Usage with Whisper:
# input_segment = from_diart_whisper(diar_turn.model_dump(), whisper_text="Let's use Redis.")
# result = process_segment(input_segment)
```

---

## 12. Integration with FastAPI Backend

Integrating into a FastAPI endpoint is instantaneous because `process_segment` and `process_transcript` accept standard dictionaries and Pydantic models:

```python
# server.py (FastAPI Backend)
from fastapi import FastAPI, WebSocket
from models.schemas import TranscriptSegmentInput, TranscriptSegmentOutput, MeetingOutput
from nlp import process_segment, process_transcript

app = FastAPI(title="Meeting Decision Extractor API")

@app.post("/api/v1/process-segment", response_model=TranscriptSegmentOutput)
async def api_process_segment(segment: TranscriptSegmentInput):
    """Processes a single real-time transcript segment."""
    return process_segment(segment)

@app.post("/api/v1/process-transcript", response_model=MeetingOutput)
async def api_process_transcript(payload: dict):
    """Batch processes a full meeting transcript."""
    segments = payload.get("segments", [])
    meeting_id = payload.get("meeting_id")
    return process_transcript(segments, meeting_id=meeting_id)

@app.websocket("/ws/meetings/{meeting_id}")
async def meeting_ws(websocket: WebSocket, meeting_id: str):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        data["meeting_id"] = meeting_id
        result = process_segment(data)
        # Broadcast structured extraction to connected clients / frontend dashboard
        await websocket.send_json(result.model_dump())
```
