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

## 10. Speech-to-Text with OpenAI Whisper

Whisper provides accurate speech-to-text transcription with timestamped intervals:
- **Responsibility**: *WHAT* was said and *WHEN* (start/end in seconds).
- **Implementation**: [`ai/transcription/whisper_provider.py`](file:///e:/meeting-decision-extracter/ai/transcription/whisper_provider.py) implements `TranscriptionProvider`.
- **Default Model**: `'base'` (configurable to `'tiny'`, `'small'`, `'medium'`, `'large'`).

### Usage
```python
from transcription.whisper_provider import WhisperProvider

provider = WhisperProvider("base")
turns = provider.transcribe("data/dummy_meeting.wav")
for turn in turns:
    print(f"[{turn.start:.2f}s - {turn.end:.2f}s]: {turn.text}")
```

---

## 11. Speaker Diarization with Diart (3-Speaker Configuration)

Diart provides real-time online speaker diarization built on Pyannote Audio:
- **Responsibility**: *WHO* was speaking and *WHEN* (start/end in seconds).
- **Implementation**: [`ai/diarization/diart_provider.py`](file:///e:/meeting-decision-extracter/ai/diarization/diart_provider.py) implements `DiarizationProvider`.
- **3-Speaker Resolution**: Diart's default clustering threshold (`delta_new=1.0`) is too permissive for clean or synthetic voices, merging the third voice into an existing cluster. Configuring `SpeakerDiarizationConfig(delta_new=0.8)` separates all 3 distinct speakers (`speaker0`, `speaker1`, `speaker2`) reliably without artificial segmentation.

```python
from diart.blocks import SpeakerDiarizationConfig
import diart

config = SpeakerDiarizationConfig(delta_new=0.8)
pipeline = diart.SpeakerDiarization(config=config)
```

### Hugging Face Setup (Required for Diart Model Weights)
1. Accept model terms on Hugging Face:
   - [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
   - [pyannote/embedding](https://huggingface.co/pyannote/embedding)
2. Generate an Access Token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Set your token environment variable:
   ```bash
   # Windows PowerShell
   $env:HF_TOKEN = "hf_your_token_here"
   # Linux / macOS
   export HF_TOKEN="hf_your_token_here"
   ```

### PyTorch Compatibility & Safe Globals
PyTorch 2.6+ defaults `torch.load` to `weights_only=True`. The `DiartProvider` automatically allowlists `omegaconf.listconfig.ListConfig` and `torch.serialization.add_safe_globals` at startup to ensure seamless checkpoint loading without disabling PyTorch security.

---

## 12. Speaker-Aware Transcript Alignment

Connects *WHO* spoke (Diart) with *WHAT* was transcribed (Whisper):
- **Implementation**: [`ai/transcription/speaker_aware.py`](file:///e:/meeting-decision-extracter/ai/transcription/speaker_aware.py)
- **Algorithm**: Maximum Temporal Overlap
  1. For each Whisper turn `[t_start, t_end]`, compute overlap against all Diart segments.
  2. Accumulate overlap durations per speaker cluster.
  3. Assign the speaker with the maximum accumulated overlap.
  4. If a Whisper turn occurs in silence with zero overlap, assigns `'UNKNOWN'`.
  5. Preserves Whisper transcribed text and timestamps verbatim.

```python
from transcription.speaker_aware import align_transcript_with_speakers

aligned_turns = align_transcript_with_speakers(
    whisper_turns=whisper_turns,
    diarization_segments=diar_segments,
)
```

---

## 13. End-to-End Audio Processing Orchestrator

The top-level orchestrator connects the entire multi-modal pipeline:
```
Audio File (WAV)
       │
  ┌────┴────┐
  │         │
Diart    Whisper
  │         │
  └────┬────┘
       ▼
Speaker Alignment
       ▼
Speaker-Aware Transcript
       ▼
Existing NLP Pipeline
       ▼
Structured AudioMeetingResult
```

- **Implementation**: [`ai/transcription/audio_processor.py`](file:///e:/meeting-decision-extracter/ai/transcription/audio_processor.py)
- **Convenience API**: `process_audio(audio_path, meeting_id=None, reference_datetime=None)`
- **Output**: Returns schema-validated `AudioMeetingResult` with:
  - `speaker_transcript`: List of `SpeakerTranscriptTurn` (speaker, start, end, text)
  - `action_items`: Extracted actions with assignees, deadlines, and statuses
  - `decisions`: Extracted meeting decisions with speakers
  - `participants`: Set of detected meeting participants

### Programmatic Usage
```python
from transcription.audio_processor import process_audio

result = process_audio("data/dummy_meeting.wav", meeting_id="meeting_001")
print(result.model_dump_json(indent=2))
```

---

## 14. Testing & Verification

### Fast Offline Unit Tests (Mocked Inference, No Model Downloads)
```bash
python -m pytest tests/ -v
```
Runs the full suite of **65 unit tests** covering sentence classification, action/person/deadline/decision/status extractors, adapters, benchmark evaluation, Diart provider mocks, speaker-aware alignment, and audio processor orchestration.

### Manual Real-Audio Verification Tests
1. **Whisper Transcription**:
   ```bash
   python test_manual_whisper.py
   ```
2. **Diart 3-Speaker Diarization**:
   ```bash
   python test_manual_diarization.py
   ```
3. **End-to-End Audio Pipeline**:
   ```bash
   python test_manual_audio_pipeline.py
   ```

---

## 15. Integration with FastAPI Backend

Integrating into a future FastAPI backend is straightforward because `process_audio`, `process_segment`, and `process_transcript` return standard Pydantic models:

```python
from fastapi import FastAPI, UploadFile, File
from transcription.audio_processor import process_audio
from models.schemas import AudioMeetingResult

app = FastAPI(title="Meeting Decision Extractor API")

@app.post("/api/v1/process-audio", response_model=AudioMeetingResult)
async def api_process_audio(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    return process_audio(temp_path)
```

---

## 16. Known Limitations

1. **CPU Inference Latency**: Running Whisper and Diart simultaneously on CPU takes ~40-60 seconds for a ~2-minute audio file. GPU acceleration (CUDA) is recommended for production deployments.
2. **Speaker Turn Granularity**: When two speakers talk simultaneously or rapid turn-taking occurs within a single Whisper sentence turn, the sentence is assigned to the dominant speaker based on maximum temporal overlap.
3. **Online Clustering Cluster IDs**: Diart assigns speaker labels (`speaker0`, `speaker1`, ...) based on chronological appearance in the stream. Mapping cluster labels to human names (e.g. "Rahul", "Sneha") occurs downstream via NLP assignee resolution or participant matching.

