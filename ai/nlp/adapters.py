"""Universal Input Adapters.

Converts all transcript sources into TranscriptSegmentInput objects
prior to NLP processing, guaranteeing identical pipeline behavior across:
- CSV datasets
- Plain text transcript logs
- Whisper speech-to-text turns
- Diart diarization + Whisper merged segments
- Live microphone streaming chunks
- FastAPI JSON request payloads
"""

import re
from typing import Any, Dict, Optional
from models.schemas import TranscriptSegmentInput


def from_dict(data: Dict[str, Any]) -> TranscriptSegmentInput:
    """Convert a generic dictionary (e.g., FastAPI request payload) into TranscriptSegmentInput."""
    return TranscriptSegmentInput.model_validate(data)


def from_csv_row(row: Dict[str, Any]) -> TranscriptSegmentInput:
    """Convert a CSV dictionary row into a validated TranscriptSegmentInput."""
    return TranscriptSegmentInput(
        meeting_id=row.get("meeting_id") or None,
        speaker_id=row.get("speaker_id") or None,
        speaker=row.get("speaker") or None,
        timestamp=str(row.get("timestamp")) if row.get("timestamp") else None,
        text=str(row.get("text", "")).strip(),
    )


def from_text_line(
    line: str,
    default_meeting_id: Optional[str] = None,
    default_timestamp: Optional[str] = None,
) -> TranscriptSegmentInput:
    """Parse a single transcript text line into TranscriptSegmentInput.
    
    Handles:
    - "[10:15:23] Rahul: I'll complete the backend by Friday."
    - "SPEAKER_00: Let's use Redis for caching."
    - "Manager: Rahul should finish the API."
    - Plain raw text without prefix.
    """
    raw = line.strip()
    if not raw:
        return TranscriptSegmentInput(text="", meeting_id=default_meeting_id)

    timestamp = default_timestamp
    speaker = None
    speaker_id = None
    text = raw

    # 1. Extract bracketed timestamp: e.g. [10:15:23] or (10:15)
    time_match = re.match(r"^\[([0-9:]+)\]\s*(.*)$", text)
    if time_match:
        timestamp = time_match.group(1).strip()
        text = time_match.group(2).strip()

    # 2. Extract Speaker: text
    speaker_match = re.match(r"^([A-Za-z0-9_\s]+?):\s*(.+)$", text)
    if speaker_match:
        cand_speaker = speaker_match.group(1).strip()
        text = speaker_match.group(2).strip()
        if cand_speaker.upper().startswith("SPEAKER_"):
            speaker_id = cand_speaker
            speaker = cand_speaker
        else:
            speaker = cand_speaker

    return TranscriptSegmentInput(
        meeting_id=default_meeting_id,
        speaker_id=speaker_id,
        speaker=speaker,
        timestamp=timestamp,
        text=text,
    )


def from_whisper_turn(
    turn: Dict[str, Any],
    speaker: Optional[str] = None,
    speaker_id: Optional[str] = None,
    meeting_id: Optional[str] = None,
) -> TranscriptSegmentInput:
    """Convert an OpenAI Whisper STT output turn into TranscriptSegmentInput.
    
    Expected turn schema: {"text": "...", "start": 0.0, "end": 4.2}
    """
    start_sec = turn.get("start", 0.0)
    # Format seconds into HH:MM:SS
    hours = int(start_sec // 3600)
    minutes = int((start_sec % 3600) // 60)
    seconds = int(start_sec % 60)
    formatted_ts = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return TranscriptSegmentInput(
        meeting_id=meeting_id,
        speaker_id=speaker_id or turn.get("speaker_id"),
        speaker=speaker or turn.get("speaker"),
        timestamp=formatted_ts,
        text=str(turn.get("text", "")).strip(),
    )


def from_diart_whisper(
    diar_segment: Dict[str, Any],
    whisper_text: str,
    meeting_id: Optional[str] = None,
) -> TranscriptSegmentInput:
    """Convert a Diart speaker diarization segment paired with Whisper transcript text.
    
    Expected diar_segment: {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.2}
    """
    speaker_id = diar_segment.get("speaker", "SPEAKER_00")
    start_sec = diar_segment.get("start", 0.0)
    hours = int(start_sec // 3600)
    minutes = int((start_sec % 3600) // 60)
    seconds = int(start_sec % 60)
    formatted_ts = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return TranscriptSegmentInput(
        meeting_id=meeting_id,
        speaker_id=speaker_id,
        speaker=speaker_id,
        timestamp=formatted_ts,
        text=whisper_text.strip(),
    )
