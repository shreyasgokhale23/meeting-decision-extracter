"""Data models and Pydantic schemas for the Meeting Decision Extractor AI/NLP module."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SentenceType(str, Enum):
    """Categorical classification of meeting utterances."""
    ACTION = "ACTION"
    DECISION = "DECISION"
    DISCUSSION = "DISCUSSION"
    INFORMATION = "INFORMATION"
    UNKNOWN = "UNKNOWN"


class TaskStatus(str, Enum):
    """Execution status for extracted action items."""
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    CANCELLED = "Cancelled"
    BLOCKED = "Blocked"
    UNKNOWN = "Unknown"


class TranscriptSegmentInput(BaseModel):
    """Normalized input segment accepted by the NLP pipeline.
    
    All ingestion sources (CSV, plain text, Whisper, Diart, FastAPI)
    must convert into this format prior to NLP processing.
    """
    meeting_id: Optional[str] = Field(default=None, description="Unique identifier for the meeting session")
    speaker_id: Optional[str] = Field(default=None, description="Diarization cluster tag, e.g. SPEAKER_00")
    speaker: Optional[str] = Field(default=None, description="Human-readable speaker name if resolved, e.g. 'Rahul'")
    timestamp: Optional[str] = Field(default=None, description="Time offset or HH:MM:SS string")
    text: str = Field(..., min_length=1, description="Raw transcription text of the utterance")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class TranscriptSegmentOutput(BaseModel):
    """Structured output returned for each processed segment."""
    meeting_id: Optional[str] = None
    speaker_id: Optional[str] = None
    speaker: Optional[str] = None
    timestamp: Optional[str] = None
    text: str
    type: SentenceType
    action: Optional[str] = None
    responsible_person: Optional[str] = None
    deadline_text: Optional[str] = None
    deadline_normalized: Optional[str] = None
    decision: Optional[str] = None
    status: TaskStatus = TaskStatus.UNKNOWN

    model_config = ConfigDict(extra="ignore", use_enum_values=True)


class ActionItem(BaseModel):
    """Meeting-level aggregated action item."""
    action: str
    responsible_person: Optional[str] = None
    deadline: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING

    model_config = ConfigDict(use_enum_values=True)


class DecisionItem(BaseModel):
    """Meeting-level aggregated decision item."""
    decision: str
    speaker: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True)


class MeetingOutput(BaseModel):
    """Comprehensive meeting summary output schema."""
    meeting_id: Optional[str] = None
    action_items: List[ActionItem] = Field(default_factory=list)
    decisions: List[DecisionItem] = Field(default_factory=list)
    transcript: List[TranscriptSegmentOutput] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)
