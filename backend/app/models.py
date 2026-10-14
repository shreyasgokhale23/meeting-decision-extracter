from datetime import UTC, date, datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SegmentType = Literal["ACTION", "DECISION", "DISCUSSION", "INFORMATION", "UNKNOWN"]
MeetingStatus = Literal["created", "processing", "completed", "failed"]


class AIResult(BaseModel):
    """The agreed AI-to-backend contract. NLP is intentionally outside this service."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    meeting_id: str = Field(min_length=1, max_length=128)
    speaker_id: str = Field(min_length=1, max_length=128)
    speaker: str = Field(min_length=1, max_length=256)
    timestamp: datetime
    text: str = Field(min_length=1, max_length=20000)
    type: SegmentType
    action: str | None = Field(default=None, max_length=4000)
    responsible_person: str | None = Field(default=None, max_length=256)
    deadline_text: str | None = Field(default=None, max_length=512)
    deadline_normalized: date | None = None
    decision: str | None = Field(default=None, max_length=4000)
    status: str | None = Field(default=None, max_length=64)

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value

    @model_validator(mode="after")
    def validate_classification_fields(self):
        if self.type == "ACTION" and not self.action:
            raise ValueError("action is required when type is ACTION")
        if self.type == "DECISION" and not self.decision:
            raise ValueError("decision is required when type is DECISION")
        if self.type != "ACTION" and any((self.action, self.responsible_person, self.deadline_text, self.deadline_normalized)):
            raise ValueError("action fields are only valid when type is ACTION")
        if self.type != "DECISION" and self.decision:
            raise ValueError("decision is only valid when type is DECISION")
        return self


class MeetingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    meeting_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=512)
    started_at: datetime | None = None

    @field_validator("started_at")
    @classmethod
    def started_at_must_have_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            return value.replace(tzinfo=UTC)
        return value


class MeetingOut(BaseModel):
    id: Optional[str] = None
    meeting_id: str
    title: str
    status: MeetingStatus = "created"
    started_at: datetime
    created_at: datetime
    ended_at: Optional[datetime] = None
    audio_path: Optional[str] = None
    participants: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class MeetingAudioUploadResponse(BaseModel):
    meeting_id: str
    status: MeetingStatus = "processing"
    message: Optional[str] = None


class MeetingResultOut(BaseModel):
    meeting: dict[str, Any]
    participants: List[str] = Field(default_factory=list)
    transcript: List[dict[str, Any]] = Field(default_factory=list)
    action_items: List[dict[str, Any]] = Field(default_factory=list)
    decisions: List[dict[str, Any]] = Field(default_factory=list)
