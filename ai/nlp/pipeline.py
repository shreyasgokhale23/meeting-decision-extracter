"""Main NLP Processing Pipeline for Meeting Decision Extractor."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from models.schemas import (
    ActionItem,
    DecisionItem,
    MeetingOutput,
    SentenceType,
    TaskStatus,
    TranscriptSegmentInput,
    TranscriptSegmentOutput,
)
from nlp.action_extractor import extract_action
from nlp.adapters import from_dict
from nlp.classifier import classify_sentence
from nlp.deadline_extractor import extract_deadline
from nlp.decision_extractor import extract_decision
from nlp.normalizer import clean_text
from nlp.person_extractor import extract_responsible_person
from nlp.status_extractor import extract_status

logger = logging.getLogger(__name__)


def process_segment(
    segment: Union[TranscriptSegmentInput, Dict[str, Any]],
    reference_datetime: Optional[datetime] = None,
) -> TranscriptSegmentOutput:
    """Process a single transcript segment and extract structured intelligence.
    
    Args:
        segment: TranscriptSegmentInput instance or dict conforming to schema.
        reference_datetime: Optional datetime for resolving relative deadlines.
        
    Returns:
        Structured TranscriptSegmentOutput.
    """
    try:
        if isinstance(segment, dict):
            inp = from_dict(segment)
        else:
            inp = segment
    except Exception as e:
        logger.error(f"Input validation error for segment {segment}: {e}")
        return TranscriptSegmentOutput(
            text=str(segment.get("text", "") if isinstance(segment, dict) else getattr(segment, "text", "")),
            type=SentenceType.UNKNOWN,
            status=TaskStatus.UNKNOWN,
        )

    raw_text = inp.text or ""
    cleaned = clean_text(raw_text)

    if not cleaned:
        return TranscriptSegmentOutput(
            meeting_id=inp.meeting_id,
            speaker_id=inp.speaker_id,
            speaker=inp.speaker,
            timestamp=inp.timestamp,
            text=raw_text,
            type=SentenceType.UNKNOWN,
            status=TaskStatus.UNKNOWN,
        )

    sentence_type = classify_sentence(cleaned)

    action: Optional[str] = None
    responsible_person: Optional[str] = None
    deadline_text: Optional[str] = None
    deadline_normalized: Optional[str] = None
    decision: Optional[str] = None
    status: TaskStatus = TaskStatus.UNKNOWN

    if sentence_type == SentenceType.ACTION:
        action = extract_action(cleaned)
        responsible_person = extract_responsible_person(
            cleaned,
            speaker=inp.speaker,
            speaker_id=inp.speaker_id,
        )
        deadline_text, deadline_normalized = extract_deadline(
            cleaned,
            reference_datetime=reference_datetime,
        )
        status = extract_status(cleaned, SentenceType.ACTION)

    elif sentence_type == SentenceType.DECISION:
        decision = extract_decision(cleaned)
        status = TaskStatus.UNKNOWN

    elif sentence_type == SentenceType.UNKNOWN:
        # All extractions remain None/Unknown
        status = TaskStatus.UNKNOWN

    else:
        # DISCUSSION / INFORMATION
        status = TaskStatus.UNKNOWN

    return TranscriptSegmentOutput(
        meeting_id=inp.meeting_id,
        speaker_id=inp.speaker_id,
        speaker=inp.speaker,
        timestamp=inp.timestamp,
        text=raw_text,
        type=sentence_type,
        action=action,
        responsible_person=responsible_person,
        deadline_text=deadline_text,
        deadline_normalized=deadline_normalized,
        decision=decision,
        status=status,
    )


def process_transcript(
    segments: List[Union[TranscriptSegmentInput, Dict[str, Any]]],
    meeting_id: Optional[str] = None,
    reference_datetime: Optional[datetime] = None,
) -> MeetingOutput:
    """Process an entire meeting transcript session in batch.
    
    Aggregates action items, decisions, and participants into a complete MeetingOutput.
    
    Args:
        segments: List of transcript segment inputs or dicts.
        meeting_id: Optional meeting ID if not specified on individual segments.
        reference_datetime: Base date for relative deadline resolution.
        
    Returns:
        Aggregated MeetingOutput.
    """
    meeting_id_resolved = meeting_id
    processed_turns: List[TranscriptSegmentOutput] = []
    action_items: List[ActionItem] = []
    decisions: List[DecisionItem] = []
    participants_set = set()

    for seg in segments:
        try:
            out = process_segment(seg, reference_datetime=reference_datetime)
            if not meeting_id_resolved and out.meeting_id:
                meeting_id_resolved = out.meeting_id

            if out.speaker:
                participants_set.add(out.speaker)
            elif out.speaker_id:
                participants_set.add(out.speaker_id)

            processed_turns.append(out)

            # Aggregate action items
            if out.type == SentenceType.ACTION and out.action:
                action_items.append(
                    ActionItem(
                        action=out.action,
                        responsible_person=out.responsible_person,
                        deadline=out.deadline_normalized or out.deadline_text,
                        status=out.status,
                    )
                )

            # Aggregate decisions
            elif out.type == SentenceType.DECISION and out.decision:
                decisions.append(
                    DecisionItem(
                        decision=out.decision,
                        speaker=out.speaker or out.speaker_id,
                    )
                )
        except Exception as e:
            logger.error(f"Error processing segment {seg}: {e}", exc_info=True)

    return MeetingOutput(
        meeting_id=meeting_id_resolved,
        action_items=action_items,
        decisions=decisions,
        transcript=processed_turns,
        participants=sorted(list(participants_set)),
    )
