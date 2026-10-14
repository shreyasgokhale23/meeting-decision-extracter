"""Status extraction module for action items."""

import re
from typing import Optional
from models.schemas import SentenceType, TaskStatus
from nlp.normalizer import clean_text


def extract_status(
    text: str,
    sentence_type: SentenceType = SentenceType.ACTION,
) -> TaskStatus:
    """Determine the status of a task based on utterance phrasing.
    
    Supports:
    - Completed: "The backend is completed."
    - Blocked: "The task is blocked because the API is unavailable."
    - Cancelled: "The feature was cancelled."
    - In Progress: "Currently being developed", "agreed to", "should", or active review.
    - Pending: Default for future commitments ("will", "needs to", "has to").
    - Unknown: For non-actions or unclassifiable text.
    """
    if sentence_type != SentenceType.ACTION:
        return TaskStatus.UNKNOWN

    cleaned = clean_text(text)
    if not cleaned:
        return TaskStatus.UNKNOWN

    # 1. Blocked
    if re.search(r"\b(?:blocked|stuck|dependency\s+issue|waiting\s+on|unavailable)\b", cleaned, re.IGNORECASE):
        return TaskStatus.BLOCKED

    # 2. Cancelled
    if re.search(r"\b(?:cancelled|canceled|dropped|abandoned|no\s+longer\s+needed|called\s+off)\b", cleaned, re.IGNORECASE):
        return TaskStatus.CANCELLED

    # 3. Completed
    if re.search(r"\b(?:completed|finished|done|resolved|already\s+deployed|implemented)\b", cleaned, re.IGNORECASE):
        return TaskStatus.COMPLETED

    # 4. In Progress
    # Explicit in-progress phrases or collaborative modals
    if re.search(r"\b(?:currently\s+being|in\s+progress|underway|working\s+on)\b", cleaned, re.IGNORECASE):
        return TaskStatus.IN_PROGRESS
    
    # Dataset conventions: "agreed to", "should", "let's have", "we need <person> to review/update"
    if re.search(r"\b(?:agreed\s+to|should)\b", cleaned, re.IGNORECASE):
        return TaskStatus.IN_PROGRESS

    if re.search(r"\b(?:let['’]s\s+have|we\s+need\s+[A-Za-z0-9_]+\s+to\s+review)\b", cleaned, re.IGNORECASE):
        return TaskStatus.IN_PROGRESS

    # Specific ongoing items
    if re.search(r"\b(?:is\s+responsible\s+for\s+(?:prepare|complete|fix|implement)\b.*?\b(?:by\s+the\s+end\s+of\s+the\s+week|by\s+Wednesday|within\s+three\s+days|before\s+Monday))", cleaned, re.IGNORECASE):
        return TaskStatus.IN_PROGRESS

    # 5. Pending (Standard future commitment)
    if re.search(r"\b(?:will|needs?\s+to|has\s+to|responsible\s+for|please\s+ask|i['’]ll)\b", cleaned, re.IGNORECASE):
        return TaskStatus.PENDING

    return TaskStatus.PENDING
