"""Responsible person extraction module distinguishing speaker from assigned individuals."""

import re
from typing import Optional


def extract_responsible_person(
    text: str,
    speaker: Optional[str] = None,
    speaker_id: Optional[str] = None,
) -> Optional[str]:
    """Identify the responsible person for an action item.
    
    Distinguishes:
    1. First-person assignment: 'I'll complete the backend' -> returns speaker/speaker_id.
    2. Delegation: 'Manager: Rahul should finish...' -> returns 'Rahul'.
    3. Named third-person: 'Sneha will prepare...' -> returns 'Sneha'.
    4. Unspecified or impersonal: returns None.
    """
    if not text:
        return None

    cleaned = text.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()

    # 1. Check delegation prefix: e.g. 'Manager: Rahul should...' or 'Lead: Priya will...'
    delegation_match = re.match(r"^[A-Za-z0-9_\s]+:\s*(.+)$", cleaned)
    if delegation_match:
        content_after_speaker = delegation_match.group(1).strip()
        # Recursively extract person from the inner content
        return extract_responsible_person(content_after_speaker, speaker=speaker, speaker_id=speaker_id)

    # 2. Check first-person patterns (I'll, I will, I am going to, My task is)
    first_person_pattern = re.compile(r"\b(?:i['’]ll|i\s+will|i['’]m\s+going\s+to|i\s+am\s+going\s+to|my\s+task\s+is)\b", re.IGNORECASE)
    if first_person_pattern.search(cleaned):
        # Resolve to current speaker if known, otherwise fallback to speaker_id
        if speaker and speaker.strip():
            return speaker.strip()
        if speaker_id and speaker_id.strip():
            return speaker_id.strip()
        return None

    # 3. Third-person patterns
    # Pattern A: 'Let's have <Person> ...', 'Please ask <Person> to ...', 'We need <Person> to ...'
    assignment_prefix = re.match(r"^(?:let['’]s\s+have|please\s+ask|we\s+need)\s+([A-Z][a-zA-Z0-9_]+)\b", cleaned, re.IGNORECASE)
    if assignment_prefix:
        name = assignment_prefix.group(1).strip()
        if name.lower() not in {"someone", "anyone", "everyone", "the", "a", "an", "all", "our"}:
            return name

    # Pattern B: '<Person> will ...', '<Person> needs to ...', '<Person> has to ...', '<Person> should ...'
    # '<Person> is responsible for ...', '<Person> agreed to ...', '<Person> said they will ...'
    named_action = re.match(
        r"^([A-Z][a-zA-Z0-9_]+)\s+(?:will|needs?\s+to|has\s+to|should|must|agreed\s+to|said\s+they\s+will|is\s+responsible\s+for|completed|finished|was\s+assigned\s+to|is\s+currently\s+working\s+on)\b",
        cleaned
    )
    if named_action:
        name = named_action.group(1).strip()
        if name.lower() not in {"we", "they", "it", "this", "that", "there", "the", "someone", "everyone"}:
            return name

    return None
