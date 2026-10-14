"""Action item extractor for normalizing imperative action clauses."""

import re
from typing import Optional
from nlp.normalizer import clean_text, remove_meeting_item_reference
from nlp.deadline_extractor import COMPILED_DEADLINE_REGEX


def extract_action(text: str) -> Optional[str]:
    """Extract and normalize the imperative action description from an utterance.
    
    Examples:
        "Sneha will prepare the dashboard UI tomorrow, as discussed in meeting item 124."
        -> "Prepare the dashboard UI"
        
        "Let's have Neha update the API documentation by Friday."
        -> "Update the API documentation"
        
        "We need Amit to review the test cases by Friday."
        -> "Review the test cases"
        
        "I'll complete the backend by Friday."
        -> "Complete the backend"
    """
    if not text:
        return None

    cleaned = clean_text(text)
    # Strip boilerplate meeting reference
    cleaned = remove_meeting_item_reference(cleaned)

    # Strip delegation prefix if present (e.g. 'Manager: Rahul should...')
    delegation_match = re.match(r"^[A-Za-z0-9_\s]+:\s*(.+)$", cleaned)
    if delegation_match:
        cleaned = delegation_match.group(1).strip()

    # Regex patterns for stripping subject and auxiliary verb markers
    subject_aux_patterns = [
        # Let's have <Person> (to) <verb>...
        r"^(?:let['’]s\s+have|please\s+ask|we\s+need)\s+[A-Za-z0-9_]+\s+(?:to\s+)?",
        # <Person> is responsible for <verb>...
        r"^[A-Za-z0-9_]+\s+is\s+responsible\s+for\s+",
        # <Person> said they will / agreed to <verb>...
        r"^[A-Za-z0-9_]+\s+(?:said\s+they\s+will|agreed\s+to)\s+",
        # <Person> will / needs to / has to / should / must <verb>...
        r"^[A-Za-z0-9_]+\s+(?:will|needs?\s+to|has\s+to|should|must)\s+",
        # First-person markers: I'll, I will, We will, We need to
        r"^(?:i['’]ll|i\s+will|we\s+will|we\s+need\s+to)\s+",
        # Impersonal: Needs to / Need to
        r"^(?:needs?\s+to\s+)?",
    ]

    action_text = cleaned
    for pattern in subject_aux_patterns:
        match = re.search(pattern, action_text, flags=re.IGNORECASE)
        if match and match.start() == 0:
            action_text = action_text[match.end():].strip()
            break

    # Strip trailing deadline if present
    # Check if a deadline occurs towards the end of the action clause
    deadline_match = COMPILED_DEADLINE_REGEX.search(action_text)
    if deadline_match:
        # If deadline is towards the end, truncate it
        action_text = action_text[:deadline_match.start()].strip()

    # Clean trailing punctuation
    action_text = re.sub(r"[,;.]+$", "", action_text).strip()

    if not action_text:
        return None

    # Capitalize the first letter while preserving acronyms
    return action_text[0].upper() + action_text[1:]
