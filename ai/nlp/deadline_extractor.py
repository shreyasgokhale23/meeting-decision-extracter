"""Deadline extraction and normalization module supporting relative, absolute, and context-dependent deadlines."""

import calendar
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
import dateparser


# Context-dependent patterns that cannot be reliably mapped to calendar dates without meeting schedule metadata
CONTEXT_DEADLINE_PATTERNS = [
    re.compile(r"\b(?:before|after|prior\s+to|by)\s+the\s+final\s+review\b", re.IGNORECASE),
    re.compile(r"\b(?:before|after|prior\s+to|by)\s+the\s+(?:release|launch|demo|presentation|milestone)\b", re.IGNORECASE),
]

# Regex patterns for identifying deadlines in utterances
DEADLINE_REGEX_PATTERNS = [
    # Context-dependent
    r"\b(?:before|prior\s+to|by)\s+the\s+final\s+review\b",
    # Relative day offsets
    r"\b(?:today|tomorrow|yesterday)\b",
    # Day names with preposition
    r"\b(?:by|before|on|until)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    # Horizon offsets
    r"\bwithin\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:days?|weeks?|months?)\b",
    r"\b(?:next\s+week|next\s+month|next\s+Monday|next\s+Friday)\b",
    r"\bby\s+(?:the\s+)?end\s+of\s+(?:the\s+)?(?:week|month)\b",
    # Absolute dates
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:,?\s+\d{4})?\b",
]

COMPILED_DEADLINE_REGEX = re.compile("|".join(f"({p})" for p in DEADLINE_REGEX_PATTERNS), re.IGNORECASE)

WORD_TO_NUM = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def extract_deadline(
    text: str,
    reference_datetime: Optional[datetime] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract deadline text and normalized YYYY-MM-DD date.
    
    Returns:
        (deadline_text, deadline_normalized)
        If context-dependent, deadline_normalized is None while deadline_text is preserved.
    """
    if not text:
        return None, None

    match = COMPILED_DEADLINE_REGEX.search(text)
    if not match:
        return None, None

    deadline_text = match.group(0).strip()

    # Check if this is a context-dependent deadline
    for ctx_pattern in CONTEXT_DEADLINE_PATTERNS:
        if ctx_pattern.search(deadline_text):
            return deadline_text, None

    ref_base = reference_datetime or datetime.now()

    # Special handling for "end of the week" -> Sunday of reference week
    if re.search(r"end\s+of\s+(?:the\s+)?week", deadline_text, re.IGNORECASE):
        days_ahead = 6 - ref_base.weekday()
        if days_ahead < 0:
            days_ahead += 7
        end_of_week = ref_base + timedelta(days=days_ahead)
        return deadline_text, end_of_week.strftime("%Y-%m-%d")

    # Special handling for "end of the month" -> Last calendar day of reference month
    if re.search(r"end\s+of\s+(?:the\s+)?month", deadline_text, re.IGNORECASE):
        last_day = calendar.monthrange(ref_base.year, ref_base.month)[1]
        end_of_month = datetime(ref_base.year, ref_base.month, last_day)
        return deadline_text, end_of_month.strftime("%Y-%m-%d")

    settings = {
        "RELATIVE_BASE": ref_base,
        "PREFER_DATES_FROM": "future",
        "RETURN_AS_TIMEZONE_AWARE": False,
    }

    # Try direct parse
    parsed_date = dateparser.parse(deadline_text, settings=settings)

    # Try transforming "within X days" -> "in X days"
    if not parsed_date and re.search(r"^within\s+", deadline_text, re.IGNORECASE):
        cand = re.sub(r"^within\s+", "in ", deadline_text, flags=re.IGNORECASE)
        for w, n in WORD_TO_NUM.items():
            cand = re.sub(rf"\b{w}\b", n, cand, flags=re.IGNORECASE)
        parsed_date = dateparser.parse(cand, settings=settings)

    # Try stripping prepositions e.g. "before Monday" -> "Monday", "by Friday" -> "Friday"
    if not parsed_date:
        cand = re.sub(r"^(?:before|by|on|until)\s+", "", deadline_text, flags=re.IGNORECASE).strip()
        parsed_date = dateparser.parse(cand, settings=settings)

    if parsed_date:
        deadline_normalized = parsed_date.strftime("%Y-%m-%d")
        return deadline_text, deadline_normalized

    return deadline_text, None
