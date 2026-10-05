"""Text normalization and sentence segmentation utilities."""

import re
from typing import List


def clean_text(text: str) -> str:
    """Clean text by stripping quotes, excessive whitespace, and edge noise."""
    if not text:
        return ""
    text = text.strip()
    # Remove outer matching quotes
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_meeting_item_reference(text: str) -> str:
    """Remove boilerplate artifacts like ', as discussed in meeting item 124'."""
    pattern = r",?\s*(?:as\s+discussed\s+in\s+meeting\s+item\s+\d+|as\s+noted\s+in\s+item\s+\d+)\.?"
    cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return cleaned.strip()


def split_into_sentences(text: str) -> List[str]:
    """Segment an utterance into individual clean sentences while preserving abbreviations."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    
    # Handle common abbreviations and decimals to prevent improper splitting
    # Split on period, exclamation, or question mark followed by space or end
    # Using regex lookbehind/lookahead
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'I])", cleaned)
    
    result = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean:
            result.append(s_clean)
            
    return result if result else [cleaned]
