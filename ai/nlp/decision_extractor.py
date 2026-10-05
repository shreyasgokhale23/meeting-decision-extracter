"""Decision extraction and normalization module."""

import re
from typing import Optional
from nlp.normalizer import clean_text, remove_meeting_item_reference


def extract_decision(text: str) -> Optional[str]:
    """Extract and normalize a concise decision statement from an utterance.
    
    Examples:
        "We agreed that FastAPI should handle the backend."
        -> "Use FastAPI for the backend"
        
        "Let's use Redis for caching."
        -> "Use Redis for caching"
        
        "Everyone agreed that Redis is the best choice for caching."
        -> "Use Redis for caching"
        
        "We have decided that PostgreSQL will be used for relational data storage."
        -> "Use PostgreSQL for relational data storage"
        
        "The team agreed to deploy on AWS."
        -> "Deploy on AWS"
        
        "The team selected Docker for deployment."
        -> "Use Docker for deployment"
    """
    if not text:
        return None

    cleaned = clean_text(text)
    cleaned = remove_meeting_item_reference(cleaned)

    # Specific canonical pattern rewrites:
    # 1. "<Tech> should handle <purpose>" e.g. "We agreed that FastAPI should handle the backend."
    m_handle = re.search(r"\b([A-Za-z0-9\.\-_/ ]+?)\s+should\s+handle\s+(.+)$", cleaned, re.IGNORECASE)
    if m_handle:
        tech = m_handle.group(1).strip()
        purpose = m_handle.group(2).strip().rstrip(".")
        # remove leading 'that' if present
        tech = re.sub(r"^(?:that|we\s+agreed\s+that)\s+", "", tech, flags=re.IGNORECASE).strip()
        return f"Use {tech} for {purpose}"

    # 2. "<Tech> is the best choice for <purpose>"
    m_best = re.search(r"\b([A-Za-z0-9\.\-_/ ]+?)\s+is\s+the\s+best\s+choice\s+for\s+(.+)$", cleaned, re.IGNORECASE)
    if m_best:
        tech = m_best.group(1).strip()
        purpose = m_best.group(2).strip().rstrip(".")
        tech = re.sub(r"^(?:that|everyone\s+agreed\s+that)\s+", "", tech, flags=re.IGNORECASE).strip()
        return f"Use {tech} for {purpose}"

    # 3. "<Tech> will be used for <purpose>"
    m_used = re.search(r"\b([A-Za-z0-9\.\-_/ ]+?)\s+will\s+be\s+used\s+for\s+(.+)$", cleaned, re.IGNORECASE)
    if m_used:
        tech = m_used.group(1).strip()
        purpose = m_used.group(2).strip().rstrip(".")
        tech = re.sub(r"^(?:that|we\s+have\s+decided\s+that)\s+", "", tech, flags=re.IGNORECASE).strip()
        return f"Use {tech} for {purpose}"

    # 4. "The team selected <Tech> for <purpose>"
    m_select = re.search(r"\bthe\s+team\s+selected\s+([A-Za-z0-9\.\-_/ ]+?)\s+for\s+(.+)$", cleaned, re.IGNORECASE)
    if m_select:
        tech = m_select.group(1).strip()
        purpose = m_select.group(2).strip().rstrip(".")
        return f"Use {tech} for {purpose}"

    # 5. General prefix stripping:
    # "Let's use ...", "We decided to ...", "We will use ...", "The final decision is to ..."
    prefix_patterns = [
        r"^(?:the\s+final\s+decision\s+is\s+to\s+)",
        r"^(?:we\s+(?:have\s+)?decided\s+to\s+)",
        r"^(?:the\s+team\s+agreed\s+to\s+)",
        r"^(?:we\s+agreed\s+to\s+)",
        r"^(?:the\s+project\s+will\s+use\s+)",
        r"^(?:we\s+will\s+use\s+)",
        r"^(?:let['’]s\s+use\s+)",
        r"^(?:let['’]s\s+go\s+with\s+)",
    ]

    for pattern in prefix_patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            # If the pattern was "the project will use" or "we will use" or "let's use", prefix with "Use "
            tail = cleaned[match.end():].strip().rstrip(".")
            if "use" in pattern:
                return f"Use {tail}"
            return tail[0].upper() + tail[1:]

    # Fallback: clean trailing period
    cleaned = cleaned.rstrip(".")
    return cleaned[0].upper() + cleaned[1:] if cleaned else None
