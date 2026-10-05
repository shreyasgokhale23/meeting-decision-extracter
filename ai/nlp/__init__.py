"""NLP processing package exports."""

from .pipeline import process_segment, process_transcript
from .adapters import (
    from_csv_row,
    from_text_line,
    from_whisper_turn,
    from_diart_whisper,
    from_dict,
)
from .classifier import RuleBasedClassifier, classify_sentence
from .action_extractor import extract_action
from .person_extractor import extract_responsible_person
from .deadline_extractor import extract_deadline
from .decision_extractor import extract_decision
from .status_extractor import extract_status
from .normalizer import clean_text, split_into_sentences, remove_meeting_item_reference

__all__ = [
    "process_segment",
    "process_transcript",
    "from_csv_row",
    "from_text_line",
    "from_whisper_turn",
    "from_diart_whisper",
    "from_dict",
    "RuleBasedClassifier",
    "classify_sentence",
    "extract_action",
    "extract_responsible_person",
    "extract_deadline",
    "extract_decision",
    "extract_status",
    "clean_text",
    "split_into_sentences",
    "remove_meeting_item_reference",
]
