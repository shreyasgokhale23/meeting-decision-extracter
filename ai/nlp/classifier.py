"""Classification module for categorizing meeting utterances into sentence types."""

import re
from abc import ABC, abstractmethod
from typing import Optional
from models.schemas import SentenceType
from nlp.normalizer import clean_text, remove_meeting_item_reference


class BaseClassifier(ABC):
    """Abstract interface for sentence classification models."""

    @abstractmethod
    def classify(self, text: str) -> SentenceType:
        """Classify input text into a SentenceType."""
        pass


class RuleBasedClassifier(BaseClassifier):
    """Hybrid rule-and-pattern sentence classifier.
    
    Categorizes utterances into:
    - ACTION: Commitments, assignments, imperative tasks.
    - DECISION: Architectural and consensus choices.
    - DISCUSSION: Debated topics and meeting focus areas.
    - INFORMATION: Factual status reports, system specs.
    - UNKNOWN: Audio checks, casual commentary, unclassifiable chatter.
    """

    def __init__(self) -> None:
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        # Decision patterns
        self.decision_patterns = [
            re.compile(r"\b(?:we\s+(?:have\s+)?decided\s+to|we\s+decided\s+that|let['’]s\s+use|let['’]s\s+go\s+with)\b", re.IGNORECASE),
            re.compile(r"\b(?:the\s+team\s+agreed\s+to|everyone\s+agreed\s+that|we\s+agreed\s+that)\b", re.IGNORECASE),
            re.compile(r"\b(?:the\s+final\s+decision\s+is\s+to|the\s+team\s+selected)\b", re.IGNORECASE),
            re.compile(r"\b(?:we\s+have\s+decided\s+that|the\s+project\s+will\s+use)\b", re.IGNORECASE),
            re.compile(r"\bwe\s+will\s+use\s+[A-Za-z0-9\.\-_/ ]+\s+for\b", re.IGNORECASE),
            re.compile(r"\b[A-Za-z0-9\.\-_/ ]+\s+(?:is\s+the\s+best\s+choice\s+for|will\s+be\s+used\s+for|should\s+handle)\b", re.IGNORECASE),
        ]

        # Discussion patterns
        self.discussion_patterns = [
            re.compile(r"\b(?:the\s+team\s+discussed|we\s+discussed|there\s+was\s+a\s+discussion\s+about|we\s+spent\s+some\s+time\s+discussing|we\s+talked\s+about)\b", re.IGNORECASE),
            re.compile(r"\b(?:the\s+meeting\s+focused\s+on|the\s+meeting\s+covered|the\s+group\s+examined\s+the\s+issues\s+related\s+to)\b", re.IGNORECASE),
            re.compile(r"\b(?:the\s+participants\s+raised\s+concerns\s+about|the\s+team\s+reviewed\s+the\s+current\s+situation\s+regarding|everyone\s+shared\s+their\s+thoughts\s+about)\b", re.IGNORECASE),
        ]

        # Information patterns
        self.information_patterns = [
            re.compile(r"\b(?:the\s+team\s+was\s+informed\s+that|the\s+current\s+status\s+is\s+that|the\s+current\s+setup\s+is\s+that)\b", re.IGNORECASE),
            re.compile(r"\b(?:we\s+know\s+that|it\s+was\s+reported\s+that|the\s+team\s+noted\s+that|the\s+meeting\s+confirmed\s+that)\b", re.IGNORECASE),
            re.compile(r"\b(?:for\s+reference|the\s+project\s+information\s+shows\s+that|it\s+was\s+mentioned\s+that)\b", re.IGNORECASE),
        ]

        # Action patterns
        self.action_patterns = [
            # First person commitment: I'll, I will, We need to finish
            re.compile(r"\b(?:i['’]ll|i\s+will|i['’]m\s+going\s+to)\s+[a-z]+", re.IGNORECASE),
            # Delegation / Assignment: Let's have X, Please ask X to, We need X to
            re.compile(r"\b(?:let['’]s\s+have|please\s+ask|we\s+need)\s+[A-Za-z0-9_]+\s+(?:to\s+)?[a-z]+", re.IGNORECASE),
            # Third person modals: X will, X needs to, X has to, X should, X is responsible for
            re.compile(r"\b[A-Za-z0-9_]+\s+(?:will|needs?\s+to|has\s+to|should|must|agreed\s+to|said\s+they\s+will|is\s+responsible\s+for|completed|finished|was\s+assigned\s+to|is\s+currently\s+working\s+on)\s+[a-z]+", re.IGNORECASE),
            # General imperative actions: We need to ..., Need to ...
            re.compile(r"\b(?:we\s+need\s+to|needs?\s+to\s+be\s+done|must\s+be\s+completed|assigned\s+to)\b", re.IGNORECASE),
        ]

        # Unknown / Chitchat patterns
        self.unknown_patterns = [
            re.compile(r"\b(?:can\s+everyone\s+hear\s+me|am\s+i\s+audible|can\s+you\s+see\s+my\s+screen|mic\s+check)\b", re.IGNORECASE),
            re.compile(r"\b(?:the\s+interface\s+looks|looks\s+much\s+better|good\s+morning|good\s+afternoon|hello\s+everyone|thank\s+you\s+all)\b", re.IGNORECASE),
            re.compile(r"\b(?:the\s+meeting\s+started\s+late|sorry\s+i['’]m\s+late|bye\s+everyone|see\s+you)\b", re.IGNORECASE),
        ]

    def classify(self, text: str) -> SentenceType:
        """Classify a single sentence into a SentenceType."""
        cleaned = clean_text(text)
        if not cleaned:
            return SentenceType.UNKNOWN

        # Pre-strip delegation prefix if present, e.g. 'Manager: Rahul should...'
        delegation_match = re.match(r"^[A-Za-z0-9_\s]+:\s*(.+)$", cleaned)
        core_text = delegation_match.group(1) if delegation_match else cleaned
        core_text_clean = remove_meeting_item_reference(core_text)

        # 1. Check Unknown / Chitchat
        for pattern in self.unknown_patterns:
            if pattern.search(core_text_clean):
                return SentenceType.UNKNOWN

        # 2. Check Decision patterns (check before action so 'we decided to use' isn't mistaken for action)
        for pattern in self.decision_patterns:
            if pattern.search(core_text_clean):
                return SentenceType.DECISION

        # 3. Check Discussion patterns
        for pattern in self.discussion_patterns:
            if pattern.search(core_text_clean):
                return SentenceType.DISCUSSION

        # 4. Check Information patterns
        for pattern in self.information_patterns:
            if pattern.search(core_text_clean):
                return SentenceType.INFORMATION

        # 5. Check Action patterns
        for pattern in self.action_patterns:
            if pattern.search(core_text_clean):
                return SentenceType.ACTION

        # Default fallback
        return SentenceType.UNKNOWN


# Default global instance
default_classifier = RuleBasedClassifier()


def classify_sentence(text: str, classifier: Optional[BaseClassifier] = None) -> SentenceType:
    """Helper to classify a sentence using default or custom classifier."""
    c = classifier or default_classifier
    return c.classify(text)
