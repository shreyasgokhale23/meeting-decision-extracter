"""Tests for sentence classifier module."""

import pytest
from models.schemas import SentenceType
from nlp.classifier import classify_sentence


def test_action_classification():
    action_examples = [
        "I'll complete the backend by Friday.",
        "Sneha will prepare the UI tomorrow.",
        "Let's have Neha update the API documentation by Friday.",
        "We need Amit to review the test cases by Friday.",
        "Manager: Rahul should complete the backend by Friday.",
        "Arjun will create the presentation within three days, as discussed in meeting item 127.",
        "Amit is responsible for update the database schema today.",
        "Please ask Riya to configure the development server before the final review.",
    ]
    for text in action_examples:
        res = classify_sentence(text)
        assert res == SentenceType.ACTION, f"Failed for text: '{text}', got {res}"


def test_decision_classification():
    decision_examples = [
        "Let's use MongoDB for the database.",
        "We decided to use Next.js.",
        "The team agreed to deploy on AWS.",
        "We agreed that FastAPI should handle the backend.",
        "The final decision is to use WebSocket for real-time communication.",
        "Everyone agreed that Redis is the best choice for caching.",
        "We have decided that PostgreSQL will be used for relational data storage.",
        "The team selected Docker for deployment.",
    ]
    for text in decision_examples:
        res = classify_sentence(text)
        assert res == SentenceType.DECISION, f"Failed for text: '{text}', got {res}"


def test_discussion_classification():
    discussion_examples = [
        "The team discussed the API response time.",
        "We spent some time discussing the API response time.",
        "The team discussed the application security.",
        "The meeting focused on the project budget.",
        "There was a discussion about the authentication flow.",
        "The group examined the issues related to the database performance.",
        "The participants raised concerns about the testing strategy.",
    ]
    for text in discussion_examples:
        res = classify_sentence(text)
        assert res == SentenceType.DISCUSSION, f"Failed for text: '{text}', got {res}"


def test_information_classification():
    information_examples = [
        "The team was informed that the API currently supports five endpoints.",
        "The current status is that the backend is written in Python.",
        "We know that the meeting starts at ten o'clock.",
        "For reference, the current version is 2.1.",
        "The team noted that the project deadline is next month.",
        "The meeting confirmed that the testing environment is ready.",
    ]
    for text in information_examples:
        res = classify_sentence(text)
        assert res == SentenceType.INFORMATION, f"Failed for text: '{text}', got {res}"


def test_unknown_classification():
    unknown_examples = [
        "Can everyone hear me?",
        "The interface looks much better now.",
        "The meeting started late today.",
        "Am I audible to everyone in the room?",
        "Good morning everyone.",
        "Can you see my screen?",
        "Thanks everyone for joining.",
    ]
    for text in unknown_examples:
        res = classify_sentence(text)
        assert res == SentenceType.UNKNOWN, f"Failed for text: '{text}', got {res}"
