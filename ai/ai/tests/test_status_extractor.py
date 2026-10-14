"""Tests for task status extractor."""

import pytest
from models.schemas import SentenceType, TaskStatus
from nlp.status_extractor import extract_status


def test_status_pending():
    assert extract_status("I'll complete the backend tomorrow.", SentenceType.ACTION) == TaskStatus.PENDING
    assert extract_status("Sneha will prepare the dashboard UI.", SentenceType.ACTION) == TaskStatus.PENDING
    assert extract_status("Rahul needs to upload the project files.", SentenceType.ACTION) == TaskStatus.PENDING


def test_status_in_progress():
    assert extract_status("The UI is currently being developed.", SentenceType.ACTION) == TaskStatus.IN_PROGRESS
    assert extract_status("Neha agreed to update the API documentation.", SentenceType.ACTION) == TaskStatus.IN_PROGRESS
    assert extract_status("Sneha should run the performance tests.", SentenceType.ACTION) == TaskStatus.IN_PROGRESS
    assert extract_status("Let's have Neha update the API documentation.", SentenceType.ACTION) == TaskStatus.IN_PROGRESS


def test_status_completed():
    assert extract_status("The backend is completed.", SentenceType.ACTION) == TaskStatus.COMPLETED
    assert extract_status("Rahul completed the authentication module.", SentenceType.ACTION) == TaskStatus.COMPLETED
    assert extract_status("The feature has been implemented and already deployed.", SentenceType.ACTION) == TaskStatus.COMPLETED


def test_status_blocked():
    assert extract_status("The task is blocked because the API is unavailable.", SentenceType.ACTION) == TaskStatus.BLOCKED
    assert extract_status("We are stuck waiting on database credentials.", SentenceType.ACTION) == TaskStatus.BLOCKED


def test_status_cancelled():
    assert extract_status("The legacy migration was cancelled.", SentenceType.ACTION) == TaskStatus.CANCELLED
    assert extract_status("The old reporting feature was dropped.", SentenceType.ACTION) == TaskStatus.CANCELLED


def test_status_unknown_for_non_actions():
    assert extract_status("Let's use Redis for caching.", SentenceType.DECISION) == TaskStatus.UNKNOWN
    assert extract_status("The API supports 5 endpoints.", SentenceType.INFORMATION) == TaskStatus.UNKNOWN
    assert extract_status("Can everyone hear me?", SentenceType.UNKNOWN) == TaskStatus.UNKNOWN
