"""Tests for decision extraction and normalization."""

import pytest
from nlp.decision_extractor import extract_decision


def test_standard_decisions():
    assert extract_decision("Let's use MongoDB for the database.") == "Use MongoDB for the database"
    assert extract_decision("We decided to use Next.js for the frontend.") == "Use Next.js for the frontend"
    assert extract_decision("The team agreed to deploy on AWS.") == "Deploy on AWS"
    assert extract_decision("The final decision is to use WebSocket for real-time communication.") == "Use WebSocket for real-time communication"


def test_handled_and_best_choice_decisions():
    assert extract_decision("We agreed that FastAPI should handle the backend.") == "Use FastAPI for the backend"
    assert extract_decision("Everyone agreed that Redis is the best choice for caching.") == "Use Redis for caching"
    assert extract_decision("We have decided that PostgreSQL will be used for relational data storage.") == "Use PostgreSQL for relational data storage"
    assert extract_decision("The team selected Docker for deployment.") == "Use Docker for deployment"


def test_empty_or_none():
    assert extract_decision("") is None
    assert extract_decision("   ") is None
