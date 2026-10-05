"""Tests for action extractor module."""

import pytest
from nlp.action_extractor import extract_action


def test_first_person_action():
    text = "I'll complete the backend by Friday."
    action = extract_action(text)
    assert action == "Complete the backend"

    text2 = "I will prepare the UI tomorrow."
    assert extract_action(text2) == "Prepare the UI"


def test_third_person_assignment_action():
    text = "Sneha will prepare the dashboard UI tomorrow, as discussed in meeting item 124."
    action = extract_action(text)
    assert action == "Prepare the dashboard UI"

    text2 = "Arjun will create the presentation within three days."
    assert extract_action(text2) == "Create the presentation"


def test_delegation_action():
    text = "Manager: Rahul should complete the backend by Friday."
    action = extract_action(text)
    assert action == "Complete the backend"


def test_auxiliary_patterns_action():
    assert extract_action("Let's have Neha update the API documentation by Friday.") == "Update the API documentation"
    assert extract_action("We need Amit to review the test cases by Friday.") == "Review the test cases"
    assert extract_action("Amit is responsible for update the database schema today.") == "Update the database schema"
    assert extract_action("Please ask Riya to configure the development server before the final review.") == "Configure the development server"
    assert extract_action("Rahul needs to complete the data preprocessing before Monday.") == "Complete the data preprocessing"


def test_edge_cases_action():
    assert extract_action("") is None
    assert extract_action("   ") is None
