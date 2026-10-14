"""Tests for responsible person extractor."""

import pytest
from nlp.person_extractor import extract_responsible_person


def test_first_person_person_extraction():
    text = "I'll complete the backend by Friday."
    person = extract_responsible_person(text, speaker="Rahul", speaker_id="SPEAKER_00")
    assert person == "Rahul"

    # Fallback to speaker_id if speaker name is None
    person_id = extract_responsible_person(text, speaker=None, speaker_id="SPEAKER_00")
    assert person_id == "SPEAKER_00"


def test_third_person_person_extraction():
    assert extract_responsible_person("Sneha will prepare the dashboard UI tomorrow.") == "Sneha"
    assert extract_responsible_person("Let's have Neha update the API documentation by Friday.") == "Neha"
    assert extract_responsible_person("We need Amit to review the test cases by Friday.") == "Amit"
    assert extract_responsible_person("Amit is responsible for update the database schema today.") == "Amit"
    assert extract_responsible_person("Please ask Riya to configure the development server.") == "Riya"
    assert extract_responsible_person("Rahul needs to complete the data preprocessing before Monday.") == "Rahul"


def test_delegation_person_extraction():
    text = "Manager: Rahul should complete the backend by Friday."
    # Speaker is Manager, but responsible person should be extracted as Rahul
    person = extract_responsible_person(text, speaker="Manager")
    assert person == "Rahul"

    text2 = "Team Lead: Priya will design the database tables."
    assert extract_responsible_person(text2, speaker="Team Lead") == "Priya"


def test_no_person():
    assert extract_responsible_person("") is None
    assert extract_responsible_person("Can everyone hear me?") is None
    assert extract_responsible_person("The API currently supports five endpoints.") is None
