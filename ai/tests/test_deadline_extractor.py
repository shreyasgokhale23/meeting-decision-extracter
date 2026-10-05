"""Tests for deadline extraction and date normalization."""

from datetime import datetime
import pytest
from nlp.deadline_extractor import extract_deadline


@pytest.fixture
def ref_date():
    # 2026-10-05 is a Monday
    return datetime(2026, 10, 5, 10, 0, 0)


def test_relative_deadlines(ref_date):
    # Today
    text, norm = extract_deadline("I'll do it today.", reference_datetime=ref_date)
    assert text.lower() == "today"
    assert norm == "2026-10-05"

    # Tomorrow
    text, norm = extract_deadline("Sneha will prepare the UI tomorrow.", reference_datetime=ref_date)
    assert text.lower() == "tomorrow"
    assert norm == "2026-10-06"

    # By Friday
    text, norm = extract_deadline("I'll complete the backend by Friday.", reference_datetime=ref_date)
    assert text.lower() == "by friday"
    assert norm == "2026-10-09"

    # Before Monday
    text, norm = extract_deadline("Priya will update the database schema before Monday.", reference_datetime=ref_date)
    assert text.lower() == "before monday"
    assert norm == "2026-10-12"

    # Within three days
    text, norm = extract_deadline("Arjun will create the presentation within three days.", reference_datetime=ref_date)
    assert text.lower() == "within three days"
    assert norm == "2026-10-08"


def test_absolute_deadlines(ref_date):
    text, norm = extract_deadline("Complete the audit by October 15, 2026.", reference_datetime=ref_date)
    assert text == "October 15, 2026"
    assert norm == "2026-10-15"

    text2, norm2 = extract_deadline("Submit the report by 2026-10-20.", reference_datetime=ref_date)
    assert text2 == "2026-10-20"
    assert norm2 == "2026-10-20"


def test_context_dependent_deadline(ref_date):
    text, norm = extract_deadline("Please configure the server before the final review.", reference_datetime=ref_date)
    assert text.lower() == "before the final review"
    # Cannot be resolved to calendar date without external milestone schedule
    assert norm is None


def test_no_deadline(ref_date):
    text, norm = extract_deadline("Rahul will complete the backend API.", reference_datetime=ref_date)
    assert text is None
    assert norm is None
