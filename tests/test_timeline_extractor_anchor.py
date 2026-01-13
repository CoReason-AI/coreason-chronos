from datetime import datetime, timedelta, timezone

import pytest

from coreason_chronos.timeline_extractor import TimelineExtractor


@pytest.fixture  # type: ignore
def extractor() -> TimelineExtractor:
    return TimelineExtractor()


@pytest.fixture  # type: ignore
def ref_date() -> datetime:
    return datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_basic_anchor_resolution(extractor: TimelineExtractor, ref_date: datetime) -> None:
    """
    Test "X days after Y" where Y is an absolute date.
    """
    text = "Admission on 2024-01-10. Symptoms started 2 days after Admission."
    events = extractor.extract_events(text, ref_date)

    assert len(events) == 2

    # Event 1: Admission
    admission = next(e for e in events if "2024-01-10" in e.description or "2024-01-10" in e.source_snippet)
    assert admission.timestamp == datetime(2024, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Event 2: Symptoms
    symptoms = next(e for e in events if "2 days after Admission" in e.source_snippet)
    expected_date = admission.timestamp + timedelta(days=2)
    assert symptoms.timestamp == expected_date


def test_anchor_resolution_before(extractor: TimelineExtractor, ref_date: datetime) -> None:
    """
    Test "X days before Y".
    """
    text = "Surgery was on Jan 10th. Pre-op check 1 day before surgery."
    events = extractor.extract_events(text, ref_date)

    # Check absolute date
    surgery = next(e for e in events if "Jan 10" in e.description or "Jan 10" in e.source_snippet)
    # dateparser resolves Jan 10th relative to ref_date (Jan 1 2024) -> Jan 10 2024
    assert surgery.timestamp.day == 10
    assert surgery.timestamp.month == 1

    # Check relative date
    pre_op = next(e for e in events if "1 day before surgery" in e.source_snippet)
    assert pre_op.timestamp == surgery.timestamp - timedelta(days=1)


def test_complex_anchor_phrase(extractor: TimelineExtractor, ref_date: datetime) -> None:
    """
    Test anchor phrase with multiple words.
    """
    text = "The Second Infusion Date was 2024-06-10. Nausea 2 days after the Second Infusion."
    events = extractor.extract_events(text, ref_date)

    infusion = next(e for e in events if "2024-06-10" in e.source_snippet)
    nausea = next(e for e in events if "2 days after the Second Infusion" in e.source_snippet)

    assert nausea.timestamp == infusion.timestamp + timedelta(days=2)


def test_overlap_handling(extractor: TimelineExtractor, ref_date: datetime) -> None:
    """
    Test that the custom anchor logic overrides dateparser's partial match.
    dateparser might see "2 days after" as a date relative to NOW or REF_DATE.
    We want it to be relative to the anchor.
    """
    # "2 days after admission"
    # If dateparser parses "2 days after", it might yield ref_date + 2 days = Jan 3.
    # But Admission is Jan 10. So result should be Jan 12.
    text = "Admission on Jan 10. 2 days after admission."
    events = extractor.extract_events(text, ref_date)

    anchored_event = next(e for e in events if "2 days after" in e.source_snippet)

    # Should be Jan 12, not Jan 3
    assert anchored_event.timestamp.day == 12


def test_unresolved_anchor(extractor: TimelineExtractor, ref_date: datetime) -> None:
    """
    Test when anchor is not found.
    """
    text = "2 days after unknown event."
    events = extractor.extract_events(text, ref_date)

    # Should not crash.
    # Logic says: if not resolved, log warning.
    # The event list should theoretically be empty or contain other stuff?
    # Actually, dateparser might have picked up "2 days after" and we invalidated it because of overlap.
    # But since we failed to resolve, do we add it back?
    # Current implementation: If overlap, we discard standard. If resolve fails, we add nothing.
    # So result should be empty (or 0 events if dateparser found nothing else).

    # Wait, dateparser might find "2 days after".
    # Regex finds "2 days after unknown event". Overlap -> discard dateparser result.
    # Resolve -> "unknown event" not found. -> Add nothing.
    # So 0 events.
    assert len(events) == 0


def test_fractional_duration(extractor: TimelineExtractor, ref_date: datetime) -> None:
    """
    Test fractional duration which currently truncates to int.
    "2.5 days after admission" -> 2 days.
    """
    text = "Admission on Jan 1. 2.5 days after admission."
    events = extractor.extract_events(text, ref_date)

    admission = next(e for e in events if "Jan 1" in e.source_snippet)
    anchored = next(e for e in events if "2.5 days after" in e.source_snippet)

    # Current implementation truncates 2.5 -> 2
    expected = admission.timestamp + timedelta(days=2)
    assert anchored.timestamp == expected
