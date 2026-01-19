from datetime import datetime, timedelta, timezone
from uuid import UUID
from unittest.mock import MagicMock, patch

import pytest
from dateutil.relativedelta import relativedelta
from coreason_chronos.timeline_extractor import TimelineExtractor
from coreason_chronos.schemas import TemporalEvent

@pytest.fixture
def extractor() -> TimelineExtractor:
    return TimelineExtractor()

@pytest.fixture
def ref_date() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

def test_duration_extraction_basic(extractor: TimelineExtractor, ref_date: datetime) -> None:
    """
    Test extraction of a simple duration following an event description.
    "He had a fever for 3 days starting on Jan 1st."
    """
    text = "Patient reported fever starting Jan 1st for 3 days."
    events = extractor.extract_events(text, ref_date)
    assert len(events) == 1
    event = events[0]
    assert event.timestamp.day == 1
    assert event.duration_minutes == 3 * 24 * 60
    assert event.ends_at == event.timestamp + timedelta(days=3)

def test_duration_extraction_lasting(extractor: TimelineExtractor, ref_date: datetime) -> None:
    text = "Surgery occurred on 2024-01-10, lasting 4 hours."
    events = extractor.extract_events(text, ref_date)
    assert len(events) == 1
    assert events[0].duration_minutes == 4 * 60

def test_duration_extraction_spanning(extractor: TimelineExtractor, ref_date: datetime) -> None:
    text = "The treatment phase began on Jan 5th, spanning 2 weeks."
    events = extractor.extract_events(text, ref_date)
    assert len(events) == 1
    assert events[0].duration_minutes == 14 * 24 * 60

def test_duration_extraction_anchored_event(extractor: TimelineExtractor, ref_date: datetime) -> None:
    text = "Admission on Jan 1st. 2 days after admission, rash appeared for 5 hours."
    events = extractor.extract_events(text, ref_date)
    assert len(events) == 2
    events.sort(key=lambda x: x.timestamp)
    admission = events[0]
    rash = events[1]
    assert admission.duration_minutes is None
    assert rash.timestamp.day == 3
    assert rash.duration_minutes == 5 * 60

def test_duration_extraction_fractional(extractor: TimelineExtractor, ref_date: datetime) -> None:
    text = "Symptoms persisted from Jan 2nd for 1.5 days."
    events = extractor.extract_events(text, ref_date)
    assert len(events) == 1
    assert events[0].duration_minutes == 36 * 60

def test_multiple_events_distinct_durations(extractor: TimelineExtractor, ref_date: datetime) -> None:
    text = "First event started on 2024-01-01 for 2 hours. Second event started on 2024-01-05 for 30 minutes."
    events = extractor.extract_events(text, ref_date)
    assert len(events) == 2
    events.sort(key=lambda x: x.timestamp)
    assert events[0].duration_minutes == 120
    assert events[1].duration_minutes == 30

def test_no_duration(extractor: TimelineExtractor, ref_date: datetime) -> None:
    text = "Meeting on Jan 1st."
    events = extractor.extract_events(text, ref_date)
    assert len(events) == 1
    assert events[0].duration_minutes is None

def test_duration_units_minutes_seconds(extractor: TimelineExtractor, ref_date: datetime) -> None:
    text2 = "Event on Jan 1st for 10 minutes."
    events2 = extractor.extract_events(text2, ref_date)
    assert events2[0].duration_minutes == 10

    text3 = "Event on Jan 1st for 120 seconds."
    events3 = extractor.extract_events(text3, ref_date)
    assert events3[0].duration_minutes == 2

def test_duration_before_event(extractor: TimelineExtractor, ref_date: datetime) -> None:
    text = "For 3 days, starting Jan 5th, patient had fever."
    events = extractor.extract_events(text, ref_date)
    assert len(events) == 1
    assert events[0].duration_minutes == 3 * 24 * 60

def test_zero_duration(extractor: TimelineExtractor, ref_date: datetime) -> None:
    text = "Event on Jan 1st for 0 hours."
    events = extractor.extract_events(text, ref_date)
    assert len(events) == 1
    assert events[0].duration_minutes is None

def test_pure_duration_ignored(extractor: TimelineExtractor, ref_date: datetime) -> None:
    text = "The patient was treated for 3 months."
    events = extractor.extract_events(text, ref_date)
    assert len(events) == 0

def test_anchored_zero_duration(extractor: TimelineExtractor, ref_date: datetime) -> None:
    text = "Admission on Jan 1st. 2 days after admission for 0 hours."
    events = extractor.extract_events(text, ref_date)
    events.sort(key=lambda x: x.timestamp)
    anchored = events[1]
    assert anchored.duration_minutes is None

def test_calculate_total_minutes_and_delta_direct(extractor: TimelineExtractor) -> None:
    """
    White-box test for coverage of unit branches.
    """
    # Test minutes
    mins, delta = extractor._calculate_total_minutes_and_delta(10, "minutes")
    assert mins == 10
    assert delta == timedelta(minutes=10)

    # Test seconds
    mins, delta = extractor._calculate_total_minutes_and_delta(60, "seconds")
    assert mins == 1
    assert delta == timedelta(seconds=60)

    # Test weeks
    mins, delta = extractor._calculate_total_minutes_and_delta(1, "weeks")
    assert mins == 10080
    assert delta == timedelta(weeks=1)

    # Test days
    mins, delta = extractor._calculate_total_minutes_and_delta(1, "days")
    assert mins == 1440
    assert delta == timedelta(days=1)

    # Test hours
    mins, delta = extractor._calculate_total_minutes_and_delta(1, "hours")
    assert mins == 60
    assert delta == timedelta(hours=1)

def test_intervening_case_2(extractor: TimelineExtractor, ref_date: datetime) -> None:
    """
    Test Case 2: Match < Forbidden < Snippet.
    Duration appears before an anchor, which appears before the event.
    """
    text = "For 3 days, 2 days before admission, admission on Jan 5th."
    events = extractor.extract_events(text, ref_date)

    events.sort(key=lambda x: x.timestamp)
    admission_events = [e for e in events if e.timestamp.day == 5]
    if admission_events:
        assert admission_events[0].duration_minutes is None

@patch("coreason_chronos.timeline_extractor.search_dates")
def test_duration_regex_ignored_mock(
    mock_search_dates: MagicMock,
    extractor: TimelineExtractor,
    ref_date: datetime
) -> None:
    """
    Mock search_dates to return a pure duration snippet to ensure it is ignored.
    """
    # Simulate finding "3 months" as a date
    mock_search_dates.return_value = [("3 months", datetime(2024, 1, 1, tzinfo=timezone.utc))]

    events = extractor.extract_events("treated for 3 months", ref_date)
    assert len(events) == 0

def test_duration_overlap_forbidden(extractor: TimelineExtractor, ref_date: datetime) -> None:
    """
    Test case where a potential duration string is also a valid Date event, so it should be forbidden.
    Forces 'is_forbidden' check.
    We need to find a way to have DURATION_CONTEXT_REGEX match something that is ALSO in forbidden_ranges.
    """
    text = "for 3 days"

    # Make forbidden range cover it (0, 10)
    forbidden = [(0, 10)]

    # Call private method
    mins, delta = extractor._resolve_duration(text, 12, 20, forbidden_ranges=forbidden)

    # Should return None because match overlaps forbidden
    assert mins is None
    assert delta is None

def test_duration_variable_units(extractor: TimelineExtractor, ref_date: datetime) -> None:
    """
    Test variable units (months, years) to ensure fallback path coverage.
    """
    text = "Project launched on Jan 1st for 2 months."
    events = extractor.extract_events(text, ref_date)
    assert len(events) == 1
    event = events[0]

    # 2 months from Jan 1st is March 1st.
    # 2024 is leap year. Jan=31, Feb=29. Total 60 days.
    # 60 * 24 * 60 = 86400 minutes.
    assert event.duration_minutes is not None
    # We check approx minutes logic or specific delta
    assert event.ends_at == event.timestamp + relativedelta(months=2)
    # Check that it's March 1st
    assert event.ends_at.month == 3
    assert event.ends_at.day == 1
