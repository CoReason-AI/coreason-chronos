# Prosperity Public License 3.0
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from coreason_chronos.schemas import TemporalEvent, TemporalGranularity
from coreason_chronos.timeline_extractor import TimelineExtractor


class TestTimelineExtractor:
    @pytest.fixture
    def extractor(self) -> TimelineExtractor:
        return TimelineExtractor()

    @pytest.fixture
    def ref_date(self) -> datetime:
        return datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

    def test_absolute_date_extraction(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        text = "Patient arrived on 2023-12-25."
        events = extractor.extract_events(text, ref_date)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, TemporalEvent)
        # Check if date is in snippet
        assert "2023-12-25" in event.source_snippet
        # 2023-12-25 00:00:00 UTC
        expected = datetime(2023, 12, 25, 0, 0, 0, tzinfo=timezone.utc)
        assert event.timestamp == expected
        assert event.granularity == TemporalGranularity.DATE_ONLY

    def test_relative_date_extraction_past(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        # ref_date is 2024-01-10
        text = "Symptoms started 2 days ago."
        events = extractor.extract_events(text, ref_date)

        assert len(events) == 1
        event = events[0]
        assert "2 days ago" in event.source_snippet

        # 2024-01-10 - 2 days = 2024-01-08.
        assert event.timestamp.year == 2024
        assert event.timestamp.month == 1
        assert event.timestamp.day == 8
        assert event.timestamp.tzinfo == timezone.utc

    def test_relative_date_extraction_future(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        # ref_date is 2024-01-10
        text = "Follow up in 1 week."
        events = extractor.extract_events(text, ref_date)

        assert len(events) == 1
        event = events[0]

        # 2024-01-10 + 7 days = 2024-01-17
        assert event.timestamp.year == 2024
        assert event.timestamp.month == 1
        assert event.timestamp.day == 17
        assert event.timestamp.tzinfo == timezone.utc

    def test_multiple_dates(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        text = "Seen on Jan 1st. Returning next Friday."
        events = extractor.extract_events(text, ref_date)

        assert len(events) == 2
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)

        assert events[0].timestamp.day == 1
        assert events[0].timestamp.month == 1
        assert events[0].timestamp.year == 2024

        assert events[1].timestamp.weekday() == 4  # Friday

    def test_no_dates_found(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        text = "No temporal info here."
        events = extractor.extract_events(text, ref_date)
        assert len(events) == 0

    def test_invalid_reference_date_tz(self, extractor: TimelineExtractor) -> None:
        naive_ref = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="reference_date must be timezone-aware"):
            extractor.extract_events("test", naive_ref)

    def test_precise_time_extraction(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        text = "Incident at 14:30."
        events = extractor.extract_events(text, ref_date)
        assert len(events) == 1
        event = events[0]
        assert event.timestamp.hour == 14
        assert event.timestamp.minute == 30
        assert event.granularity == TemporalGranularity.PRECISE

    def test_timezone_aware_extraction(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        # Provide a date string with explicit timezone
        text = "Event at 2023-12-25T10:00:00+05:00"
        events = extractor.extract_events(text, ref_date)

        assert len(events) == 1
        event = events[0]

        # Should be converted to UTC
        # 10:00 +05:00 is 05:00 UTC
        assert event.timestamp.tzinfo == timezone.utc
        assert event.timestamp.hour == 5
        assert event.timestamp.day == 25
        assert event.timestamp.month == 12
        assert event.timestamp.year == 2023

    def test_naive_date_fallback(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        # Mock search_dates to return a naive date
        # to test the branch where date_obj.tzinfo is None
        naive_date = datetime(2024, 1, 1, 10, 0)
        with patch("coreason_chronos.timeline_extractor.search_dates") as mock_search:
            mock_search.return_value = [("test date", naive_date)]

            events = extractor.extract_events("test date", ref_date)

            assert len(events) == 1
            event = events[0]
            # Should be forced to UTC
            assert event.timestamp.tzinfo == timezone.utc
            assert event.timestamp.year == 2024
            assert event.timestamp.hour == 10
