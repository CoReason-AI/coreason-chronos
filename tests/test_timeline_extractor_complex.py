# Prosperity Public License 3.0
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from coreason_chronos.timeline_extractor import TimelineExtractor


class TestTimelineExtractorComplex:
    @pytest.fixture
    def extractor(self) -> TimelineExtractor:
        return TimelineExtractor()

    @pytest.fixture
    def ref_date(self) -> datetime:
        return datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

    def test_age_filtering(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        text = "Patient is 50 years old."
        events = extractor.extract_events(text, ref_date)
        # Should NOT extract 1974
        assert len(events) == 0

    def test_duration_filtering(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        text = "Symptoms lasted for 3 months."
        events = extractor.extract_events(text, ref_date)
        # Should NOT extract a date 3 months ago
        assert len(events) == 0

    def test_relative_with_direction(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        text = "Symptoms started 3 months ago."
        events = extractor.extract_events(text, ref_date)
        assert len(events) == 1
        assert "3 months ago" in events[0].source_snippet

    def test_future_relative(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        text = "Return in 2 weeks."
        events = extractor.extract_events(text, ref_date)
        assert len(events) == 1

    def test_explicit_year(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        text = "Diagnosed in 2020."
        events = extractor.extract_events(text, ref_date)
        assert len(events) == 1
        assert events[0].timestamp.year == 2020

    def test_partial_date(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        text = "Occurred in May 2024."
        events = extractor.extract_events(text, ref_date)
        assert len(events) == 1
        assert events[0].timestamp.month == 5
        assert events[0].timestamp.year == 2024

    def test_leap_year_handling(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        # Valid leap year
        text = "Feb 29 2024"
        events = extractor.extract_events(text, ref_date)
        assert len(events) == 1
        assert events[0].timestamp.day == 29

    def test_invalid_leap_year(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        # Invalid leap year Feb 29 2023
        # dateparser might extract "Feb" (current year) and "2023" (Jan 1st?) or similar
        # We just want to ensure it doesn't crash, behavior might be undefined/noisy
        text = "Feb 29 2023"
        events = extractor.extract_events(text, ref_date)
        # We accept whatever dateparser gives, or 0 if it filters.
        # Just ensure no exception.
        assert isinstance(events, list)

    def test_range_handling(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        text = "From Jan 1st to Jan 5th"
        events = extractor.extract_events(text, ref_date)
        assert len(events) >= 2

    def test_none_date_result(self, extractor: TimelineExtractor, ref_date: datetime) -> None:
        # Mock search_dates to return a tuple with None date
        with patch("coreason_chronos.timeline_extractor.search_dates") as mock_search:
            mock_search.return_value = [("some text", None)]
            events = extractor.extract_events("some text", ref_date)
            assert len(events) == 0
