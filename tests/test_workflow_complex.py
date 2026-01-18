# Prosperity Public License 3.0
from datetime import datetime, timedelta, timezone
from typing import Generator, Optional
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from dateutil import tz

from coreason_chronos.agent import ChronosTimekeeper
from coreason_chronos.schemas import TemporalEvent, TemporalGranularity
from coreason_chronos.validator import MaxDelayRule


class TestWorkflowComplex:
    """
    Advanced workflow scenarios covering chained dependencies,
    timezone complexities, and ambiguity resolution.
    """

    @pytest.fixture
    def mock_forecaster_pipeline(self) -> Generator[MagicMock, None, None]:
        with patch("coreason_chronos.forecaster.ChronosPipeline") as mock_pipeline_cls:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance
            yield mock_pipeline_instance

    @pytest.fixture
    def agent(self, mock_forecaster_pipeline: MagicMock) -> ChronosTimekeeper:
        return ChronosTimekeeper(model_name="mock-model", device="cpu")

    def test_chained_dependencies(self, agent: ChronosTimekeeper) -> None:
        """
        Scenario: Daisy-chained events.
        A (Jan 1) -> B (2 days later) -> C (3 days after B) -> D (1 week after C).
        Verifies that the extractor can resolve anchors iteratively.
        """
        narrative = (
            "Project started on Jan 1st 2024. "
            "Phase 1 began 2 days later. "
            "Phase 2 kicked off 3 days after Phase 1. "
            "Final Review happened 1 week after Phase 2."
        )
        ref_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

        events = agent.extract_from_text(narrative, ref_date)
        events.sort(key=lambda x: x.timestamp)

        # Expected:
        # 1. Start: Jan 1
        # 2. Phase 1: Jan 3
        # 3. Phase 2: Jan 6 (Jan 3 + 3d)
        # 4. Final: Jan 13 (Jan 6 + 7d)

        def find(snippet_part: str) -> Optional[TemporalEvent]:
            return next((e for e in events if snippet_part in e.source_snippet), None)

        start = find("Jan 1st")
        p1 = find("2 days later")
        p2 = find("3 days after")
        final = find("1 week after")

        assert start and p1 and p2 and final

        t_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert start.timestamp == t_start
        assert p1.timestamp == t_start + timedelta(days=2)
        assert p2.timestamp == p1.timestamp + timedelta(days=3)
        assert final.timestamp == p2.timestamp + timedelta(days=7)

    def test_timezone_crossing_compliance(self, agent: ChronosTimekeeper) -> None:
        """
        Scenario:
        Event A: Tokyo Time (JST, UTC+9).
        Event B: New York Time (EST, UTC-5).
        Check if compliance logic correctly normalizes to UTC.
        """
        # Event: 10:00 AM JST on Jan 1st
        jst = tz.gettz("Asia/Tokyo")
        est = tz.gettz("America/New_York")

        # Note: We manually construct events to test the Validator component specifically
        # in a complex timezone scenario, rather than relying on extraction which defaults to UTC.

        event_time_jst = datetime(2024, 1, 1, 10, 0, 0, tzinfo=jst)
        # In UTC: Jan 1, 01:00 UTC

        # Deadline: 24 hours later -> Jan 2, 01:00 UTC.

        # Report: 8:00 PM EST on Jan 1st.
        # Jan 1, 20:00 EST -> Jan 2, 01:00 UTC.
        # This is EXACTLY 24 hours later.

        report_time_est = datetime(2024, 1, 1, 20, 0, 0, tzinfo=est)

        target = TemporalEvent(
            id=uuid4(),
            description="Report EST",
            timestamp=report_time_est,
            granularity=TemporalGranularity.PRECISE,
            source_snippet="",
        )
        reference = TemporalEvent(
            id=uuid4(),
            description="Event JST",
            timestamp=event_time_jst,
            granularity=TemporalGranularity.PRECISE,
            source_snippet="",
        )

        rule = MaxDelayRule(max_delay=timedelta(hours=24), name="24h Cross-Zone")

        result = agent.check_compliance(target, reference, rule)

        # Should be exactly compliant (Drift = 0)
        assert result.is_compliant is True
        assert abs(result.drift.total_seconds()) < 1  # Float tolerance

    def test_ambiguous_anchors(self, agent: ChronosTimekeeper) -> None:
        """
        Scenario: Recurring event names ("Scan") acting as anchors.
        "Patient had a Scan on Jan 1st. Results came 2 days after the Scan.
         Patient had another Scan on Feb 1st. Report came 1 day after the Scan."

        Verifies that 'after the Scan' resolves to the NEAREST 'Scan' event
        (Proximity Logic) rather than just the first one.
        """
        narrative = (
            "Patient had a Scan on Jan 1st 2024. "
            "Results came 2 days after the Scan. "
            "Patient had another Scan on Feb 1st 2024. "
            "Report came 1 day after the Scan."
        )
        ref_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

        events = agent.extract_from_text(narrative, ref_date)
        events.sort(key=lambda x: x.timestamp)

        # Expected:
        # 1. Jan 1: Scan 1
        # 2. Jan 3: Results (Jan 1 + 2d)
        # 3. Feb 1: Scan 2
        # 4. Feb 2: Report (Feb 1 + 1d)

        jan_events = [e for e in events if e.timestamp.month == 1]
        feb_events = [e for e in events if e.timestamp.month == 2]

        # If proximity fails and it anchors to first scan always:
        # Results -> Jan 3.
        # Report -> Jan 2 (Jan 1 + 1d).
        # Then all 4 events would be in Jan.

        assert len(jan_events) == 2, f"Expected 2 Jan events, got {len(jan_events)}"
        assert len(feb_events) == 2, f"Expected 2 Feb events, got {len(feb_events)}"

        # Check specific dates
        assert jan_events[0].timestamp.day == 1
        assert jan_events[1].timestamp.day == 3

        assert feb_events[0].timestamp.day == 1
        assert feb_events[1].timestamp.day == 2
