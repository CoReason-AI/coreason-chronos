# Prosperity Public License 3.0
from datetime import datetime, timedelta, timezone
from typing import Generator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from coreason_chronos.agent import ChronosTimekeeper
from coreason_chronos.schemas import TemporalEvent, TemporalGranularity
from coreason_chronos.validator import MaxDelayRule


class TestAgentEdgeCases:
    """
    Integration tests for ChronosTimekeeper covering complex scenarios
    and edge cases not covered in the simple smoke test.
    """

    @pytest.fixture
    def mock_forecaster_pipeline(self) -> Generator[MagicMock, None, None]:
        """
        Mocks the internal pipeline to avoid loading the T5 model.
        """
        with patch("coreason_chronos.forecaster.ChronosPipeline") as mock_pipeline_cls:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance
            yield mock_pipeline_instance

    @pytest.fixture
    def agent(self, mock_forecaster_pipeline: MagicMock) -> ChronosTimekeeper:
        return ChronosTimekeeper(model_name="mock-model", device="cpu")

    def test_multi_patient_context_isolation(self, agent: ChronosTimekeeper) -> None:
        """
        Scenario: Two patients with distinct events.
        Verify that relative anchors resolve to the semantically correct event
        rather than just the closest one, avoiding cross-patient pollution.

        Note: Anchors must share tokens with the target event description
        because the current TimelineExtractor uses simple token overlap (Recall),
        not semantic embeddings.
        """
        narrative = (
            "Patient Alpha had a Surgery on Jan 1st 2024. "
            "He reported pain 2 days after the Surgery. "
            "Patient Beta had a Discharge on Feb 1st 2024. "
            "She reported recovery 3 days after Discharge."
        )
        ref_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

        events = agent.extract_from_text(narrative, ref_date)
        events.sort(key=lambda x: x.timestamp)

        # Expected:
        # 1. Alpha Surgery: Jan 1
        # 2. Alpha Pain: Jan 3 (Jan 1 + 2 days) -> Anchored to "Surgery"
        # 3. Beta Discharge: Feb 1
        # 4. Beta Recovery: Feb 4 (Feb 1 + 3 days) -> Anchored to "Discharge"

        alpha_surgery = next(e for e in events if "Jan 1st" in e.source_snippet)
        beta_discharge = next(e for e in events if "Feb 1st" in e.source_snippet)

        alpha_pain = next(e for e in events if "2 days after" in e.source_snippet)
        beta_recovery = next(e for e in events if "3 days after" in e.source_snippet)

        # Verify Alpha
        assert alpha_surgery.timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert alpha_pain.timestamp == alpha_surgery.timestamp + timedelta(days=2)

        # Verify Beta
        assert beta_discharge.timestamp == datetime(2024, 2, 1, tzinfo=timezone.utc)
        assert beta_recovery.timestamp == beta_discharge.timestamp + timedelta(days=3)

    def test_leap_year_boundary_compliance(self, agent: ChronosTimekeeper) -> None:
        """
        Scenario: Event occurs on Feb 28th, deadline is 48 hours later.
        In a leap year (2024), this lands on Feb 29th then March 1st.
        Verify compliance logic handles this correctly.
        """
        # Event A: Feb 28 2024 12:00 UTC
        start_time = datetime(2024, 2, 28, 12, 0, 0, tzinfo=timezone.utc)

        event_occurrence = TemporalEvent(
            id=uuid4(),
            description="Occurrence",
            timestamp=start_time,
            granularity=TemporalGranularity.PRECISE,
            source_snippet="",
        )

        # Deadline: 48 hours -> Feb 29 12:00 -> March 1 12:00

        rule = MaxDelayRule(max_delay=timedelta(hours=48), name="48h Rule")

        # Case 1: Exact Deadline (Compliant)
        report_on_time = TemporalEvent(
            id=uuid4(),
            description="Report On Time",
            timestamp=start_time + timedelta(hours=48),
            granularity=TemporalGranularity.PRECISE,
            source_snippet="",
        )

        res_ok = agent.check_compliance(report_on_time, event_occurrence, rule)
        assert res_ok.is_compliant is True
        assert res_ok.drift == timedelta(0)

        # Case 2: 1 Second Late (Non-Compliant)
        report_late = TemporalEvent(
            id=uuid4(),
            description="Report Late",
            timestamp=start_time + timedelta(hours=48, seconds=1),
            granularity=TemporalGranularity.PRECISE,
            source_snippet="",
        )

        res_fail = agent.check_compliance(report_late, event_occurrence, rule)
        assert res_fail.is_compliant is False
        assert res_fail.drift == timedelta(seconds=1)

        # Verify date math was correct (Leap Day existence)
        expected_deadline = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert (start_time + timedelta(hours=48)) == expected_deadline

    def test_causality_impossible_timeline(self, agent: ChronosTimekeeper) -> None:
        """
        Scenario: Effect happens strictly BEFORE Cause.
        Verify CausalityEngine returns False.
        """
        t0 = datetime(2024, 1, 10, 10, 0, tzinfo=timezone.utc)

        cause_event = TemporalEvent(
            id=uuid4(), description="Cause", timestamp=t0, granularity=TemporalGranularity.PRECISE, source_snippet=""
        )

        effect_event = TemporalEvent(
            id=uuid4(),
            description="Effect",
            timestamp=t0 - timedelta(minutes=1),
            granularity=TemporalGranularity.PRECISE,
            source_snippet="",
        )

        is_plausible = agent.analyze_causality(cause_event, effect_event)
        assert is_plausible is False
