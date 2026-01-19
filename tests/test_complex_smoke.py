# Prosperity Public License 3.0
from datetime import datetime, timedelta, timezone
from typing import Generator
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure

from coreason_chronos.agent import ChronosTimekeeper
from coreason_chronos.forecaster import ChronosForecaster
from coreason_chronos.schemas import ComplianceResult, ForecastResult
from coreason_chronos.validator import MaxDelayRule


class TestComplexSmoke:
    """
    End-to-End Smoke Test for the 'Incident Response' workflow.
    Verifies: Extraction -> Causality -> Compliance -> Forecasting -> Visualization.
    Mocks the heavy Forecaster to ensure CI stability.
    """

    @pytest.fixture
    def mock_forecaster(self) -> MagicMock:
        """
        Mocks the ChronosForecaster component using dependency injection.
        """
        mock = MagicMock(spec=ChronosForecaster)
        # Setup default return for forecast
        mock.forecast.return_value = ForecastResult(
            median=[105.0] * 5,
            lower_bound=[100.0] * 5,
            upper_bound=[110.0] * 5,
            confidence_level=0.9,
        )
        return mock

    @pytest.fixture
    def agent(self, mock_forecaster: MagicMock) -> ChronosTimekeeper:
        """
        Initializes the Timekeeper agent using dependency injection.
        """
        return ChronosTimekeeper(forecaster=mock_forecaster)

    def test_incident_response_workflow(self, agent: ChronosTimekeeper) -> None:
        """
        Scenario:
        1. Parse a clinical narrative to extract events.
        2. Verify causal order (Enrollment -> Adverse Event).
        3. Check compliance (Reporting Latency).
        4. Forecast future enrollment (Mocked).
        5. Visualize the forecast.
        """

        # --- 1. Extraction ---
        # "Study Start on 2024-01-01. Patient A enrolled 5 days later.
        #  Severe Fever occurred 2 days after enrollment.
        #  Reported to safety board 30 hours after Severe Fever."

        narrative = (
            "Study Start on Jan 1st 2024. "
            "Patient A enrollment 5 days later. "
            "The patient was monitored closely. "
            "Severe Fever occurred 2 days after enrollment. "
            "Reported to safety board 30 hours after Severe Fever."
        )

        ref_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

        events = agent.extract_from_text(narrative, ref_date)

        # Sort by time to ensure easier assertions
        events.sort(key=lambda x: x.timestamp)

        # Debug print for visibility if test fails
        for e in events:
            print(f"Event: {e.description} | Time: {e.timestamp} | Src: {e.source_snippet}")

        # We expect 4 events:
        # 1. Study Start (2024-01-01)
        # 2. Enrollment (Start + 5d = 2024-01-06)
        # 3. Fever (Enroll + 2d = 2024-01-08)
        # 4. Report (Fever + 30h = 2024-01-09 06:00)

        assert len(events) >= 4, "Failed to extract all chained events"

        # Locate specific events by description keywords
        # Note: we use source_snippet as description includes context which can overlap
        start_event = next(e for e in events if "Jan 1st" in e.source_snippet)
        enroll_event = next(e for e in events if "5 days later" in e.source_snippet)
        fever_event = next(e for e in events if "2 days after" in e.source_snippet)
        report_event = next(e for e in events if "30 hours after" in e.source_snippet)

        # Verify Timestamps
        assert start_event.timestamp == ref_date
        assert enroll_event.timestamp == ref_date + timedelta(days=5)
        assert fever_event.timestamp == enroll_event.timestamp + timedelta(days=2)
        assert report_event.timestamp == fever_event.timestamp + timedelta(hours=30)

        # --- 2. Causality ---
        # Check: Did Enrollment plausible cause Fever? (Yes, Enroll < Fever)
        is_plausible = agent.analyze_causality(enroll_event, fever_event)
        assert is_plausible is True, "Enrollment should be a plausible cause for Fever (Precedence)"

        # Check: Did Report cause Fever? (No, Report > Fever)
        is_reverse_plausible = agent.analyze_causality(report_event, fever_event)
        assert is_reverse_plausible is False, "Report (future) cannot cause Fever (past)"

        # --- 3. Compliance ---
        # Rule: SAE must be reported within 24 hours.
        # Actual: Reported in 30 hours.
        # Expectation: Non-compliant, drift = 6 hours.

        rule = MaxDelayRule(max_delay=timedelta(hours=24), name="24h Reporting Window")
        compliance: ComplianceResult = agent.check_compliance(target=report_event, reference=fever_event, rule=rule)

        assert compliance.is_compliant is False, "Should be non-compliant (30h > 24h)"
        assert compliance.drift == timedelta(hours=6)
        assert compliance.message is not None and "Violation" in compliance.message

        # --- 4. Forecasting (Mocked) ---
        # Context: Trial Manager asks for enrollment forecast.
        # We mock the Forecaster output via DI.

        history = [10.0, 12.0, 15.0, 18.0]
        prediction_length = 5

        forecast_result = agent.forecast_series(history=history, prediction_length=prediction_length)

        assert len(forecast_result.median) == prediction_length
        assert forecast_result.confidence_level == 0.9

        # Verify the mock was called
        # agent.forecaster is the mock object passed in __init__
        agent.forecaster.forecast.assert_called_once()

        # --- 5. Visualization ---
        from coreason_chronos.schemas import ForecastRequest
        from coreason_chronos.visualizer import plot_forecast

        req = ForecastRequest(history=history, prediction_length=prediction_length, confidence_level=0.9)

        fig = plot_forecast(req, forecast_result, title="Enrollment Forecast")
        assert isinstance(fig, Figure)
        # plt.close(fig) # Good practice
