# Prosperity Public License 3.0
from datetime import datetime, timedelta, timezone
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from coreason_chronos.agent import ChronosTimekeeper
from coreason_chronos.schemas import ComplianceResult
from coreason_chronos.validator import MaxDelayRule
from matplotlib.figure import Figure


class TestComplexSmoke:
    """
    End-to-End Smoke Test for the 'Incident Response' workflow.
    Verifies: Extraction -> Causality -> Compliance -> Forecasting -> Visualization.
    Mocks the heavy Forecaster to ensure CI stability.
    """

    @pytest.fixture
    def mock_forecaster_pipeline(self) -> Generator[MagicMock, None, None]:
        """
        Mocks the internal pipeline of the ChronosForecaster to avoid loading the T5 model.
        """
        with patch("coreason_chronos.forecaster.ChronosPipeline") as mock_pipeline_cls:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance
            yield mock_pipeline_instance

    @pytest.fixture
    def agent(self, mock_forecaster_pipeline: MagicMock) -> ChronosTimekeeper:
        """
        Initializes the Timekeeper agent (which will use the mocked pipeline).
        """
        # parameters don't matter much since pipeline is mocked
        return ChronosTimekeeper(model_name="mock-model", device="cpu")

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
        # We mock the Forecaster output because we don't want to load the T5 model.

        # Setup the mock on the forecaster instance specifically
        # (The agent.forecaster.pipeline is the mock object from the fixture)
        mock_pipeline = agent.forecaster.pipeline

        # shape: [num_series, num_samples, prediction_length]
        # We simulate 1 series, 20 samples, 5 steps
        import torch

        mock_output = torch.rand((1, 20, 5))
        # Adjust values to be somewhat realistic for the median check
        mock_output = mock_output * 10 + 100  # range 100-110
        mock_pipeline.predict.return_value = mock_output

        history = [10.0, 12.0, 15.0, 18.0]
        prediction_length = 5

        forecast_result = agent.forecast_series(history=history, prediction_length=prediction_length)

        assert len(forecast_result.median) == prediction_length
        assert forecast_result.confidence_level == 0.9
        # Verify the mock was called
        mock_pipeline.predict.assert_called_once()

        # --- 5. Visualization ---
        from coreason_chronos.schemas import ForecastRequest
        from coreason_chronos.visualizer import plot_forecast

        req = ForecastRequest(history=history, prediction_length=prediction_length, confidence_level=0.9)

        fig = plot_forecast(req, forecast_result, title="Enrollment Forecast")
        assert isinstance(fig, Figure)
        # plt.close(fig) # Good practice
