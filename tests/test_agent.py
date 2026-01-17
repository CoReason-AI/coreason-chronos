from datetime import datetime, timezone
from typing import Generator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from coreason_chronos.agent import ChronosTimekeeper
from coreason_chronos.schemas import ComplianceResult, ForecastResult, TemporalEvent, TemporalGranularity
from coreason_chronos.validator import MaxDelayRule


class TestChronosTimekeeper:
    @pytest.fixture
    def mock_components(self) -> Generator[tuple[MagicMock, MagicMock, MagicMock], None, None]:
        with (
            patch("coreason_chronos.agent.TimelineExtractor") as mock_ext_cls,
            patch("coreason_chronos.agent.ChronosForecaster") as mock_fc_cls,
            patch("coreason_chronos.agent.CausalityEngine") as mock_causal_cls,
        ):
            mock_ext = mock_ext_cls.return_value
            mock_fc = mock_fc_cls.return_value
            mock_causal = mock_causal_cls.return_value

            yield mock_ext, mock_fc, mock_causal

    def test_initialization(self, mock_components: tuple[MagicMock, MagicMock, MagicMock]) -> None:
        """Test that components are initialized with correct parameters."""
        mock_ext, mock_fc, mock_causal = mock_components

        # Init agent
        ChronosTimekeeper(model_name="test-model", device="cuda")

        # Check Forecaster init
        # We need to check the call to the CLASS, not the instance
        # mock_fc_cls is what was patched.
        # But in the fixture we yielded the instance.
        # Let's verify via the class patch which we need to capture if we want to check args.
        pass  # We rely on the fixture setup implying successful init, detailed arg check below.

    def test_initialization_args(self) -> None:
        with (
            patch("coreason_chronos.agent.ChronosForecaster") as mock_fc_cls,
            patch("coreason_chronos.agent.TimelineExtractor"),
            patch("coreason_chronos.agent.CausalityEngine"),
        ):
            # Update verification to include quantization=None default
            ChronosTimekeeper(model_name="custom-model", device="cuda")
            mock_fc_cls.assert_called_with(model_name="custom-model", device="cuda", quantization=None)

            # Update verification for explicit quantization
            ChronosTimekeeper(model_name="custom-model", device="cuda", quantization="int8")
            mock_fc_cls.assert_called_with(model_name="custom-model", device="cuda", quantization="int8")

    def test_agent_lifecycle_with_quantization(self) -> None:
        """
        Test that initialization with quantization yields a working agent
        that uses the quantized forecaster (mocked).
        """
        with (
            patch("coreason_chronos.agent.ChronosForecaster") as mock_fc_cls,
            patch("coreason_chronos.agent.TimelineExtractor"),
            patch("coreason_chronos.agent.CausalityEngine"),
        ):
            # 1. Initialize
            agent = ChronosTimekeeper(model_name="q-model", device="cuda", quantization="int8")

            # Verify initialization
            mock_fc_cls.assert_called_with(model_name="q-model", device="cuda", quantization="int8")

            # 2. Mock behavior for forecast
            mock_forecaster_instance = mock_fc_cls.return_value
            expected_result = MagicMock(spec=ForecastResult)
            mock_forecaster_instance.forecast.return_value = expected_result

            # 3. Use the agent
            result = agent.forecast_series(history=[1.0, 2.0], prediction_length=3)

            assert result == expected_result
            mock_forecaster_instance.forecast.assert_called_once()

    def test_extract_from_text(self, mock_components: tuple[MagicMock, MagicMock, MagicMock]) -> None:
        mock_ext, _, _ = mock_components
        agent = ChronosTimekeeper()

        ref_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        expected_events = [MagicMock(spec=TemporalEvent)]
        mock_ext.extract_events.return_value = expected_events

        result = agent.extract_from_text("some text", ref_date)

        assert result == expected_events
        mock_ext.extract_events.assert_called_with("some text", ref_date)

    def test_forecast_series(self, mock_components: tuple[MagicMock, MagicMock, MagicMock]) -> None:
        _, mock_fc, _ = mock_components
        agent = ChronosTimekeeper()

        history = [1.0, 2.0, 3.0]
        expected_result = MagicMock(spec=ForecastResult)
        mock_fc.forecast.return_value = expected_result

        result = agent.forecast_series(history, prediction_length=5, confidence_level=0.8)

        assert result == expected_result

        # Verify arguments passed to forecast via ForecastRequest
        call_args = mock_fc.forecast.call_args
        assert call_args is not None
        req = call_args[0][0]
        assert req.history == history
        assert req.prediction_length == 5
        assert req.confidence_level == 0.8

    def test_check_compliance(self, mock_components: tuple[MagicMock, MagicMock, MagicMock]) -> None:
        # Compliance check uses a passed rule instance, so we don't mock the rule class in the agent init,
        # but we mock the rule instance passed to the method.
        agent = ChronosTimekeeper()

        evt_target = TemporalEvent(
            id=uuid4(),
            description="T",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            granularity=TemporalGranularity.PRECISE,
            source_snippet="",
        )
        evt_ref = TemporalEvent(
            id=uuid4(),
            description="R",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            granularity=TemporalGranularity.PRECISE,
            source_snippet="",
        )

        mock_rule = MagicMock(spec=MaxDelayRule)
        expected_res = ComplianceResult(is_compliant=True, drift=datetime.now() - datetime.now())  # Zero delta
        mock_rule.validate.return_value = expected_res

        result = agent.check_compliance(evt_target, evt_ref, mock_rule)

        assert result == expected_res
        mock_rule.validate.assert_called_with(evt_target.timestamp, evt_ref.timestamp)

    def test_analyze_causality(self, mock_components: tuple[MagicMock, MagicMock, MagicMock]) -> None:
        _, _, mock_causal = mock_components
        agent = ChronosTimekeeper()

        evt_a = MagicMock(spec=TemporalEvent)
        evt_b = MagicMock(spec=TemporalEvent)
        evt_a.description = "A"
        evt_b.description = "B"

        mock_causal.is_plausible_cause.return_value = True

        result = agent.analyze_causality(evt_a, evt_b)

        assert result is True
        mock_causal.is_plausible_cause.assert_called_with(evt_a, evt_b)

    def test_full_workflow_extract_validate_cause(
        self, mock_components: tuple[MagicMock, MagicMock, MagicMock]
    ) -> None:
        """
        Test a realistic workflow: Extract 2 events, check causality, then check compliance.
        """
        mock_ext, _, mock_causal = mock_components
        agent = ChronosTimekeeper()

        # 1. Setup Mock Extraction
        evt_start = TemporalEvent(
            id=uuid4(),
            description="Start",
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            granularity=TemporalGranularity.PRECISE,
            source_snippet="Start at 10:00",
        )
        evt_end = TemporalEvent(
            id=uuid4(),
            description="End",
            timestamp=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            granularity=TemporalGranularity.PRECISE,
            source_snippet="End at 11:00",
        )

        mock_ext.extract_events.return_value = [evt_start, evt_end]

        # 2. Run Extraction
        events = agent.extract_from_text("Start at 10:00. End at 11:00.", datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert len(events) == 2

        # 3. Check Causality (Is Start plausible cause of End?)
        mock_causal.is_plausible_cause.return_value = True
        is_cause = agent.analyze_causality(events[0], events[1])
        assert is_cause is True
        mock_causal.is_plausible_cause.assert_called_with(evt_start, evt_end)

        # 4. Check Compliance (End should be within 2 hours of Start)
        mock_rule = MagicMock(spec=MaxDelayRule)
        # Mocking the result of validate
        mock_rule.validate.return_value = ComplianceResult(is_compliant=True, drift=datetime.now() - datetime.now())

        res = agent.check_compliance(events[1], events[0], mock_rule)
        assert res.is_compliant is True
        mock_rule.validate.assert_called_with(evt_end.timestamp, evt_start.timestamp)

    def test_component_error_propagation(self, mock_components: tuple[MagicMock, MagicMock, MagicMock]) -> None:
        """Test that exceptions from sub-components are propagated correctly."""
        mock_ext, _, _ = mock_components
        agent = ChronosTimekeeper()

        # Simulate Extraction Failure
        mock_ext.extract_events.side_effect = ValueError("Extraction Failed")

        with pytest.raises(ValueError, match="Extraction Failed"):
            agent.extract_from_text("bad text", datetime.now(timezone.utc))

    def test_empty_text_extraction(self, mock_components: tuple[MagicMock, MagicMock, MagicMock]) -> None:
        """Test behavior with empty input."""
        mock_ext, _, _ = mock_components
        agent = ChronosTimekeeper()

        # Mock extracting nothing
        mock_ext.extract_events.return_value = []

        result = agent.extract_from_text("", datetime.now(timezone.utc))
        assert result == []
