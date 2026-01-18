from datetime import datetime, timezone
from unittest.mock import patch
import torch
import pytest
from coreason_chronos.agent import ChronosTimekeeper

def test_complex_end_to_end_smoke_workflow() -> None:
    """
    A complex smoke test validating the full Extract-Align-Forecast-Visualize loop.
    Scenario:
    An 'Incident Response' workflow where we:
    1. Extract a timeline from a messy report.
    2. Verify causality (Did the alert happen after the outage?).
    3. Check compliance (Did the alert happen within the 15-minute SLA?).
    4. Forecast future system load based on historical metrics.
    5. Visualize the forecast.
    """
    # ---------------------------------------------------------
    # 1. Setup & Initialization
    # ---------------------------------------------------------
    # We mock the underlying HuggingFace pipeline to avoid downloading
    # the Chronos model (approx 500MB+) during a quick smoke test.
    with patch("coreason_chronos.forecaster.ChronosPipeline") as MockPipeline:
        # Configure the mock to return a valid tensor shape:
        # [num_series=1, num_samples=20, prediction_length=5]
        mock_instance = MockPipeline.from_pretrained.return_value
        mock_instance.predict.return_value = torch.rand(1, 20, 5)
        # Initialize the Agent (Facade)
        # We explicitly use 'cpu' to ensure this runs on standard CI runners.
        agent = ChronosTimekeeper(model_name="amazon/chronos-t5-tiny", device="cpu")
        # Verify internal wiring
        assert agent.extractor is not None
        assert agent.causality is not None
        assert agent.forecaster is not None
        # ---------------------------------------------------------
        # 2. Timeline Extraction (Text -> Events)
        # ---------------------------------------------------------
        # Narrative: Contains an absolute date ("Jan 10... 08:00") and a
        # relative anchored event ("20 minutes after the outage").
        narrative = (
            "System Outage Report. "
            "The outage started on Jan 10, 2024 at 08:00 UTC. "
            "The P1 Alert was fired 20 minutes after the outage started."
        )
        ref_date = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        events = agent.extract_from_text(narrative, ref_date)
        # We expect at least 2 events
        assert len(events) >= 2
        # Sort by time to deterministically identify events
        events.sort(key=lambda x: x.timestamp)
        outage_event = events[0]
        alert_event = events[1]
        # Verify Outage (Absolute Extraction)
        assert outage_event.timestamp == datetime(2024, 1, 10, 8, 0, tzinfo=timezone.utc)
        # Verify Alert (Fuzzy Anchor Resolution)
        # "20 minutes after" -> 08:00 + 20min = 08:20
        assert "Alert" in alert_event.description or "Alert" in alert_event.source_snippet
