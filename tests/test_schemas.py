from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from coreason_chronos.schemas import ForecastRequest, TemporalEvent, TemporalGranularity


class TestTemporalGranularity:
    def test_values(self) -> None:
        assert TemporalGranularity.PRECISE == "PRECISE"
        assert TemporalGranularity.DATE_ONLY == "DATE_ONLY"
        assert TemporalGranularity.FUZZY == "FUZZY"


class TestTemporalEvent:
    def test_valid_temporal_event(self) -> None:
        event = TemporalEvent(
            id=uuid4(),
            description="Test Event",
            timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            granularity=TemporalGranularity.PRECISE,
            duration_minutes=60,
            ends_at=datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
            source_snippet="Test snippet",
        )
        assert event.description == "Test Event"
        assert event.granularity == TemporalGranularity.PRECISE

    def test_minimal_temporal_event(self) -> None:
        event = TemporalEvent(
            id=uuid4(),
            description="Minimal Event",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            granularity=TemporalGranularity.DATE_ONLY,
            source_snippet="snippet",
        )
        assert event.duration_minutes is None
        assert event.ends_at is None

    def test_invalid_uuid(self) -> None:
        with pytest.raises(ValidationError):
            TemporalEvent(
                id="not-a-uuid",
                description="Invalid ID",
                timestamp=datetime.now(timezone.utc),
                granularity=TemporalGranularity.PRECISE,
                source_snippet="snippet",
            )

    def test_invalid_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            TemporalEvent(
                id=uuid4(),
                description="Invalid Timestamp",
                timestamp="not-a-timestamp",
                granularity=TemporalGranularity.PRECISE,
                source_snippet="snippet",
            )


class TestForecastRequest:
    def test_valid_forecast_request(self) -> None:
        req = ForecastRequest(history=[1.0, 2.0, 3.0], prediction_length=5, confidence_level=0.95, covariates=[1, 0, 1])
        assert req.history == [1.0, 2.0, 3.0]
        assert req.prediction_length == 5
        assert req.confidence_level == 0.95
        assert req.covariates == [1, 0, 1]

    def test_minimal_forecast_request(self) -> None:
        req = ForecastRequest(history=[1.0, 2.0, 3.0], prediction_length=5, confidence_level=0.95)
        assert req.covariates is None

    def test_invalid_history_type(self) -> None:
        with pytest.raises(ValidationError):
            ForecastRequest(
                history=["a", "b"],
                prediction_length=5,
                confidence_level=0.95,
            )

    def test_invalid_confidence_level_type(self) -> None:
        with pytest.raises(ValidationError):
            ForecastRequest(
                history=[1.0, 2.0],
                prediction_length=5,
                confidence_level="high",
            )
