from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from coreason_chronos.schemas import ForecastRequest, TemporalEvent, TemporalGranularity
from pydantic import ValidationError


class TestSchemas:
    def test_temporal_event_valid(self) -> None:
        event = TemporalEvent(
            id=uuid4(),
            description="Test",
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            granularity=TemporalGranularity.PRECISE,
            source_snippet="snippet",
        )
        assert event.description == "Test"

    def test_temporal_event_valid_duration(self) -> None:
        event = TemporalEvent(
            id=uuid4(),
            description="Test",
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            granularity=TemporalGranularity.PRECISE,
            source_snippet="snippet",
            duration_minutes=10,
        )
        assert event.duration_minutes == 10

    def test_temporal_event_naive_timestamp(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            TemporalEvent(
                id=uuid4(),
                description="Test",
                timestamp=datetime(2024, 1, 1, 10, 0),  # naive
                granularity=TemporalGranularity.PRECISE,
                source_snippet="snippet",
            )
        assert "timestamp must be timezone-aware" in str(excinfo.value)

    def test_temporal_event_negative_duration(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            TemporalEvent(
                id=uuid4(),
                description="Test",
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                granularity=TemporalGranularity.PRECISE,
                duration_minutes=-5,
                source_snippet="snippet",
            )
        assert "duration_minutes must be non-negative" in str(excinfo.value)

    def test_temporal_event_ends_at_before_timestamp(self) -> None:
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        ends = ts - timedelta(hours=1)
        with pytest.raises(ValidationError) as excinfo:
            TemporalEvent(
                id=uuid4(),
                description="Test",
                timestamp=ts,
                ends_at=ends,
                granularity=TemporalGranularity.PRECISE,
                source_snippet="snippet",
            )
        assert "ends_at must be after timestamp" in str(excinfo.value)

    def test_temporal_event_ends_at_valid(self) -> None:
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        ends = ts + timedelta(hours=1)
        event = TemporalEvent(
            id=uuid4(),
            description="Test",
            timestamp=ts,
            ends_at=ends,
            granularity=TemporalGranularity.PRECISE,
            source_snippet="snippet",
        )
        assert event.ends_at == ends

    def test_forecast_request_valid(self) -> None:
        req = ForecastRequest(history=[1.0, 2.0], prediction_length=5, confidence_level=0.9)
        assert req.prediction_length == 5

    def test_forecast_request_invalid_prediction_length(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ForecastRequest(history=[1.0], prediction_length=0, confidence_level=0.9)
        assert "prediction_length must be positive" in str(excinfo.value)

    def test_forecast_request_invalid_confidence_level(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ForecastRequest(history=[1.0], prediction_length=5, confidence_level=1.5)
        assert "confidence_level must be between 0.0 and 1.0" in str(excinfo.value)

        with pytest.raises(ValidationError) as excinfo:
            ForecastRequest(history=[1.0], prediction_length=5, confidence_level=0.0)
        assert "confidence_level must be between 0.0 and 1.0" in str(excinfo.value)
