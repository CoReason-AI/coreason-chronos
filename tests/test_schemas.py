from datetime import datetime, timedelta, timezone
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

    def test_timestamp_timezone_aware(self) -> None:
        """Test that naive timestamp raises ValidationError"""
        with pytest.raises(ValidationError) as excinfo:
            TemporalEvent(
                id=uuid4(),
                description="Naive Timestamp",
                timestamp=datetime(2024, 1, 1, 10, 0, 0),  # Naive
                granularity=TemporalGranularity.PRECISE,
                source_snippet="snippet",
            )
        assert "timestamp must be timezone-aware" in str(excinfo.value)

    def test_duration_non_negative(self) -> None:
        """Test that negative duration raises ValidationError"""
        with pytest.raises(ValidationError) as excinfo:
            TemporalEvent(
                id=uuid4(),
                description="Negative Duration",
                timestamp=datetime.now(timezone.utc),
                granularity=TemporalGranularity.PRECISE,
                duration_minutes=-10,
                source_snippet="snippet",
            )
        assert "duration_minutes must be non-negative" in str(excinfo.value)

    def test_ends_at_after_timestamp(self) -> None:
        """Test that ends_at before timestamp raises ValidationError"""
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValidationError) as excinfo:
            TemporalEvent(
                id=uuid4(),
                description="Invalid Ends At",
                timestamp=ts,
                granularity=TemporalGranularity.PRECISE,
                ends_at=ts - timedelta(hours=1),
                source_snippet="snippet",
            )
        assert "ends_at must be after timestamp" in str(excinfo.value)

    def test_serialization_roundtrip(self) -> None:
        """Complex scenario: Verify strict JSON serialization and deserialization."""
        original = TemporalEvent(
            id=uuid4(),
            description="Roundtrip Event",
            timestamp=datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc),
            granularity=TemporalGranularity.PRECISE,
            duration_minutes=120,
            ends_at=datetime(2024, 6, 15, 16, 30, 0, tzinfo=timezone.utc),
            source_snippet="roundtrip test",
        )

        json_str = original.model_dump_json()
        restored = TemporalEvent.model_validate_json(json_str)

        assert original == restored
        assert restored.timestamp.tzinfo is not None


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

    def test_prediction_length_positive(self) -> None:
        """Test that prediction_length must be positive."""
        with pytest.raises(ValidationError) as excinfo:
            ForecastRequest(history=[1.0, 2.0], prediction_length=0, confidence_level=0.9)
        assert "prediction_length must be positive" in str(excinfo.value)

        with pytest.raises(ValidationError) as excinfo:
            ForecastRequest(history=[1.0, 2.0], prediction_length=-5, confidence_level=0.9)
        assert "prediction_length must be positive" in str(excinfo.value)

    def test_confidence_level_range(self) -> None:
        """Test that confidence_level must be between 0 and 1."""
        with pytest.raises(ValidationError) as excinfo:
            ForecastRequest(history=[1.0], prediction_length=1, confidence_level=1.1)
        assert "confidence_level must be between 0.0 and 1.0" in str(excinfo.value)

        with pytest.raises(ValidationError) as excinfo:
            ForecastRequest(history=[1.0], prediction_length=1, confidence_level=-0.1)
        assert "confidence_level must be between 0.0 and 1.0" in str(excinfo.value)
