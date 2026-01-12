# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_chronos

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from coreason_chronos.models import ForecastRequest, TemporalEvent, TemporalGranularity


def test_temporal_granularity_values() -> None:
    """Test that TemporalGranularity has the correct values."""
    assert TemporalGranularity.PRECISE == "PRECISE"
    assert TemporalGranularity.DATE_ONLY == "DATE_ONLY"
    assert TemporalGranularity.FUZZY == "FUZZY"


def test_temporal_event_creation() -> None:
    """Test creating a valid TemporalEvent."""
    event_id = uuid4()
    now = datetime.now(timezone.utc)
    event = TemporalEvent(
        id=event_id,
        description="Test Event",
        timestamp=now,
        granularity=TemporalGranularity.PRECISE,
        source_snippet="test snippet",
    )

    assert event.id == event_id
    assert event.description == "Test Event"
    assert event.timestamp == now
    assert event.granularity == TemporalGranularity.PRECISE
    assert event.source_snippet == "test snippet"
    assert event.duration_minutes is None
    assert event.ends_at is None


def test_temporal_event_utc_enforcement() -> None:
    """Test that timestamps must be timezone aware and are converted to UTC."""
    event_id = uuid4()

    # Naive datetime should fail
    naive_dt = datetime(2024, 1, 1, 10, 0, 0)
    with pytest.raises(ValidationError, match="Timestamp must be timezone-aware"):
        TemporalEvent(
            id=event_id,
            description="Naive Event",
            timestamp=naive_dt,
            granularity=TemporalGranularity.PRECISE,
            source_snippet="snippet",
        )

    # Non-UTC aware datetime should be converted
    tz_offset = timezone(timedelta(hours=5))
    aware_dt = datetime(2024, 1, 1, 10, 0, 0, tzinfo=tz_offset)
    event = TemporalEvent(
        id=event_id,
        description="Aware Event",
        timestamp=aware_dt,
        granularity=TemporalGranularity.PRECISE,
        source_snippet="snippet",
    )

    assert event.timestamp.tzinfo == timezone.utc
    assert event.timestamp == aware_dt  # Equality holds across timezones

    # Already UTC should remain UTC
    utc_dt = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    event_utc = TemporalEvent(
        id=event_id,
        description="UTC Event",
        timestamp=utc_dt,
        granularity=TemporalGranularity.PRECISE,
        source_snippet="snippet",
    )
    assert event_utc.timestamp == utc_dt
    assert event_utc.timestamp.tzinfo == timezone.utc


def test_temporal_event_optional_fields() -> None:
    """Test TemporalEvent with optional fields populated."""
    now = datetime.now(timezone.utc)
    # Ensure consistency: ends_at = timestamp + duration (60m)
    end = now + timedelta(minutes=60)

    event = TemporalEvent(
        id=uuid4(),
        description="Event with duration",
        timestamp=now,
        granularity=TemporalGranularity.DATE_ONLY,
        source_snippet="duration snippet",
        duration_minutes=60,
        ends_at=end,
    )

    assert event.duration_minutes == 60
    assert event.ends_at == end


def test_temporal_event_consistency_validation() -> None:
    """Test that inconsistent duration/ends_at raises error."""
    now = datetime.now(timezone.utc)
    end = now + timedelta(minutes=60)

    # Mismatch: Duration says 30 mins, ends_at says 60 mins later
    with pytest.raises(ValidationError, match="Inconsistent time definition"):
        TemporalEvent(
            id=uuid4(),
            description="Inconsistent Event",
            timestamp=now,
            granularity=TemporalGranularity.PRECISE,
            source_snippet="snippet",
            duration_minutes=30,
            ends_at=end,
        )


def test_temporal_event_negative_duration() -> None:
    """Test that duration cannot be negative."""
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        TemporalEvent(
            id=uuid4(),
            description="Negative Duration",
            timestamp=now,
            granularity=TemporalGranularity.PRECISE,
            source_snippet="snippet",
            duration_minutes=-10,
        )


def test_forecast_request_creation() -> None:
    """Test creating a valid ForecastRequest."""
    request = ForecastRequest(
        history=[10.0, 20.0, 30.0],
        prediction_length=12,
        confidence_level=0.95,
    )

    assert request.history == [10.0, 20.0, 30.0]
    assert request.prediction_length == 12
    assert request.confidence_level == 0.95
    assert request.covariates is None


def test_forecast_request_validation_errors() -> None:
    """Test validation constraints for ForecastRequest."""

    # Empty history
    with pytest.raises(ValidationError):
        ForecastRequest(
            history=[],
            prediction_length=5,
            confidence_level=0.9,
        )

    # Infinite history
    with pytest.raises(ValidationError, match="History must contain finite numbers"):
        ForecastRequest(
            history=[10.0, float("inf")],
            prediction_length=5,
            confidence_level=0.9,
        )

    # Zero prediction length
    with pytest.raises(ValidationError):
        ForecastRequest(
            history=[10.0],
            prediction_length=0,
            confidence_level=0.9,
        )

    # Invalid confidence level (>= 1.0)
    with pytest.raises(ValidationError):
        ForecastRequest(
            history=[10.0],
            prediction_length=5,
            confidence_level=1.0,
        )

    # Invalid confidence level (<= 0.0)
    with pytest.raises(ValidationError):
        ForecastRequest(
            history=[10.0],
            prediction_length=5,
            confidence_level=0.0,
        )
