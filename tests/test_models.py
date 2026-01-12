# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_chronos

from datetime import datetime, timezone
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


def test_temporal_event_optional_fields() -> None:
    """Test TemporalEvent with optional fields populated."""
    now = datetime.now(timezone.utc)
    end = datetime.now(timezone.utc)
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


def test_temporal_event_validation_error() -> None:
    """Test validation failure when required fields are missing."""
    with pytest.raises(ValidationError):
        # Missing required fields
        TemporalEvent(
            id=uuid4(),
            description="Invalid Event",
        )  # type: ignore


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


def test_forecast_request_with_covariates() -> None:
    """Test ForecastRequest with covariates."""
    request = ForecastRequest(
        history=[1.0, 2.0],
        prediction_length=5,
        confidence_level=0.90,
        covariates=[1, 0, 1],
    )

    assert request.covariates == [1, 0, 1]


def test_forecast_request_validation_error() -> None:
    """Test validation failure for ForecastRequest."""
    with pytest.raises(ValidationError):
        # Missing history
        ForecastRequest(
            prediction_length=5,
            confidence_level=0.90,
        )  # type: ignore
