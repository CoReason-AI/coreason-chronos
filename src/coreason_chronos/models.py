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
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class TemporalGranularity(str, Enum):
    """Enumeration for the granularity of a temporal event."""

    PRECISE = "PRECISE"  # "2024-01-01 10:00"
    DATE_ONLY = "DATE_ONLY"  # "2024-01-01"
    FUZZY = "FUZZY"  # "Early January 2024"


class TemporalEvent(BaseModel):
    """
    Represents a normalized event on the timeline.

    Attributes:
        id: Unique identifier for the event.
        description: Textual description of the event.
        timestamp: The normalized absolute timestamp (ISO 8601 UTC).
        granularity: How precise the timestamp is.
        duration_minutes: Optional duration of the event.
        ends_at: Optional end timestamp.
        source_snippet: The original text snippet that generated this event.
    """

    id: UUID
    description: str
    timestamp: datetime
    granularity: TemporalGranularity
    duration_minutes: Optional[int] = Field(default=None, ge=0)
    ends_at: Optional[datetime] = Field(default=None, validate_default=True)
    source_snippet: str

    @field_validator("timestamp", "ends_at")
    @classmethod
    def enforce_utc(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure that datetime fields are timezone-aware and set to UTC."""
        if v is None:
            return v
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        if v.tzinfo != timezone.utc:
            return v.astimezone(timezone.utc)
        return v

    @model_validator(mode="after")
    def check_duration_consistency(self) -> "TemporalEvent":
        """
        Validate consistency between timestamp, duration_minutes, and ends_at.
        If all three are present, timestamp + duration must equal ends_at.
        """
        if self.duration_minutes is not None and self.ends_at is not None and self.timestamp is not None:
            calculated_end = self.timestamp + timedelta(minutes=self.duration_minutes)
            # Allow for very minor precision differences if strictly necessary,
            # but date math should be exact here.
            if calculated_end != self.ends_at:
                raise ValueError("Inconsistent time definition: timestamp + duration_minutes != ends_at")
        return self


class ForecastRequest(BaseModel):
    """
    Request object for generating time-series forecasts.

    Attributes:
        history: Historical data points.
        prediction_length: Number of future steps to predict.
        confidence_level: The confidence interval (e.g., 0.90 for P90).
        covariates: Optional external factors affecting the forecast.
    """

    history: List[float] = Field(..., min_length=1)
    prediction_length: int = Field(..., gt=0)
    confidence_level: float = Field(..., gt=0.0, lt=1.0)
    covariates: Optional[List[int]] = None

    @field_validator("history")
    @classmethod
    def check_history_finite(cls, v: List[float]) -> List[float]:
        """Ensure history contains valid finite numbers."""
        import math

        for x in v:
            if not math.isfinite(x):
                raise ValueError("History must contain finite numbers (no NaN or Inf).")
        return v
