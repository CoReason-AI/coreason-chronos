# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_chronos

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel


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
    duration_minutes: Optional[int] = None
    ends_at: Optional[datetime] = None
    source_snippet: str


class ForecastRequest(BaseModel):
    """
    Request object for generating time-series forecasts.

    Attributes:
        history: Historical data points.
        prediction_length: Number of future steps to predict.
        confidence_level: The confidence interval (e.g., 0.90 for P90).
        covariates: Optional external factors affecting the forecast.
    """

    history: List[float]
    prediction_length: int
    confidence_level: float
    covariates: Optional[List[int]] = None
