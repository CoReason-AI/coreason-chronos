from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TemporalGranularity(str, Enum):
    """
    Granularity of a temporal event.
    """

    PRECISE = "PRECISE"  # "2024-01-01 10:00"
    DATE_ONLY = "DATE_ONLY"  # "2024-01-01"
    FUZZY = "FUZZY"  # "Early January 2024"


class TemporalEvent(BaseModel):
    """
    Represents a single event in the timeline.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    description: str  # "Headache onset"
    timestamp: datetime
    granularity: TemporalGranularity

    # Allen's Algebra
    duration_minutes: Optional[int] = None
    ends_at: Optional[datetime] = None

    source_snippet: str  # "onset 2 hours later"


class ForecastRequest(BaseModel):
    """
    Request payload for the forecasting model.
    """

    model_config = ConfigDict(frozen=True)

    history: List[float]  # [10, 12, 15, 18...]
    prediction_length: int  # 12 steps
    confidence_level: float  # 0.90 (P90)

    # SOTA: Contextual Covariates
    # e.g., "Is this a holiday?"
    covariates: Optional[List[int]] = None
