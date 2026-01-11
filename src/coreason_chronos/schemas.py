from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


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

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return v

    @field_validator("duration_minutes")
    @classmethod
    def duration_must_be_non_negative(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("duration_minutes must be non-negative")
        return v

    @model_validator(mode="after")
    def ends_at_must_be_after_timestamp(self) -> "TemporalEvent":
        if self.ends_at is not None:
            # Ensure ends_at is timezone aware if timestamp is (which is enforced above)
            # If timestamp passed validation, it has tzinfo.
            # We should check ends_at tzinfo too if we want strictness, or pydantic handles it.
            # Let's check logic first.
            if self.ends_at <= self.timestamp:
                raise ValueError("ends_at must be after timestamp")
        return self


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

    @field_validator("prediction_length")
    @classmethod
    def prediction_length_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("prediction_length must be positive")
        return v

    @field_validator("confidence_level")
    @classmethod
    def confidence_level_must_be_valid(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("confidence_level must be between 0.0 and 1.0")
        return v


class ForecastResult(BaseModel):
    """
    Result payload from the forecasting model.
    """

    model_config = ConfigDict(frozen=True)

    median: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    confidence_level: float
