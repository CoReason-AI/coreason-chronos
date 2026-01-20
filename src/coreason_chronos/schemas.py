from datetime import datetime, timedelta
from enum import Enum
from math import isinf, isnan
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class TemporalGranularity(str, Enum):
    """
    Enumeration representing the granularity or precision of a temporal event.

    Attributes:
        PRECISE: Exact timestamp (e.g., "2024-01-01 10:00").
        DATE_ONLY: Date precision only (e.g., "2024-01-01").
        FUZZY: Approximate or descriptive time (e.g., "Early January 2024").
    """

    PRECISE = "PRECISE"
    DATE_ONLY = "DATE_ONLY"
    FUZZY = "FUZZY"


class TemporalEvent(BaseModel):
    """
    Represents a single discrete event in a timeline, including its temporal attributes.

    This model supports both point events and interval events (via `ends_at` or `duration_minutes`).
    It serves as the core data structure for the 'Longitudinal Reconstruction' capability.

    Attributes:
        id: Unique identifier for the event.
        description: Natural language description of the event (e.g., "Headache onset").
        timestamp: The start time of the event (must be timezone-aware).
        granularity: The precision level of the timestamp.
        duration_minutes: Optional duration of the event in minutes.
        ends_at: Optional explicit end time of the event.
        source_snippet: The original text snippet from which this event was extracted.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    description: str
    timestamp: datetime
    granularity: TemporalGranularity

    # Allen's Algebra
    duration_minutes: Optional[int] = None
    ends_at: Optional[datetime] = None

    source_snippet: str

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, v: datetime) -> datetime:
        """
        Validates that the timestamp is timezone-aware.

        Args:
            v: The datetime object to validate.

        Returns:
            The validated datetime object.

        Raises:
            ValueError: If the timestamp is naive (no timezone info).
        """
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return v

    @field_validator("duration_minutes")
    @classmethod
    def duration_must_be_non_negative(cls, v: Optional[int]) -> Optional[int]:
        """
        Validates that the duration is non-negative.

        Args:
            v: The duration in minutes.

        Returns:
            The validated duration.

        Raises:
            ValueError: If the duration is negative.
        """
        if v is not None and v < 0:
            raise ValueError("duration_minutes must be non-negative")
        return v

    @model_validator(mode="after")
    def ends_at_must_be_after_timestamp(self) -> "TemporalEvent":
        """
        Validates that the end time is chronologically after the start time.

        Returns:
            The validated TemporalEvent instance.

        Raises:
            ValueError: If ends_at is less than or equal to timestamp.
        """
        if self.ends_at is not None:
            if self.ends_at <= self.timestamp:
                raise ValueError("ends_at must be after timestamp")
        return self


class ForecastRequest(BaseModel):
    """
    Represents a request payload for the forecasting model (The Oracle).

    Attributes:
        history: A sequence of historical data points (floats).
        prediction_length: The number of future time steps to predict.
        confidence_level: The desired probability for the prediction interval (e.g., 0.90 for P90).
        covariates: Optional external factors influencing the forecast (e.g., [0, 1] for holidays).
    """

    model_config = ConfigDict(frozen=True)

    history: List[float]
    prediction_length: int
    confidence_level: float

    # SOTA: Contextual Covariates
    covariates: Optional[List[int]] = None

    @field_validator("history")
    @classmethod
    def history_must_be_valid(cls, v: List[float]) -> List[float]:
        """
        Validates that the history list is not empty and contains valid numbers.

        Args:
            v: The list of historical values.

        Returns:
            The validated list.

        Raises:
            ValueError: If the list is empty or contains NaN/Inf values.
        """
        if not v:
            raise ValueError("history must not be empty")
        for x in v:
            if isnan(x) or isinf(x):
                raise ValueError("history must not contain NaN or Inf values")
        return v

    @field_validator("prediction_length")
    @classmethod
    def prediction_length_must_be_positive(cls, v: int) -> int:
        """
        Validates that the prediction horizon is positive.

        Args:
            v: The number of steps to predict.

        Returns:
            The validated integer.

        Raises:
            ValueError: If prediction_length is <= 0.
        """
        if v <= 0:
            raise ValueError("prediction_length must be positive")
        return v

    @field_validator("confidence_level")
    @classmethod
    def confidence_level_must_be_valid(cls, v: float) -> float:
        """
        Validates that the confidence level is between 0 and 1 exclusive.

        Args:
            v: The confidence probability.

        Returns:
            The validated float.

        Raises:
            ValueError: If not in (0.0, 1.0).
        """
        if not (0.0 < v < 1.0):
            raise ValueError("confidence_level must be between 0.0 and 1.0")
        return v


class ForecastResult(BaseModel):
    """
    Represents the output from the forecasting model.

    Attributes:
        median: The median predicted values (point forecast).
        lower_bound: The lower bound of the confidence interval.
        upper_bound: The upper bound of the confidence interval.
        confidence_level: The confidence level used for the intervals.
    """

    model_config = ConfigDict(frozen=True)

    median: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    confidence_level: float


class ComplianceResult(BaseModel):
    """
    Represents the result of a temporal compliance check.

    Attributes:
        is_compliant: True if the rule was satisfied, False otherwise.
        drift: The time difference between the target and the deadline/reference.
        message: A human-readable explanation of the result, especially on failure.
    """

    model_config = ConfigDict(frozen=True)

    is_compliant: bool
    drift: timedelta
    message: Optional[str] = None
