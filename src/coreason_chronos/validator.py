from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional

from coreason_chronos.schemas import ComplianceResult


class ValidationRule(ABC):
    """
    Abstract base class for compliance validation rules (The Compliance Clock).
    """

    @abstractmethod
    def validate(self, target_time: datetime, reference_time: datetime) -> ComplianceResult:
        """
        Validates the target_time against the reference_time according to the rule logic.

        Args:
            target_time: The time of the event being checked (e.g., Report Submission Time).
            reference_time: The base time for the rule (e.g., Adverse Event Occurrence Time).

        Returns:
            ComplianceResult object indicating status (pass/fail), drift, and a descriptive message.
        """
        pass  # pragma: no cover


class MaxDelayRule(ValidationRule):
    """
    A specific rule ensuring an event happens within a maximum allowed delay after a reference event.

    Formula: target_time <= reference_time + max_delay
    """

    def __init__(self, max_delay: timedelta, name: Optional[str] = None) -> None:
        """
        Initializes the MaxDelayRule.

        Args:
            max_delay: The maximum allowed duration between reference_time and target_time.
            name: Optional descriptive name for the rule.

        Raises:
            ValueError: If max_delay is negative.
        """
        if max_delay.total_seconds() < 0:
            raise ValueError("max_delay must be non-negative")
        self.max_delay = max_delay
        self.name = name or f"Max Delay {max_delay}"

    def validate(self, target_time: datetime, reference_time: datetime) -> ComplianceResult:
        """
        Checks if the target event occurred within the max_delay window of the reference event.

        Args:
            target_time: The timestamp to validate.
            reference_time: The start of the window.

        Returns:
            ComplianceResult with drift calculated as (target_time - deadline).

        Raises:
            ValueError: If inputs are not timezone-aware.
        """
        if target_time.tzinfo is None or reference_time.tzinfo is None:
            raise ValueError("Both target_time and reference_time must be timezone-aware")

        # Normalize to UTC for absolute time arithmetic (avoids DST wall-clock issues)
        ref_utc = reference_time.astimezone(timezone.utc)
        target_utc = target_time.astimezone(timezone.utc)

        deadline = ref_utc + self.max_delay
        drift = target_utc - deadline
        is_compliant = target_utc <= deadline

        message = None
        if not is_compliant:
            message = f"Violation: Event occurred {drift} after the deadline."

        return ComplianceResult(
            is_compliant=is_compliant,
            drift=drift,
            message=message,
        )
