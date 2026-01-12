from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

from coreason_chronos.schemas import ComplianceResult


class ValidationRule(ABC):
    """
    Abstract base class for compliance rules.
    """

    @abstractmethod
    def validate(self, target_time: datetime, reference_time: datetime) -> ComplianceResult:
        """
        Validates the target_time against the reference_time according to the rule.

        Args:
            target_time: The time of the event being checked (e.g. Report Time).
            reference_time: The base time for the rule (e.g. Occurrence Time).

        Returns:
            ComplianceResult indicating status and drift.
        """
        pass  # pragma: no cover


class MaxDelayRule(ValidationRule):
    """
    Rule ensuring an event happens within a maximum delay after a reference event.
    Formula: target_time <= reference_time + max_delay
    """

    def __init__(self, max_delay: timedelta, name: Optional[str] = None) -> None:
        if max_delay.total_seconds() < 0:
            raise ValueError("max_delay must be non-negative")
        self.max_delay = max_delay
        self.name = name or f"Max Delay {max_delay}"

    def validate(self, target_time: datetime, reference_time: datetime) -> ComplianceResult:
        if target_time.tzinfo is None or reference_time.tzinfo is None:
            raise ValueError("Both target_time and reference_time must be timezone-aware")

        # Normalize to UTC for absolute time arithmetic (avoids DST wall-clock issues)
        from datetime import timezone

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
