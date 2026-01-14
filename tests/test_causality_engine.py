from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

from coreason_chronos.causality import CausalityEngine
from coreason_chronos.schemas import TemporalEvent, TemporalGranularity


class TestCausalityEngine:
    def setup_method(self) -> None:
        self.engine = CausalityEngine()
        self.base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def _create_event(self, description: str, offset_start_hours: float, duration_minutes: int = 60) -> TemporalEvent:
        start = self.base + timedelta(hours=offset_start_hours)
        return TemporalEvent(
            id=uuid4(),
            description=description,
            timestamp=start,
            duration_minutes=duration_minutes,
            granularity=TemporalGranularity.PRECISE,
            source_snippet="test",
        )

    def test_plausible_cause_before(self) -> None:
        # Cause: [12:00, 13:00], Effect: [14:00, 15:00]
        cause = self._create_event("Cause", 0, 60)
        effect = self._create_event("Effect", 2, 60)
        assert self.engine.is_plausible_cause(cause, effect) is True

    def test_plausible_cause_meets(self) -> None:
        # Cause: [12:00, 13:00], Effect: [13:00, 14:00]
        cause = self._create_event("Cause", 0, 60)
        effect = self._create_event("Effect", 1, 60)
        assert self.engine.is_plausible_cause(cause, effect) is True

    def test_plausible_cause_overlaps(self) -> None:
        # Cause: [12:00, 14:00], Effect: [13:00, 15:00]
        cause = self._create_event("Cause", 0, 120)
        effect = self._create_event("Effect", 1, 120)
        assert self.engine.is_plausible_cause(cause, effect) is True

    def test_plausible_cause_contains(self) -> None:
        # Cause: [12:00, 15:00], Effect: [13:00, 14:00]
        cause = self._create_event("Cause", 0, 180)
        effect = self._create_event("Effect", 1, 60)
        assert self.engine.is_plausible_cause(cause, effect) is True

    def test_plausible_cause_starts(self) -> None:
        # Cause: [12:00, 13:00], Effect: [12:00, 14:00]
        cause = self._create_event("Cause", 0, 60)
        effect = self._create_event("Effect", 0, 120)
        assert self.engine.is_plausible_cause(cause, effect) is True

    def test_implausible_cause_after(self) -> None:
        # Cause: [14:00, 15:00], Effect: [12:00, 13:00]
        cause = self._create_event("Cause", 2, 60)
        effect = self._create_event("Effect", 0, 60)
        assert self.engine.is_plausible_cause(cause, effect) is False

    def test_implausible_cause_during(self) -> None:
        # Cause: [13:00, 14:00], Effect: [12:00, 15:00]
        # Cause starts strictly AFTER Effect starts.
        cause = self._create_event("Cause", 1, 60)
        effect = self._create_event("Effect", 0, 180)
        assert self.engine.is_plausible_cause(cause, effect) is False

    def test_point_events_handling(self) -> None:
        # Point events (duration 0 implies epsilon duration in engine)

        # Cause at 12:00, Effect at 12:01 -> Plausible
        cause = self._create_event("Point Cause", 0, 0)
        effect = self._create_event("Point Effect", 0.02, 0)  # 1.2 min later
        assert self.engine.is_plausible_cause(cause, effect) is True

        # Cause at 12:01, Effect at 12:00 -> Implausible
        cause = self._create_event("Point Cause Late", 0.02, 0)
        effect = self._create_event("Point Effect Early", 0, 0)
        assert self.engine.is_plausible_cause(cause, effect) is False

        # Simultaneous Point Events -> Plausible (e.g. EQUALS or OVERLAPS depending on microsecond math)
        # 12:00 and 12:00
        cause = self._create_event("Point A", 0, 0)
        effect = self._create_event("Point B", 0, 0)
        assert self.engine.is_plausible_cause(cause, effect) is True

    def test_error_handling(self) -> None:
        """Test that errors in get_relation are handled and return False."""
        cause = self._create_event("Cause", 0, 60)
        effect = self._create_event("Effect", 2, 60)

        with patch("coreason_chronos.causality.CausalityEngine.get_relation", side_effect=ValueError("Test Error")):
            assert self.engine.is_plausible_cause(cause, effect) is False

    def test_explicit_ends_at(self) -> None:
        """Test event with explicit ends_at instead of duration."""
        start = self.base
        end = self.base + timedelta(hours=1)
        event = TemporalEvent(
            id=uuid4(),
            description="Explicit End",
            timestamp=start,
            ends_at=end,
            granularity=TemporalGranularity.PRECISE,
            source_snippet="test"
        )

        # Self-relation should be EQUALS
        assert self.engine.get_relation(event, event) == "EQUALS"
