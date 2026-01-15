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
            source_snippet="test",
        )

        # Self-relation should be EQUALS
        assert self.engine.get_relation(event, event) == "EQUALS"

    def test_timezone_crossing_plausibility(self) -> None:
        """
        Test that is_plausible_cause respects absolute time across timezones.
        """
        # UTC Time: 12:00
        utc_start = self.base

        # JST (UTC+9) Time: 21:00 (Same as 12:00 UTC)
        jst_tz = timezone(timedelta(hours=9))
        jst_start = datetime(2024, 1, 1, 21, 0, 0, tzinfo=jst_tz)

        # EST (UTC-5) Time: 07:00 (Same as 12:00 UTC)
        est_tz = timezone(timedelta(hours=-5))
        est_start = datetime(2024, 1, 1, 7, 0, 0, tzinfo=est_tz)

        # Cause in EST (07:00 = 12:00 UTC), Effect in UTC (12:00)
        # Should be EQUALS/STARTS -> Plausible
        cause = TemporalEvent(
            id=uuid4(), description="Cause EST", timestamp=est_start, duration_minutes=60,
            granularity=TemporalGranularity.PRECISE, source_snippet="test"
        )
        effect = TemporalEvent(
            id=uuid4(), description="Effect UTC", timestamp=utc_start, duration_minutes=60,
            granularity=TemporalGranularity.PRECISE, source_snippet="test"
        )
        assert self.engine.is_plausible_cause(cause, effect) is True

        # Cause in UTC (12:00), Effect in JST (20:00 = 11:00 UTC)
        # 12:00 > 11:00 -> Implausible (Cause happens AFTER Effect)
        jst_early = datetime(2024, 1, 1, 20, 0, 0, tzinfo=jst_tz)
        effect_early = TemporalEvent(
            id=uuid4(), description="Effect JST Early", timestamp=jst_early, duration_minutes=60,
            granularity=TemporalGranularity.PRECISE, source_snippet="test"
        )
        assert self.engine.is_plausible_cause(cause, effect_early) is False

    def test_interval_resolution_precedence(self) -> None:
        """
        Test that ends_at takes precedence over duration_minutes in _resolve_interval.
        """
        start = self.base
        # Duration says 5 hours (ends 17:00)
        duration = 300
        # ends_at says 1 hour (ends 13:00)
        explicit_end = self.base + timedelta(hours=1)

        event = TemporalEvent(
            id=uuid4(),
            description="Ambiguous Event",
            timestamp=start,
            duration_minutes=duration,
            ends_at=explicit_end,
            granularity=TemporalGranularity.PRECISE,
            source_snippet="test"
        )

        # If it used duration, it would be [12:00, 17:00].
        # If it uses ends_at, it is [12:00, 13:00].

        # Compare with an event at 13:00-14:00.
        # If precedence is ends_at: [12,13] meets [13,14] -> MEETS
        # If precedence is duration: [12,17] overlaps [13,14] -> CONTAINS/OVERLAPS? [13,14] is inside [12,17] -> CONTAINS.

        target_event = self._create_event("Target", 1, 60) # [13:00, 14:00]

        relation = self.engine.get_relation(event, target_event)

        # Expecting MEETS because logic checks ends_at first
        assert relation == "MEETS"

    def test_causality_transitivity(self) -> None:
        """
        Test chain of events A -> B -> C.
        """
        # A: 10:00-11:00
        event_a = self._create_event("A", -2, 60)
        # B: 11:00-12:00
        event_b = self._create_event("B", -1, 60)
        # C: 12:00-13:00
        event_c = self._create_event("C", 0, 60)

        # A meets B (Plausible)
        assert self.engine.is_plausible_cause(event_a, event_b) is True
        # B meets C (Plausible)
        assert self.engine.is_plausible_cause(event_b, event_c) is True

        # A before C (Plausible)
        assert self.engine.is_plausible_cause(event_a, event_c) is True
