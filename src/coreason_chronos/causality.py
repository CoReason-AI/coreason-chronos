from datetime import datetime, timedelta
from enum import Enum

from coreason_chronos.schemas import TemporalEvent
from coreason_chronos.utils.logger import logger


class AllenRelation(str, Enum):
    """
    The 13 basic relations of Allen's Interval Algebra.
    """

    BEFORE = "BEFORE"
    AFTER = "AFTER"
    MEETS = "MEETS"
    MET_BY = "MET_BY"
    OVERLAPS = "OVERLAPS"
    OVERLAPPED_BY = "OVERLAPPED_BY"
    STARTS = "STARTS"
    STARTED_BY = "STARTED_BY"
    FINISHES = "FINISHES"
    FINISHED_BY = "FINISHED_BY"
    DURING = "DURING"
    CONTAINS = "CONTAINS"
    EQUALS = "EQUALS"


def get_interval_relation(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> AllenRelation:
    """
    Determines the Allen Interval Algebra relation between two intervals A and B.

    Args:
        start_a: Start time of interval A.
        end_a: End time of interval A.
        start_b: Start time of interval B.
        end_b: End time of interval B.

    Returns:
        The AllenRelation describing the relationship A -> B.

    Raises:
        ValueError: If any datetime is naive, or if any interval is invalid (start >= end).
    """
    # Validation
    for dt, name in [(start_a, "start_a"), (end_a, "end_a"), (start_b, "start_b"), (end_b, "end_b")]:
        if dt.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware")

    if start_a >= end_a:
        raise ValueError("Interval A is invalid: start_a must be strictly before end_a (no point events allowed)")
    if start_b >= end_b:
        raise ValueError("Interval B is invalid: start_b must be strictly before end_b (no point events allowed)")

    # Logic
    # 1. BEFORE (A < B): A ends before B starts
    if end_a < start_b:
        return AllenRelation.BEFORE

    # 2. AFTER (A > B): A starts after B ends
    if start_a > end_b:
        return AllenRelation.AFTER

    # 3. MEETS (A m B): A ends exactly when B starts
    if end_a == start_b:
        return AllenRelation.MEETS

    # 4. MET_BY (A mi B): A starts exactly when B ends
    if start_a == end_b:
        return AllenRelation.MET_BY

    # 5. OVERLAPS (A o B): A starts before B, A ends after B starts but before B ends
    if start_a < start_b and start_b < end_a < end_b:
        return AllenRelation.OVERLAPS

    # 6. OVERLAPPED_BY (A oi B): B starts before A, B ends after A starts but before A ends
    if start_b < start_a and start_a < end_b < end_a:
        return AllenRelation.OVERLAPPED_BY

    # 7. STARTS (A s B): A and B start together, A ends before B
    if start_a == start_b and end_a < end_b:
        return AllenRelation.STARTS

    # 8. STARTED_BY (A si B): A and B start together, A ends after B
    if start_a == start_b and end_a > end_b:
        return AllenRelation.STARTED_BY

    # 9. FINISHES (A f B): A ends with B, A starts after B starts
    if end_a == end_b and start_a > start_b:
        return AllenRelation.FINISHES

    # 10. FINISHED_BY (A fi B): A ends with B, A starts before B starts
    if end_a == end_b and start_a < start_b:
        return AllenRelation.FINISHED_BY

    # 11. DURING (A d B): A starts after B starts and ends before B ends
    if start_a > start_b and end_a < end_b:
        return AllenRelation.DURING

    # 12. CONTAINS (A di B): A starts before B starts and ends after B ends
    if start_a < start_b and end_a > end_b:
        return AllenRelation.CONTAINS

    # 13. EQUALS (A = B): A and B start and end together
    if start_a == start_b and end_a == end_b:
        return AllenRelation.EQUALS

    # Should be unreachable if logic is complete, but let's see.
    # Are there any other cases?
    # Logic covers:
    # Disjoint: Before, After
    # Touching: Meets, Met By
    # Overlapping boundaries: Overlaps, Overlapped By
    # Matching Start: Starts, Started By, Equals
    # Matching End: Finishes, Finished By, Equals
    # Nested: During, Contains

    # It seems complete.
    raise ValueError("Unable to determine relation. This code path should be unreachable.")  # pragma: no cover


class CausalityEngine:
    """
    High-level engine for determining causal relationships between events.
    """

    def _resolve_interval(self, event: TemporalEvent) -> tuple[datetime, datetime]:
        """
        Resolves a TemporalEvent into a strict [start, end) interval.
        Point events are converted to [timestamp, timestamp + 1 microsecond].
        """
        start = event.timestamp
        end = None

        if event.ends_at:
            end = event.ends_at
        elif event.duration_minutes is not None:
            end = start + timedelta(minutes=event.duration_minutes)

        # Handle point events (or if calculated end <= start due to 0 duration)
        if end is None or end <= start:
            # Promote to epsilon interval for Algebra compliance
            end = start + timedelta(microseconds=1)

        return start, end

    def get_relation(self, event_a: TemporalEvent, event_b: TemporalEvent) -> AllenRelation:
        """
        Determines the Allen Relation between two TemporalEvents.
        """
        start_a, end_a = self._resolve_interval(event_a)
        start_b, end_b = self._resolve_interval(event_b)

        return get_interval_relation(start_a, end_a, start_b, end_b)

    def is_plausible_cause(self, cause: TemporalEvent, effect: TemporalEvent) -> bool:
        """
        Determines if 'cause' is temporally plausible as a cause for 'effect'.

        Plausibility Rule: Cause must start BEFORE or SIMULTANEOUSLY with the Effect's start.
        (Cause Start <= Effect Start)

        This maps to the following Allen Relations:
        - BEFORE (Cause < Effect)
        - MEETS (Cause touches Effect start)
        - OVERLAPS (Cause starts before, overlaps Effect)
        - FINISHED_BY (Cause ends with Effect, starts before)
        - CONTAINS (Cause contains Effect)
        - STARTS (Cause starts with Effect)
        - STARTED_BY (Cause starts with Effect)
        - EQUALS (Same interval)

        Implausible:
        - AFTER
        - MET_BY
        - OVERLAPPED_BY
        - FINISHES
        - DURING (Cause starts AFTER Effect starts)
        """
        try:
            relation = self.get_relation(cause, effect)
        except ValueError as e:
            logger.error(f"Failed to calculate interval relation: {e}")
            return False

        plausible_relations = {
            AllenRelation.BEFORE,
            AllenRelation.MEETS,
            AllenRelation.OVERLAPS,
            AllenRelation.FINISHED_BY,
            AllenRelation.CONTAINS,
            AllenRelation.STARTS,
            AllenRelation.STARTED_BY,
            AllenRelation.EQUALS,
        }

        is_plausible = relation in plausible_relations
        logger.debug(
            f"Checking causality: '{cause.description}' vs '{effect.description}' -> {relation}"
            f" -> Plausible: {is_plausible}"
        )
        return is_plausible
