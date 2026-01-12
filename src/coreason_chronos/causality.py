from datetime import datetime
from enum import Enum


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
