from datetime import datetime
from enum import Enum


class IntervalRelation(str, Enum):
    """
    The 13 basic relations of Allen's Interval Algebra.
    """

    # X equals Y
    EQUALS = "EQUALS"

    # X before Y (Y after X)
    BEFORE = "BEFORE"
    AFTER = "AFTER"

    # X meets Y (Y met by X)
    MEETS = "MEETS"
    MET_BY = "MET_BY"

    # X overlaps Y (Y overlapped by X)
    OVERLAPS = "OVERLAPS"
    OVERLAPPED_BY = "OVERLAPPED_BY"

    # X starts Y (Y started by X)
    STARTS = "STARTS"
    STARTED_BY = "STARTED_BY"

    # X finishes Y (Y finished by X)
    FINISHES = "FINISHES"
    FINISHED_BY = "FINISHED_BY"

    # X during Y (Y contains X)
    DURING = "DURING"
    CONTAINS = "CONTAINS"


def get_interval_relation(start1: datetime, end1: datetime, start2: datetime, end2: datetime) -> IntervalRelation:
    """
    Determines the Allen Interval Algebra relationship between two intervals (X and Y).

    Args:
        start1: Start of interval X
        end1: End of interval X
        start2: Start of interval Y
        end2: End of interval Y

    Returns:
        The IntervalRelation describing X's relationship to Y.

    Raises:
        ValueError: If start >= end for either interval.
    """
    if start1 >= end1:
        raise ValueError(f"Interval 1 invalid: start ({start1}) must be strictly before end ({end1})")
    if start2 >= end2:
        raise ValueError(f"Interval 2 invalid: start ({start2}) must be strictly before end ({end2})")

    # X equals Y
    if start1 == start2 and end1 == end2:
        return IntervalRelation.EQUALS

    # X before Y
    if end1 < start2:
        return IntervalRelation.BEFORE

    # X after Y
    if start1 > end2:
        return IntervalRelation.AFTER

    # X meets Y
    if end1 == start2:
        return IntervalRelation.MEETS

    # X met by Y
    if start1 == end2:
        return IntervalRelation.MET_BY

    # X starts Y
    if start1 == start2 and end1 < end2:
        return IntervalRelation.STARTS

    # X started by Y
    if start1 == start2 and end1 > end2:
        return IntervalRelation.STARTED_BY

    # X finishes Y
    if end1 == end2 and start1 > start2:
        return IntervalRelation.FINISHES

    # X finished by Y
    if end1 == end2 and start1 < start2:
        return IntervalRelation.FINISHED_BY

    # X during Y
    if start1 > start2 and end1 < end2:
        return IntervalRelation.DURING

    # X contains Y
    if start1 < start2 and end1 > end2:
        return IntervalRelation.CONTAINS

    # X overlaps Y
    if start1 < start2 < end1 < end2:
        return IntervalRelation.OVERLAPS

    # X overlapped by Y
    if start2 < start1 < end2 < end1:
        return IntervalRelation.OVERLAPPED_BY

    # Should be unreachable if logic is complete, but for safety:
    raise ValueError(
        f"Could not determine relationship between [{start1}, {end1}] and [{start2}, {end2}]"
    )  # pragma: no cover
