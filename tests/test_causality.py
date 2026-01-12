from datetime import datetime, timedelta, timezone

import pytest

from coreason_chronos.causality import AllenRelation, get_interval_relation


# Helper for creating UTC datetimes
def dt(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 1, 1, hour, minute, tzinfo=timezone.utc)


def test_before() -> None:
    # A: 10:00-11:00, B: 12:00-13:00
    r = get_interval_relation(dt(10), dt(11), dt(12), dt(13))
    assert r == AllenRelation.BEFORE


def test_after() -> None:
    # A: 12:00-13:00, B: 10:00-11:00
    r = get_interval_relation(dt(12), dt(13), dt(10), dt(11))
    assert r == AllenRelation.AFTER


def test_meets() -> None:
    # A: 10:00-11:00, B: 11:00-12:00
    r = get_interval_relation(dt(10), dt(11), dt(11), dt(12))
    assert r == AllenRelation.MEETS


def test_met_by() -> None:
    # A: 11:00-12:00, B: 10:00-11:00
    r = get_interval_relation(dt(11), dt(12), dt(10), dt(11))
    assert r == AllenRelation.MET_BY


def test_overlaps() -> None:
    # A: 10:00-12:00, B: 11:00-13:00
    r = get_interval_relation(dt(10), dt(12), dt(11), dt(13))
    assert r == AllenRelation.OVERLAPS


def test_overlapped_by() -> None:
    # A: 11:00-13:00, B: 10:00-12:00
    r = get_interval_relation(dt(11), dt(13), dt(10), dt(12))
    assert r == AllenRelation.OVERLAPPED_BY


def test_starts() -> None:
    # A: 10:00-11:00, B: 10:00-12:00
    r = get_interval_relation(dt(10), dt(11), dt(10), dt(12))
    assert r == AllenRelation.STARTS


def test_started_by() -> None:
    # A: 10:00-12:00, B: 10:00-11:00
    r = get_interval_relation(dt(10), dt(12), dt(10), dt(11))
    assert r == AllenRelation.STARTED_BY


def test_finishes() -> None:
    # A: 11:00-12:00, B: 10:00-12:00
    r = get_interval_relation(dt(11), dt(12), dt(10), dt(12))
    assert r == AllenRelation.FINISHES


def test_finished_by() -> None:
    # A: 10:00-12:00, B: 11:00-12:00
    r = get_interval_relation(dt(10), dt(12), dt(11), dt(12))
    assert r == AllenRelation.FINISHED_BY


def test_during() -> None:
    # A: 11:00-12:00, B: 10:00-13:00
    r = get_interval_relation(dt(11), dt(12), dt(10), dt(13))
    assert r == AllenRelation.DURING


def test_contains() -> None:
    # A: 10:00-13:00, B: 11:00-12:00
    r = get_interval_relation(dt(10), dt(13), dt(11), dt(12))
    assert r == AllenRelation.CONTAINS


def test_equals() -> None:
    # A: 10:00-11:00, B: 10:00-11:00
    r = get_interval_relation(dt(10), dt(11), dt(10), dt(11))
    assert r == AllenRelation.EQUALS


def test_invalid_naive_datetime() -> None:
    naive = datetime(2024, 1, 1, 10, 0)
    aware = dt(10)
    with pytest.raises(ValueError, match="must be timezone-aware"):
        get_interval_relation(naive, aware, aware, aware)


def test_invalid_point_event_A() -> None:
    # start == end
    with pytest.raises(ValueError, match="Interval A is invalid"):
        get_interval_relation(dt(10), dt(10), dt(10), dt(11))


def test_invalid_inverted_event_A() -> None:
    # start > end
    with pytest.raises(ValueError, match="Interval A is invalid"):
        get_interval_relation(dt(11), dt(10), dt(10), dt(11))


def test_invalid_point_event_B() -> None:
    # start == end
    with pytest.raises(ValueError, match="Interval B is invalid"):
        get_interval_relation(dt(10), dt(11), dt(10), dt(10))


def test_timezone_conversion() -> None:
    # Compare UTC with UTC+1
    # 10:00 UTC
    start_a = dt(10)
    end_a = dt(11)

    # 10:00 UTC is 11:00 UTC+1
    tz_plus_1 = timezone(timedelta(hours=1))
    start_b = datetime(2024, 1, 1, 11, 0, tzinfo=tz_plus_1)
    # 11:00 UTC is 12:00 UTC+1
    end_b = datetime(2024, 1, 1, 12, 0, tzinfo=tz_plus_1)

    # Should be EQUALS because they represent the same absolute time
    r = get_interval_relation(start_a, end_a, start_b, end_b)
    assert r == AllenRelation.EQUALS
