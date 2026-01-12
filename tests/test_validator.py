from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from coreason_chronos.validator import MaxDelayRule


def test_max_delay_compliant() -> None:
    # Rule: Max delay 24 hours
    rule = MaxDelayRule(max_delay=timedelta(hours=24))

    # Reference: Today 12:00 UTC
    reference = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Target: Today 13:00 UTC (1 hour later) -> Compliant
    target = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

    result = rule.validate(target, reference)

    assert result.is_compliant is True
    # Drift: target - deadline
    # deadline = 12:00 + 24h = Jan 2 12:00
    # target = Jan 1 13:00
    # drift = Jan 1 13:00 - Jan 2 12:00 = -23 hours
    assert result.drift == timedelta(hours=-23)
    assert result.message is None


def test_max_delay_boundary() -> None:
    # Rule: Max delay 24 hours
    rule = MaxDelayRule(max_delay=timedelta(hours=24))

    reference = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Target: Exact deadline
    target = reference + timedelta(hours=24)

    result = rule.validate(target, reference)

    assert result.is_compliant is True
    assert result.drift == timedelta(0)


def test_max_delay_violation() -> None:
    # Rule: Max delay 24 hours
    rule = MaxDelayRule(max_delay=timedelta(hours=24))

    reference = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Target: 25 hours later -> Violation
    target = reference + timedelta(hours=25)

    result = rule.validate(target, reference)

    assert result.is_compliant is False
    # Drift = 25h - 24h = 1h
    assert result.drift == timedelta(hours=1)
    assert "Violation" in (result.message or "")


def test_timezone_awareness_check() -> None:
    rule = MaxDelayRule(max_delay=timedelta(hours=1))

    aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    naive = datetime(2024, 1, 1, 12, 0, 0)

    # Should raise ValueError if naive
    with pytest.raises(ValueError, match="must be timezone-aware"):
        rule.validate(naive, aware)

    with pytest.raises(ValueError, match="must be timezone-aware"):
        rule.validate(aware, naive)


def test_mixed_timezones() -> None:
    # Verify that different timezones are handled correctly
    rule = MaxDelayRule(max_delay=timedelta(hours=1))

    # Reference: 12:00 UTC
    reference = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Target: 12:30 UTC+1 (which is 11:30 UTC) -> Compliant
    # Wait, 12:30 UTC+1 is 11:30 UTC.
    # Reference is 12:00 UTC.
    # Target is BEFORE reference.
    # Logic: target <= reference + delay
    # 11:30 <= 12:00 + 1h (13:00). True.

    target_tz = timezone(timedelta(hours=1))
    target = datetime(2024, 1, 1, 12, 30, 0, tzinfo=target_tz)

    result = rule.validate(target, reference)
    assert result.is_compliant is True

    # Drift calculation
    # Deadline: 13:00 UTC
    # Target: 11:30 UTC
    # Drift: 11:30 - 13:00 = -1.5 hours
    assert result.drift == timedelta(hours=-1.5)


def test_negative_delay_init_error() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MaxDelayRule(max_delay=timedelta(hours=-1))


def test_leap_year_crossing() -> None:
    """
    Test delay calculation across a leap year boundary (Feb 29).
    """
    rule = MaxDelayRule(max_delay=timedelta(days=2))

    # Reference: 2024 is a leap year. Feb 28th.
    reference = datetime(2024, 2, 28, 12, 0, 0, tzinfo=timezone.utc)

    # Deadline: Feb 28 + 2 days = March 1st. (28 -> 29 -> 1)

    # Target: March 1st 12:00 UTC -> Compliant (Exact boundary)
    target = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    result = rule.validate(target, reference)
    assert result.is_compliant is True
    assert result.drift == timedelta(0)

    # Verify non-leap year behavior for contrast
    # 2023 is not a leap year. Feb 28 + 2 days = March 2nd.
    reference_non_leap = datetime(2023, 2, 28, 12, 0, 0, tzinfo=timezone.utc)
    target_non_leap = datetime(2023, 3, 2, 12, 0, 0, tzinfo=timezone.utc)

    result = rule.validate(target_non_leap, reference_non_leap)
    assert result.is_compliant is True
    assert result.drift == timedelta(0)


def test_dst_transition() -> None:
    """
    Test delay calculation crossing a DST transition.

    Example: US/Eastern springs forward in March. 2am becomes 3am.
    Gap: 2024-03-10 02:00:00 doesn't exist.

    Reference: 2024-03-10 01:30:00 EST (-0500)
    Delay: 1 hour.
    Deadline: 02:30:00 local time... NO.
    UTC Reference: 06:30:00 UTC.
    UTC Deadline: 07:30:00 UTC.

    If we stick to converting to UTC internally (which the code does implicitly by comparing timestamps),
    arithmetic works on absolute time, ignoring the wall clock shift.

    Target (Wall Clock): 03:30:00 EDT (-0400).
    03:30 EDT = 07:30 UTC.

    So 01:30 EST + 1 hour -> 03:30 EDT.
    """
    # Use ZoneInfo for proper DST handling
    try:
        nyc = ZoneInfo("America/New_York")
    except Exception:
        pytest.skip("ZoneInfo data not available")

    rule = MaxDelayRule(max_delay=timedelta(hours=1))

    # 1:30 AM EST on DST switch day (Spring Forward)
    reference = datetime(2024, 3, 10, 1, 30, 0, tzinfo=nyc)

    # Target: 3:30 AM EDT (which is 1 hour absolute time later)
    target = datetime(2024, 3, 10, 3, 30, 0, tzinfo=nyc)

    result = rule.validate(target, reference)
    assert result.is_compliant is True
    assert result.drift == timedelta(0)


def test_microsecond_precision() -> None:
    """
    Test strict adherence down to microseconds.
    """
    rule = MaxDelayRule(max_delay=timedelta(seconds=1))

    reference = datetime(2024, 1, 1, 12, 0, 0, 0, tzinfo=timezone.utc)

    # Target: 1 second + 1 microsecond later -> Violation
    target = reference + timedelta(seconds=1, microseconds=1)

    result = rule.validate(target, reference)
    assert result.is_compliant is False
    assert result.drift == timedelta(microseconds=1)


def test_target_before_reference() -> None:
    """
    Test case where target happens BEFORE reference.
    Since delay must be positive, this should always be compliant.
    """
    rule = MaxDelayRule(max_delay=timedelta(hours=1))

    reference = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    target = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)

    result = rule.validate(target, reference)
    assert result.is_compliant is True
    # Drift: 11:00 - 13:00 = -2 hours
    assert result.drift == timedelta(hours=-2)


def test_zero_delay_rule() -> None:
    """
    Test rule with 0 delay (Reference >= Target).
    """
    rule = MaxDelayRule(max_delay=timedelta(0))

    reference = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Exact match -> Compliant
    assert rule.validate(reference, reference).is_compliant is True

    # 1 microsecond late -> Violation
    target_late = reference + timedelta(microseconds=1)
    assert rule.validate(target_late, reference).is_compliant is False
