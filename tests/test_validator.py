from datetime import datetime, timedelta, timezone

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
