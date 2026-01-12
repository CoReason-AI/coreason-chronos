from datetime import datetime, timedelta, timezone

import pytest

from coreason_chronos.utils.algebra import IntervalRelation, get_interval_relation


class TestAllenAlgebra:
    def setup_method(self) -> None:
        self.base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Define X as [base, base + 2h]
        self.x_start = self.base
        self.x_end = self.base + timedelta(hours=2)

    def test_equals(self) -> None:
        # Y is same as X
        assert get_interval_relation(self.x_start, self.x_end, self.x_start, self.x_end) == IntervalRelation.EQUALS

    def test_before(self) -> None:
        # X: [12, 14], Y: [15, 17]
        y_start = self.base + timedelta(hours=3)
        y_end = self.base + timedelta(hours=5)
        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.BEFORE

    def test_after(self) -> None:
        # X: [12, 14], Y: [09, 11]
        y_start = self.base - timedelta(hours=3)
        y_end = self.base - timedelta(hours=1)
        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.AFTER

    def test_meets(self) -> None:
        # X: [12, 14], Y: [14, 16]
        y_start = self.x_end
        y_end = y_start + timedelta(hours=2)
        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.MEETS

    def test_met_by(self) -> None:
        # X: [12, 14], Y: [10, 12]
        y_end = self.x_start
        y_start = y_end - timedelta(hours=2)
        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.MET_BY

    def test_starts(self) -> None:
        # X: [12, 14], Y: [12, 15]
        y_start = self.x_start
        y_end = self.x_end + timedelta(hours=1)
        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.STARTS

    def test_started_by(self) -> None:
        # X: [12, 14], Y: [12, 13]
        y_start = self.x_start
        y_end = self.x_end - timedelta(hours=1)
        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.STARTED_BY

    def test_finishes(self) -> None:
        # X: [12, 14], Y: [11, 14]
        y_end = self.x_end
        y_start = self.x_start - timedelta(hours=1)
        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.FINISHES

    def test_finished_by(self) -> None:
        # X: [12, 14], Y: [13, 14]
        y_end = self.x_end
        y_start = self.x_start + timedelta(hours=1)
        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.FINISHED_BY

    def test_during(self) -> None:
        # X: [12, 14], Y: [11, 15]
        y_start = self.x_start - timedelta(hours=1)
        y_end = self.x_end + timedelta(hours=1)
        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.DURING

    def test_contains(self) -> None:
        # X: [12, 14], Y: [12:30, 13:30]
        y_start = self.x_start + timedelta(minutes=30)
        y_end = self.x_end - timedelta(minutes=30)
        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.CONTAINS

    def test_overlaps(self) -> None:
        # X: [12, 14], Y: [13, 15]
        y_start = self.x_start + timedelta(hours=1)
        y_end = self.x_end + timedelta(hours=1)
        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.OVERLAPS

    def test_overlapped_by(self) -> None:
        # X: [12, 14], Y: [11, 13]
        y_start = self.x_start - timedelta(hours=1)
        y_end = self.x_end - timedelta(hours=1)
        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.OVERLAPPED_BY

    def test_invalid_interval(self) -> None:
        # Start >= End
        with pytest.raises(ValueError):
            get_interval_relation(self.x_end, self.x_start, self.x_start, self.x_end)

        with pytest.raises(ValueError):
            get_interval_relation(self.x_start, self.x_end, self.x_end, self.x_start)

    def test_timezone_equivalence(self) -> None:
        """Test that intervals in different timezones but representing same physical time are treated correctly."""
        # 12:00 UTC = 07:00 EST (UTC-5)
        est_tz = timezone(timedelta(hours=-5))
        start_est = datetime(2024, 1, 1, 7, 0, 0, tzinfo=est_tz)
        end_est = datetime(2024, 1, 1, 9, 0, 0, tzinfo=est_tz)

        # Should be EQUALS to [12:00 UTC, 14:00 UTC]
        assert get_interval_relation(self.x_start, self.x_end, start_est, end_est) == IntervalRelation.EQUALS

    def test_microsecond_precision(self) -> None:
        """Test high precision boundaries."""
        # Y starts 1 microsecond AFTER X ends. Should be BEFORE, not MEETS.
        y_start = self.x_end + timedelta(microseconds=1)
        y_end = y_start + timedelta(hours=1)

        assert get_interval_relation(self.x_start, self.x_end, y_start, y_end) == IntervalRelation.BEFORE

        # Y starts exactly at X end (MEETS)
        y_start_meets = self.x_end
        y_end_meets = y_start_meets + timedelta(hours=1)
        assert get_interval_relation(self.x_start, self.x_end, y_start_meets, y_end_meets) == IntervalRelation.MEETS

    def test_leap_year_spanning(self) -> None:
        """Test intervals crossing leap day (Feb 29)."""
        # 2024 is a leap year.
        # X: Feb 28 to Mar 1
        x_start = datetime(2024, 2, 28, 12, 0, 0, tzinfo=timezone.utc)
        x_end = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Duration should be 48 hours (28th -> 29th -> 1st)
        assert (x_end - x_start).total_seconds() == 48 * 3600

        # Y: Feb 29 12:00 to Feb 29 13:00
        y_start = datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc)
        y_end = datetime(2024, 2, 29, 13, 0, 0, tzinfo=timezone.utc)

        # Y is strictly DURING X
        assert get_interval_relation(x_start, x_end, y_start, y_end) == IntervalRelation.CONTAINS

    def test_zero_duration_explicit_failure(self) -> None:
        """Ensure that point events (duration=0) are strictly rejected as per Interval Algebra."""
        # Point event
        point_start = self.base
        point_end = self.base

        with pytest.raises(ValueError) as excinfo:
            get_interval_relation(point_start, point_end, self.x_start, self.x_end)
        assert "must be strictly before end" in str(excinfo.value)
