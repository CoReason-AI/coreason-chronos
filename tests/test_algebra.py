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
