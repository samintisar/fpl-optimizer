from datetime import UTC, datetime, timedelta

from fplopt.ingest.schedule import deadlines_from_bootstrap, pre_deadline_due

D = datetime(2026, 10, 10, 10, 0, tzinfo=UTC)


def test_parses_deadlines():
    bootstrap = {"events": [{"deadline_time": "2026-10-10T10:00:00Z"}]}
    assert deadlines_from_bootstrap(bootstrap) == [D]


def test_due_inside_window_without_snapshot():
    assert pre_deadline_due(D - timedelta(minutes=110), [D], []) == D


def test_not_due_before_window():
    assert pre_deadline_due(D - timedelta(hours=3), [D], []) is None


def test_not_due_after_deadline():
    assert pre_deadline_due(D + timedelta(minutes=1), [D], []) is None


def test_not_due_when_window_already_has_snapshot():
    taken = [D - timedelta(minutes=100)]
    assert pre_deadline_due(D - timedelta(minutes=50), [D], taken) is None


def test_snapshot_before_window_does_not_count():
    taken = [D - timedelta(hours=8)]
    assert pre_deadline_due(D - timedelta(minutes=50), [D], taken) == D
