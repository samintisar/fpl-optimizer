from datetime import UTC, datetime, timedelta

import pytest

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


def test_deadline_without_offset_is_treated_as_utc():
    assert deadlines_from_bootstrap({"events": [{"deadline_time": "2026-10-10T10:00:00"}]}) == [D]


def test_deadline_with_offset_is_normalised_to_utc():
    bootstrap = {"events": [{"deadline_time": "2026-10-10T11:00:00+01:00"}]}
    [deadline] = deadlines_from_bootstrap(bootstrap)
    assert deadline == D
    assert deadline.tzinfo == UTC


def test_naive_now_rejected():
    with pytest.raises(ValueError):
        pre_deadline_due(datetime(2026, 10, 10, 9, 0), [D], [])


def test_window_start_inclusive_deadline_exclusive():
    assert pre_deadline_due(D - timedelta(hours=2), [D], []) == D
    assert pre_deadline_due(D, [D], []) is None


def test_snapshot_at_window_edges():
    assert pre_deadline_due(D - timedelta(minutes=30), [D], [D - timedelta(hours=2)]) is None
    assert pre_deadline_due(D - timedelta(minutes=30), [D], [D]) == D


def test_picks_the_deadline_whose_window_contains_now():
    later = D + timedelta(days=7)
    assert pre_deadline_due(later - timedelta(minutes=30), [D, later], []) == later
    assert pre_deadline_due(D, [], []) is None


def test_postponed_deadline_reopens_window_once_old_snapshot_is_outside_it():
    old_snapshot = D - timedelta(minutes=110)
    moved = D + timedelta(hours=3)
    assert pre_deadline_due(moved - timedelta(minutes=60), [moved], [old_snapshot]) == moved


def test_custom_lead():
    assert pre_deadline_due(D - timedelta(minutes=30), [D], [], lead=timedelta(hours=1)) == D
    assert pre_deadline_due(D - timedelta(minutes=90), [D], [], lead=timedelta(hours=1)) is None
