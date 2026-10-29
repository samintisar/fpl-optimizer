import json
from datetime import UTC, datetime

import pytest

from fplopt.ingest.espn import (
    archive_period,
    backfill_espn,
    espn_team_ids,
    events_to_fetch,
    recent_periods,
    snapshot_espn_recent,
    summary_endpoint,
)
from fplopt.ingest.jobs import SnapshotError
from fplopt.ingest.raw_store import RawStore

NOW = datetime(2026, 10, 27, 3, 0, tzinfo=UTC)


def event(event_id, date, home, away, completed=True, state="post"):
    return {
        "id": event_id,
        "date": date,
        "status": {
            "type": {
                "name": "STATUS_FULL_TIME" if completed else "STATUS_SCHEDULED",
                "completed": completed,
                "state": state,
            }
        },
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"id": home}},
                    {"homeAway": "away", "team": {"id": away}},
                ]
            }
        ],
    }


def summary(event_id, completed=True):
    return {
        "header": {
            "id": event_id,
            "competitions": [{"status": {"type": {"completed": completed}}}],
        },
        "rosters": [],
        "keyEvents": [],
    }


class FakeEspn:
    def __init__(self, boards, fail=()):
        self.boards = boards  # (league, period) -> list of events
        self.fail = set(fail)
        self.calls = []

    def scoreboard(self, league, period):
        self.calls.append(("scoreboard", league, period))
        return json.dumps({"events": self.boards.get((league, period), [])}).encode()

    def summary(self, league, event_id):
        self.calls.append(("summary", league, event_id))
        if event_id in self.fail:
            raise RuntimeError(f"boom {event_id}")
        return json.dumps(summary(event_id)).encode()


PL = frozenset({"359", "363"})


def test_events_to_fetch_keeps_completed_events_with_a_pl_club_after_july_2016():
    events = [
        event("1", "2016-10-25T18:45Z", "359", "1000"),  # Arsenal v lower club
        event("2", "2016-10-25T18:45Z", "1000", "1001"),  # no PL club
        event("3", "2016-05-01T18:45Z", "359", "1000"),  # before 2016/17
        event("4", "2026-11-04T20:00Z", "363", "1000", completed=False, state="pre"),
    ]
    assert events_to_fetch({"events": events}, "eng.league_cup", PL) == ["1"]


def test_events_to_fetch_keeps_every_pl_match_in_eng_1():
    events = [event("9", "2016-08-13T14:00Z", "1000", "1001")]
    assert events_to_fetch({"events": events}, "eng.1", PL) == ["9"]


def test_archive_period_writes_the_scoreboard_and_new_summaries_once(tmp_path):
    store = RawStore(tmp_path)
    espn = FakeEspn({("eng.league_cup", "2016"): [event("1", "2016-10-25T18:45Z", "359", "1000")]})
    sleeps = []
    assert (
        archive_period(store, espn, "eng.league_cup", "2016", PL, NOW, lambda: NOW, sleeps.append)
        == 1
    )
    assert store.latest("espn", summary_endpoint("eng.league_cup", "1")) is not None
    assert store.latest("espn", "eng.league_cup/scoreboard/2016") is not None
    # A second run re-reads the scoreboard but skips the completed summary.
    later = datetime(2026, 10, 28, tzinfo=UTC)
    assert (
        archive_period(
            store, espn, "eng.league_cup", "2016", PL, later, lambda: later, sleeps.append
        )
        == 0
    )
    assert [c[0] for c in espn.calls] == ["scoreboard", "summary", "scoreboard"]


def test_archive_period_fetches_the_others_when_one_summary_fails(tmp_path):
    store = RawStore(tmp_path)
    events = [event(i, "2016-10-25T18:45Z", "359", "1000") for i in ("1", "2")]
    espn = FakeEspn({("eng.league_cup", "2016"): events}, fail={"1"})
    with pytest.raises(SnapshotError, match="eng.league_cup/1"):
        archive_period(store, espn, "eng.league_cup", "2016", PL, NOW, lambda: NOW, lambda s: None)
    assert store.latest("espn", summary_endpoint("eng.league_cup", "2")) is not None


def test_recent_periods_cover_a_week_back_and_sixty_days_ahead():
    assert recent_periods(NOW) == ["202610", "202611", "202612"]


def test_backfill_walks_every_league_and_year(tmp_path):
    espn = FakeEspn({})
    backfill_espn(
        RawStore(tmp_path), espn, PL, first_year=2025, now=lambda: NOW, sleep=lambda s: None
    )
    boards = {(c[1], c[2]) for c in espn.calls if c[0] == "scoreboard"}
    assert ("eng.1", "2025") in boards and ("uefa.super_cup", "2026") in boards
    assert len(boards) == 16


def test_snapshot_recent_fetches_the_recent_months_of_every_league(tmp_path):
    espn = FakeEspn({})
    snapshot_espn_recent(RawStore(tmp_path), espn, PL, now=lambda: NOW, sleep=lambda s: None)
    assert len([c for c in espn.calls if c[0] == "scoreboard"]) == 8 * 3


def test_espn_team_ids_reads_the_config_column():
    from synthetic_raw import REPO_CONFIG

    ids = espn_team_ids(REPO_CONFIG / "teams.csv")
    assert {"359", "363", "360"} <= ids and len(ids) == 35
