"""Known double GWs from the as-of schedule (chip policy 2, V5)."""

import pandas as pd

from fplopt.backtest.chip_doubles import DOUBLE_LEAD, known_doubles


class FakeView:
    def __init__(self, rows):
        self.rows = rows

    def schedule(self, season):
        frame = pd.DataFrame(
            self.rows, columns=["season", "gw_index", "home_team_key", "away_team_key"]
        )
        return frame[frame["season"] == season].astype({"gw_index": "Int64"})


def test_known_doubles_within_the_lead():
    rows = [
        (2020, 5, 1, 2),
        (2020, 12, 1, 2),
        (2020, 12, 1, 3),  # club 1 twice in GW12: a double
        (2020, 30, 4, 5),
        (2020, 30, 6, 4),  # club 4 twice in GW30: beyond the lead from GW5
        (2020, None, 7, 8),  # unscheduled
        (2020, 8, 0, 9),
        (2020, 8, 0, 8),  # placeholder side (team_key 0) twice: not a double
        (2021, 13, 1, 2),
        (2021, 13, 1, 3),  # another season
    ]
    view = FakeView(rows)
    assert DOUBLE_LEAD == 10
    assert known_doubles(view, 2020, 5) == frozenset({12})
    assert known_doubles(view, 2020, 2) == frozenset({12})  # exactly at gw_index + lead
    assert known_doubles(view, 2020, 1) == frozenset()  # one beyond the lead
    assert known_doubles(view, 2020, 12) == frozenset()  # only GWs after the decision GW
    assert known_doubles(view, 2020, 25) == frozenset({30})
    assert known_doubles(view, 2020, 5, lead=30) == frozenset({12, 30})


def test_known_doubles_by_gw_on_a_synthetic_store():
    from synthetic_season import synthetic_tables

    from fplopt.backtest.chip_values import known_doubles_by_gw
    from fplopt.features.store import DataStore

    store = DataStore(tables=synthetic_tables(seasons=(2022, 2023)))
    table = known_doubles_by_gw(store, [2023], DOUBLE_LEAD)
    assert table and all(season == 2023 for season, _ in table)
    assert sorted(gw for _, gw in table) == list(range(1, len(table) + 1))
    assert all(isinstance(v, frozenset) for v in table.values())


def test_known_doubles_by_gw_sees_a_double_only_from_the_snapshot_that_announces_it():
    from synthetic_season import synthetic_tables

    from fplopt.backtest.chip_values import known_doubles_by_gw
    from fplopt.features.store import DataStore

    tables = synthetic_tables(seasons=(2023,))
    schedule, gameweek = tables["schedule"], tables["gameweek"]
    assert known_doubles_by_gw(DataStore(tables=tables), [2023], lead=38)[(2023, 1)] == frozenset()
    k = 20  # snapshot published just after the deadline of GW k moves a GW k+5 fixture to k+6
    moved = schedule[schedule["gw_index"] == k + 5].iloc[0]
    club = moved["home_team_key"]
    snapshot_at = gameweek.loc[gameweek["gw_index"] == k, "deadline_time"].iloc[0] + pd.Timedelta(
        hours=1
    )
    snap = schedule[
        ["season", "fixture_key", "gw", "kickoff_time", "home_team_key", "away_team_key"]
    ].copy()
    snap["gw"] = snap["gw"].mask(snap["fixture_key"] == moved["fixture_key"], k + 6)
    # the club's own fixture in GW k+6 stays: it now has two there
    later = schedule[schedule["gw_index"] == k + 6][["home_team_key", "away_team_key"]]
    assert (later == club).any(axis=None)
    snap["snapshot_at"] = snapshot_at
    snap["fpl_fixture_id"] = snap["fixture_key"]
    for flag in ("started", "finished", "finished_provisional"):
        snap[flag] = pd.array([False] * len(snap), dtype="boolean")
    snap["event_time"] = snap["available_at"] = snapshot_at
    tables["fixture_snapshot"] = snap.astype(tables["fixture_snapshot"].dtypes.to_dict())

    table = known_doubles_by_gw(DataStore(tables=tables), [2023], lead=38)
    assert all(k + 6 not in table[(2023, g)] for g in range(1, k + 1))
    assert all(k + 6 in table[(2023, g)] for g in range(k + 1, k + 6))
