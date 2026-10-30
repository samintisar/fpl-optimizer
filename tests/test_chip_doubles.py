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
        (2021, 13, 1, 2),
        (2021, 13, 1, 3),  # another season
    ]
    view = FakeView(rows)
    assert DOUBLE_LEAD == 10
    assert known_doubles(view, 2020, 5) == frozenset({12})
    assert known_doubles(view, 2020, 12) == frozenset()  # only GWs after the decision GW
    assert known_doubles(view, 2020, 25) == frozenset({30})
    assert known_doubles(view, 2020, 5, lead=30) == frozenset({12, 30})


def test_known_doubles_by_gw_on_a_synthetic_store():
    from synthetic_season import synthetic_tables

    from fplopt.backtest.chip_doubles import known_doubles_by_gw
    from fplopt.features.store import DataStore

    store = DataStore(tables=synthetic_tables(seasons=(2022, 2023)))
    table = known_doubles_by_gw(store, [2023])
    assert table and all(season == 2023 for season, _ in table)
    assert sorted(gw for _, gw in table) == list(range(1, len(table) + 1))
    assert all(isinstance(v, frozenset) for v in table.values())
