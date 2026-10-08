"""Realized outcomes for the xP evaluation (fplopt.evaluate.outcomes) on a synthetic league."""

from __future__ import annotations

import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.backtest.rules import backtest_rules
from fplopt.backtest.scoring import score_matches
from fplopt.backtest.simulator import read_outcomes
from fplopt.evaluate.outcomes import GW_TOTAL_COLUMNS, OUTCOME_COLUMNS, fixture_outcomes, gw_totals
from fplopt.features.store import DataStore

SEASON = 2023
RULES = backtest_rules(SEASON)
BLANK = (SEASON, 5, 1)  # club 1's GW5 fixture is played in GW7
DOUBLE = (SEASON, 9, 4)  # club 4's GW11 fixture is played in GW9


@pytest.fixture(scope="module")
def league():
    tables = synthetic_tables(seasons=(SEASON,), n_clubs=10, blank=BLANK, double=DOUBLE)
    return tables, DataStore(tables=tables)


def lockdown(tables, gw):
    gameweeks = tables["gameweek"]
    return gameweeks.loc[(gameweeks["season"] == SEASON) & (gameweeks["gw"] == gw)].iloc[0]


def test_fixture_outcomes_are_rescored_player_match_rows(league):
    tables, store = league
    row = lockdown(tables, 3)
    out = fixture_outcomes(store, RULES, SEASON, 3, row["lockdown_time"])
    assert list(out.columns) == [name for name, _ in OUTCOME_COLUMNS]
    assert out.dtypes.astype(str).tolist() == [str(dtype) for _, dtype in OUTCOME_COLUMNS]
    matches = tables["player_match"]
    matches = matches[(matches["season"] == SEASON) & (matches["gw"] == 3)]
    assert len(out) == len(matches)
    assert (out["gw_index"] == row["gw_index"]).all()
    positions = tables["player_season"].set_index("player_key")["element_type"]
    expected = score_matches(
        matches.assign(element_type=matches["player_key"].map(positions)), RULES
    )["points"]
    merged = out.merge(
        matches.assign(expected=expected.to_numpy())[["player_key", "fixture_key", "expected"]],
        on=["player_key", "fixture_key"],
    )
    assert (merged["points"] == merged["expected"]).all()
    # The synthetic total_points follow the backtest rules.
    assert out["points"].sum() == matches["total_points"].sum()


def test_outcomes_need_the_lockdown(league):
    tables, store = league
    row = lockdown(tables, 3)
    early = fixture_outcomes(store, RULES, SEASON, 3, row["lockdown_time"] - pd.Timedelta("1s"))
    assert early.empty and list(early.columns) == [name for name, _ in OUTCOME_COLUMNS]


@pytest.mark.parametrize("gw", [5, 7, 9, 11])
def test_gw_totals_sum_doubles_and_omit_blanks(league, gw):
    tables, store = league
    when = lockdown(tables, gw)["lockdown_time"]
    fixtures = fixture_outcomes(store, RULES, SEASON, gw, when)
    totals = gw_totals(fixtures)
    assert list(totals.columns) == [name for name, _ in GW_TOTAL_COLUMNS]
    assert totals["player_key"].is_unique
    backtest = read_outcomes(store, RULES, SEASON, gw, when).realized
    assert (
        dict(
            zip(
                totals["player_key"],
                zip(totals["points"], totals["minutes"], strict=True),
                strict=True,
            )
        )
        == backtest
    )
    club = totals.set_index("player_key")["team_key"]
    counts = totals.set_index("player_key")["n_fixtures"]
    if gw == 5:  # club 1 (and its GW5 opponent) blank
        assert 1 not in set(club)
    if gw in (7, 9):  # the moved fixtures make doubles
        double = 1 if gw == 7 else 4
        assert (counts[club == double] == 2).all()
        assert (counts[club != double].isin([1, 2])).all()
    if gw == 11:
        assert 4 not in set(club)


def test_gw_totals_of_nothing_is_empty():
    empty = gw_totals(pd.DataFrame({name: pd.Series(dtype=d) for name, d in OUTCOME_COLUMNS}))
    assert empty.empty and list(empty.columns) == [name for name, _ in GW_TOTAL_COLUMNS]
