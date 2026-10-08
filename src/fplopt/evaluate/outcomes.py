"""Realized outcomes for xP evaluation (Phase 5 plan, *Evaluation (criterion 1)*).

Outcomes of (season, gw) are read exactly as the backtester reads them (PLAN §5 *Mechanics*):
from `store.as_of(lockdown + 1 µs)` (results are available at the GW lockdown and views are
strict), the GW's `player_match` rows with the player's `player_season` position
(`fplopt.backtest.simulator.gw_matches`; rows without one are dropped with a warning), and
points re-scored with `score_matches` under the season's backtest rules (`backtest_rules`),
so they are on the same scale as `rescored_points`, the models' xP and the backtest.

- `fixture_outcomes`: one row per player-fixture (`OUTCOME_COLUMNS`).
- `gw_totals`: one row per player-GW, doubles summed (`n_fixtures` = 2); a player without a
  row in the GW (his club blanks, or he isn't registered) has no row: 0 points, 0 minutes.
"""

from __future__ import annotations

import pandas as pd

from fplopt.backtest.rules import Rules
from fplopt.backtest.scoring import score_matches
from fplopt.backtest.simulator import MATCH_COLUMNS, OUTCOME_DELAY, gw_matches
from fplopt.features.store import DataStore

__all__ = ("GW_TOTAL_COLUMNS", "OUTCOME_COLUMNS", "STAT_COLUMNS", "fixture_outcomes", "gw_totals")

# Summed per player-GW by `gw_totals` (with `points`).
STAT_COLUMNS = (
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
)
OUTCOME_COLUMNS = (
    ("player_key", "int64"),
    ("fixture_key", "int64"),
    ("season", "int64"),
    ("gw", "int64"),
    ("gw_index", "int64"),
    ("team_key", "int64"),
    ("element_type", "int64"),
    ("starts", "Int64"),  # null where the source has no starts (before 2022/23 GW16)
    *((name, "int64") for name in STAT_COLUMNS),
    ("points", "int64"),
)
GW_TOTAL_COLUMNS = (
    ("player_key", "int64"),
    ("season", "int64"),
    ("gw", "int64"),
    ("gw_index", "int64"),
    ("team_key", "int64"),
    ("element_type", "int64"),
    ("n_fixtures", "int64"),
    ("starts", "Int64"),  # null if every fixture's is
    *((name, "int64") for name in STAT_COLUMNS),
    ("points", "int64"),
)
_READ_COLUMNS = (*MATCH_COLUMNS, "team_key", "starts")


def _empty(columns: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns})


def fixture_outcomes(
    store: DataStore, rules: Rules, season: int, gw: int, lockdown: pd.Timestamp
) -> pd.DataFrame:
    """Realized per player-fixture outcomes of (season, gw) as of `lockdown + 1 µs`, points
    re-scored under `rules` (module docstring). Columns `OUTCOME_COLUMNS`, sorted by
    (fixture_key, player_key); empty if the GW has no visible rows (not played yet)."""
    view = store.as_of(pd.Timestamp(lockdown) + OUTCOME_DELAY)
    matches = gw_matches(view, season, gw, _READ_COLUMNS)
    if matches.empty:
        return _empty(OUTCOME_COLUMNS)
    gameweeks = view.table("gameweek", columns=["season", "gw", "gw_index"])
    row = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)]
    if len(row) != 1:
        raise LookupError(f"season {season}: {len(row)} gameweek rows for GW {gw}")
    out = matches.assign(
        gw_index=int(row["gw_index"].iloc[0]),
        points=score_matches(matches, rules)["points"].to_numpy(dtype="int64"),
    )
    out = out[[name for name, _ in OUTCOME_COLUMNS]].astype(dict(OUTCOME_COLUMNS))
    return out.sort_values(["fixture_key", "player_key"], kind="mergesort").reset_index(drop=True)


def gw_totals(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Per player-GW sums of `fixture_outcomes` rows (one or more GWs): `n_fixtures` = rows,
    `STAT_COLUMNS` and `points` summed, club and position from the first fixture. Columns
    `GW_TOTAL_COLUMNS`, sorted by (season, gw_index, player_key)."""
    if fixtures.empty:
        return _empty(GW_TOTAL_COLUMNS)
    keys = ["season", "gw", "gw_index", "player_key"]
    ordered = fixtures.sort_values([*keys, "fixture_key"], kind="mergesort")
    grouped = ordered.groupby(keys, sort=True)
    sums = grouped[[*STAT_COLUMNS, "points"]].sum()
    first = grouped[["team_key", "element_type"]].first()
    starts = grouped["starts"].sum(min_count=1)
    out = sums.join(first).assign(n_fixtures=grouped.size(), starts=starts).reset_index()
    out = out[[name for name, _ in GW_TOTAL_COLUMNS]].astype(dict(GW_TOTAL_COLUMNS))
    return out.sort_values(["season", "gw_index", "player_key"], kind="mergesort").reset_index(
        drop=True
    )
