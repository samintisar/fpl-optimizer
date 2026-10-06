"""Baseline feature builders (Phase 2 plan, Task 3).

Each builder takes exactly one argument, an `AsOfView`, reads only through it and returns a
deterministic frame (fixed columns and dtypes, sorted by its key, RangeIndex). The target
gameweek is the one whose deadline is the view's deadline.

Pool coverage: before player snapshots exist (2016/17 – 2020/21 GW32), `player_pool` is
built from `player_gw` rows visible at the deadline, and some clubs can have no honest
as-of source at all: at 2020/21 GW1 Man Utd, Man City, Burnley and Aston Villa had their
GW1 match postponed, so they have no GW1 row and no earlier row that season, and the pool
lacks them entirely. `pool_coverage` reports per club its fixtures and pool size, and
`player_pool` logs a warning when a club with a fixture in the horizon has no players. The
backtester must not start from such deadlines (or must flag them; PLAN §5). Measured on
data/ 2026-10-06 over every non-holdout deadline: 2020/21 GW1 is the only one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from fplopt.features.store import AsOfView

log = logging.getLogger(__name__)

# Module-level values are immutable (tuples; dtypes as (column, dtype) pairs): feature
# modules keep no state between calls (tests/test_features_architecture.py).
Dtypes = Mapping[str, object] | tuple[tuple[str, object], ...]

UTC_US = pd.DatetimeTZDtype("us", "UTC")
SNAPSHOT_COLUMNS = (
    "snapshot_at",
    "season",
    "team_key",
    "element_type",
    "now_cost",
    "status",
    "chance_of_playing_next_round",
    "news_added",
    "ep_next",
)
FORM_STATS = (
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "us_npxg",
    "us_xa",
    "fpl_xg",
    "fpl_xa",
)
RATE_STATS = tuple(s for s in FORM_STATS if s not in ("minutes", "starts"))
FORM_WINDOW = 5
HORIZON = 5  # upcoming_fixtures: the target GW and the next HORIZON GWs
OUTCOMES = ("home", "draw", "away")
ODDS_GROUP = ("fixture_key", "source", "bookmaker", "market", "outcome", "line")


def _finish(df: pd.DataFrame, dtypes: Dtypes, sort_by: list[str]) -> pd.DataFrame:
    """Exactly `dtypes`' columns, in that order and with those dtypes, sorted (stable)."""
    dtypes = dict(dtypes)
    out = df[list(dtypes)].astype(dtypes)
    return out.sort_values(sort_by, kind="mergesort").reset_index(drop=True)


def _target(view: AsOfView) -> tuple[int, int, int]:
    """(season, gw, gw_index) of the gameweek whose deadline is the view's deadline."""
    season, gw = view.gameweek_for_deadline()
    gameweeks = _gameweeks(view, season)
    return season, gw, int(gameweeks.loc[gameweeks["gw"] == gw, "gw_index"].iloc[0])


def _gameweeks(view: AsOfView, season: int) -> pd.DataFrame:
    gameweeks = view.table("gameweek", columns=["season", "gw", "gw_index"])
    return gameweeks[gameweeks["season"] == season].reset_index(drop=True)


def _newest_snapshot(view: AsOfView, season: int) -> pd.DataFrame:
    """Rows of `season`'s newest player snapshot before the deadline (empty if none). Per
    player the newest row across sources, then only players present at the newest time:
    a player missing from it has left the game."""
    snaps = view.latest("player_snapshot", by=["player_key"], columns=list(SNAPSHOT_COLUMNS))
    snaps = snaps[snaps["season"] == season]
    if snaps.empty:
        return snaps
    return snaps[snaps["snapshot_at"] == snaps["snapshot_at"].max()]


# --- players -----------------------------------------------------------------------------

POOL_DTYPES = (
    ("player_key", "int64"),
    ("element_type", "int64"),
    ("team_key", "int64"),
    ("price", "int64"),
    ("source", "str"),
)


def player_pool(view: AsOfView) -> pd.DataFrame:
    """Players pickable for the target GW with position, club and price (tenths of £m).
    From the newest player snapshot of the season when one exists before the deadline
    (source 'snapshot'); otherwise from `player_gw` (source 'player_gw'): the target GW's
    rows visible at the deadline (players registered with that club by then) plus, for
    players of clubs without a fixture in the target GW (blank), their newest earlier row
    of the season. Logs a warning when a club with a fixture in the horizon has no players
    (see the module docstring and `pool_coverage`)."""
    pool = _pool(view)
    gaps = _coverage(view, pool)
    gaps = gaps[(gaps["n_fixtures_horizon"] > 0) & (gaps["n_pool_players"] == 0)]
    if len(gaps):
        log.warning(
            "player_pool at %s: %d club(s) with a fixture in the horizon have no pool "
            "players (team_key %s); the pool is incomplete",
            view.deadline,
            len(gaps),
            gaps["team_key"].tolist(),
        )
    return pool


def _pool(view: AsOfView) -> pd.DataFrame:
    season, gw, gw_index = _target(view)
    snaps = _newest_snapshot(view, season)
    if not snaps.empty:
        pool = snaps.rename(columns={"now_cost": "price"}).assign(source="snapshot")
        return _finish(pool, POOL_DTYPES, ["player_key"])

    columns = ["player_key", "season", "gw", "team_key", "element_type", "value"]
    rows = view.table("player_gw", columns=columns)
    rows = rows[rows["season"] == season]
    gameweeks = _gameweeks(view, season)
    index = dict(zip(gameweeks["gw"], gameweeks["gw_index"], strict=True))
    rows = rows.assign(gw_index=rows["gw"].map(index))
    current = rows[rows["gw"] == gw]
    earlier = rows[(rows["gw_index"] < gw_index) & ~rows["player_key"].isin(current["player_key"])]
    # Newest earlier row per player, kept only if the club blanks in the target GW (else
    # the player has left the game).
    earlier = earlier.sort_values(["player_key", "gw_index"], kind="mergesort")
    earlier = earlier.drop_duplicates("player_key", keep="last")
    schedule = view.schedule(season)
    teams = set(schedule["home_team_key"]) | set(schedule["away_team_key"])
    playing = schedule[schedule["gw"] == gw]
    blank = teams - set(playing["home_team_key"]) - set(playing["away_team_key"])
    earlier = earlier[earlier["team_key"].isin(blank)]
    pool = pd.concat([current, earlier], ignore_index=True)
    pool = pool.rename(columns={"value": "price"}).assign(source="player_gw")
    return _finish(pool, POOL_DTYPES, ["player_key"])


COVERAGE_DTYPES = (
    ("season", "int64"),
    ("gw", "int64"),
    ("team_key", "int64"),
    ("n_fixtures", "int64"),
    ("n_fixtures_horizon", "int64"),
    ("n_pool_players", "int64"),
)


def _coverage(view: AsOfView, pool: pd.DataFrame) -> pd.DataFrame:
    season, gw, gw_index = _target(view)
    schedule = view.schedule(season)
    gameweeks = _gameweeks(view, season)
    horizon = gameweeks.loc[gameweeks["gw_index"].between(gw_index, gw_index + HORIZON), "gw"]
    sides = _sides(schedule[schedule["gw"].notna()].astype({"gw": "int64"}))
    teams = sorted(set(schedule["home_team_key"]) | set(schedule["away_team_key"]))
    out = pd.DataFrame({"team_key": teams})
    target = sides.loc[sides["gw"] == gw, "team_key"].value_counts()
    ahead = sides.loc[sides["gw"].isin(horizon), "team_key"].value_counts()
    players = pool["team_key"].value_counts()
    out = out.assign(
        season=season,
        gw=gw,
        n_fixtures=out["team_key"].map(target).fillna(0),
        n_fixtures_horizon=out["team_key"].map(ahead).fillna(0),
        n_pool_players=out["team_key"].map(players).fillna(0),
    )
    return _finish(out, COVERAGE_DTYPES, ["team_key"])


def pool_coverage(view: AsOfView) -> pd.DataFrame:
    """Per club of the season's as-of schedule: its fixtures in the target GW and in the
    horizon (target GW and the next HORIZON GWs) and its number of `player_pool` players.
    A club with fixtures and 0 players means the pool is incomplete at this deadline."""
    return _coverage(view, _pool(view))


AVAILABILITY_DTYPES = (
    ("player_key", "int64"),
    ("status", "str"),
    ("chance_of_playing_next_round", "Int64"),
    ("news_added", UTC_US),
)


def _pool_with_snapshot(view: AsOfView, columns: list[str]) -> pd.DataFrame:
    """Pool players with `columns` from the newest snapshot (null where there is none)."""
    season, _, _ = _target(view)
    pool = _pool(view)[["player_key"]]
    snaps = _newest_snapshot(view, season)[["player_key", *columns]]
    return pool.merge(snaps, on="player_key", how="left", validate="one_to_one")


def availability(view: AsOfView) -> pd.DataFrame:
    """Per pool player: status flag, chance of playing next round and when the news was
    added, from the newest snapshot (null without one, i.e. before 2021)."""
    df = _pool_with_snapshot(view, ["status", "chance_of_playing_next_round", "news_added"])
    return _finish(df, AVAILABILITY_DTYPES, ["player_key"])


def ep_next(view: AsOfView) -> pd.DataFrame:
    """Per pool player: FPL's `ep_next` from the newest snapshot (null without one)."""
    df = _pool_with_snapshot(view, ["ep_next"])
    return _finish(df, {"player_key": "int64", "ep_next": "Float64"}, ["player_key"])


def _window(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Per player sums over the rows of `df`, with counts and per-90 rates (each rate over
    the minutes of matches where the stat is known; null when those minutes are 0)."""
    groups = df["player_key"]
    out = pd.DataFrame(
        {
            f"matches_{suffix}": groups.groupby(groups).size(),
            f"apps_{suffix}": (df["minutes"] > 0).groupby(groups).sum(),
        }
    )
    for stat in FORM_STATS:
        values = df[stat].astype("Float64")
        out[f"{stat}_{suffix}"] = values.groupby(groups).sum(min_count=1)
    for stat in RATE_STATS:
        values = df[stat].astype("Float64")
        minutes = df["minutes"].where(values.notna(), 0).groupby(groups).sum()
        total = values.groupby(groups).sum(min_count=1)
        out[f"{stat}_per90_{suffix}"] = (total / minutes * 90).where(minutes > 0)
    return out


def _form_dtypes() -> dict[str, object]:
    dtypes: dict[str, object] = {"player_key": "int64"}
    for suffix in ("last5", "season"):
        dtypes[f"matches_{suffix}"] = "int64"
        dtypes[f"apps_{suffix}"] = "int64"
        dtypes.update({f"{stat}_{suffix}": "Float64" for stat in FORM_STATS})
    for suffix in ("last5", "season"):
        dtypes.update({f"{stat}_per90_{suffix}": "Float64" for stat in RATE_STATS})
    return dtypes


FORM_DTYPES = tuple(_form_dtypes().items())


def recent_form(view: AsOfView) -> pd.DataFrame:
    """Per player with a visible match this season: sums of FORM_STATS, matches,
    appearances (minutes > 0) and per-90 rates over the last 5 matches and the season so
    far. The target GW's matches are not visible before its deadline."""
    season, _, _ = _target(view)
    columns = ["player_key", "season", "fixture_key", "kickoff_time", *FORM_STATS]
    matches = view.table("player_match", columns=columns)
    matches = matches[matches["season"] == season]
    matches = matches.sort_values(["player_key", "kickoff_time", "fixture_key"], kind="mergesort")
    from_end = matches.groupby("player_key").cumcount(ascending=False)
    last = _window(matches[from_end < FORM_WINDOW], "last5")
    form = last.join(_window(matches, "season"), how="outer")
    form = form.rename_axis("player_key").reset_index()
    return _finish(form, FORM_DTYPES, ["player_key"])


# --- teams -------------------------------------------------------------------------------

UPCOMING_DTYPES = (
    ("season", "int64"),
    ("team_key", "int64"),
    ("gw", "int64"),
    ("gw_index", "int64"),
    ("horizon", "int64"),
    ("n_fixtures", "int64"),
    ("fixture_key", "Int64"),
    ("opponent_team_key", "Int64"),
    ("is_home", "boolean"),
    ("kickoff_time", UTC_US),
    ("schedule_source", "str"),
)


def _sides(fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture and team: team_key, opponent_team_key, is_home."""
    columns = ["fixture_key", "gw", "kickoff_time"]
    home = fixtures[columns].assign(
        team_key=fixtures["home_team_key"],
        opponent_team_key=fixtures["away_team_key"],
        is_home=True,
    )
    away = fixtures[columns].assign(
        team_key=fixtures["away_team_key"],
        opponent_team_key=fixtures["home_team_key"],
        is_home=False,
    )
    return pd.concat([home, away], ignore_index=True)


def upcoming_fixtures(view: AsOfView) -> pd.DataFrame:
    """Per club of the season and GW from the target GW to HORIZON GWs later (by
    gw_index): its fixtures in the as-of schedule, one row each (opponent, home flag), or a
    single row with null fixture columns for a blank; `n_fixtures` 0 = blank, 2 = double."""
    season, _, gw_index = _target(view)
    schedule = view.schedule(season)
    gameweeks = _gameweeks(view, season)
    gameweeks = gameweeks[gameweeks["gw_index"].between(gw_index, gw_index + HORIZON)]
    teams = pd.DataFrame(
        {"team_key": sorted(set(schedule["home_team_key"]) | set(schedule["away_team_key"]))}
    )
    grid = teams.merge(gameweeks[["gw", "gw_index"]], how="cross")
    sides = _sides(schedule[schedule["gw"].notna()].astype({"gw": "int64"}))
    out = grid.merge(sides, on=["team_key", "gw"], how="left", validate="one_to_many")
    out["n_fixtures"] = out.groupby(["team_key", "gw"])["fixture_key"].transform("count")
    source = schedule["schedule_source"].iloc[0] if len(schedule) else pd.NA
    out = out.assign(season=season, horizon=out["gw_index"] - gw_index, schedule_source=source)
    sort_by = ["team_key", "gw_index", "kickoff_time", "fixture_key"]
    return _finish(out, UPCOMING_DTYPES, sort_by)


STRENGTH_DTYPES = (
    ("season", "int64"),
    ("gw", "int64"),
    ("fixture_key", "int64"),
    ("kickoff_time", UTC_US),
    ("team_key", "int64"),
    ("opponent_team_key", "int64"),
    ("is_home", "bool"),
    ("elo", "Float64"),
    ("opponent_elo", "Float64"),
    ("p_win", "Float64"),
    ("p_draw", "Float64"),
    ("p_loss", "Float64"),
    ("odds_source", "str"),
)


def _complete(prices: pd.DataFrame) -> pd.DataFrame:
    """fixture_key x OUTCOMES prices, only fixtures with all three known and positive."""
    prices = prices.reindex(columns=list(OUTCOMES))
    return prices[(prices > 0).all(axis=1)]


def _match_probabilities(view: AsOfView, fixture_keys: list[int]) -> pd.DataFrame:
    """Per fixture: overround-normalised h2h probabilities (p_home, p_draw, p_away) from the
    newest odds before the deadline: football-data's market average if complete, else the
    median across Odds API bookmakers of the fixture's newest Odds API snapshot."""
    odds = view.latest("odds_snapshot", by=list(ODDS_GROUP), columns=["price", "snapshot_at"])
    odds = odds[
        (odds["market"] == "h2h")
        & odds["fixture_key"].isin(fixture_keys)
        & odds["outcome"].isin(OUTCOMES)
    ]
    fd = odds[(odds["source"] == "football-data") & (odds["bookmaker"] == "avg")]
    fd = _complete(fd.pivot_table("price", "fixture_key", "outcome", aggfunc="median"))
    api = odds[odds["source"] == "odds-api"]
    api = api[api["snapshot_at"] == api.groupby("fixture_key")["snapshot_at"].transform("max")]
    api = _complete(api.pivot_table("price", "fixture_key", "outcome", aggfunc="median"))
    api = api[~api.index.isin(fd.index)]
    prices = pd.concat([fd.assign(odds_source="football-data"), api.assign(odds_source="odds-api")])
    implied = 1 / prices[list(OUTCOMES)].astype("float64")
    probabilities = implied.div(implied.sum(axis=1), axis=0)
    probabilities.columns = ["p_home", "p_draw", "p_away"]
    probabilities["odds_source"] = prices["odds_source"]
    return probabilities.rename_axis("fixture_key").reset_index()


def team_strength(view: AsOfView) -> pd.DataFrame:
    """Per target-GW fixture and side: both clubs' newest Elo `rating_after` before the
    deadline, and the match's as-of h2h probabilities from the side's view (p_win, p_draw,
    p_loss; null without odds)."""
    season, gw, _ = _target(view)
    schedule = view.schedule(season)
    fixtures = schedule[schedule["gw"] == gw].astype({"gw": "int64"})
    sides = _sides(fixtures)
    ratings = view.table("team_rating", columns=["team_key", "event_time", "rating_after"])
    ratings = ratings.sort_values(["team_key", "event_time"], kind="mergesort")
    elo = ratings.drop_duplicates("team_key", keep="last").set_index("team_key")["rating_after"]
    probabilities = _match_probabilities(view, fixtures["fixture_key"].tolist())
    out = sides.merge(probabilities, on="fixture_key", how="left", validate="many_to_one")
    home = out["is_home"].to_numpy(dtype=bool)
    out = out.assign(
        season=season,
        elo=out["team_key"].map(elo),
        opponent_elo=out["opponent_team_key"].map(elo),
        p_win=np.where(home, out["p_home"], out["p_away"]),
        p_draw=out["p_draw"],
        p_loss=np.where(home, out["p_away"], out["p_home"]),
    )
    return _finish(out, STRENGTH_DTYPES, ["fixture_key", "team_key"])
