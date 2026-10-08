"""Per player-fixture history for the minutes model (Phase 5 plan, Task 3; PLAN §6.3).

Two frames, both read only through the view:
- `training_frame(view)`: one row per visible `player_match` row (every registered player
  and fixture, 0-minute rows included) with its lag features and targets;
- `prediction_frame(view)`: one row per `player_pool` player and horizon fixture of his
  club (`upcoming_fixtures`: the target GW and the next `HORIZON` GWs; none in a blank,
  two in a double) with the lags as of the view's deadline.

**Starts.** `player_match.starts` where it is known (2022/23 GW16 on); before that the
inferred starters (`infer_starts`): per (fixture, team) the 11 players with the most
minutes, ties broken by `player_key`; with fewer than 11 players on the pitch, every one of
them. A starter sent off early, with fewer minutes than a substitute, is the inference's
known error (measured in PLAN §6.3).

**Earlier fixtures only.** A training row's lags use the player's rows of earlier gameweeks
(by `(season, gw_index)`, then kickoff), never his rows of the same GW: both fixtures of a
double see the history as of the GW's deadline, like a prediction does. The team's earlier
fixtures (rest days, fixtures this season) are those of earlier GWs / earlier kickoffs.
"Rows" are the player's own registered fixtures (any club, any season): a new signing's
window holds his old club's fixtures, and `fixtures_at_club` says how long he has been at
the new one. The prediction frame computes the same lags for a query row placed after every
visible row (the target GW), so training and prediction features match by construction;
every horizon fixture gets the lags as of the deadline (frozen), and only `rest_days` (from
the as-of schedule) and the keys differ per fixture.

**Features (`LAG_FEATURES`):**
- `element_type`, `price` (`player_gw.value` at that GW; the pool's price when predicting);
- `start_rate_l{1,3,5,10}`, `minutes_l{1,3,5,10}`: mean starts / minutes over the last k
  rows (NaN without rows), `start_rate_long` and `n_long` over the last `LONG_WINDOW`;
- `start_share_season`: his starts at his current club this season / the club's fixtures
  this season before the GW (NaN before the club's first fixture); `start_share_last`: his
  starts last season (any club) / 38 (NaN if he had no row last season);
- `days_since_app`: days from the GW deadline back to his last appearance (minutes > 0,
  capped at 365; NaN if none); `rest_days`: days since his club's previous fixture (capped
  at 30; 30 without one);
- `absent_run`: consecutive 0-minute rows just before (capped at 10); `returned`: 1 if one
  of his last `RETURN_WINDOW` rows is an appearance right after ≥ `ABSENCE_RUN` 0-minute
  rows;
- `yellows_season`: yellow cards this season; `reds_l3`: red cards in his last 3 rows;
- `comp_start_sum`: the sum of his same-position teammates' `start_rate_l5` (teammates =
  the other registered players of that team-fixture; the pool's club-mates when
  predicting); `comp_rank`: his rank by `start_rate_l5` among them (1 = highest);
- `fixtures_at_club`: his consecutive earlier rows at his current club (capped at 10; 0 for
  a new signing, also when the pool's club differs from his last row's club).

**Dropped (no source):** European/cup matches (no data; rest days count league fixtures
only), manager tenure, age. `team_join_date` is not a feature: the snapshot carries it only
from 2024/25, and the pool's club (a snapshot from 2020/21 GW32) already marks a club change
(`fixtures_at_club` = 0).

**Suspensions.** `ban_remaining`: league fixtures still to serve, from `BAN_RULES` (PLAN
§6.3; simplified, see the constants) and red cards (`RED_BAN` = 1 match: our data can't tell
a straight red from a second yellow), served by the player's subsequent rows. Training rows
carry `banned` (the row's fixture falls within a ban); the prediction frame carries
`ban_remaining` and `club_fixture_order` (0 = his club's first horizon fixture), so a model
can zero the first `ban_remaining` fixtures.

**Targets** (training): `minutes`, `start` (0/1), `sixty` (minutes ≥ 60), `sub` (minutes >
0 without a start); `start_inferred` marks inferred starts.

Vectorized: every lag is a difference of cumulative sums over rows sorted by (player,
season, gw_index, kickoff, fixture); all seasons (~257k rows) take a few seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fplopt.features.baseline import HORIZON, player_pool, upcoming_fixtures
from fplopt.features.store import AsOfView

__all__ = (
    "BAN_RULES",
    "HORIZON",
    "LAG_FEATURES",
    "PREDICTION_COLUMNS",
    "PREDICTION_KEYS",
    "TARGETS",
    "TRAINING_KEYS",
    "infer_starts",
    "prediction_frame",
    "training_frame",
)

UTC_US = pd.DatetimeTZDtype("us", "UTC")
EPOCH = pd.Timestamp(0, tz="UTC")
N_STARTERS = 11
WINDOWS = (1, 3, 5, 10)
LONG_WINDOW = 38
SEASON_FIXTURES = 38  # league fixtures per club and season (start_share_last)
REST_CAP_DAYS = 30.0
APPEARANCE_CAP_DAYS = 365.0
RUN_CAP = 10
CLUB_CAP = 10
ABSENCE_RUN = 3  # 0-minute rows before an appearance that make it a return
RETURN_WINDOW = 3
RED_WINDOW = 3
# (yellow cards this season, last gw_index at which reaching them bans, matches banned).
# Simplified from the Premier League rules: the real cut-offs are the club's 19th / 32nd
# league match (here gw_index 19 / 32), cup and European cards and matches don't count, and
# the ban starts with the next league fixture (in reality two days after the card is
# confirmed, and cup matches serve it).
BAN_RULES = ((5, 19, 1), (10, 32, 2), (15, 99, 3))
RED_BAN = 1  # any red card bans the next league fixture (straight reds get 3 in reality)
MAX_BAN = 4  # longest ban from one fixture (15th yellow + red)

MATCH_COLUMNS = (
    "player_key",
    "season",
    "fixture_key",
    "gw",
    "team_key",
    "kickoff_time",
    "minutes",
    "starts",
    "yellow_cards",
    "red_cards",
)
SORT_KEYS = ("player_key", "season", "gw_index", "kickoff_time", "fixture_key")
LAG_FEATURES = (
    "element_type",
    "price",
    *(f"start_rate_l{k}" for k in WINDOWS),
    *(f"minutes_l{k}" for k in WINDOWS),
    "start_rate_long",
    "n_long",
    "start_share_season",
    "start_share_last",
    "days_since_app",
    "rest_days",
    "absent_run",
    "returned",
    "yellows_season",
    "reds_l3",
    "comp_start_sum",
    "comp_rank",
    "fixtures_at_club",
)
TARGETS = ("minutes", "start", "sixty", "sub")
TRAINING_KEYS = (
    ("player_key", "int64"),
    ("season", "int64"),
    ("fixture_key", "int64"),
    ("gw", "int64"),
    ("gw_index", "int64"),
    ("team_key", "int64"),
    ("kickoff_time", UTC_US),
)
PREDICTION_KEYS = (
    ("player_key", "int64"),
    ("fixture_key", "int64"),
    ("team_key", "int64"),
    ("season", "int64"),
    ("gw", "int64"),
    ("gw_index", "int64"),
    ("horizon", "int64"),
    ("kickoff_time", UTC_US),
)
PREDICTION_COLUMNS = (
    *(name for name, _ in PREDICTION_KEYS),
    *LAG_FEATURES,
    "ban_remaining",
    "club_fixture_order",
)


def _days(times: pd.Series) -> np.ndarray:
    """Float days since the epoch (NaN for NaT)."""
    return ((pd.to_datetime(times, utc=True) - EPOCH) / pd.Timedelta(days=1)).to_numpy(
        dtype="float64", na_value=np.nan
    )


# --- starts ------------------------------------------------------------------------------


def infer_starts(matches: pd.DataFrame) -> pd.Series:
    """Inferred starters (bool, aligned to `matches`): per (fixture_key, team_key) the
    `N_STARTERS` rows with the most minutes, ties by player_key; only rows with minutes > 0
    (fewer than 11 players on the pitch: all of them)."""
    order = matches.sort_values(
        ["fixture_key", "team_key", "minutes", "player_key"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    rank = order.groupby(["fixture_key", "team_key"], sort=False).cumcount()
    started = (rank < N_STARTERS) & (order["minutes"] > 0)
    return started.reindex(matches.index)


def _starts(matches: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(start as float 0/1, inferred flag): real `starts` where known, else inferred."""
    real = pd.to_numeric(matches["starts"], errors="coerce").astype("float64")
    inferred = infer_starts(matches).astype("float64")
    known = real.notna()
    return real.where(known, inferred).clip(0, 1), ~known


# --- inputs ------------------------------------------------------------------------------


def _gameweeks(view: AsOfView) -> pd.DataFrame:
    return view.table("gameweek", columns=["season", "gw", "gw_index", "deadline_time"])


def _matches(view: AsOfView, gameweeks: pd.DataFrame) -> pd.DataFrame:
    """Visible player_match rows with gw_index, the GW deadline, start and position/price."""
    matches = view.table("player_match", columns=list(MATCH_COLUMNS))
    matches = matches.merge(gameweeks, on=["season", "gw"], how="inner", validate="many_to_one")
    start, inferred = _starts(matches)
    matches = matches.assign(
        minutes=matches["minutes"].astype("float64"),
        start=start,
        start_inferred=inferred,
        yellow_cards=matches["yellow_cards"].fillna(0).astype("float64"),
        red_cards=matches["red_cards"].fillna(0).astype("float64"),
    )
    registered = view.table(
        "player_gw", columns=["player_key", "season", "gw", "element_type", "value"]
    )
    registered = registered.drop_duplicates(["player_key", "season", "gw"], keep="last")
    matches = matches.merge(
        registered, on=["player_key", "season", "gw"], how="left", validate="many_to_one"
    )
    positions = view.table("player_season", columns=["player_key", "season", "element_type"])
    positions = positions.drop_duplicates(["player_key", "season"], keep="last")
    fallback = matches[["player_key", "season"]].merge(
        positions, on=["player_key", "season"], how="left", validate="many_to_one"
    )["element_type"]
    element_type = matches["element_type"].astype("float64")
    matches["element_type"] = element_type.fillna(fallback.astype("float64")).to_numpy()
    return matches.rename(columns={"value": "price"})


def _team_fixtures(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (team_key, fixture_key) of the matches, with season, gw_index, kickoff."""
    columns = ["team_key", "fixture_key", "season", "gw_index", "kickoff_time"]
    fixtures = matches[columns].drop_duplicates(["team_key", "fixture_key"])
    return fixtures.sort_values(
        ["team_key", "kickoff_time", "fixture_key"], kind="mergesort"
    ).reset_index(drop=True)


def _fixtures_before(fixtures: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Per row (team_key, season, gw_index): the team's fixtures of that season in earlier
    GWs (by gw_index)."""
    counts = fixtures.groupby(["team_key", "season", "gw_index"]).size().rename("n")
    counts = counts.groupby(level=["team_key", "season"]).cumsum().reset_index()
    counts = counts.sort_values("gw_index", kind="mergesort")
    left = rows[["team_key", "season", "gw_index"]].assign(_row=np.arange(len(rows)))
    left = left.astype({"team_key": "int64", "season": "int64", "gw_index": "int64"})
    counts = counts.astype({"team_key": "int64", "season": "int64", "gw_index": "int64"})
    merged = pd.merge_asof(
        left.sort_values("gw_index", kind="mergesort"),
        counts,
        on="gw_index",
        by=["team_key", "season"],
        allow_exact_matches=False,
    )
    out = np.zeros(len(rows))
    out[merged["_row"].to_numpy()] = merged["n"].fillna(0).to_numpy(dtype="float64")
    return out


# --- lags --------------------------------------------------------------------------------


def _windowed(values: np.ndarray, block: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Sum of values[lo:block] per row (exclusive cumulative sums)."""
    cumulative = np.concatenate([[0.0], np.cumsum(values, dtype="float64")])
    return cumulative[block] - cumulative[lo]


def _lags(rows: pd.DataFrame) -> pd.DataFrame:
    """Lag features of `rows` (sorted by SORT_KEYS, RangeIndex) from each player's rows of
    earlier GWs. Needs player_key, season, gw_index, team_key, minutes, start, yellow_cards,
    red_cards, kickoff_time and ref_time (the GW deadline)."""
    n = len(rows)
    index = np.arange(n)
    player = rows["player_key"].to_numpy()
    first = index - rows.groupby("player_key", sort=False).cumcount().to_numpy()
    block_keys = ["player_key", "season", "gw_index"]
    in_block = rows.groupby(block_keys, sort=False).cumcount().to_numpy()
    block = index - in_block
    season_first = index - rows.groupby(["player_key", "season"], sort=False).cumcount().to_numpy()
    minutes = rows["minutes"].to_numpy(dtype="float64")
    start = rows["start"].to_numpy(dtype="float64")
    appeared = (minutes > 0).astype("float64")
    has_prior = block > first
    prev = np.where(has_prior, block - 1, 0)
    out = {}

    for k in (*WINDOWS, LONG_WINDOW):
        lo = np.maximum(block - k, first)
        count = (block - lo).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = _windowed(start, block, lo) / count
            mean_minutes = _windowed(minutes, block, lo) / count
        if k == LONG_WINDOW:
            out["start_rate_long"] = rate
            out["n_long"] = count
        else:
            out[f"start_rate_l{k}"] = rate
            out[f"minutes_l{k}"] = mean_minutes

    # Starts at the current club this season, over the club's fixtures this season.
    club_season = ["player_key", "season", "team_key"]
    club_starts = pd.Series(start).groupby([rows[c] for c in club_season]).cumsum()
    block_starts = pd.Series(start).groupby([rows[c] for c in block_keys]).cumsum()
    out["_club_starts"] = (club_starts - block_starts).to_numpy(dtype="float64")

    # Last season (any club): starts / SEASON_FIXTURES, NaN without a row.
    last = rows[["player_key", "season"]].assign(start=start)
    per_season = last.groupby(["player_key", "season"]).agg(starts=("start", "sum"))
    per_season = per_season.reset_index().assign(season=lambda d: d["season"] + 1)
    shifted = rows[["player_key", "season"]].merge(
        per_season, on=["player_key", "season"], how="left", validate="many_to_one"
    )
    out["start_share_last"] = shifted["starts"].to_numpy(dtype="float64") / SEASON_FIXTURES

    # Days since the last appearance before the block, from the GW deadline.
    kickoff = _days(rows["kickoff_time"])
    appearance = pd.Series(np.where(appeared > 0, kickoff, np.nan))
    last_appearance = appearance.groupby(player).ffill().to_numpy()
    since = _days(rows["ref_time"]) - np.where(has_prior, last_appearance[prev], np.nan)
    out["days_since_app"] = np.clip(since, 0, APPEARANCE_CAP_DAYS)

    # Consecutive 0-minute rows ending at each row; the run just before the block.
    nonzero = pd.Series(np.where(appeared > 0, index, np.nan)).groupby(player).ffill()
    run = index - nonzero.fillna(pd.Series(first - 1)).to_numpy()
    out["absent_run"] = np.minimum(np.where(has_prior, run[prev], 0), RUN_CAP).astype("float64")
    previous_run = np.where(index > first, run[np.maximum(index - 1, 0)], 0)
    came_back = ((appeared > 0) & (previous_run >= ABSENCE_RUN)).astype("float64")
    lo = np.maximum(block - RETURN_WINDOW, first)
    out["returned"] = (_windowed(came_back, block, lo) > 0).astype("float64")

    # Cards: yellows this season before the block, reds in the last RED_WINDOW rows.
    yellow = rows["yellow_cards"].to_numpy(dtype="float64")
    red = rows["red_cards"].to_numpy(dtype="float64")
    out["yellows_season"] = _windowed(yellow, block, season_first)
    out["reds_l3"] = _windowed(red, block, np.maximum(block - RED_WINDOW, first))

    # Bans: matches each row triggers, still to serve at the block (served by later rows).
    total = pd.Series(yellow).groupby([rows["player_key"], rows["season"]]).cumsum().to_numpy()
    before = total - yellow
    gw_index = rows["gw_index"].to_numpy()
    length = RED_BAN * (red > 0)
    for cards, last_gw, matches in BAN_RULES:
        length = length + matches * ((before < cards) & (total >= cards) & (gw_index <= last_gw))
    remaining = np.zeros(n)
    for d in range(1, MAX_BAN + 1):
        source = block - d
        valid = source >= first
        left = np.where(valid, length[np.maximum(source, 0)] - (d - 1), 0)
        remaining = np.maximum(remaining, left)
    out["ban_remaining"] = remaining
    out["banned"] = (in_block < remaining).astype("float64")

    # Consecutive earlier rows at the current club (the spell), capped.
    team = rows["team_key"].to_numpy()
    changed = (index == first) | (team != np.roll(team, 1))
    spell_first = pd.Series(np.where(changed, index, np.nan)).ffill().to_numpy()
    at_club = np.where(block >= spell_first, block - spell_first, 0)
    out["fixtures_at_club"] = np.minimum(at_club, CLUB_CAP).astype("float64")
    return pd.DataFrame(out)


def _competition(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """comp_start_sum / comp_rank of each row among the rows sharing `by` (+ position)."""
    keys = [*by, "element_type"]
    rate = frame["start_rate_l5"].fillna(0.0)
    total = rate.groupby([frame[k] for k in keys], dropna=False).transform("sum")
    rank = rate.groupby([frame[k] for k in keys], dropna=False).rank(ascending=False, method="min")
    return frame.assign(comp_start_sum=total - rate, comp_rank=rank.astype("float64"))


def _sorted(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values(list(SORT_KEYS), kind="mergesort").reset_index(drop=True)


# --- training ----------------------------------------------------------------------------


def training_frame(view: AsOfView) -> pd.DataFrame:
    """One row per visible player_match row: keys (TRAINING_KEYS), LAG_FEATURES, the
    targets (TARGETS), `start_inferred` and `banned`; sorted by (player_key, season,
    gw_index, kickoff_time, fixture_key), RangeIndex."""
    gameweeks = _gameweeks(view)
    matches = _sorted(_matches(view, gameweeks).rename(columns={"deadline_time": "ref_time"}))
    lags = _lags(matches)
    frame = pd.concat([matches, lags], axis=1)
    fixtures = _team_fixtures(matches)
    frame["_club_fixtures"] = _fixtures_before(fixtures, frame)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = frame["_club_starts"] / frame["_club_fixtures"]
    frame["start_share_season"] = share.where(frame["_club_fixtures"] > 0)
    fixtures["previous"] = fixtures.groupby("team_key")["kickoff_time"].shift(1)
    rest = (fixtures["kickoff_time"] - fixtures["previous"]) / pd.Timedelta(days=1)
    fixtures["rest_days"] = rest.clip(upper=REST_CAP_DAYS).fillna(REST_CAP_DAYS)
    frame = frame.merge(
        fixtures[["team_key", "fixture_key", "rest_days"]],
        on=["team_key", "fixture_key"],
        how="left",
        validate="many_to_one",
    )
    frame = _competition(frame, ["fixture_key", "team_key"])
    minutes = frame["minutes"]
    frame = frame.assign(
        sixty=(minutes >= 60).astype("float64"),
        sub=((minutes > 0) & (frame["start"] == 0)).astype("float64"),
    )
    columns = [
        *(name for name, _ in TRAINING_KEYS),
        *LAG_FEATURES,
        *TARGETS,
        "start_inferred",
        "banned",
    ]
    dtypes = {
        **dict(TRAINING_KEYS),
        **{name: "float64" for name in (*LAG_FEATURES, *TARGETS, "banned")},
        "start_inferred": "bool",
    }
    out = frame[columns].astype(dtypes)
    return out.sort_values(list(SORT_KEYS), kind="mergesort").reset_index(drop=True)


# --- prediction --------------------------------------------------------------------------


def _horizon_fixtures(view: AsOfView, season: int) -> pd.DataFrame:
    """Per club and horizon fixture: keys, kickoff and rest days (from the as-of schedule:
    days since the club's previous scheduled fixture of the season, capped)."""
    upcoming = upcoming_fixtures(view)
    upcoming = upcoming[upcoming["fixture_key"].notna()]
    schedule = view.schedule(season)
    schedule = schedule[schedule["kickoff_time"].notna()]
    sides = pd.concat(
        [
            schedule[["fixture_key", "kickoff_time"]].assign(team_key=schedule["home_team_key"]),
            schedule[["fixture_key", "kickoff_time"]].assign(team_key=schedule["away_team_key"]),
        ],
        ignore_index=True,
    )
    sides = sides.sort_values(["team_key", "kickoff_time", "fixture_key"], kind="mergesort")
    previous = sides.groupby("team_key")["kickoff_time"].shift(1)
    rest = (sides["kickoff_time"] - previous) / pd.Timedelta(days=1)
    sides = sides.assign(rest_days=rest.clip(upper=REST_CAP_DAYS).fillna(REST_CAP_DAYS))
    upcoming = upcoming.astype({"fixture_key": "int64"})
    out = upcoming.merge(
        sides[["team_key", "fixture_key", "rest_days"]],
        on=["team_key", "fixture_key"],
        how="left",
        validate="one_to_one",
    )
    out["rest_days"] = out["rest_days"].fillna(REST_CAP_DAYS)
    out = out.sort_values(["team_key", "kickoff_time", "fixture_key"], kind="mergesort")
    out["club_fixture_order"] = out.groupby("team_key").cumcount().astype("float64")
    return out


def prediction_frame(view: AsOfView) -> pd.DataFrame:
    """One row per pool player and horizon fixture of his club: PREDICTION_COLUMNS (keys,
    LAG_FEATURES as of the deadline, `ban_remaining`, `club_fixture_order`); sorted by
    (player_key, horizon, kickoff_time, fixture_key), RangeIndex."""
    season, gw = view.gameweek_for_deadline()
    gameweeks = _gameweeks(view)
    target = int(
        gameweeks.loc[(gameweeks["season"] == season) & (gameweeks["gw"] == gw), "gw_index"].iloc[0]
    )
    pool = player_pool(view)
    matches = _matches(view, gameweeks)
    fixtures = _team_fixtures(matches)
    history = matches[matches["player_key"].isin(pool["player_key"])]
    query = pd.DataFrame(
        {
            "player_key": pool["player_key"].to_numpy(),
            "season": season,
            "gw": gw,
            "gw_index": target,
            "fixture_key": -1,
            "team_key": pool["team_key"].to_numpy(),
            "kickoff_time": view.deadline,
            "ref_time": view.deadline,
            "minutes": 0.0,
            "start": 0.0,
            "yellow_cards": 0.0,
            "red_cards": 0.0,
            "_query": True,
        }
    )
    history = history.drop(columns=["deadline_time"]).assign(
        ref_time=history["deadline_time"], _query=False
    )
    rows = _sorted(pd.concat([history, query], ignore_index=True))
    lags = _lags(rows)
    lags = lags[rows["_query"].to_numpy(dtype=bool)].reset_index(drop=True)
    queried = rows.loc[rows["_query"].to_numpy(dtype=bool), ["player_key", "team_key"]]
    frame = pd.concat([queried.reset_index(drop=True), lags], axis=1)
    frame = frame.merge(
        pool[["player_key", "element_type", "price"]],
        on="player_key",
        how="left",
        validate="one_to_one",
    )
    frame["element_type"] = frame["element_type"].astype("float64")
    frame["price"] = frame["price"].astype("float64")
    club = frame[["team_key"]].assign(season=season, gw_index=target)
    club_fixtures = _fixtures_before(fixtures, club)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = frame["_club_starts"] / club_fixtures
    frame["start_share_season"] = share.where(club_fixtures > 0)
    frame = _competition(frame, ["team_key"])
    horizon = _horizon_fixtures(view, season)
    out = frame.merge(
        horizon[
            [
                "team_key",
                "fixture_key",
                "season",
                "gw",
                "gw_index",
                "horizon",
                "kickoff_time",
                "rest_days",
                "club_fixture_order",
            ]
        ],
        on="team_key",
        how="inner",
        validate="many_to_many",
    )
    dtypes = {
        **dict(PREDICTION_KEYS),
        **{name: "float64" for name in (*LAG_FEATURES, "ban_remaining", "club_fixture_order")},
    }
    out = out[list(PREDICTION_COLUMNS)].astype(dtypes)
    sort_by = ["player_key", "horizon", "kickoff_time", "fixture_key"]
    return out.sort_values(sort_by, kind="mergesort").reset_index(drop=True)
