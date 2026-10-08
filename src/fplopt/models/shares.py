"""Player goal and assist shares, and penalties (PLAN §6.2; Phase 5 plan, *Player model →
Shares* and Task 4).

**npxG and xA per player-match** (`match_xg`, first source available):
- npxG: Understat `us_npxg`; else FPL's `fpl_xg` (Opta, penalties included) minus
  `FPL_PEN_XG` (0.79, Opta's xG of every penalty: each FPL row whose only shot was a
  penalty has exactly 0.79) per penalty attempt, times `k_goals` (Opta → Understat scale);
  else missing. Penalty attempts = penalty goals + `penalties_missed` (FPL). Penalty goals:
  Understat `us_goals − us_npg` where it exists; otherwise inferred from FPL alone: the
  candidates are min(goals, ⌊(fpl_xg − 0.79 · missed) / 0.79⌋) (each penalty adds 0.79 to
  the match's FPL xG), and the expected penalty goals are candidates × π, with π_main for
  his club's main taker at kickoff (rank 1 of the club's `penalties_order` listing in the
  newest snapshot before the kickoff) and π_other for everyone else.
- xA: `us_xa`, else `fpl_xa` × `k_assists`, else missing.
- k and π are fitted on the visible rows where both sources exist (2022/23 GW16 on):
  k = Σ Understat / Σ FPL estimate; π = Σ Understat penalty goals / Σ candidates per group;
  each shrunk toward its default (`DEFAULT_PEN_PI`; k toward 1) with a few pseudo-rows.
  Before any overlap is visible no row needs them. `dev/shares_eval.py npxg-check` measures
  the substitution on the 2022/23 GW16–2024/25 overlap (PLAN §3 VERIFY).

**Shares** (`goal_share`, `assist_share`): the player's npxG (xA) per 90 over the team's
npxG per 90 while he is on the pitch, approximated as team npxG × minutes / 90. Per played
row: X = his npxG, D = team npxG × minutes / 90 (team npxG: Understat's team value, else
the sum of the players' npxG when every player who played has one). Without the team's
npxG (2016/17–2018/19: no Understat team data), D uses the team's goals × `ratio_team`
(Σ team npxG / Σ team goals). Without his npxG (the 2016/17–2018/19 gaps: 54/40/28% of
played rows), X = his goals × `ratio_goals` (Σ npxG / Σ goals over rows with both) and the
row weighs `goals_weight` (stronger shrinkage). Assists the same with xA / FPL assists
(`ratio_assists`). The fallback's goals include penalties (they can't be separated without
Understat); its lower weight limits that. `fit_shares` logs the coverage per season.

**Marcel prior** (`marcel_shares`): share = (Σ w·X + μ·D0) / (Σ w·D + D0), over his rows
of the current season (in-season data up to the deadline) and the 3 before, with
w = `season_weights`[age] (age 0 = the current season, whose rows enter as they become
visible, so the in-season weight grows with the minutes played; tuned default 2-2-1-1: the
previous seasons 2-1-1 and the current season level with the last one) ×
`club_change_weight` for rows at a club other than his current one (a club change keeps the
old share with extra shrinkage) × `goals_weight` for fallback rows; D0 = `prior_minutes` /
90 × the league's mean team output per 90 (`team_output`), i.e. `prior_minutes` minutes at
the prior mean μ (tuned: 1920, stronger than the plan's ~480). μ is the position + price
prior: per position a quasi-Poisson regression X ~ D·exp(a + b·z), z = (price − the
position's mean price) / 10, fitted on the last `FIT_SEASONS` seasons' player-seasons (price
= his first `player_gw` price of the season; at prediction his pool price). The plan's
variant, the league average of his position (Σ X / Σ D) for players with history and the
price prior for new players only (no played minute in the window), is
`price_prior_all=False`; it scored a little worse on develop.

**Team consistency** (`predict_shares`, per team-fixture over the club's pool players, f =
e_minutes / 90 from the minutes frame, the expected XI by minutes):
- λ_np = max(λ_for · (1 − `own_goal_fraction`) − E[team penalty goals], 0) with λ_for from
  `team_lambdas`; own goals are taken out because no player of the club is credited with
  them (an addition to the plan's "λ_for minus expected penalty goals"; ~3.5% of goals);
- E[np goals_i] = λ_np · g_i f_i / Σ_j g_j f_j;
- E[FPL assists_i] = λ_for · `assist_fraction` · a_i f_i / Σ_j a_j f_j, with the assist
  fraction = Σ FPL assists / Σ team goals (FPL assists ≠ xA: penalties won, rebounds, …);
- output shares are normalized: goal_share_i = g_i / Σ_j g_j f_j (so Σ goal_share · f = 1).

**Penalties:** E[pen goals_i] = P(on pitch) · P(taker) · team attempts per match ·
conversion.
- Takers: the club's listing from `penalties_order` in the season's newest snapshot (from
  2020/21 GW32), ranked by order; each listed player is the designated taker with
  probability q = `taker_q` (fitted: the share of a club's attempts taken by its main taker
  in matches he played in full, from the snapshot era). Before snapshots (or a club with no
  listing): his recent attempts (penalty goals + misses, any club, weighted like the share
  rows) s_i, ranked by s, q_i = s_i / (s_i + `taker_prior`).
- Per fixture, along the ranking: P_i = f_i q_i Π_{j ranked above} (1 − f_j q_j) (the
  first-choice taker takes it when on the pitch, else the next one); the leftover Π (1 −
  f_j q_j) goes to all club players by g_i f_i / Σ g f. So Σ P_i = 1 and Σ E[pen goals] =
  the team's expected penalty goals. `p_taker` = P_i / f_i (0 when f_i = 0).
- Team attempts per match and conversion: per club over the last `FIT_SEASONS` seasons'
  team-fixtures with known attempts (Understat team xG − npxG over 0.76, or the players'
  sum when every player who played has known attempts), shrunk toward the league's rate
  (`pen_team_prior` pseudo-matches) and conversion (`conversion_prior` pseudo-attempts);
  the league values are shrunk toward `DEFAULT_PEN_RATE` / `DEFAULT_CONVERSION` (general
  EPL figures, used where no attempts are visible: 2016/17–2018/19 lack team-level data).

API: `fit_shares(view) -> SharesFit` (frozen; the walk-forward fit at the refit cutoff,
`fplopt.models.fitted`) and `predict_shares(view, fit, minutes, team) -> frame`, where
`minutes` is the `predict_minutes` / `adjust_minutes` frame and `team` the `team_lambdas`
frame of the same view. One row per row of `minutes`: `SHARES_COLUMNS`, sorted by
(player_key, horizon, fixture_key), RangeIndex. Player data enters at predict time (every
row visible at the deadline); the fit holds the league-level parameters. No state, no file
access; not registered (Task 5 does that).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fplopt.features.baseline import player_pool
from fplopt.features.store import AsOfView

log = logging.getLogger(__name__)

__all__ = (
    "FPL_PEN_XG",
    "SHARES_COLUMNS",
    "SharesFit",
    "SharesParams",
    "fit_shares",
    "marcel_shares",
    "match_xg",
    "predict_shares",
    "taker_probabilities",
)

FPL_PEN_XG = 0.79  # Opta's xG of a penalty (FPL expected_goals)
US_PEN_XG = 0.76  # Understat's xG of a penalty (team us_xg − us_npxg per attempt)
FIT_SEASONS = 3  # seasons of rows behind the league-level parameters
DEFAULT_PEN_RATE = 0.13  # penalty attempts per team-match (EPL long run)
DEFAULT_CONVERSION = 0.78
DEFAULT_ASSIST_FRACTION = 0.88
DEFAULT_OWN_GOAL_FRACTION = 0.035
DEFAULT_TAKER_Q = 0.85
DEFAULT_PEN_PI = (0.5, 0.2)  # P(an inferred candidate is a penalty): main taker, others
LEAGUE_PRIOR_MATCHES = 100.0  # pseudo team-matches at DEFAULT_PEN_RATE
LEAGUE_PRIOR_ATTEMPTS = 20.0  # pseudo attempts at DEFAULT_CONVERSION
TAKER_PRIOR_ATTEMPTS = 10.0  # pseudo attempts at DEFAULT_TAKER_Q
PI_PRIOR = 10.0  # pseudo candidates at DEFAULT_PEN_PI
K_PRIOR = 5.0  # pseudo npxG / xA at k = 1
SNAPSHOT_TOLERANCE = pd.Timedelta(days=14)  # a listing older than this at kickoff is stale
PRICE_SCALE = 10.0  # price units (£0.1m) per regression unit (£1m)
PRICE_ITERATIONS = 30
PRICE_RIDGE = 1.0
RATE_EPS = 1e-9
KEYS = ("player_key", "fixture_key", "team_key", "season", "gw", "gw_index", "horizon")
VALUES = (
    "e_np_goals",
    "e_pen_goals",
    "e_goals",
    "e_assists",
    "goal_share",
    "assist_share",
    "p_taker",
)
SHARES_COLUMNS = (*KEYS, *VALUES)
SORT_BY = ("player_key", "horizon", "fixture_key")
MATCH_COLUMNS = (
    "player_key",
    "season",
    "fixture_key",
    "gw",
    "team_key",
    "kickoff_time",
    "minutes",
    "goals_scored",
    "assists",
    "penalties_missed",
    "own_goals",
    "fpl_xg",
    "fpl_xa",
    "us_goals",
    "us_npg",
    "us_npxg",
    "us_xa",
)
FLOAT_COLUMNS = MATCH_COLUMNS[6:]


@dataclass(frozen=True)
class SharesParams:
    """Shares settings. `prior_minutes`, `season_weights`, `club_change_weight`,
    `price_prior_all` and `taker_prior` act at predict time only (so variants can share a
    fit); `goals_weight` also weighs the fallback rows in the fitted position means, and
    `pen_team_prior` / `conversion_prior` shrink the fitted club penalty rates. Tuned on
    develop by `dev/shares_eval.py` (2017/18–2022/23, 38 variants; objective: mean over
    seasons of the goals + FPL-assists Poisson log-likelihood per player-fixture, horizons
    0–5): pseudo-minutes 240 < 480 < 960 < 1440 ≈ 2880 ≈ 1920 (best; the surface is flat
    from 960 up); current-season weight 2 > 3 > 5 and > 1 (previous seasons 2-1-1); club
    change weight 0.5 ≈ 0.25 > 1; the price prior for everyone beats it for new players
    only (+0.0003); goals_weight 0.5 ≥ 0.25 > 1."""

    prior_minutes: float = 1920.0  # pseudo-minutes at the prior mean
    season_weights: tuple[float, ...] = (2.0, 2.0, 1.0, 1.0)  # by age: current, −1, −2, −3
    club_change_weight: float = 0.5  # rows at a club other than his current one
    goals_weight: float = 0.5  # rows without his xG (goals / assists fallback)
    price_prior_all: bool = True  # price prior for everyone, not only new players
    pen_team_prior: float = 38.0  # pseudo team-matches at the league's penalty rate
    conversion_prior: float = 20.0  # pseudo attempts at the league's conversion
    taker_prior: float = 1.0  # history takers: q = s / (s + taker_prior)

    def __post_init__(self) -> None:
        if not self.season_weights or any(w < 0 for w in self.season_weights):
            raise ValueError("season_weights must be non-empty and >= 0")
        if self.prior_minutes <= 0:
            raise ValueError("prior_minutes must be > 0")


@dataclass(frozen=True)
class SharesFit:
    """League-level parameters fitted at `cutoff` (frozen; no state). Position tables are
    `(element_type, value)` pairs; `price_*` are `(element_type, a, b, center)`; `team_pen`
    is `(team_key, attempts per match, conversion)` for clubs with visible attempts."""

    cutoff: pd.Timestamp
    params: SharesParams
    k_goals: float
    k_assists: float
    pen_pi: tuple[float, float]
    ratio_goals: float
    ratio_assists: float
    ratio_team: float
    team_output: float
    position_goal: tuple[tuple[int, float], ...]
    position_assist: tuple[tuple[int, float], ...]
    price_goal: tuple[tuple[int, float, float, float], ...]
    price_assist: tuple[tuple[int, float, float, float], ...]
    pen_rate: float
    conversion: float
    team_pen: tuple[tuple[int, float, float], ...]
    assist_fraction: float
    own_goal_fraction: float
    taker_q: float
    n_rows: int
    coverage: tuple[tuple[int, float, float], ...] = field(default=())  # season, xG, xA share


# --- inputs ---------------------------------------------------------------------------------


def _floats(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].astype("Float64").to_numpy(dtype="float64", na_value=np.nan)


def _matches(view: AsOfView, min_season: int | None = None) -> pd.DataFrame:
    """Visible player_match rows (float stats with NaN) of seasons >= `min_season` (all if
    None), with element_type and the GW price (only when all seasons are read: the fit)."""
    matches = view.table("player_match", columns=list(MATCH_COLUMNS))
    if min_season is not None:
        matches = matches[matches["season"] >= min_season].reset_index(drop=True)
    matches = matches.assign(**{c: _floats(matches, c) for c in FLOAT_COLUMNS})
    sort_by = ["player_key", "season", "kickoff_time", "fixture_key"]
    if min_season is not None:
        matches = matches.assign(element_type=np.nan, price=np.nan)
        return matches.sort_values(sort_by, kind="mergesort").reset_index(drop=True)
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
    matches["price"] = matches["value"].astype("float64")
    return matches.sort_values(sort_by, kind="mergesort").reset_index(drop=True)


def _main_takers(view: AsOfView) -> pd.DataFrame:
    """Per snapshot time and club, the rank-1 player of the club's `penalties_order` listing
    (lowest order, ties by player_key): snapshot_at, team_key, taker_key; sorted by time."""
    snaps = view.table(
        "player_snapshot", columns=["snapshot_at", "player_key", "team_key", "penalties_order"]
    )
    snaps = snaps[snaps["penalties_order"].notna()]
    snaps = snaps.sort_values(
        ["snapshot_at", "team_key", "penalties_order", "player_key"], kind="mergesort"
    )
    first = snaps.drop_duplicates(["snapshot_at", "team_key"], keep="first")
    out = first.rename(columns={"player_key": "taker_key"})[
        ["snapshot_at", "team_key", "taker_key"]
    ]
    return out.astype({"team_key": "int64", "taker_key": "int64"}).reset_index(drop=True)


def _main_taker_at_kickoff(rows: pd.DataFrame, takers: pd.DataFrame) -> np.ndarray:
    """Per row (team_key, kickoff_time): the club's main taker in the newest listing before
    kickoff (within SNAPSHOT_TOLERANCE), −1 without one; aligned to `rows`."""
    out = np.full(len(rows), -1, dtype="int64")
    if rows.empty or takers.empty:
        return out
    left = rows[["team_key", "kickoff_time"]].assign(_row=np.arange(len(rows)))
    left = left.astype({"team_key": "int64"}).sort_values("kickoff_time", kind="mergesort")
    merged = pd.merge_asof(
        left,
        takers,
        left_on="kickoff_time",
        right_on="snapshot_at",
        by="team_key",
        allow_exact_matches=False,
        tolerance=SNAPSHOT_TOLERANCE,
    )
    found = merged["taker_key"].notna().to_numpy()
    out[merged["_row"].to_numpy()[found]] = merged["taker_key"].to_numpy()[found].astype("int64")
    return out


def _main_taker_flags(view: AsOfView, matches: pd.DataFrame) -> np.ndarray:
    """Per row: True if he was his club's main taker at kickoff. Only rows with FPL xG need
    it (penalty inference), so only those are looked up (False elsewhere)."""
    flags = np.zeros(len(matches), dtype=bool)
    need = ~np.isnan(matches["fpl_xg"].to_numpy(dtype="float64"))
    if not need.any():
        return flags
    takers = _main_takers(view)
    main = _main_taker_at_kickoff(matches[need], takers)
    flags[need] = main == matches.loc[need, "player_key"].to_numpy(dtype="int64")
    return flags


# --- npxG / xA sources ----------------------------------------------------------------------


def _candidates(frame: pd.DataFrame) -> np.ndarray:
    """Penalty-goal candidates from FPL alone: min(goals, ⌊(fpl_xg − 0.79·missed)/0.79⌋)."""
    fpl_xg = frame["fpl_xg"].to_numpy(dtype="float64")
    missed = np.nan_to_num(frame["penalties_missed"].to_numpy(dtype="float64"))
    goals = np.nan_to_num(frame["goals_scored"].to_numpy(dtype="float64"))
    room = np.floor((fpl_xg - FPL_PEN_XG * missed + 1e-6) / FPL_PEN_XG)
    return np.minimum(goals, np.clip(np.nan_to_num(room), 0, None))


def match_xg(
    frame: pd.DataFrame,
    k_goals: float = 1.0,
    k_assists: float = 1.0,
    pen_pi: tuple[float, float] = DEFAULT_PEN_PI,
) -> pd.DataFrame:
    """Per player-match row the npxG / xA from the first source available (see the module
    docstring). `frame` has the float columns `minutes, goals_scored, penalties_missed,
    fpl_xg, fpl_xa, us_goals, us_npg, us_npxg, us_xa` (NaN = missing) and the bool
    `main_taker`. Returns a frame aligned to it: `npxg`, `xa`, `pen_goals`, `pen_attempts`
    (NaN when unknown) and `xg_source` / `xa_source` ('understat', 'fpl' or 'none')."""
    us_npxg = frame["us_npxg"].to_numpy(dtype="float64")
    fpl_xg = frame["fpl_xg"].to_numpy(dtype="float64")
    missed = np.nan_to_num(frame["penalties_missed"].to_numpy(dtype="float64"))
    us_pen = frame["us_goals"].to_numpy(dtype="float64") - frame["us_npg"].to_numpy(dtype="float64")
    main = frame["main_taker"].to_numpy(dtype=bool)
    inferred = _candidates(frame) * np.where(main, pen_pi[0], pen_pi[1])
    pen_goals = np.where(~np.isnan(us_pen), us_pen, np.where(np.isnan(fpl_xg), np.nan, inferred))
    attempts = pen_goals + missed
    fpl_npxg = k_goals * np.clip(fpl_xg - FPL_PEN_XG * attempts, 0.0, None)
    has_us = ~np.isnan(us_npxg)
    has_fpl = ~np.isnan(fpl_xg)
    npxg = np.where(has_us, us_npxg, np.where(has_fpl, fpl_npxg, np.nan))
    us_xa = frame["us_xa"].to_numpy(dtype="float64")
    fpl_xa = frame["fpl_xa"].to_numpy(dtype="float64")
    xa = np.where(~np.isnan(us_xa), us_xa, k_assists * fpl_xa)
    sources = np.array(["none", "fpl", "understat"])
    return pd.DataFrame(
        {
            "npxg": npxg,
            "xa": xa,
            "pen_goals": pen_goals,
            "pen_attempts": attempts,
            "xg_source": sources[np.where(has_us, 2, np.where(has_fpl, 1, 0))],
            "xa_source": sources[np.where(~np.isnan(us_xa), 2, np.where(np.isnan(xa), 0, 1))],
        },
        index=frame.index,
    )


def _shrunk_ratio(numerator: float, denominator: float, prior: float, weight: float) -> float:
    return float((numerator + prior * weight) / (denominator + weight))


def _source_constants(matches: pd.DataFrame) -> tuple[float, float, tuple[float, float]]:
    """k_goals, k_assists and π from the visible rows where Understat and FPL both exist."""
    both = matches[
        matches["us_npxg"].notna() & matches["fpl_xg"].notna() & (matches["minutes"] > 0)
    ]
    if both.empty:
        return 1.0, 1.0, DEFAULT_PEN_PI
    candidates = _candidates(both)
    us_pen = (both["us_goals"] - both["us_npg"]).to_numpy(dtype="float64")
    main = both["main_taker"].to_numpy(dtype=bool)
    pi = tuple(
        _shrunk_ratio(
            float(us_pen[group & (candidates > 0)].sum()),
            float(candidates[group].sum()),
            DEFAULT_PEN_PI[i],
            PI_PRIOR,
        )
        for i, group in enumerate((main, ~main))
    )
    attempts = (both["us_goals"] - both["us_npg"] + both["penalties_missed"]).to_numpy()
    fpl_npxg = np.clip(both["fpl_xg"].to_numpy() - FPL_PEN_XG * attempts, 0.0, None)
    k_goals = _shrunk_ratio(float(both["us_npxg"].sum()), float(fpl_npxg.sum()), 1.0, K_PRIOR)
    xa = both[both["us_xa"].notna() & both["fpl_xa"].notna()]
    k_assists = _shrunk_ratio(float(xa["us_xa"].sum()), float(xa["fpl_xa"].sum()), 1.0, K_PRIOR)
    return k_goals, k_assists, (pi[0], pi[1])


# --- rows: X and D per player-match ---------------------------------------------------------


def _team_fixtures(view: AsOfView, rows: pd.DataFrame, min_season: int | None) -> pd.DataFrame:
    """Per visible team-fixture: season, goals_for, team npxG (Understat, else the players'
    sum with full coverage), penalty attempts (Understat team, else the players' sum with
    full coverage), penalty goals (players' sum, full coverage), FPL assists and own goals
    of the club's players."""
    teams = view.table(
        "team_match",
        columns=["fixture_key", "team_key", "season", "goals_for", "us_xg", "us_npxg"],
    )
    teams = teams.assign(
        goals_for=teams["goals_for"].astype("float64"),
        us_xg=_floats(teams, "us_xg"),
        us_npxg=_floats(teams, "us_npxg"),
    )
    if min_season is not None:
        teams = teams[teams["season"] >= min_season].reset_index(drop=True)
    played = rows[rows["minutes"] > 0]
    played = played.assign(
        npxg_missing=played["npxg"].isna(), att_missing=played["pen_attempts"].isna()
    )
    sums = played.groupby(["fixture_key", "team_key"]).agg(
        npxg_sum=("npxg", "sum"),
        npxg_missing=("npxg_missing", "sum"),
        att_sum=("pen_attempts", "sum"),
        pen_sum=("pen_goals", "sum"),
        att_missing=("att_missing", "sum"),
    )
    sums = sums.assign(
        npxg_known=sums["npxg_missing"] == 0, att_known=sums["att_missing"] == 0
    ).drop(columns=["npxg_missing", "att_missing"])
    totals = rows.groupby(["fixture_key", "team_key"]).agg(
        assists=("assists", "sum"), own_goals=("own_goals", "sum")
    )
    teams = teams.merge(sums.reset_index(), on=["fixture_key", "team_key"], how="left")
    teams = teams.merge(totals.reset_index(), on=["fixture_key", "team_key"], how="left")
    npxg_known = teams["npxg_known"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    att_known = teams["att_known"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    team_npxg = np.where(
        ~np.isnan(teams["us_npxg"].to_numpy()),
        teams["us_npxg"].to_numpy(),
        np.where(npxg_known, teams["npxg_sum"].to_numpy(dtype="float64"), np.nan),
    )
    us_attempts = np.round((teams["us_xg"].to_numpy() - teams["us_npxg"].to_numpy()) / US_PEN_XG)
    attempts = np.where(
        ~np.isnan(us_attempts),
        us_attempts,
        np.where(att_known, teams["att_sum"].to_numpy(dtype="float64"), np.nan),
    )
    return teams.assign(
        team_npxg=team_npxg,
        team_attempts=attempts,
        team_pen_goals=np.where(att_known, teams["pen_sum"].to_numpy(dtype="float64"), np.nan),
        assists=teams["assists"].fillna(0.0).astype("float64"),
        own_goals=teams["own_goals"].fillna(0.0).astype("float64"),
    )[
        [
            "fixture_key",
            "team_key",
            "season",
            "goals_for",
            "team_npxg",
            "team_attempts",
            "team_pen_goals",
            "assists",
            "own_goals",
        ]
    ]


def _share_rows(
    rows: pd.DataFrame, teams: pd.DataFrame, ratios: tuple[float, float, float]
) -> pd.DataFrame:
    """Per played row: x_goals, d_goals, fb_goals (fallback: no xG of his), and the same for
    assists (see the module docstring)."""
    ratio_goals, ratio_assists, ratio_team = ratios
    played = rows[rows["minutes"] > 0]
    played = played.merge(
        teams[["fixture_key", "team_key", "goals_for", "team_npxg"]],
        on=["fixture_key", "team_key"],
        how="inner",
        validate="many_to_one",
    )
    on_pitch = played["minutes"].to_numpy(dtype="float64") / 90.0
    team_npxg = played["team_npxg"].to_numpy(dtype="float64")
    team_goals = played["goals_for"].to_numpy(dtype="float64") * ratio_team
    denominator = np.where(np.isnan(team_npxg), team_goals, team_npxg) * on_pitch
    npxg = played["npxg"].to_numpy(dtype="float64")
    xa = played["xa"].to_numpy(dtype="float64")
    goals = np.nan_to_num(played["goals_scored"].to_numpy(dtype="float64")) * ratio_goals
    assists = np.nan_to_num(played["assists"].to_numpy(dtype="float64")) * ratio_assists
    return played.assign(
        x_goals=np.where(np.isnan(npxg), goals, npxg),
        fb_goals=np.isnan(npxg),
        x_assists=np.where(np.isnan(xa), assists, xa),
        fb_assists=np.isnan(xa),
        d=denominator,
        on_pitch=on_pitch,
    )


def _rows(
    view: AsOfView, fit: SharesFit | None = None, min_season: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float, tuple[float, float]]]:
    """(player rows with npxG/xA columns, team-fixtures, (k_goals, k_assists, π)) of seasons
    >= `min_season` (all if None). With `fit` None the source constants are estimated here
    (the fit), else taken from it."""
    matches = _matches(view, min_season)
    matches["main_taker"] = _main_taker_flags(view, matches)
    if fit is None:
        k_goals, k_assists, pen_pi = _source_constants(matches)
    else:
        k_goals, k_assists, pen_pi = fit.k_goals, fit.k_assists, fit.pen_pi
    xg = match_xg(matches, k_goals, k_assists, pen_pi)
    rows = pd.concat([matches, xg], axis=1)
    return rows, _team_fixtures(view, rows, min_season), (k_goals, k_assists, pen_pi)


# --- the Marcel prior -----------------------------------------------------------------------


def _price_model(seasons: pd.DataFrame, x: str, d: str) -> tuple[float, float, float]:
    """(a, b, center): quasi-Poisson X ~ D·exp(a + b·(price − center)/10) by Newton steps
    from a = log ΣX/ΣD, b = 0, with a small ridge on b; `seasons` has price, x, d."""
    rows = seasons[(seasons[d] > 0) & seasons["price"].notna()]
    if rows.empty or rows[x].sum() <= 0:
        total = float(seasons[x].sum()) / max(float(seasons[d].sum()), RATE_EPS)
        return float(np.log(max(total, RATE_EPS))), 0.0, 0.0
    weights = rows[d].to_numpy(dtype="float64")
    price = rows["price"].to_numpy(dtype="float64")
    center = float(np.average(price, weights=weights))
    z = (price - center) / PRICE_SCALE
    target = rows[x].to_numpy(dtype="float64")
    theta = np.array([np.log(target.sum() / weights.sum()), 0.0])
    design = np.column_stack([np.ones_like(z), z])
    ridge = np.diag([0.0, PRICE_RIDGE])
    for _ in range(PRICE_ITERATIONS):
        mu = weights * np.exp(design @ theta)
        gradient = design.T @ (mu - target) + ridge @ theta
        hessian = (design * mu[:, None]).T @ design + ridge
        theta = theta - np.linalg.solve(hessian, gradient)
    return float(theta[0]), float(theta[1]), center


def _price_prior(table: tuple, element_type: np.ndarray, price: np.ndarray) -> np.ndarray:
    out = np.full(len(element_type), np.nan)
    for position, a, b, center in table:
        mask = element_type == position
        out[mask] = np.exp(a + b * (price[mask] - center) / PRICE_SCALE)
    return out


def _position_prior(table: tuple, element_type: np.ndarray) -> np.ndarray:
    lookup = dict(table)
    fallback = float(np.mean(list(lookup.values()))) if lookup else 0.0
    return np.array([lookup.get(int(e), fallback) if e == e else fallback for e in element_type])


def marcel_shares(
    rows: pd.DataFrame, pool: pd.DataFrame, fit: SharesFit, season: int
) -> pd.DataFrame:
    """Per pool player (`player_key, team_key, element_type, price`): the posterior goal and
    assist shares g, a (see the module docstring), the prior means used, `new` (no played
    minute in the window) and his recent penalty attempts `attempts` (weighted like the
    rows). `rows` are `_share_rows` frames (season, team_key, x_*, d, fb_*, on_pitch,
    pen_attempts, penalties_missed)."""
    params = fit.params
    weights = np.asarray(params.season_weights, dtype="float64")
    window = rows[rows["player_key"].isin(pool["player_key"])]
    age = season - window["season"].to_numpy(dtype="int64")
    window = window[(age >= 0) & (age < len(weights))]
    age = season - window["season"].to_numpy(dtype="int64")
    club = pool.set_index("player_key")["team_key"]
    moved = window["team_key"].to_numpy() != window["player_key"].map(club).to_numpy()
    base = weights[age] * np.where(moved, params.club_change_weight, 1.0)
    w_goals = base * np.where(window["fb_goals"], params.goals_weight, 1.0)
    w_assists = base * np.where(window["fb_assists"], params.goals_weight, 1.0)
    d = window["d"].to_numpy(dtype="float64")
    attempts = np.nan_to_num(window["pen_goals"].to_numpy(dtype="float64")) + np.nan_to_num(
        window["penalties_missed"].to_numpy(dtype="float64")
    )
    sums = (
        pd.DataFrame(
            {
                "player_key": window["player_key"].to_numpy(),
                "xg": w_goals * window["x_goals"].to_numpy(dtype="float64"),
                "dg": w_goals * d,
                "xa": w_assists * window["x_assists"].to_numpy(dtype="float64"),
                "da": w_assists * d,
                "minutes": window["on_pitch"].to_numpy(dtype="float64"),
                "attempts": base * attempts,
            }
        )
        .groupby("player_key")
        .sum()
    )
    out = pool[["player_key", "team_key", "element_type", "price"]].merge(
        sums, left_on="player_key", right_index=True, how="left"
    )
    out = out.fillna({c: 0.0 for c in ("xg", "dg", "xa", "da", "minutes", "attempts")})
    element_type = out["element_type"].to_numpy(dtype="float64")
    price = out["price"].to_numpy(dtype="float64")
    new = out["minutes"].to_numpy() <= 0
    use_price = np.full(len(out), True) if params.price_prior_all else new
    prior_g = np.where(
        use_price,
        _price_prior(fit.price_goal, element_type, price),
        _position_prior(fit.position_goal, element_type),
    )
    prior_a = np.where(
        use_price,
        _price_prior(fit.price_assist, element_type, price),
        _position_prior(fit.position_assist, element_type),
    )
    prior_g = np.where(np.isnan(prior_g), _position_prior(fit.position_goal, element_type), prior_g)
    prior_a = np.where(
        np.isnan(prior_a), _position_prior(fit.position_assist, element_type), prior_a
    )
    d0 = params.prior_minutes / 90.0 * fit.team_output
    return out.assign(
        g=(out["xg"] + prior_g * d0) / (out["dg"] + d0),
        a=(out["xa"] + prior_a * d0) / (out["da"] + d0),
        prior_g=prior_g,
        prior_a=prior_a,
        new=new,
    )


# --- the fit --------------------------------------------------------------------------------


def _coverage(rows: pd.DataFrame) -> tuple[tuple[int, float, float], ...]:
    played = rows[rows["minutes"] > 0]
    out = []
    for season, group in played.groupby("season", sort=True):
        out.append(
            (int(season), float(group["npxg"].notna().mean()), float(group["xa"].notna().mean()))
        )
    return tuple(out)


def _ratio(numerator: pd.Series, denominator: pd.Series, default: float) -> float:
    total = float(denominator.sum())
    return float(numerator.sum()) / total if total > 0 else default


def _taker_q(rows: pd.DataFrame, teams: pd.DataFrame, takers: pd.DataFrame) -> float:
    """Share of a club's penalty attempts taken by its main taker (as listed at kickoff) in
    the fixtures he played in full, shrunk toward DEFAULT_TAKER_Q."""
    known = teams[(teams["team_attempts"] > 0)][["fixture_key", "team_key", "team_attempts"]]
    if known.empty or takers.empty:
        return DEFAULT_TAKER_Q
    kickoffs = rows.drop_duplicates(["fixture_key", "team_key"])[
        ["fixture_key", "team_key", "kickoff_time"]
    ]
    known = known.merge(kickoffs, on=["fixture_key", "team_key"], how="inner")
    known = known.reset_index(drop=True)
    known["taker_key"] = _main_taker_at_kickoff(known, takers)
    own = rows[["fixture_key", "player_key", "minutes", "pen_attempts"]].rename(
        columns={"player_key": "taker_key"}
    )
    known = known.merge(own, on=["fixture_key", "taker_key"], how="inner")
    full = known[(known["minutes"] >= 90) & known["pen_attempts"].notna()]
    return _shrunk_ratio(
        float(full["pen_attempts"].sum()),
        float(full["team_attempts"].sum()),
        DEFAULT_TAKER_Q,
        TAKER_PRIOR_ATTEMPTS,
    )


def fit_shares(view: AsOfView, params: SharesParams | None = None) -> SharesFit:
    """League-level parameters from every row visible in `view` (see the module docstring)."""
    params = SharesParams() if params is None else params
    season, _ = view.gameweek_for_deadline()
    rows, teams, (k_goals, k_assists, pen_pi) = _rows(view)
    played = rows[rows["minutes"] > 0]
    with_xg = played[played["npxg"].notna()]
    with_xa = played[played["xa"].notna()]
    ratio_goals = _ratio(with_xg["npxg"], with_xg["goals_scored"], 1.0)
    ratio_assists = _ratio(with_xa["xa"], with_xa["assists"], 1.0)
    team_xg = teams[teams["team_npxg"].notna()]
    ratio_team = _ratio(team_xg["team_npxg"], team_xg["goals_for"], ratio_goals)
    share_rows = _share_rows(rows, teams, (ratio_goals, ratio_assists, ratio_team))

    recent = share_rows[share_rows["season"] > season - FIT_SEASONS]
    team_output = _ratio(recent["d"], recent["on_pitch"], 1.3)
    position_goal, position_assist, price_goal, price_assist = [], [], [], []
    for position, group in recent.groupby("element_type", sort=True):
        if position != position:
            continue
        wg = np.where(group["fb_goals"], params.goals_weight, 1.0)
        wa = np.where(group["fb_assists"], params.goals_weight, 1.0)
        mean_goal = _ratio(group["x_goals"] * wg, group["d"] * wg, 0.0)
        mean_assist = _ratio(group["x_assists"] * wa, group["d"] * wa, 0.0)
        position_goal.append((int(position), mean_goal))
        position_assist.append((int(position), mean_assist))
        first_price = (
            rows[rows["element_type"] == position]
            .dropna(subset=["price"])
            .drop_duplicates(["player_key", "season"])[["player_key", "season", "price"]]
        )
        seasons = (
            group.assign(
                xg=group["x_goals"] * wg,
                dg=group["d"] * wg,
                xa_=group["x_assists"] * wa,
                da=group["d"] * wa,
            )
            .groupby(["player_key", "season"])[["xg", "dg", "xa_", "da"]]
            .sum()
            .reset_index()
            .merge(first_price, on=["player_key", "season"], how="left")
        )
        price_goal.append((int(position), *_price_model(seasons, "xg", "dg")))
        price_assist.append((int(position), *_price_model(seasons, "xa_", "da")))

    window = teams[teams["season"] > season - FIT_SEASONS]
    known = window[window["team_attempts"].notna()]
    pen_rate = _shrunk_ratio(
        float(known["team_attempts"].sum()),
        float(len(known)),
        DEFAULT_PEN_RATE,
        LEAGUE_PRIOR_MATCHES,
    )
    scored = window[window["team_pen_goals"].notna() & window["team_attempts"].notna()]
    conversion = _shrunk_ratio(
        float(scored["team_pen_goals"].sum()),
        float(scored["team_attempts"].sum()),
        DEFAULT_CONVERSION,
        LEAGUE_PRIOR_ATTEMPTS,
    )
    team_pen = []
    for team_key, group in known.groupby("team_key", sort=True):
        goals = group[group["team_pen_goals"].notna()]
        team_pen.append(
            (
                int(team_key),
                _shrunk_ratio(
                    float(group["team_attempts"].sum()),
                    float(len(group)),
                    pen_rate,
                    params.pen_team_prior,
                ),
                _shrunk_ratio(
                    float(goals["team_pen_goals"].sum()),
                    float(goals["team_attempts"].sum()),
                    conversion,
                    params.conversion_prior,
                ),
            )
        )
    assist_fraction = _ratio(window["assists"], window["goals_for"], DEFAULT_ASSIST_FRACTION)
    own_goal_fraction = _ratio(window["own_goals"], window["goals_for"], DEFAULT_OWN_GOAL_FRACTION)
    taker_q = _taker_q(rows, teams, _main_takers(view))
    coverage = _coverage(rows)
    for item_season, xg_share, xa_share in coverage:
        log.info(
            "shares: season %d: npxG on %.0f%%, xA on %.0f%% of played rows (the rest: goals"
            " / assists fallback)",
            item_season,
            100 * xg_share,
            100 * xa_share,
        )
    return SharesFit(
        cutoff=view.deadline,
        params=params,
        k_goals=k_goals,
        k_assists=k_assists,
        pen_pi=pen_pi,
        ratio_goals=ratio_goals,
        ratio_assists=ratio_assists,
        ratio_team=ratio_team,
        team_output=team_output,
        position_goal=tuple(position_goal),
        position_assist=tuple(position_assist),
        price_goal=tuple(price_goal),
        price_assist=tuple(price_assist),
        pen_rate=pen_rate,
        conversion=conversion,
        team_pen=tuple(team_pen),
        assist_fraction=assist_fraction,
        own_goal_fraction=own_goal_fraction,
        taker_q=taker_q,
        n_rows=len(share_rows),
        coverage=coverage,
    )


# --- prediction -----------------------------------------------------------------------------


def taker_probabilities(
    groups: np.ndarray, rank: np.ndarray, q: np.ndarray, f: np.ndarray, g: np.ndarray
) -> np.ndarray:
    """Per row (a club's player in one fixture; `groups` identifies the club-fixture) the
    probability that he takes a penalty the club is awarded: along the ranking (`rank`
    ascending, NaN = not a candidate), P_i = f_i q_i Π_{j above} (1 − f_j q_j); the
    leftover Π (1 − f_j q_j) is shared by g_i f_i / Σ g f (by f / Σ f if Σ g f = 0)."""
    groups = np.asarray(groups)
    rank, q, f, g = (np.asarray(v, dtype="float64") for v in (rank, q, f, g))
    n = len(groups)
    codes, uniques = pd.factorize(groups)
    listed = ~np.isnan(rank)
    log_term = np.log(np.clip(np.where(listed, 1.0 - np.clip(f * q, 0.0, 1.0), 1.0), 1e-300, None))
    order = np.lexsort((np.where(listed, rank, np.inf), codes))
    cumulative = np.cumsum(log_term[order])
    sorted_codes = codes[order]
    first = np.searchsorted(sorted_codes, sorted_codes, side="left")
    before = np.empty(n)
    before[order] = cumulative - log_term[order] - np.concatenate([[0.0], cumulative])[first]
    leftover = np.exp(np.bincount(codes, weights=log_term, minlength=len(uniques)))[codes]
    direct = np.where(listed, f * q * np.exp(before), 0.0)
    weight = g * f
    sums = np.bincount(codes, weights=weight, minlength=len(uniques))[codes]
    on_pitch = np.bincount(codes, weights=f, minlength=len(uniques))[codes]
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = np.where(sums > 0, weight / sums, np.where(on_pitch > 0, f / on_pitch, 0.0))
    return direct + leftover * spread


def _listing(view: AsOfView, season: int, players: pd.DataFrame, fit: SharesFit) -> pd.DataFrame:
    """Per pool player: `rank` (NaN = not a candidate) and `q` in his club's taker list: the
    season's newest snapshot `penalties_order` where the club has one, else his recent
    attempts (see the module docstring)."""
    snaps = view.latest(
        "player_snapshot",
        by=["player_key"],
        columns=["snapshot_at", "season", "team_key", "penalties_order"],
    )
    snaps = snaps[snaps["season"] == season]
    if not snaps.empty:
        snaps = snaps[snaps["snapshot_at"] == snaps["snapshot_at"].max()]
    listed = snaps[snaps["penalties_order"].notna()][["player_key", "team_key", "penalties_order"]]
    out = players[["player_key", "team_key", "attempts"]].merge(
        listed, on=["player_key", "team_key"], how="left", validate="one_to_one"
    )
    order = out["penalties_order"].astype("Float64").to_numpy(dtype="float64", na_value=np.nan)
    from_snapshot = out["team_key"].isin(set(listed["team_key"])).to_numpy()
    attempts = out["attempts"].to_numpy(dtype="float64")
    # Snapshot clubs rank by order; the others by attempts (only players with attempts).
    key = np.where(from_snapshot, order, np.where(attempts > 0, -attempts, np.nan))
    ranked = pd.DataFrame(
        {"team_key": out["team_key"].to_numpy(), "key": key, "player_key": out["player_key"]}
    )
    ranked = ranked[ranked["key"].notna()].sort_values(
        ["team_key", "key", "player_key"], kind="mergesort"
    )
    ranked["rank"] = ranked.groupby("team_key").cumcount().astype("float64") + 1.0
    rank = out[["player_key"]].merge(ranked[["player_key", "rank"]], on="player_key", how="left")
    rank = rank["rank"].to_numpy(dtype="float64")
    q = np.where(from_snapshot, fit.taker_q, attempts / (attempts + fit.params.taker_prior))
    return pd.DataFrame(
        {
            "player_key": out["player_key"].to_numpy(),
            "rank": rank,
            "q": np.where(np.isnan(rank), 0.0, q),
        }
    )


def predict_shares(
    view: AsOfView, fit: SharesFit, minutes: pd.DataFrame, team: pd.DataFrame
) -> pd.DataFrame:
    """Per row of `minutes` (pool player × horizon fixture): `SHARES_COLUMNS` (see the module
    docstring), sorted by (player_key, horizon, fixture_key), RangeIndex."""
    season, _ = view.gameweek_for_deadline()
    rows, teams, _ = _rows(view, fit, season - len(fit.params.season_weights) + 1)
    share_rows = _share_rows(rows, teams, (fit.ratio_goals, fit.ratio_assists, fit.ratio_team))
    pool = player_pool(view)[["player_key", "team_key", "element_type", "price"]]
    players = marcel_shares(share_rows, pool, fit, season)
    players = players.merge(
        _listing(view, season, players, fit), on="player_key", how="left", validate="one_to_one"
    )

    frame = minutes[[*KEYS, "e_minutes"]].merge(
        players[["player_key", "g", "a", "rank", "q"]],
        on="player_key",
        how="left",
        validate="many_to_one",
    )
    sides = team[["fixture_key", "team_key", "lambda_for"]]
    frame = frame.merge(sides, on=["fixture_key", "team_key"], how="left", validate="many_to_one")
    missing = int(frame["lambda_for"].isna().sum())
    if missing:
        log.warning("shares: %d row(s) have no team λ (not in the team frame); set to 0", missing)
    penalties = {key: (rate, conversion) for key, rate, conversion in fit.team_pen}
    clubs = frame["team_key"].to_numpy(dtype="int64")
    team_pen_goals = np.array(
        [
            penalties.get(int(t), (fit.pen_rate, fit.conversion))[0]
            * penalties.get(int(t), (fit.pen_rate, fit.conversion))[1]
            for t in clubs
        ],
        dtype="float64",
    )
    lambda_for = frame["lambda_for"].fillna(0.0).to_numpy(dtype="float64")
    lambda_np = np.clip(lambda_for * (1.0 - fit.own_goal_fraction) - team_pen_goals, 0.0, None)

    f = np.clip(frame["e_minutes"].to_numpy(dtype="float64") / 90.0, 0.0, None)
    g = frame["g"].fillna(0.0).to_numpy(dtype="float64")
    a = frame["a"].fillna(0.0).to_numpy(dtype="float64")
    groups = frame["fixture_key"].to_numpy(dtype="int64") * 1_000_000 + clubs
    codes, uniques = pd.factorize(groups)
    sum_g = np.bincount(codes, weights=g * f, minlength=len(uniques))[codes]
    sum_a = np.bincount(codes, weights=a * f, minlength=len(uniques))[codes]
    with np.errstate(divide="ignore", invalid="ignore"):
        goal_share = np.where(sum_g > 0, g / sum_g, 0.0)
        assist_share = np.where(sum_a > 0, a / sum_a, 0.0)
    p_take = taker_probabilities(
        groups,
        frame["rank"].to_numpy(dtype="float64"),
        frame["q"].fillna(0.0).to_numpy(dtype="float64"),
        f,
        g,
    )
    e_np_goals = lambda_np * goal_share * f
    e_pen_goals = p_take * team_pen_goals
    with np.errstate(divide="ignore", invalid="ignore"):
        p_taker = np.where(f > 0, p_take / f, 0.0)
    out = frame[list(KEYS)].assign(
        e_np_goals=e_np_goals,
        e_pen_goals=e_pen_goals,
        e_goals=e_np_goals + e_pen_goals,
        e_assists=lambda_for * fit.assist_fraction * assist_share * f,
        goal_share=goal_share,
        assist_share=assist_share,
        p_taker=p_taker,
    )
    dtypes = {**{name: "int64" for name in KEYS}, **{name: "float64" for name in VALUES}}
    out = out[list(SHARES_COLUMNS)].astype(dtypes)
    return out.sort_values(list(SORT_BY), kind="mergesort").reset_index(drop=True)
