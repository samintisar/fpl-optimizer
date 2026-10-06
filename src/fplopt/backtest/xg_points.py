"""xG-scored points per player-match (PLAN §5 Comparing policies; Phase 3 plan, Decisions).

A second, lower-variance backtest metric: minutes are realized, and the luck-heavy
components are replaced by their expectation given the chances created and conceded.

- Goals = player xG × goal points of the position; assists = player xA × assist points.
- Clean sheet (players with ≥ `long_play_minutes`) = CS points × P(N = 0) and goals conceded
  (any minutes > 0, mirroring `score_matches`, which uses FPL's on-pitch `goals_conceded`
  stat regardless of minutes) = GC points × E[floor(N / goals_conceded_per_point)], with
  N ~ Poisson(opponent team xG × minutes / 90).
- Everything else is realized, from `score_matches`: appearance, saves, penalties saved and
  missed, cards, own goals, bonus, defcon.

Sources:
- Player xG/xA come as a pair from one source: Understat (`us_xg`, `us_xa`) when both are
  non-null, else FPL (`fpl_xg`, `fpl_xa`) when both are non-null. Never xG from one source
  and xA from the other. Both are full xG (penalties included), matching the realized
  `penalties_missed` deduction being kept as is.
- Team xG (`team_xg_table`): per (fixture, side) the first non-null of Understat (`us_xg`),
  summed FPL (`fpl_xg`), football-data (`fd_xg`). A player's opponent xG is the row of
  (fixture_key, opponent_team_key).

Minutes are always FPL's (`minutes`); Understat's `us_minutes` can differ and is not used.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fplopt.backtest.rules import Rules
from fplopt.backtest.scoring import score_matches

UNDERSTAT, FPL, FOOTBALL_DATA = "understat", "fpl", "football-data"
# (xG column, xA column, source) in precedence order.
PLAYER_XG_SOURCES = (("us_xg", "us_xa", UNDERSTAT), ("fpl_xg", "fpl_xa", FPL))
# (team_match column, source) in precedence order.
TEAM_XG_SOURCES = (("us_xg", UNDERSTAT), ("fpl_xg", FPL), ("fd_xg", FOOTBALL_DATA))
XG_REQUIRED = ("fixture_key", "opponent_team_key", "us_xg", "us_xa", "fpl_xg", "fpl_xa")
# Realized components kept as is (everything but goals, assists, CS and goals conceded).
REALIZED = (
    "appearance",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "bonus",
    "defcon",
)
# Poisson sums run over N = 0..POISSON_CAP. The neglected tail is P(N > 40), below 1e-15 for
# any rate under 10 goals (one team's xG in a match has never come close).
POISSON_CAP = 40


def _floats(series: pd.Series) -> np.ndarray:
    return series.astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)


def _strings(values: np.ndarray, index: pd.Index | None = None) -> pd.Series:
    """A str Series (missing values stay missing) from an object array of str/None."""
    return pd.Series(values, index=index, dtype="str")


def poisson_pmf(rate: np.ndarray, cap: int = POISSON_CAP) -> np.ndarray:
    """P(N = n) for n = 0..cap, one row per rate (shape (len(rate), cap + 1)). Built by the
    recursion p(n) = p(n-1) × rate / n, so rate 0 gives p(0) = 1 exactly."""
    rate = np.asarray(rate, dtype=np.float64)
    steps = rate[:, None] / np.arange(1, cap + 1)[None, :]
    ones = np.ones((len(rate), 1))
    return np.exp(-rate)[:, None] * np.concatenate([ones, np.cumprod(steps, axis=1)], axis=1)


def poisson_expected_floor(rate: np.ndarray, k: int, cap: int = POISSON_CAP) -> np.ndarray:
    """E[floor(N / k)] for N ~ Poisson(rate), by explicit summation over N = 0..cap."""
    return poisson_pmf(rate, cap) @ (np.arange(cap + 1) // k)


def team_xg_table(team_match: pd.DataFrame) -> pd.DataFrame:
    """One row per team_match row: fixture_key, team_key, xg (Float64), source (str).

    `xg` is the first non-null of Understat, summed FPL, football-data xG; `source` names it.
    Both are null when no source has the match (2016/17-2018/19)."""
    missing = [
        c for c in ("fixture_key", "team_key", *dict(TEAM_XG_SOURCES)) if c not in team_match
    ]
    if missing:
        raise KeyError(f"team_xg_table: missing columns {missing}")
    xg = np.full(len(team_match), np.nan)
    source = np.full(len(team_match), None, dtype=object)
    for column, name in reversed(TEAM_XG_SOURCES):
        values = _floats(team_match[column])
        present = ~np.isnan(values)
        xg = np.where(present, values, xg)
        source = np.where(present, name, source)
    return pd.DataFrame(
        {
            "fixture_key": team_match["fixture_key"].to_numpy(),
            "team_key": team_match["team_key"].to_numpy(),
            "xg": pd.array(xg, dtype="Float64"),
            "source": _strings(source, team_match.index),
        }
    )


def _by_position(element_type: np.ndarray, values: Mapping[int, int]) -> np.ndarray:
    lookup = np.zeros(5, dtype=np.int64)
    for et, points in values.items():
        lookup[et] = points
    return lookup[element_type]


def xg_score_matches(matches: pd.DataFrame, team_xg: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """xG-scored points of each player-match row, on the input's index.

    `matches`: player_match rows with `element_type` (everything `score_matches` needs) plus
    `XG_REQUIRED`. `team_xg`: `team_xg_table` output, unique per (fixture_key, team_key).

    Output columns:
    - `xg_points` (Float64): realized components + the xG terms (module docstring). Rows with
      0 minutes need no xG and score their realized points. Null when a needed input is
      missing: player xG/xA (any minutes > 0), or opponent xG when a Poisson term is needed,
      i.e. minutes ≥ `long_play_minutes` and the position has CS points, or minutes > 0 and
      it has goals-conceded points. (A MID under 60 minutes or any FWD needs no team xG.)
    - `xg_source` (str): the player xG/xA source used ("understat" | "fpl"); null when not
      needed (0 minutes) or missing.
    - `opp_xg_source` (str): the opponent team xG source used ("understat" | "fpl" |
      "football-data"); null when not needed or missing.
    """
    missing = [c for c in XG_REQUIRED if c not in matches]
    if missing:
        raise KeyError(f"xg_score_matches: missing columns {missing}")
    realized = score_matches(matches, rules)  # validates the stat columns
    element_type = matches["element_type"].to_numpy(dtype=np.int64)
    minutes = matches["minutes"].to_numpy(dtype=np.int64)
    played = minutes > 0
    long_play = minutes >= rules.long_play_minutes

    # Player xG/xA: one source for both, Understat first.
    xg = np.full(len(matches), np.nan)
    xa = np.full(len(matches), np.nan)
    source = np.full(len(matches), None, dtype=object)
    for xg_column, xa_column, name in reversed(PLAYER_XG_SOURCES):
        xg_values, xa_values = _floats(matches[xg_column]), _floats(matches[xa_column])
        pair = ~np.isnan(xg_values) & ~np.isnan(xa_values)
        xg = np.where(pair, xg_values, xg)
        xa = np.where(pair, xa_values, xa)
        source = np.where(pair, name, source)
    has_player = ~np.isnan(xg)

    # Opponent team xG.
    teams = team_xg.set_index(["fixture_key", "team_key"])
    if not teams.index.is_unique:
        raise ValueError("xg_score_matches: team_xg has duplicate (fixture_key, team_key)")
    opponent = teams.reindex(
        pd.MultiIndex.from_arrays([matches["fixture_key"], matches["opponent_team_key"]])
    )
    opp_xg = _floats(opponent["xg"])
    opp_source = opponent["source"].to_numpy(dtype=object)
    has_opp = ~np.isnan(opp_xg)

    cs_points = _by_position(element_type, rules.clean_sheets)
    gc_points = _by_position(element_type, rules.goals_conceded)
    needs_cs = long_play & (cs_points != 0)
    needs_gc = played & (gc_points != 0)
    needs_opp = needs_cs | needs_gc

    rate = np.where(has_opp, opp_xg, 0.0) * minutes / 90
    cs_term = np.where(needs_cs, cs_points * np.exp(-rate), 0.0)
    gc_term = np.where(
        needs_gc, gc_points * poisson_expected_floor(rate, rules.goals_conceded_per_point), 0.0
    )
    goal_points = _by_position(element_type, rules.goals_scored)
    attack_term = np.where(played & has_player, xg * goal_points + xa * rules.assists, 0.0)
    total = realized[list(REALIZED)].sum(axis=1).to_numpy(dtype=np.float64)
    total = total + attack_term + cs_term + gc_term
    null = (played & ~has_player) | (needs_opp & ~has_opp)

    out = pd.DataFrame(index=matches.index)
    out["xg_points"] = pd.array(np.where(null, np.nan, total), dtype="Float64")
    out["xg_source"] = _strings(np.where(played & has_player, source, None), matches.index)
    out["opp_xg_source"] = _strings(np.where(needs_opp & has_opp, opp_source, None), matches.index)
    return out
