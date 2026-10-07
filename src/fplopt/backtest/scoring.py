"""Points per player-match, by component, under a `Rules` (PLAN §5 Scoring the backtest).

The backtest re-scores every match from its stat line rather than trusting the archived
`total_points`, so that all develop/validate seasons are scored under one rule set (the
current rules without defcon). The scorer is checked against FPL's own totals with
`legacy_rules()` on 2016/17-2024/25 and with the native rules on 2026/27
(tests/test_backtest_scoring.py, `-m realdata`).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fplopt.backtest.rules import Rules

REQUIRED = (
    "element_type",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
)
COMPONENTS = (
    "appearance",
    "goals",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "bonus",
    "defcon",
)
# What counts towards the defcon threshold when FPL's own `defensive_contribution` count is
# missing (seasons before 2025/26, or a source without it): DEF CBIT, MID/FWD CBIRT.
DEFCON_STATS = {
    2: ("clearances_blocks_interceptions", "tackles"),
    3: ("clearances_blocks_interceptions", "tackles", "recoveries"),
    4: ("clearances_blocks_interceptions", "tackles", "recoveries"),
}


def _by_position(element_type: pd.Series, values: Mapping[int, int]) -> np.ndarray:
    return element_type.map(dict(values)).to_numpy(dtype=np.int64)


def _defcon_count(matches: pd.DataFrame) -> pd.Series:
    """Defensive-contribution count per row (nullable Int64): FPL's count when present,
    else the sum of the position's stats; null if neither is available."""
    count = pd.Series(pd.NA, index=matches.index, dtype="Int64")
    for et, stats in DEFCON_STATS.items():
        rows = matches["element_type"] == et
        if all(stat in matches for stat in stats):
            # A null component makes the sum null (no partial counts).
            count[rows] = sum(matches.loc[rows, stat].astype("Int64") for stat in stats)
    if "defensive_contribution" in matches:
        fpl = matches["defensive_contribution"].astype("Int64")
        count = fpl.where(fpl.notna(), count)
    return count


def score_matches(matches: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """Points of each player-match row, by component, plus their sum `points`.

    Input columns: `REQUIRED` (non-null; element_type 1-4), optionally
    `defensive_contribution`, `clearances_blocks_interceptions`, `tackles`, `recoveries`.
    Output: int64 columns `COMPONENTS` + `points`, on the input's index.

    - `clean_sheets` is used as is: FPL already applies the 60-minute rule to the stat.
    - Saves and goals conceded score per `saves_per_point` / `goals_conceded_per_point`
      (floor), each per match.
    - Defcon only when `rules.defcon_enabled`, never for GKs: the award when the count
      reaches the position's threshold, the count being FPL's `defensive_contribution` when
      non-null, else CBI + tackles (DEF) or CBI + tackles + recoveries (MID/FWD). A missing
      count scores 0.
    """
    missing = [column for column in REQUIRED if column not in matches]
    if missing:
        raise KeyError(f"score_matches: missing columns {missing}")
    nulls = [column for column in REQUIRED if matches[column].isna().any()]
    if nulls:
        raise ValueError(f"score_matches: null values in {nulls}")
    element_type = matches["element_type"].astype(np.int64)
    bad = sorted(set(element_type) - {1, 2, 3, 4})
    if bad:
        raise ValueError(f"score_matches: element_type must be 1-4, got {bad}")

    def stat(column: str) -> np.ndarray:
        return matches[column].to_numpy(dtype=np.int64)

    minutes = stat("minutes")
    out = pd.DataFrame(index=matches.index)
    out["appearance"] = np.where(
        minutes >= rules.long_play_minutes,
        rules.long_play,
        np.where(minutes > 0, rules.short_play, 0),
    )
    out["goals"] = stat("goals_scored") * _by_position(element_type, rules.goals_scored)
    out["assists"] = stat("assists") * rules.assists
    out["clean_sheets"] = stat("clean_sheets") * _by_position(element_type, rules.clean_sheets)
    out["goals_conceded"] = (stat("goals_conceded") // rules.goals_conceded_per_point) * (
        _by_position(element_type, rules.goals_conceded)
    )
    out["saves"] = (stat("saves") // rules.saves_per_point) * rules.saves
    out["penalties_saved"] = stat("penalties_saved") * rules.penalties_saved
    out["penalties_missed"] = stat("penalties_missed") * rules.penalties_missed
    out["yellow_cards"] = stat("yellow_cards") * rules.yellow_cards
    out["red_cards"] = stat("red_cards") * rules.red_cards
    out["own_goals"] = stat("own_goals") * rules.own_goals
    out["bonus"] = stat("bonus") * rules.bonus
    out["defcon"] = 0
    if rules.defcon_enabled:
        count = _defcon_count(matches)
        threshold = element_type.map(
            {et: t for et, t in rules.defcon_threshold.items() if t is not None}
        ).astype("Int64")
        reached = (count >= threshold).fillna(False).to_numpy(dtype=bool)
        award = _by_position(element_type, rules.defensive_contribution)
        out["defcon"] = np.where(reached, award, 0)
    out = out.astype(np.int64)
    out["points"] = out[list(COMPONENTS)].sum(axis=1).astype(np.int64)
    return out
