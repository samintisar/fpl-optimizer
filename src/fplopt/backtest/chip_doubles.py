"""Double GWs known at a deadline (chip policy 2, V5: double-aware terminal values,
docs/superpowers/specs/2026-10-30-chip-policy-2-design.md).

A GW (`gw_index`) is a known double when some club has ≥ 2 league fixtures in it in the
as-of schedule (`AsOfView.schedule`), within `lead` GWs after the decision GW.
`fixture_snapshot` only covers seasons archived live (2026/27 on), so every develop
season's as-of schedule is the final one: it includes rescheduling announced after the
deadline. The lead caps how far ahead that hindsight reaches.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from fplopt.backtest.simulator import FAR_FUTURE, season_schedule

__all__ = ("DOUBLE_LEAD", "known_doubles", "known_doubles_by_gw")

DOUBLE_LEAD = 10  # GWs after the decision GW (pre-registered in the spec)


def known_doubles(view: Any, season: int, gw_index: int, lead: int = DOUBLE_LEAD) -> frozenset[int]:
    """The `gw_index`s in (gw_index, gw_index + lead] where a club has ≥ 2 fixtures in
    `view.schedule(season)` (unscheduled fixtures and placeholder sides, team_key ≤ 0,
    do not count)."""
    schedule = view.schedule(season)
    schedule = schedule[schedule["gw_index"].notna()]
    sides = pd.concat(
        [
            schedule[["gw_index", side]].rename(columns={side: "team_key"})
            for side in ("home_team_key", "away_team_key")
        ]
    )
    sides = sides[sides["team_key"] > 0]
    counts = sides.groupby(["gw_index", "team_key"]).size()
    doubles = {int(g) for (g, _), n in counts.items() if n >= 2}
    return frozenset(g for g in doubles if gw_index < g <= gw_index + lead)


def known_doubles_by_gw(
    store: Any, seasons: Iterable[int], lead: int = DOUBLE_LEAD
) -> dict[tuple[int, int], frozenset[int]]:
    """(season, gw_index) → `known_doubles` at that GW's deadline, for the estimator. Each
    deadline is the one `season_schedule` finds for the GW (visible at its own deadline), and
    the schedule is read through `store.as_of(deadline)` only."""
    out = {}
    gameweeks = store.as_of(FAR_FUTURE).table("gameweek", columns=["season", "gw_index"])
    for season in seasons:
        indices = gameweeks[gameweeks["season"] == season]["gw_index"].dropna().unique()
        for gw_index in sorted(int(i) for i in indices):
            deadline = season_schedule(store, season, gw_index)["deadline_time"].iloc[0]
            view = store.as_of(deadline)
            out[(int(season), gw_index)] = known_doubles(view, season, gw_index, lead)
    return out
