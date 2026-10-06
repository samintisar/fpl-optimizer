"""Baseline xP models (Phase 3 plan, Task 4; PLAN §5 *Baselines*).

Each model takes exactly one argument, an `AsOfView`, reads only through it (feature
builders and view methods; tests/test_features_architecture.py) and returns an xP frame
(Phase 3 Decisions): one row per `player_pool` player and GW of the season from the target
GW to `HORIZON` GWs later by `gw_index` (from `upcoming_fixtures`; fewer near the season's
end), columns `player_key, gw, gw_index, horizon (0 = target), xp`, dtypes int64 x 4 and
float64, sorted by (player_key, horizon), RangeIndex. `xp` is 0 when unknown and in a GW
where the player's club blanks; in a double it counts both fixtures.

- `xp_rolling`: per-fixture rate = mean `total_points` over the player's last `ROLLING_N`
  visible `player_match` rows (any season, by kickoff; 0-minute rows count), 0 without rows;
  xP = rate x his club's fixtures in the GW.
- `xp_ep_next`: FPL's `ep_next` (from the newest player snapshot). Per-fixture rate =
  `ep_next / n_fixtures` of the target GW if his club plays in it, else the snapshot's `form`
  (FPL's mean points per match over the last 30 days); nulls count as 0. xP = rate x
  fixtures, so the target GW's xP is `ep_next` (0 in a blank: our as-of schedule decides,
  like every other GW). Without a snapshot of the season before the deadline (before
  2020/21 GW32) every xP is 0 and a warning is logged.

The models keep no state and cache nothing: callers (the simulator) cache per deadline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fplopt.features.baseline import HORIZON, ep_next, player_pool, upcoming_fixtures
from fplopt.features.store import AsOfView

log = logging.getLogger(__name__)

ROLLING_N = 5
XP_DTYPES = (
    ("player_key", "int64"),
    ("gw", "int64"),
    ("gw_index", "int64"),
    ("horizon", "int64"),
    ("xp", "float64"),
)
__all__ = ("HORIZON", "ROLLING_N", "XP_DTYPES", "xp_ep_next", "xp_rolling")


def _grid(pool: pd.DataFrame, view: AsOfView) -> pd.DataFrame:
    """Pool player x horizon GW with the club's number of fixtures in that GW (0 = blank)."""
    fixtures = upcoming_fixtures(view)
    fixtures = fixtures[["team_key", "gw", "gw_index", "horizon", "n_fixtures"]].drop_duplicates()
    gameweeks = fixtures[["gw", "gw_index", "horizon"]].drop_duplicates()
    grid = pool[["player_key", "team_key"]].merge(gameweeks, how="cross")
    grid = grid.merge(
        fixtures, on=["team_key", "gw", "gw_index", "horizon"], how="left", validate="many_to_one"
    )
    grid["n_fixtures"] = grid["n_fixtures"].fillna(0).astype("int64")
    return grid


def _finish(grid: pd.DataFrame, xp: pd.Series | np.ndarray) -> pd.DataFrame:
    out = grid.assign(xp=np.nan_to_num(np.asarray(xp, dtype="float64"), nan=0.0))
    out = out[[name for name, _ in XP_DTYPES]].astype(dict(XP_DTYPES))
    return out.sort_values(["player_key", "horizon"], kind="mergesort").reset_index(drop=True)


def xp_rolling(view: AsOfView) -> pd.DataFrame:
    """Rolling-average xP: mean points of the player's last ROLLING_N matches x fixtures."""
    pool = player_pool(view)
    grid = _grid(pool, view)
    columns = ["player_key", "kickoff_time", "fixture_key", "total_points"]
    matches = view.table("player_match", columns=columns)
    matches = matches[matches["player_key"].isin(pool["player_key"])]
    matches = matches.sort_values(["player_key", "kickoff_time", "fixture_key"], kind="mergesort")
    recent = matches[matches.groupby("player_key").cumcount(ascending=False) < ROLLING_N]
    rate = recent.groupby("player_key")["total_points"].mean().astype("float64")
    xp = grid["player_key"].map(rate).fillna(0.0) * grid["n_fixtures"]
    return _finish(grid, xp)


def xp_ep_next(view: AsOfView) -> pd.DataFrame:
    """FPL `ep_next` xP: ep_next in the target GW, extended per fixture (form after a blank)."""
    pool = player_pool(view)
    grid = _grid(pool, view)
    if not (pool["source"] == "snapshot").any():
        log.warning(
            "xp_ep_next at %s: no player snapshot of the season before the deadline "
            "(ep_next exists from 2020/21 GW32); every xP is 0",
            view.deadline,
        )
        return _finish(grid, np.zeros(len(grid)))
    snapshot = ep_next(view).set_index("player_key")
    target = grid[grid["horizon"] == 0].set_index("player_key")["n_fixtures"]
    n_target = grid["player_key"].map(target).fillna(0).astype("int64")
    expected = grid["player_key"].map(snapshot["ep_next"]).astype("float64").fillna(0.0)
    form = grid["player_key"].map(snapshot["form"]).astype("float64").fillna(0.0)
    rate = np.where(n_target > 0, expected / n_target.where(n_target > 0, 1), form)
    return _finish(grid, rate * grid["n_fixtures"])
