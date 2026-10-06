"""Start states for backtests (PLAN §5 *Starts from any state*; Phase 3 plan, Task 5).

Past real-manager squads aren't available, so seasons are replayed from generated squads
at a deadline, built from what is visible in the deadline view:

- `template_state`: the most-owned valid squad. Ownership = the newest player snapshot's
  `selected_by_percent` of the season, else each player's newest visible
  `player_gw_ownership.selected` of the season (GW t's value is visible from GW t+1). Players
  are added in descending ownership (ties by player_key) when their position has a slot,
  their club is under the cap, and the money left can still fill the remaining slots with
  the cheapest eligible players. Before 2021/22 no ownership is visible at GW1, so template
  starts begin at GW2 there.
- `random_state`: a seeded random valid squad. Slots are visited in random order and each
  is filled with a feasible player (same test) drawn with probability ∝ price, so squads
  spend realistically. `numpy.random.default_rng(seed)`; deterministic per (view, seed).

Both return a `SquadState` for the view's GW: purchase price = price, bank = budget − cost,
1 free transfer (0 at `gw_index == 1`, where transfers are unlimited anyway), no chips used.
Both raise `StartStateError` where `pool_coverage` shows a club with a fixture in the
horizon and no pool players (only 2020/21 GW1 on real data), when the pool can't form a
valid squad, and (template) when no ownership is visible.

Like the policies, this module reads data only through the view
(tests/test_features_architecture.py).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from fplopt.backtest.rules import Rules
from fplopt.backtest.state import Holding, SquadState
from fplopt.features.baseline import player_pool, pool_coverage
from fplopt.features.store import AsOfView

__all__ = (
    "StartStateError",
    "check_coverage",
    "ownership",
    "random_state",
    "target_gameweek",
    "template_state",
)


class StartStateError(ValueError):
    """No start state can be generated at this deadline."""


def target_gameweek(view: AsOfView) -> tuple[int, int, int]:
    """(season, gw, gw_index) of the GW whose deadline is the view's deadline."""
    season, gw = view.gameweek_for_deadline()
    gameweeks = view.table("gameweek", columns=["season", "gw", "gw_index"])
    row = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)]
    return season, gw, int(row["gw_index"].iloc[0])


def check_coverage(view: AsOfView) -> None:
    """StartStateError if a club with a fixture in the horizon has no pool players."""
    coverage = pool_coverage(view)
    gaps = coverage[(coverage["n_fixtures_horizon"] > 0) & (coverage["n_pool_players"] == 0)]
    if len(gaps):
        raise StartStateError(
            f"pool coverage gap at {view.deadline}: club(s) {gaps['team_key'].tolist()} "
            "have fixtures in the horizon but no pool players"
        )


def ownership(view: AsOfView) -> pd.Series | None:
    """player_key -> ownership visible at the deadline (float64; snapshot percent, else the
    newest visible `player_gw_ownership.selected` of the season), or None if none is
    visible for the season."""
    season, _, _ = target_gameweek(view)
    columns = ["snapshot_at", "season", "selected_by_percent"]
    snaps = view.latest("player_snapshot", by=["player_key"], columns=columns)
    snaps = snaps[(snaps["season"] == season) & snaps["selected_by_percent"].notna()]
    if len(snaps):
        snaps = snaps[snaps["snapshot_at"] == snaps["snapshot_at"].max()]
        return pd.Series(
            snaps["selected_by_percent"].to_numpy(dtype="float64"),
            index=snaps["player_key"].to_numpy(dtype="int64"),
        )
    rows = view.table("player_gw_ownership", columns=["player_key", "season", "gw", "selected"])
    rows = rows[rows["season"] == season]
    if rows.empty:
        return None
    gameweeks = view.table("gameweek", columns=["season", "gw", "gw_index"])
    gameweeks = gameweeks[gameweeks["season"] == season]
    index = dict(zip(gameweeks["gw"], gameweeks["gw_index"], strict=True))
    rows = rows.assign(gw_index=rows["gw"].map(index))
    rows = rows.sort_values(["player_key", "gw_index"], kind="mergesort")
    rows = rows.drop_duplicates("player_key", keep="last")
    return pd.Series(
        rows["selected"].to_numpy(dtype="float64"),
        index=rows["player_key"].to_numpy(dtype="int64"),
    )


class _Builder:
    """A squad under construction from the pool (arrays sorted by player_key)."""

    def __init__(self, pool: pd.DataFrame, rules: Rules) -> None:
        self.rules = rules
        self.keys = pool["player_key"].to_numpy(dtype="int64")
        self.et = pool["element_type"].to_numpy(dtype="int64")
        self.team = pool["team_key"].to_numpy(dtype="int64")
        self.price = pool["price"].to_numpy(dtype="int64")
        self.chosen = np.zeros(len(self.keys), dtype=bool)
        self.slots = Counter({et: n for et, n in rules.squad_select.items() if n})
        self.clubs: Counter[int] = Counter()
        self.spent = 0
        # Per position: pool positions sorted by price (then key) for the cheapest fill.
        self.by_price = {
            et: np.flatnonzero(self.et == et)[
                np.lexsort((self.keys[self.et == et], self.price[self.et == et]))
            ]
            for et in self.slots
        }

    @property
    def full(self) -> bool:
        return sum(self.slots.values()) == 0

    def _fill_cost(self, slots: Counter[int], blocked: np.ndarray) -> float:
        """Cheapest cost of filling `slots` with players not `blocked` (inf if impossible)."""
        total = 0
        for et, n in slots.items():
            if n <= 0:
                continue
            idx = self.by_price[et]
            idx = idx[~blocked[idx]][:n]
            if len(idx) < n:
                return float("inf")
            total += int(self.price[idx].sum())
        return total

    def feasible(self, i: int) -> bool:
        """Player i fits: his position has a slot, his club is under the cap, and the
        remaining budget can fill the remaining slots with the cheapest eligible players."""
        et, team = int(self.et[i]), int(self.team[i])
        if self.chosen[i] or self.slots[et] <= 0 or self.clubs[team] >= self.rules.team_limit:
            return False
        left = self.rules.budget - self.spent - int(self.price[i])
        if left < 0:
            return False
        slots = self.slots.copy()
        slots[et] -= 1
        blocked = self.chosen.copy()
        blocked[i] = True
        # Clubs at the cap once i is in: none of their players can fill a slot.
        full = [t for t, n in self.clubs.items() if n >= self.rules.team_limit]
        if self.clubs[team] + 1 >= self.rules.team_limit:
            full.append(team)
        if full:
            blocked |= np.isin(self.team, full)
        return self._fill_cost(slots, blocked) <= left

    def add(self, i: int) -> None:
        self.chosen[i] = True
        self.slots[int(self.et[i])] -= 1
        self.clubs[int(self.team[i])] += 1
        self.spent += int(self.price[i])

    def state(self, view: AsOfView) -> SquadState:
        season, _, gw_index = target_gameweek(view)
        idx = np.flatnonzero(self.chosen)
        holdings = tuple(
            Holding(
                int(self.keys[i]),
                int(self.et[i]),
                int(self.team[i]),
                int(self.price[i]),
                int(self.price[i]),
            )
            for i in idx
        )
        return SquadState(
            season=season,
            gw_index=gw_index,
            holdings=holdings,
            bank=self.rules.budget - self.spent,
            free_transfers=1 if gw_index > 1 else 0,
        )


def _pool(view: AsOfView) -> pd.DataFrame:
    check_coverage(view)
    pool = player_pool(view)
    return pool.sort_values("player_key", kind="mergesort").reset_index(drop=True)


def _no_squad(view: AsOfView, builder: _Builder) -> StartStateError:
    return StartStateError(
        f"the pool at {view.deadline} cannot form a valid squad "
        f"(open slots {dict(+builder.slots)}, spent {builder.spent})"
    )


def template_state(view: AsOfView, rules: Rules) -> SquadState:
    """The most-owned valid squad at the view's deadline (see the module docstring)."""
    pool = _pool(view)
    owned = ownership(view)
    if owned is None:
        raise StartStateError(f"no ownership visible at {view.deadline}: no template squad")
    builder = _Builder(pool, rules)
    share = pd.Series(builder.keys).map(owned).fillna(0.0).to_numpy(dtype="float64")
    for i in np.lexsort((builder.keys, -share)):
        if builder.full:
            break
        if builder.feasible(int(i)):
            builder.add(int(i))
    if not builder.full:
        raise _no_squad(view, builder)
    return builder.state(view)


def random_state(view: AsOfView, rules: Rules, seed: int) -> SquadState:
    """A seeded random valid squad at the view's deadline, players drawn ∝ price (see the
    module docstring)."""
    pool = _pool(view)
    builder = _Builder(pool, rules)
    rng = np.random.default_rng(seed)
    slots = [et for et, n in sorted(builder.slots.items()) for _ in range(n)]
    for k in rng.permutation(len(slots)):
        et = slots[int(k)]
        candidates = [int(i) for i in np.flatnonzero(builder.et == et) if builder.feasible(int(i))]
        if not candidates:
            raise _no_squad(view, builder)
        weights = builder.price[candidates].astype("float64")
        weights = np.where(weights > 0, weights, 0.0)
        if weights.sum() <= 0:
            weights = np.ones(len(candidates))
        builder.add(candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))])
    return builder.state(view)
