"""The planner's input at one deadline (Phase 4 plan, Task 1; PLAN §7).

`PlanInput.from_context(state, pool, xp, rules, params)` turns the backtester's decision
context into plain, immutable data the MILP is built from:

- the squad is first refreshed from the pool (`state.refresh`: clubs and prices as of the
  deadline); an owned player missing from the pool (left the game) keeps his last price and
  club, can be sold and can't be bought (as `state.apply_decision`);
- **prices are fixed over the horizon** at the deadline price (no price prediction, PLAN
  §1). An owned player's first sale earns his selling price (`selling_price` with the rules'
  `sell_on_fee`); any other sale (a player bought, or bought back, within the plan) earns
  his buy price;
- the horizon GWs are those of the xP frame with `horizon < params.horizon`, so near the
  season's end the horizon is shorter. Missing xP counts as 0 (NaN too);
- candidates are pruned (`prune.prune`); owned players are always candidates.

Pure: no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fplopt.backtest.rules import Rules
from fplopt.backtest.state import POOL_COLUMNS, SquadState, refresh, selling_price
from fplopt.optimize.params import OptimizerParams
from fplopt.optimize.prune import prune

XP_COLUMNS = ("player_key", "gw", "gw_index", "horizon", "xp")


@dataclass(frozen=True)
class Player:
    """A candidate. `price` is the buy price (pool price; a departed owned player's last
    price), `sell_price` what his first sale earns (his selling price if owned, else
    `price`). `xp[t]` is his xP in horizon GW t."""

    player_key: int
    element_type: int
    team_key: int
    price: int
    sell_price: int
    owned: bool
    buyable: bool
    xp: tuple[float, ...]


@dataclass(frozen=True)
class HorizonGw:
    """One GW of the horizon: `horizon` is its offset from the target GW by `gw_index`
    (0 = the GW being decided), which is also its decay exponent."""

    gw: int
    gw_index: int
    horizon: int


@dataclass(frozen=True)
class PlanInput:
    """Everything the MILP needs, immutable. `players` are the candidates (owned first
    included), sorted by `player_key`; `n_pool` is the pool size before pruning. `bank` is in
    tenths of £m and `free_transfers` the FTs for the first horizon GW (ignored when
    `gw1`, the unlimited pre-season GW). `state` is the refreshed squad state (chips used,
    for the chip scenarios)."""

    state: SquadState
    rules: Rules
    players: tuple[Player, ...]
    gws: tuple[HorizonGw, ...]
    bank: int
    free_transfers: int
    gw1: bool
    n_pool: int

    @property
    def owned_keys(self) -> tuple[int, ...]:
        return tuple(p.player_key for p in self.players if p.owned)

    @property
    def player_by_key(self) -> dict[int, Player]:
        return {p.player_key: p for p in self.players}

    @classmethod
    def from_context(
        cls,
        state: SquadState,
        pool: pd.DataFrame,
        xp: pd.DataFrame,
        rules: Rules,
        params: OptimizerParams,
    ) -> PlanInput:
        """Build the input from a (pre- or post-refresh) state, the deadline's pool
        (`player_key, element_type, team_key, price`), an xP frame (`player_key, gw,
        gw_index, horizon, xp`) and the rules."""
        if state.freehit_backup is not None:
            raise ValueError("state is a Free Hit gameweek state; call next_state first")
        missing = [c for c in POOL_COLUMNS if c not in pool.columns]
        missing += [c for c in XP_COLUMNS if c not in xp.columns]
        if missing:
            raise ValueError(f"pool/xp frames are missing columns {missing}")
        state = refresh(state, pool)
        gws = _horizon_gws(xp, state, params)
        horizons = [g.horizon for g in gws]

        pool = pool[list(POOL_COLUMNS)].astype("int64")
        pool = pool.sort_values("player_key", kind="mergesort").reset_index(drop=True)
        owned = {h.player_key: h for h in state.holdings}
        rows = xp[xp["horizon"].isin(horizons)]
        matrix = (
            rows.assign(xp=rows["xp"].astype("float64").fillna(0.0))
            .pivot_table(index="player_key", columns="horizon", values="xp", aggfunc="sum")
            .reindex(columns=horizons)
            .fillna(0.0)
        )

        # Every pool player plus departed owned players, as one frame for pruning.
        in_pool = set(pool["player_key"].tolist())
        departed = [h for h in state.holdings if h.player_key not in in_pool]
        frame = pd.concat(
            [
                pool,
                pd.DataFrame(
                    [(h.player_key, h.element_type, h.team_key, h.price) for h in departed],
                    columns=list(POOL_COLUMNS),
                    dtype="int64",
                ),
            ],
            ignore_index=True,
        ).sort_values("player_key", kind="mergesort")
        keys = frame["player_key"].to_numpy(dtype="int64")
        xp_rows = matrix.reindex(keys).fillna(0.0).to_numpy(dtype="float64")
        if xp_rows.shape[1] != len(horizons):  # empty matrix: no xP rows at all
            xp_rows = np.zeros((len(keys), len(horizons)))
        buyable = np.isin(keys, pool["player_key"].to_numpy())
        kept = prune(
            keys=keys,
            element_type=frame["element_type"].to_numpy(dtype="int64"),
            team_key=frame["team_key"].to_numpy(dtype="int64"),
            price=frame["price"].to_numpy(dtype="int64"),
            xp=xp_rows,
            owned=np.isin(keys, list(owned)),
            buyable=buyable,
            decay=params.decay,
            horizons=np.asarray(horizons, dtype="int64"),
            rules=rules,
            prune_n=params.prune_n,
            dominated=params.prune_dominated,
        )
        kept_set = set(kept.tolist())

        players = []
        for row, (key, et, team, price) in enumerate(
            zip(*(frame[c].tolist() for c in POOL_COLUMNS), strict=True)
        ):
            if key not in kept_set:
                continue
            holding = owned.get(key)
            players.append(
                Player(
                    player_key=int(key),
                    element_type=int(et),
                    team_key=int(team),
                    price=int(price),
                    sell_price=selling_price(holding, rules.sell_on_fee)
                    if holding is not None
                    else int(price),
                    owned=holding is not None,
                    buyable=bool(buyable[row]),
                    xp=tuple(float(v) for v in xp_rows[row]),
                )
            )
        return cls(
            state=state,
            rules=rules,
            players=tuple(players),
            gws=gws,
            bank=state.bank,
            free_transfers=state.free_transfers,
            gw1=state.gw_index == 1,
            n_pool=len(pool),
        )


def _horizon_gws(xp: pd.DataFrame, state: SquadState, params: OptimizerParams) -> tuple:
    """The xP frame's GWs with horizon < params.horizon, by horizon. The first must be the
    state's GW (horizon 0, gw_index == state.gw_index)."""
    gws = (
        xp.loc[xp["horizon"] < params.horizon, ["gw", "gw_index", "horizon"]]
        .drop_duplicates()
        .sort_values("horizon", kind="mergesort")
    )
    if gws["horizon"].duplicated().any():
        raise ValueError("xp frame maps one horizon to several GWs")
    out = tuple(
        HorizonGw(int(g), int(i), int(h))
        for g, i, h in zip(gws["gw"], gws["gw_index"], gws["horizon"], strict=True)
    )
    if not out or out[0].horizon != 0 or out[0].gw_index != state.gw_index:
        raise ValueError(
            f"xp frame must start at the state's GW (gw_index {state.gw_index}, horizon 0); "
            f"got {out[:1]}"
        )
    return out
