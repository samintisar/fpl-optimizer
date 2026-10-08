"""Decision policies for the backtester (PLAN §5 *Decision policy is pluggable*; Phase 3
plan, Task 5).

A policy is an immutable parameter holder with a `name`, the `xp_model` it reads (a
`fplopt.models.MODELS` key) and `decide(ctx) -> Decision`. The simulator builds the
`DecisionContext` at each deadline: the deadline view, the (refreshed) squad state, the
season's rules, the pool (`player_pool(view)`) and the xP frame of `xp_model` for that view.
Policies keep no state between calls and read data only through the context (the static
rules of tests/test_features_architecture.py apply to this module).

- `best_lineup`: the optimal XI for target-GW xP under the formation limits, bench GK first
  then outfield by xP, captain/vice = the two best starters.
- `RollPolicy`: no transfers, `best_lineup`.
- `GreedyPolicy`: the best single same-position transfer by discounted horizon xP if its
  gain beats a threshold and a free transfer is available (never a hit); repeated up to
  `max_transfers` while FTs last, and at `gw_index == 1` (unlimited free transfers) while the
  gain beats the threshold, up to a whole squad. No chips.
- `OptimizerPolicy`: the MILP planner (`fplopt.optimize`, PLAN §7) over the xP frame's
  horizon; executes plan #1's first-GW decision (transfers, lineup, captain, chip if
  `chips`). One plan, no roll plan (top-3 is for the manager, not the backtest).

Ties are broken by `player_key` everywhere, so decisions are deterministic.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from fplopt.backtest.gw_score import Lineup
from fplopt.backtest.rules import Rules
from fplopt.backtest.state import (
    GOALKEEPER,
    Decision,
    SquadState,
    Transfer,
    refresh,
    selling_price,
)
from fplopt.features.store import AsOfView
from fplopt.models import MODELS
from fplopt.optimize import OptimizerParams, PlanInput, optimize

__all__ = (
    "DecisionContext",
    "GreedyPolicy",
    "OptimizerPolicy",
    "Policy",
    "RollPolicy",
    "best_lineup",
    "greedy_transfers",
    "horizon_xp",
    "target_xp",
)


@dataclass(frozen=True)
class DecisionContext:
    """Everything a policy may use for one GW's decision. `pool` is `player_pool(view)`
    (`player_key, element_type, team_key, price, ...`), `xp` the policy's xP frame
    (`player_key, gw, gw_index, horizon, xp`) for the same view."""

    view: AsOfView
    state: SquadState
    rules: Rules
    pool: pd.DataFrame
    xp: pd.DataFrame


class Policy(Protocol):
    """A decision policy: immutable, no state between calls."""

    xp_model: str  # a fplopt.models.MODELS key

    @property
    def name(self) -> str: ...

    def decide(self, ctx: DecisionContext) -> Decision: ...


# --- xP helpers ---------------------------------------------------------------------------


def target_xp(xp: pd.DataFrame) -> dict[int, float]:
    """player_key -> xP of the target GW (horizon 0)."""
    target = xp[xp["horizon"] == 0]
    return dict(
        zip(target["player_key"].tolist(), target["xp"].astype(float).tolist(), strict=True)
    )


def horizon_xp(xp: pd.DataFrame, horizon: int, decay: float) -> pd.Series:
    """Per player_key Σ decay^h · xp_h over the horizons h < `horizon` (float64)."""
    rows = xp[xp["horizon"] < horizon]
    weights = np.power(float(decay), rows["horizon"].to_numpy(dtype="float64"))
    weighted = rows["xp"].to_numpy(dtype="float64") * weights
    return pd.Series(weighted, index=rows["player_key"].to_numpy()).groupby(level=0).sum()


def _value(xp: Mapping[int, float], key: int) -> float:
    value = float(xp.get(key, 0.0))
    return 0.0 if value != value else value  # NaN counts as 0


# --- lineup -------------------------------------------------------------------------------


def best_lineup(squad: pd.DataFrame, xp_target: Mapping[int, float], rules: Rules) -> Lineup:
    """The XI with the most target-GW xP under the formation limits, for a squad frame with
    `player_key, element_type` (players missing from `xp_target` count as 0).

    Each position's `play_min` best players start (so the best GK), then the best of the
    rest fill the remaining slots while their position is under `play_max`; this is optimal
    because the only constraints are per-position lower and upper bounds on a fixed-size
    selection. Bench: the other GK first, then the outfield players by xP descending.
    Captain = the best starter by xP, vice = the second. Ties by player_key ascending.
    Starters are listed by position, then xP descending."""
    players = [
        (int(k), int(et))
        for k, et in zip(squad["player_key"].tolist(), squad["element_type"].tolist(), strict=True)
    ]

    def rank(player: tuple[int, int]) -> tuple[float, int]:
        return (-_value(xp_target, player[0]), player[0])

    order = sorted(players, key=rank)
    starters: list[tuple[int, int]] = []
    counts: Counter[int] = Counter()
    for et, n in sorted(rules.play_min.items()):
        picked = [p for p in order if p[1] == et][:n]
        starters += picked
        counts[et] += len(picked)
    for player in order:
        if len(starters) >= rules.squad_play:
            break
        if player not in starters and counts[player[1]] < rules.play_max[player[1]]:
            starters.append(player)
            counts[player[1]] += 1
    if len(starters) != rules.squad_play:
        raise ValueError(f"squad cannot field {rules.squad_play} starters: {dict(counts)}")
    chosen = set(starters)
    bench = [p for p in order if p not in chosen and p[1] == GOALKEEPER]
    bench += [p for p in order if p not in chosen and p[1] != GOALKEEPER]
    captain, vice = sorted(starters, key=rank)[:2]
    starters = sorted(starters, key=lambda p: (p[1], *rank(p)))
    return Lineup(
        starters=tuple(k for k, _ in starters),
        bench=tuple(k for k, _ in bench),
        captain=captain[0],
        vice=vice[0],
    )


def _squad_frame(state: SquadState) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_key": [h.player_key for h in state.holdings],
            "element_type": [h.element_type for h in state.holdings],
        },
        dtype="int64",
    )


# --- transfers ----------------------------------------------------------------------------


def greedy_transfers(
    state: SquadState,
    pool: pd.DataFrame,
    hxp: pd.Series | Mapping[int, float],
    rules: Rules,
    threshold: float,
    limit: int,
) -> tuple[Transfer, ...]:
    """Up to `limit` transfers, each the best (out i, in j) by gain hxp_j − hxp_i among: i
    held at the start (not bought in this decision), j in the pool and neither held at the
    start nor already bought, same position, price_j ≤ bank + selling_price(i), and j's
    club stays within `team_limit` after the swap (state.py's club rule). Stops when the best
    gain is not above `threshold`. Ties: smallest in_key, then smallest out_key. Players
    without hxp count as 0. Every prefix of the result passes `apply_decision`."""
    hxp = pd.Series(hxp, dtype="float64") if not isinstance(hxp, pd.Series) else hxp
    pool_keys = pool["player_key"].to_numpy(dtype="int64")
    pool_et = pool["element_type"].to_numpy(dtype="int64")
    pool_team = pool["team_key"].to_numpy(dtype="int64")
    pool_price = pool["price"].to_numpy(dtype="int64")
    pool_hxp = hxp.reindex(pool_keys).fillna(0.0).to_numpy(dtype="float64")
    original = set(state.player_keys)
    buyable = ~np.isin(pool_keys, list(original))

    # The current squad: (key, element_type, team, selling price, hxp, sellable).
    squad = [
        (
            h.player_key,
            h.element_type,
            h.team_key,
            selling_price(h, rules.sell_on_fee),
            float(hxp.get(h.player_key, 0.0)),
        )
        for h in state.holdings
    ]
    sellable = [True] * len(squad)
    bank = state.bank
    transfers: list[Transfer] = []
    while len(transfers) < limit:
        rows = [i for i, ok in enumerate(sellable) if ok]
        if not rows or not buyable.any():
            break
        out_key = np.array([squad[i][0] for i in rows], dtype="int64")
        out_et = np.array([squad[i][1] for i in rows], dtype="int64")
        out_team = np.array([squad[i][2] for i in rows], dtype="int64")
        out_sell = np.array([squad[i][3] for i in rows], dtype="int64")
        out_hxp = np.array([squad[i][4] for i in rows], dtype="float64")
        clubs = Counter(p[2] for p in squad)
        club_count = np.array([clubs.get(int(t), 0) for t in pool_team], dtype="int64")
        # After selling i and buying j, j's club has count_j - [team_i == team_j] + 1.
        after = club_count[None, :] - (out_team[:, None] == pool_team[None, :]) + 1
        ok = (
            buyable[None, :]
            & (out_et[:, None] == pool_et[None, :])
            & (pool_price[None, :] <= bank + out_sell[:, None])
            & (after <= rules.team_limit)
        )
        if not ok.any():
            break
        gain = pool_hxp[None, :] - out_hxp[:, None]
        oi, pj = np.nonzero(ok)
        g = gain[oi, pj]
        # Max gain; ties by in_key, then out_key (lexsort: last key is primary).
        best = np.lexsort((out_key[oi], pool_keys[pj], -g))[0]
        if not g[best] > threshold:
            break
        i, j = rows[int(oi[best])], int(pj[best])
        transfers.append(Transfer(int(out_key[oi[best]]), int(pool_keys[j])))
        bank += int(out_sell[oi[best]]) - int(pool_price[j])
        squad[i] = (int(pool_keys[j]), int(pool_et[j]), int(pool_team[j]), 0, float(pool_hxp[j]))
        sellable[i] = False
        buyable[j] = False
    return tuple(transfers)


def _after(state: SquadState, transfers: tuple[Transfer, ...], pool: pd.DataFrame) -> pd.DataFrame:
    """player_key, element_type of the squad after `transfers`."""
    squad = _squad_frame(state)
    if not transfers:
        return squad
    outs = {t.out_key for t in transfers}
    ins = [t.in_key for t in transfers]
    bought = pool.loc[pool["player_key"].isin(ins), ["player_key", "element_type"]]
    kept = squad[~squad["player_key"].isin(outs)]
    return pd.concat([kept, bought.astype("int64")], ignore_index=True)


# --- policies -----------------------------------------------------------------------------


def _check_model(xp_model: str) -> None:
    if xp_model not in MODELS:
        raise ValueError(f"unknown xp_model {xp_model!r} (MODELS: {sorted(MODELS)})")


@dataclass(frozen=True)
class RollPolicy:
    """Never transfers; plays `best_lineup` on the target GW's xP."""

    xp_model: str = "rolling"

    def __post_init__(self) -> None:
        _check_model(self.xp_model)

    @property
    def name(self) -> str:
        return f"roll({self.xp_model})"

    def decide(self, ctx: DecisionContext) -> Decision:
        lineup = best_lineup(_squad_frame(ctx.state), target_xp(ctx.xp), ctx.rules)
        return Decision(transfers=(), lineup=lineup, chip=None)


@dataclass(frozen=True)
class GreedyPolicy:
    """Best same-position transfer by horizon xP (Σ_{h<horizon} decay^h · xp_h) if the gain
    is above `threshold` and a free transfer is left; up to `max_transfers` per GW (all free,
    never a hit). At `gw_index == 1` transfers are unlimited and free: it keeps going while
    the gain is above the threshold, up to `squad_size` transfers. Then `best_lineup` on the
    target GW's xP. No chips."""

    xp_model: str = "rolling"
    threshold: float = 1.0
    horizon: int = 6
    decay: float = 0.85
    max_transfers: int = 1

    def __post_init__(self) -> None:
        _check_model(self.xp_model)
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.max_transfers < 0:
            raise ValueError(f"max_transfers must be >= 0, got {self.max_transfers}")

    @property
    def name(self) -> str:
        extra = ""
        if self.horizon != 6:
            extra += f",h={self.horizon}"
        if self.decay != 0.85:
            extra += f",d={float(self.decay)!r}"
        if self.max_transfers != 1:
            extra += f",n={self.max_transfers}"
        return f"greedy({self.xp_model},t={float(self.threshold)!r}{extra})"

    def decide(self, ctx: DecisionContext) -> Decision:
        state = refresh(ctx.state, ctx.pool)
        if state.gw_index == 1:
            limit = ctx.rules.squad_size
        else:
            limit = min(self.max_transfers, max(state.free_transfers, 0))
        hxp = horizon_xp(ctx.xp, self.horizon, self.decay)
        transfers = greedy_transfers(state, ctx.pool, hxp, ctx.rules, self.threshold, limit)
        squad = _after(state, transfers, ctx.pool)
        lineup = best_lineup(squad, target_xp(ctx.xp), ctx.rules)
        return Decision(transfers=transfers, lineup=lineup, chip=None)


def _optimizer_extras(params: OptimizerParams) -> list[str]:
    """`OptimizerPolicy.name` parts for the parameters other than the hit safeguards that
    differ from `OptimizerParams()`'s (so distinct policies get distinct names)."""
    default = OptimizerParams()
    parts = []
    if params.horizon != default.horizon:
        parts.append(f"h={params.horizon}")
    if params.decay != default.decay:
        parts.append(f"d={float(params.decay)!r}")
    if params.itb_value != default.itb_value:
        parts.append(f"itb={float(params.itb_value)!r}")
    if params.ft_value != default.ft_value:
        parts.append("ftv=" + "/".join(f"{k}:{v!r}" for k, v in params.ft_value.items()))
    if params.bench_weights != default.bench_weights:
        parts.append("bw=" + "/".join(repr(w) for w in params.bench_weights))
    if params.chip_value != default.chip_value:
        parts.append("cv=" + "/".join(f"{k}:{v!r}" for k, v in params.chip_value.items()))
    if params.prune_n != default.prune_n:
        prune = "all" if params.prune_n is None else "/".join(map(str, params.prune_n.values()))
        parts.append(f"prune={prune}")
    if params.prune_dominated != default.prune_dominated:
        parts.append(f"dom={int(params.prune_dominated)}")
    if params.mip_gap != default.mip_gap:
        parts.append(f"gap={float(params.mip_gap)!r}")
    if params.threads != default.threads:
        parts.append(f"threads={params.threads}")
    if params.time_limit != default.time_limit:
        parts.append(f"tl={float(params.time_limit)!r}")
    return parts


@dataclass(frozen=True)
class OptimizerPolicy:
    """The MILP planner (`fplopt.optimize`): `PlanInput.from_context` on the context's
    (refreshed) state, pool and xP frame, then `optimize(top_k=1, roll=False,
    chips=chips)`; the decision is plan #1's first GW. `chips=False` plans without chips
    (the no-chip scenario only). Deterministic for a given context (HiGHS with fixed
    threads and gap-based stopping; the time limit is a safety net only).

    `name`: `optimizer(<xp>,mh=<max_hits|inf>,m=<hit_margin>[,...][,chips])`; the hit
    safeguards are always listed (their defaults are being tuned), other parameters only
    when they differ from `OptimizerParams()`."""

    xp_model: str = "ep_next"
    params: OptimizerParams = field(default_factory=OptimizerParams)
    chips: bool = False

    def __post_init__(self) -> None:
        _check_model(self.xp_model)
        if not isinstance(self.params, OptimizerParams):
            raise TypeError(f"params must be OptimizerParams, got {type(self.params).__name__}")

    @property
    def name(self) -> str:
        params = self.params
        max_hits = "inf" if params.max_hits is None else str(params.max_hits)
        parts = [self.xp_model, f"mh={max_hits}", f"m={float(params.hit_margin)!r}"]
        parts += _optimizer_extras(params)
        if self.chips:
            parts.append("chips")
        return f"optimizer({','.join(parts)})"

    def decide(self, ctx: DecisionContext) -> Decision:
        problem = PlanInput.from_context(ctx.state, ctx.pool, ctx.xp, ctx.rules, self.params)
        plans = optimize(problem, self.params, top_k=1, chips=self.chips, roll=False)
        return plans.decision()
