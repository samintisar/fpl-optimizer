"""Hand-made pools, states and xP frames for the optimizer tests, plus an exact brute-force
scorer of a squad's best lineup for one GW."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from fplopt.backtest.rules import load_rules
from fplopt.backtest.state import Holding, SquadState

RULES = load_rules("2026-27")
SEASON = 2023
BASE_PRICE = {1: 45, 2: 50, 3: 70, 4: 75}


def key(club: int, et: int, i: int) -> int:
    """club * 100 + element_type * 10 + i (as tests/test_backtest_state.py)."""
    return club * 100 + et * 10 + i


def make_pool(rows: Iterable[tuple[int, int, int, int]]) -> pd.DataFrame:
    """Pool frame from (player_key, element_type, team_key, price) rows."""
    pool = pd.DataFrame(list(rows), columns=["player_key", "element_type", "team_key", "price"])
    return pool.astype("int64").sort_values("player_key").reset_index(drop=True)


def grid_pool(
    clubs: Iterable[int], per_pos: Mapping[int, int], seed: int | None = None
) -> pd.DataFrame:
    """Every club gets per_pos[et] players per position; prices BASE_PRICE[et] (+ a seeded
    spread of 0..30 when seed is given)."""
    rng = np.random.default_rng(seed)
    rows = []
    for club in clubs:
        for et, n in sorted(per_pos.items()):
            for i in range(n):
                spread = int(rng.integers(0, 31)) if seed is not None else 0
                rows.append((key(club, et, i), et, club, BASE_PRICE[et] + spread))
    return make_pool(rows)


def make_state(
    pool: pd.DataFrame,
    keys: Sequence[int],
    *,
    gw_index: int = 5,
    ft: int = 1,
    bank: int | None = None,
    purchase: Mapping[int, int] | None = None,
    extra: Sequence[Holding] = (),
) -> SquadState:
    """A state holding `keys` (bought at pool price unless `purchase` says otherwise) plus
    `extra` holdings (e.g. a player who left the game). Bank: what's left of the budget."""
    rows = pool.set_index("player_key")
    holdings = [
        Holding(
            k,
            int(rows.at[k, "element_type"]),
            int(rows.at[k, "team_key"]),
            int((purchase or {}).get(k, rows.at[k, "price"])),
            int(rows.at[k, "price"]),
        )
        for k in keys
    ] + list(extra)
    if bank is None:
        bank = RULES.budget - sum(h.purchase_price for h in holdings)
    return SquadState(SEASON, gw_index, tuple(holdings), bank, ft)


def xp_frame(
    values: Mapping[int, Sequence[float]], *, gw_index: int = 5, gw: int | None = None
) -> pd.DataFrame:
    """xP frame from player_key -> xP per horizon GW (horizon 0 = gw_index)."""
    gw = gw_index if gw is None else gw
    rows = [
        (k, gw + h, gw_index + h, h, float(x))
        for k, xs in sorted(values.items())
        for h, x in enumerate(xs)
    ]
    frame = pd.DataFrame(rows, columns=["player_key", "gw", "gw_index", "horizon", "xp"])
    return frame.astype(
        {"player_key": "int64", "gw": "int64", "gw_index": "int64", "horizon": "int64"}
    )


def random_xp(
    pool: pd.DataFrame, n_gws: int, seed: int, *, gw_index: int = 5, scale: float = 1.0
) -> pd.DataFrame:
    """Seeded xP per player and GW, loosely increasing with price."""
    rng = np.random.default_rng(seed)
    values = {}
    for k, price in zip(pool["player_key"], pool["price"], strict=True):
        base = (price - 35) / 12 * scale
        values[int(k)] = np.round(rng.gamma(2.0, base / 2.0 + 0.2, n_gws), 2).tolist()
    return xp_frame(values, gw_index=gw_index)


def correlated_xp(pool: pd.DataFrame, n_gws: int, seed: int, *, gw_index: int = 5) -> pd.DataFrame:
    """Seeded, realistic-ish xP: a per-player rate rising with price, times a per-club
    fixture factor per GW (so xP is correlated across GWs, as real projections are)."""
    rng = np.random.default_rng(seed)
    clubs = sorted(set(pool["team_key"].tolist()))
    factor = {c: rng.uniform(0.7, 1.3, n_gws) for c in clubs}
    values = {}
    for k, price, club in zip(pool["player_key"], pool["price"], pool["team_key"], strict=True):
        rate = max(0.0, (price - 40) / 10 + rng.normal(0.0, 1.0))
        values[int(k)] = np.round(rate * factor[int(club)], 2).tolist()
    return xp_frame(values, gw_index=gw_index)


def first_valid_squad(pool: pd.DataFrame, budget: int = 1000) -> list[int]:
    """The cheapest 2/5/5/3 squad with at most team_limit per club (greedy by price)."""
    chosen: list[int] = []
    clubs: Counter[int] = Counter()
    for et, n in sorted(RULES.squad_select.items()):
        rows = pool[pool["element_type"] == et].sort_values(["price", "player_key"])
        picked = 0
        for k, club in zip(rows["player_key"], rows["team_key"], strict=True):
            if picked < n and clubs[club] < RULES.team_limit:
                chosen.append(int(k))
                clubs[club] += 1
                picked += 1
    assert len(chosen) == RULES.squad_size
    assert sum(pool.set_index("player_key").loc[chosen, "price"]) <= budget
    return chosen


def best_lineup_value(
    squad: Sequence[int],
    xp: Mapping[int, float],
    et: Mapping[int, int],
    bench_weights: Sequence[float],
) -> float:
    """Exact max over formations of Σ XI xP + captain xP + Σ_k w_k · bench_k xP (bench GK in
    slot 0, outfield bench by xP descending in slots 1..3)."""
    by_pos = {p: sorted((xp[k] for k in squad if et[k] == p), reverse=True) for p in (1, 2, 3, 4)}
    best = -np.inf
    for d, m, f in itertools.product(
        *(range(RULES.play_min[p], RULES.play_max[p] + 1) for p in (2, 3, 4))
    ):
        if 1 + d + m + f != RULES.squad_play:
            continue
        starters = by_pos[1][:1] + by_pos[2][:d] + by_pos[3][:m] + by_pos[4][:f]
        bench_out = sorted(by_pos[2][d:] + by_pos[3][m:] + by_pos[4][f:], reverse=True)
        value = sum(starters) + max(starters) + bench_weights[0] * by_pos[1][1]
        value += sum(w * x for w, x in zip(bench_weights[1:], bench_out, strict=True))
        best = max(best, value)
    return float(best)
