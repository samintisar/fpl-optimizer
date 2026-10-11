"""Candidate pruning (Phase 4 plan, Decisions *Pruning*; PLAN §7 *Practicalities*).

Which players the MILP may use, decided per position (element_type):

1. **Owned players are always kept** (they can be held, sold or benched).
1a. **Minutes floor** (only with `minutes`, PLAN §7 *Pruning*): a non-owned player with
   fewer than `min_minutes` expected minutes over the horizon is dropped first, so top-N
   and dominance see only the players above the floor.
2. **Top-N:** the `prune_n[et]` best by horizon xP (Σ_t decay**h_t · xp_t) and the
   `prune_n[et]` best by horizon xP per price. A position missing from `prune_n` keeps every
   player; `prune_n=None` skips this step.
3. **Dominance:** among the players left, a non-owned player j is dropped when the buyable
   players of his position that dominate him (price ≤ j's and xP ≥ j's in every horizon
   GW, strictly better somewhere or, identical, a smaller `player_key`) come from at least
   `squad_select[et] + squad_size // team_limit` distinct clubs (7 for goalkeepers, 10
   for defenders and midfielders, 8 for forwards under the 2/5/5/3 rules). The relation
   is transitive, so every dropped player keeps dominators from that many clubs among
   the kept players (by induction: a dropped dominator's own dominators are j's too). In
   a single GW a plan holding j can then swap in a kept dominator outside the squad from
   a club below the cap: at most `squad_select[et] − 1` clubs hold the position's other
   squad players and at most `squad_size // team_limit` clubs are full. Counting
   dominators instead of clubs (as first implemented) lost up to 6 points on real data
   (Task 3 benchmark): at GW1 many identical ep_next projections, so the dominators of a
   mid-priced player were often all of one club, which the cap blocks. Over several GWs
   the swap may still be blocked (not checked).

Both steps are heuristics; Task 3's benchmark (`optimize.bench`) measures their loss.

Ties are broken by `player_key`, so the result is deterministic. Pure numpy, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fplopt.backtest.rules import Rules


def horizon_xp(xp: np.ndarray, horizons: np.ndarray, decay: float) -> np.ndarray:
    """Σ_t decay**horizons[t] · xp[:, t] per row."""
    weights = np.power(float(decay), horizons.astype("float64"))
    return xp.astype("float64") @ weights if xp.shape[1] else np.zeros(xp.shape[0])


def _top(idx: np.ndarray, score: np.ndarray, keys: np.ndarray, n: int) -> np.ndarray:
    """The n entries of idx with the highest score (ties: smaller key first)."""
    order = np.lexsort((keys[idx], -score[idx]))
    return idx[order[:n]]


def prune(
    *,
    keys: np.ndarray,
    element_type: np.ndarray,
    team_key: np.ndarray,
    price: np.ndarray,
    xp: np.ndarray,
    owned: np.ndarray,
    buyable: np.ndarray,
    decay: float,
    horizons: np.ndarray,
    rules: Rules,
    prune_n: Mapping[int, int] | None,
    dominated: bool,
    minutes: np.ndarray | None = None,
    min_minutes: float = 0.0,
) -> np.ndarray:
    """The kept `player_key`s, sorted. Arrays are aligned per player (`xp` is players ×
    horizon GWs, `team_key` the club); `owned`/`buyable` are boolean masks. `minutes`
    (expected minutes over the horizon, per player) with `min_minutes` > 0 applies the
    minutes floor; None (a model without minutes) or a floor of 0 skips it."""
    keys = np.asarray(keys, dtype="int64")
    if len(np.unique(keys)) != len(keys):
        raise ValueError("prune: duplicate player keys")
    element_type = np.asarray(element_type, dtype="int64")
    team_key = np.asarray(team_key, dtype="int64")
    price = np.asarray(price, dtype="int64")
    owned = np.asarray(owned, dtype=bool)
    buyable = np.asarray(buyable, dtype=bool)
    xp = np.asarray(xp, dtype="float64").reshape(len(keys), -1)
    hxp = horizon_xp(xp, np.asarray(horizons), decay)
    eligible = np.ones(len(keys), dtype=bool)
    if minutes is not None and min_minutes > 0:
        minutes = np.asarray(minutes, dtype="float64")
        eligible = owned | (np.nan_to_num(minutes, nan=0.0) >= min_minutes)

    keep = owned.copy()
    if prune_n is None:
        keep[:] = eligible
    else:
        value = hxp / np.maximum(price, 1)
        for et in np.unique(element_type).tolist():
            idx = np.flatnonzero((element_type == et) & ~owned & buyable & eligible)
            n = prune_n.get(int(et))
            if n is None:
                keep[idx] = True
                continue
            keep[_top(idx, hxp, keys, n)] = True
            keep[_top(idx, value, keys, n)] = True

    if dominated:
        full_clubs = rules.squad_size // max(rules.team_limit, 1)
        for et in np.unique(element_type).tolist():
            idx = np.flatnonzero(keep & (element_type == et))
            need = int(rules.squad_select.get(int(et), 0)) + full_clubs
            if len(idx) <= need:
                continue
            p, x, k = price[idx], xp[idx], keys[idx]
            # dom[d, j]: d dominates j.
            weakly = (p[:, None] <= p[None, :]) & (x[:, None, :] >= x[None, :, :]).all(axis=2)
            strictly = (p[:, None] < p[None, :]) | (x[:, None, :] > x[None, :, :]).any(axis=2)
            dom = weakly & (strictly | (k[:, None] < k[None, :]))
            dom &= buyable[idx][:, None]
            np.fill_diagonal(dom, False)
            # Distinct clubs among each player's dominators.
            _, club = np.unique(team_key[idx], return_inverse=True)
            one_hot = np.zeros((len(idx), club.max() + 1), dtype=bool)
            one_hot[np.arange(len(idx)), club] = True
            clubs = (dom.T.astype("int64") @ one_hot.astype("int64") > 0).sum(axis=1)
            drop = (clubs >= need) & ~owned[idx]
            keep[idx[drop]] = False
    return np.sort(keys[keep])
