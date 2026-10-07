"""Candidate pruning (Phase 4 plan, Decisions *Pruning*; PLAN §7 *Practicalities*).

Which players the MILP may use, decided per position (element_type):

1. **Owned players are always kept** (they can be held, sold or benched).
2. **Top-N:** the `prune_n[et]` best by horizon xP (Σ_t decay**h_t · xp_t) and the
   `prune_n[et]` best by horizon xP per price. A position missing from `prune_n` keeps every
   player; `prune_n=None` skips this step.
3. **Dominance:** among the players left, a non-owned player j is dropped when at least
   `squad_select[et]` buyable players of his position dominate him: price ≤ j's and xP ≥
   j's in every horizon GW, strictly better somewhere or (identical) a smaller
   `player_key`. The relation is transitive, so every dropped player keeps at least
   `squad_select[et]` dominators (a dominated player with fewest dominators has all of
   his dominators kept). In a single GW a plan holding j can then swap in a dominator that
   isn't in the squad, at no loss of xP or bank; over several GWs, or when the club cap
   blocks every free dominator, it may not (not checked).

Both steps are heuristics. On synthetic tests dominance alone never changed the optimum;
top-N with the default N occasionally costs a little (a mid-ranked player who fits the
budget), within the default MIP gap; Task 3 tunes N on real data.

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
    price: np.ndarray,
    xp: np.ndarray,
    owned: np.ndarray,
    buyable: np.ndarray,
    decay: float,
    horizons: np.ndarray,
    rules: Rules,
    prune_n: Mapping[int, int] | None,
    dominated: bool,
) -> np.ndarray:
    """The kept `player_key`s, sorted. Arrays are aligned per player (`xp` is players ×
    horizon GWs); `owned`/`buyable` are boolean masks."""
    keys = np.asarray(keys, dtype="int64")
    if len(np.unique(keys)) != len(keys):
        raise ValueError("prune: duplicate player keys")
    element_type = np.asarray(element_type, dtype="int64")
    price = np.asarray(price, dtype="int64")
    owned = np.asarray(owned, dtype=bool)
    buyable = np.asarray(buyable, dtype=bool)
    xp = np.asarray(xp, dtype="float64").reshape(len(keys), -1)
    hxp = horizon_xp(xp, np.asarray(horizons), decay)

    keep = owned.copy()
    if prune_n is None:
        keep[:] = True
    else:
        value = hxp / np.maximum(price, 1)
        for et in np.unique(element_type).tolist():
            idx = np.flatnonzero((element_type == et) & ~owned & buyable)
            n = prune_n.get(int(et))
            if n is None:
                keep[idx] = True
                continue
            keep[_top(idx, hxp, keys, n)] = True
            keep[_top(idx, value, keys, n)] = True

    if dominated:
        for et in np.unique(element_type).tolist():
            idx = np.flatnonzero(keep & (element_type == et))
            need = int(rules.squad_select.get(int(et), 0))
            if len(idx) <= need:
                continue
            p, x, k = price[idx], xp[idx], keys[idx]
            # dom[d, j]: d dominates j.
            weakly = (p[:, None] <= p[None, :]) & (x[:, None, :] >= x[None, :, :]).all(axis=2)
            strictly = (p[:, None] < p[None, :]) | (x[:, None, :] > x[None, :, :]).any(axis=2)
            dom = weakly & (strictly | (k[:, None] < k[None, :]))
            dom &= buyable[idx][:, None]
            np.fill_diagonal(dom, False)
            drop = (dom.sum(axis=0) >= need) & ~owned[idx]
            keep[idx[drop]] = False
    return np.sort(keys[keep])
