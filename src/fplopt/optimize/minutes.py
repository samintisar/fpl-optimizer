"""The minutes model in the planner (Phase 5 plan, Task 8; PLAN §7 *Bench*, *Pruning*):
per-GW bench weights from P(starter doesn't play), and the expected-minutes floor.

**Bench weights** (`minutes_bench_weights`, used when `OptimizerParams.bench_from_minutes`
and the xP frame carries `p_play`, i.e. `v1`; the baseline models have no minutes, so they
keep the fixed `params.bench_weights` exactly). Per horizon GW t:

- **projected XI**: the incumbent squad's (the refreshed state's holdings) best lineup by
  GW t's xP (`projected_xi`: the same rule as `backtest.policies.best_lineup`, each
  position's minimum by xP, then the best of the rest within the maximums; ties by
  player_key). It is computed once for the incumbent squad, not for each plan's squad, so
  the weights are constants and the MILP stays linear. A plan that changes the squad
  (transfers, a Wildcard or Free Hit, GW1 rebuilds) is weighted with the incumbent's
  starters' risks, an approximation;
- **q_i = P(starter i doesn't play)** in GW t: 1 − `p_play_gw` (P(plays in at least one
  of the GW's fixtures); `v1` computes it as 1 − Π_f (1 − p_play_f), fixtures independent),
  else 1 − `p_play`. In a double GW `p_play` is NaN and `p_play_gw` is the product rule,
  which ignores that absences (injuries, bans) are correlated across a GW's two fixtures:
  P(neither) is then underestimated (the comonotone bound is min_f q_f). A starter with
  no row in the frame (left the game) has q = 1. If a projected starter's q is unknown
  (NaN: a double GW in a frame without `p_play_gw`), the GW falls back to the fixed
  weights (conservative: the fixed weights are the Phase 4 defaults);
- **GK slot** (bench slot 0): P(the projected starting GK doesn't play);
- **outfield slot k** (1..3), with FPL's skip rule (a bench player who doesn't play passes
  his turn to the next slot): P(M ≥ 1 + S_{k−1}), M = the number of the 10 projected
  outfield starters who don't play (Poisson-binomial over their q_i) and S_{k−1} = the
  number of the projected bench outfielders in slots 1..k−1 who do play (the incumbent's
  bench by xP, best first; Poisson-binomial over 1 − q_j), all independent
  (`skip_aware_tail`). Without skips this is P(M ≥ k); develop 2017/18–2022/23 (898 squad
  GWs, `dev/bench_weights_eval.py`): Brier vs "the slot was needed" 0.155 / 0.124 / 0.130
  for slots 1–3 with skips, 0.155 / 0.135 / 0.247 without. The formation limits on
  autosubs are ignored (a missing defender in a 3-defender XI can only be replaced by a
  bench defender), which slightly overstates the weights;
- **Bench Boost** keeps every bench weight at 1 (`model._Model.bench_weights`).

The tail P(N ≥ k) is non-increasing in k, so the slots' order still follows from the
weights (the MILP puts the best outfield bench player in slot 1).

**Minutes floor** (`horizon_minutes`, used by `problem.PlanInput.from_context` when
`OptimizerParams.min_minutes` > 0 and the frame carries `e_minutes`): a non-owned player
whose expected minutes summed (undecayed) over the horizon GWs are below the floor is not a
candidate (`prune.prune`, before top-N and dominance pruning). The floor is for a full
`params.horizon`-GW horizon, pro rata on a shorter one (the season's end). Owned players
are always candidates.

Pure: no I/O.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from fplopt.backtest.rules import Rules
from fplopt.backtest.state import GOALKEEPER

P_PLAY = "p_play"  # P(plays) per player-GW; NaN in a double GW (v1)
P_PLAY_GW = "p_play_gw"  # P(plays in at least one of the GW's fixtures) (v1)
E_MINUTES = "e_minutes"  # expected minutes per player-GW, summed over the GW's fixtures


def poisson_binomial_tail(q: Iterable[float], k_max: int) -> tuple[float, ...]:
    """(P(N ≥ 1), ..., P(N ≥ k_max)) for N = Σ_i Bernoulli(q_i), independent (q clipped
    to [0, 1])."""
    dist = np.ones(1)  # dist[n] = P(N = n) over the q seen so far
    for value in q:
        p = min(max(float(value), 0.0), 1.0)
        nxt = np.append(dist * (1.0 - p), 0.0)
        nxt[1:] += dist * p
        dist = nxt
    tail = np.cumsum(dist[::-1])[::-1]  # tail[k] = P(N >= k)
    return tuple(
        float(min(max(tail[k], 0.0), 1.0)) if k < len(tail) else 0.0 for k in range(1, k_max + 1)
    )


def _xp(xp: Mapping[int, float], key: int) -> float:
    """A player's xP from a key → xP mapping; missing or NaN counts as 0."""
    x = float(xp.get(key, 0.0))
    return 0.0 if x != x else x


def skip_aware_tail(q_starters: Iterable[float], p_bench: Sequence[float]) -> tuple[float, ...]:
    """Per bench slot k (1..len(p_bench)): P(M ≥ 1 + S_{k−1}), M = Σ Bernoulli(q_starters)
    (starters who don't play) and S_{k−1} = Σ_{j<k} Bernoulli(p_bench[j]) (earlier bench
    players who do play), all independent; probabilities clipped to [0, 1]."""
    q = [float(v) for v in q_starters]
    tail = (1.0, *poisson_binomial_tail(q, len(q) + len(p_bench) + 1))  # tail[m] = P(M ≥ m)
    out = []
    dist = np.ones(1)  # P(S = s) over the bench players seen so far
    for value in p_bench:
        reach = [tail[s + 1] if s + 1 < len(tail) else 0.0 for s in range(len(dist))]
        out.append(float(min(max(np.dot(dist, reach), 0.0), 1.0)))
        p = min(max(float(value), 0.0), 1.0)
        nxt = np.append(dist * (1.0 - p), 0.0)
        nxt[1:] += dist * p
        dist = nxt
    return tuple(out)


def projected_xi(
    players: Sequence[tuple[int, int]], xp: Mapping[int, float], rules: Rules
) -> tuple[int, ...]:
    """The keys of the XI with the most xP among `players` ((player_key, element_type)),
    under the formation limits: each position's `play_min` best, then the best of the rest
    while their position is under `play_max` (as `backtest.policies.best_lineup`). Missing
    or NaN xP counts as 0; ties by player_key."""

    order = sorted(players, key=lambda p: (-_xp(xp, p[0]), p[0]))
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
    return tuple(k for k, _ in starters)


def not_play_probabilities(xp: pd.DataFrame) -> pd.Series | None:
    """P(doesn't play) per (player_key, horizon) from the frame's `p_play_gw` (else
    `p_play`), as 1 − P; several rows of one player-horizon (one per fixture) multiply. NaN
    where unknown. None when the frame has no `p_play` (a model without minutes)."""
    if P_PLAY not in xp.columns:
        return None
    p = xp[P_PLAY].astype("float64")
    if P_PLAY_GW in xp.columns:
        gw = xp[P_PLAY_GW].astype("float64")
        p = gw.where(gw.notna(), p)
    q = (1.0 - p.clip(0.0, 1.0)).to_numpy()
    frame = pd.DataFrame(
        {
            "player_key": xp["player_key"].to_numpy(dtype="int64"),
            "horizon": xp["horizon"].to_numpy(dtype="int64"),
            "q": q,
            "unknown": np.isnan(q),
        }
    )
    grouped = frame.groupby(["player_key", "horizon"], sort=True)
    out = grouped["q"].prod()
    out[grouped["unknown"].any()] = np.nan
    return out


def minutes_bench_weights(
    squad: Sequence[tuple[int, int]],
    xp: pd.DataFrame,
    horizons: Sequence[int],
    rules: Rules,
    fallback: Sequence[float],
) -> tuple[tuple[float, ...], ...] | None:
    """Bench weights (GK slot, outfield slots 1..) per horizon GW for the incumbent squad
    ((player_key, element_type)) from the frame's minutes (module docstring); `fallback`
    (the fixed weights) for a GW where a projected starter's P(play) is unknown. None when
    the frame carries no `p_play`."""
    q = not_play_probabilities(xp)
    if q is None:
        return None
    n_outfield = len(fallback) - 1
    q_map = q.to_dict()
    et = dict(squad)
    out = []
    for h in horizons:
        rows = xp[xp["horizon"] == h]
        # The GW's xP per player: rows of one player (one per fixture) add up.
        target = {
            int(k): float(v)
            for k, v in rows.groupby("player_key", sort=True)["xp"].sum(min_count=1).items()
        }
        xi = projected_xi(squad, target, rules)
        # No row in the frame: the player left the game (or has no fixture data): q = 1.
        # Projected bench outfielders in slot order: by xP, best first (ties by key).
        bench = sorted(
            (k for k, _ in squad if k not in xi and et[k] != GOALKEEPER),
            key=lambda k: (-_xp(target, k), k),
        )[:n_outfield]
        risks = {k: float(q_map.get((k, int(h)), 1.0)) for k in (*xi, *bench)}
        if any(r != r for r in risks.values()):
            out.append(tuple(float(w) for w in fallback))
            continue
        keepers = [risks[k] for k in xi if et[k] == GOALKEEPER]
        outfield = [risks[k] for k in xi if et[k] != GOALKEEPER]
        gk = poisson_binomial_tail(keepers, 1)[0]
        plays = [1.0 - risks[k] for k in bench]
        plays += [0.0] * (n_outfield - len(plays))  # an empty slot never comes on
        out.append((gk, *skip_aware_tail(outfield, plays)))
    return tuple(out)


def horizon_minutes(
    xp: pd.DataFrame, horizons: Sequence[int], keys: np.ndarray
) -> np.ndarray | None:
    """Expected minutes per player of `keys`, summed over `horizons` (NaN and missing rows
    count 0); None when the frame has no `e_minutes`."""
    if E_MINUTES not in xp.columns:
        return None
    rows = xp[xp["horizon"].isin(list(horizons))]
    minutes = (
        pd.Series(
            rows[E_MINUTES].astype("float64").fillna(0.0).to_numpy(), index=rows["player_key"]
        )
        .groupby(level=0)
        .sum()
    )
    return minutes.reindex(keys).fillna(0.0).to_numpy(dtype="float64")
