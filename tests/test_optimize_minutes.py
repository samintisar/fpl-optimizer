"""The minutes model in the planner (fplopt.optimize.minutes; Phase 5 plan, Task 8): bench
weights from P(starter doesn't play) and the expected-minutes candidate floor."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pulp")
pytest.importorskip("highspy")

from optimize_fixtures import (  # noqa: E402
    RULES,
    best_lineup_value,
    correlated_xp,
    first_valid_squad,
    grid_pool,
    make_state,
)

from fplopt.optimize import OptimizerParams, PlanInput  # noqa: E402
from fplopt.optimize.minutes import (  # noqa: E402
    minutes_bench_weights,
    not_play_probabilities,
    poisson_binomial_tail,
    projected_xi,
)
from fplopt.optimize.model import solve_plan  # noqa: E402
from fplopt.optimize.params import DEFAULT_BENCH_WEIGHTS  # noqa: E402
from fplopt.optimize.search import chip_extra  # noqa: E402

EXACT = OptimizerParams(mip_gap=0.0, prune_n=None, prune_dominated=False, tie_epsilon=0.0)


def instance(seed: int, *, n_gws: int = 3, gw_index: int = 5):
    pool = grid_pool(range(1, 9), {1: 2, 2: 4, 3: 4, 4: 3}, seed=seed)
    xp = correlated_xp(pool, n_gws, seed, gw_index=gw_index)
    state = make_state(pool, first_valid_squad(pool), ft=1, gw_index=gw_index)
    return state, pool, xp


def with_minutes(xp: pd.DataFrame, p_play, e_minutes=None) -> pd.DataFrame:
    """The frame with `p_play` (a scalar or player_key -> per-horizon list) and, optionally,
    `e_minutes` (same forms)."""

    def column(values):
        if not isinstance(values, dict):
            return np.full(len(xp), float(values))
        return np.array(
            [
                float(values[k][h]) if k in values else 1.0
                for k, h in zip(xp["player_key"], xp["horizon"], strict=True)
            ]
        )

    out = xp.assign(p_play=column(p_play))
    if e_minutes is not None:
        out = out.assign(e_minutes=column(e_minutes))
    return out


def squad_of(state) -> list[tuple[int, int]]:
    return [(h.player_key, h.element_type) for h in state.holdings]


def plan_summary(plan) -> list:
    """Everything a plan decides plus its objective (no timings)."""
    return [
        (g.transfers, g.starters, g.bench, g.captain, g.vice, g.xp, g.bench_xp, g.objective)
        for g in plan.gws
    ] + [plan.objective, plan.status]


# --- Poisson-binomial ---------------------------------------------------------------------


def test_poisson_binomial_tail_hand_values() -> None:
    assert poisson_binomial_tail([0.1, 0.2], 3) == pytest.approx((0.28, 0.02, 0.0))
    assert poisson_binomial_tail([0.5] * 3, 3) == pytest.approx((0.875, 0.5, 0.125))
    # 0.1, 0.2, 0.3: P(0) = 0.504, P(1) = 0.398, P(2) = 0.092, P(3) = 0.006.
    assert poisson_binomial_tail([0.1, 0.2, 0.3], 3) == pytest.approx((0.496, 0.098, 0.006))
    assert poisson_binomial_tail([], 2) == (0.0, 0.0)
    assert poisson_binomial_tail([1.0, 0.0, 1.0], 3) == pytest.approx((1.0, 1.0, 0.0))
    assert poisson_binomial_tail([1.5, -0.2], 2) == pytest.approx((1.0, 0.0))  # clipped


def test_projected_xi_follows_the_formation_rules() -> None:
    # 2 GK, 5 DEF, 5 MID, 3 FWD; DEFs score most, but at most 5 DEF and 1 GK start.
    squad = [(1, 1), (2, 1)] + [(10 + i, 2) for i in range(5)]
    squad += [(20 + i, 3) for i in range(5)] + [(30 + i, 4) for i in range(3)]
    xp = {1: 1.0, 2: 9.0} | {10 + i: 8.0 for i in range(5)}
    xp |= {20 + i: float(i) for i in range(5)} | {30: 0.5, 31: 0.4, 32: 0.3}
    xi = projected_xi(squad, xp, RULES)
    assert set(xi) == {2, 10, 11, 12, 13, 14, 24, 23, 22, 30, 21}


# --- bench weights ------------------------------------------------------------------------


def test_weights_from_the_projected_xi() -> None:
    state, pool, xp = instance(0, n_gws=1)
    squad = squad_of(state)
    xi = projected_xi(squad, dict(zip(xp["player_key"], xp["xp"], strict=True)), RULES)
    et = dict(squad)
    gk = next(k for k in xi if et[k] == 1)
    outfield = [k for k in xi if et[k] != 1]
    bench = [k for k, _ in squad if k not in xi]
    # Starters: GK plays w.p. 0.7, outfield starters w.p. 0.9, 0.8 and 1 for the rest.
    # Bench players' P(play) must not matter.
    p = {gk: [0.7], outfield[0]: [0.9], outfield[1]: [0.8]}
    p |= {k: [1.0] for k in outfield[2:]} | {k: [0.1] for k in bench}
    weights = minutes_bench_weights(squad, with_minutes(xp, p), [0], RULES, DEFAULT_BENCH_WEIGHTS)
    assert weights == (pytest.approx((0.3, 0.28, 0.02, 0.0)),)


def test_gk_slot_is_the_starting_keepers_risk_only() -> None:
    state, pool, xp = instance(1, n_gws=2)
    squad = squad_of(state)
    keepers = [k for k, et in squad if et == 1]
    # Every outfielder certain to play; the keepers 0.6 / 0.4 in GW 0 and 1.0 / 0.5 in GW 1.
    p = {k: [1.0, 1.0] for k, et in squad if et != 1}
    p |= {keepers[0]: [0.6, 1.0], keepers[1]: [0.4, 0.5]}
    frame = with_minutes(xp, p)
    weights = minutes_bench_weights(squad, frame, [0, 1], RULES, DEFAULT_BENCH_WEIGHTS)
    for t, w in enumerate(weights):
        target = frame[frame["horizon"] == t]
        xi = projected_xi(squad, dict(zip(target["player_key"], target["xp"], strict=True)), RULES)
        starter = next(k for k in xi if k in keepers)
        assert w[0] == pytest.approx(1.0 - p[starter][t])
        assert w[1:] == pytest.approx((0.0, 0.0, 0.0))


def test_double_gw_uses_p_play_gw_and_unknown_falls_back_to_fixed_weights() -> None:
    state, pool, xp = instance(2, n_gws=2)
    squad = squad_of(state)
    frame = with_minutes(xp, 1.0)
    # GW 1 is a double: per-GW p_play is NaN, as in v1's frame.
    double = frame["horizon"] == 1
    frame.loc[double, "p_play"] = np.nan
    weights = minutes_bench_weights(squad, frame, [0, 1], RULES, DEFAULT_BENCH_WEIGHTS)
    assert weights[0] == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert weights[1] == DEFAULT_BENCH_WEIGHTS  # unknown: the fixed weights for that GW
    # With p_play_gw (P(plays in at least one fixture)) the double is weighted from it.
    frame["p_play_gw"] = np.where(double, 0.5, frame["p_play"])
    weights = minutes_bench_weights(squad, frame, [0, 1], RULES, DEFAULT_BENCH_WEIGHTS)
    assert weights[1] == pytest.approx((0.5, *poisson_binomial_tail([0.5] * 10, 3)))


def test_rows_per_fixture_multiply_and_missing_players_never_play() -> None:
    frame = pd.DataFrame(
        {
            "player_key": [1, 1, 2, 3],
            "horizon": [0, 0, 0, 0],
            "xp": [1.0, 1.0, 2.0, 0.5],
            "p_play": [0.5, 0.6, np.nan, 0.9],
        }
    )
    q = not_play_probabilities(frame)
    assert q[(1, 0)] == pytest.approx(0.5 * 0.4)  # plays in neither fixture
    assert np.isnan(q[(2, 0)])
    assert q[(3, 0)] == pytest.approx(0.1)
    assert not_play_probabilities(frame.drop(columns="p_play")) is None
    # A starter missing from the frame (left the game) counts as not playing: with every
    # xP 0, the keeper with the smaller key starts (ties by key), rows or not.
    state, pool, xp = instance(3, n_gws=1)
    squad = squad_of(state)
    frame = with_minutes(xp.assign(xp=0.0), 1.0)
    gk = min(k for k, et in squad if et == 1)
    assert gk in projected_xi(squad, {}, RULES)
    frame = frame[frame["player_key"] != gk].reset_index(drop=True)
    weights = minutes_bench_weights(squad, frame, [0], RULES, DEFAULT_BENCH_WEIGHTS)
    assert weights == ((1.0, 0.0, 0.0, 0.0),)


# --- the planner --------------------------------------------------------------------------

# Plan of `instance(3, n_gws=3)` with OptimizerParams(horizon=3) on the Phase 4 code
# (commit 1e10dfe, before the minutes-based weights existed): objective, then per GW the
# transfers (out, in), the bench and repr(bench_xp).
GOLDEN_OBJECTIVE = "124.30529374999998"
GOLDEN_GWS = [
    (((230, 833),), (111, 541, 420, 822), "0.6159"),
    (((541, 542),), (111, 730, 420, 822), "0.6282"),
    (((730, 632),), (711, 842, 420, 822), "0.8016"),
]


def test_without_p_play_the_fixed_weights_and_plans_are_unchanged() -> None:
    """Baseline-model frames (no p_play, no e_minutes): the Phase 3/4 plans exactly."""
    state, pool, xp = instance(3, n_gws=3)
    params = OptimizerParams(horizon=3)
    problem = PlanInput.from_context(state, pool, xp, RULES, params)
    assert problem.bench_weights is None
    assert problem.gw_bench_weights(1, params) == DEFAULT_BENCH_WEIGHTS
    plan = solve_plan(problem, params)
    off = replace(params, bench_from_minutes=False, min_minutes=60.0)
    same = solve_plan(PlanInput.from_context(state, pool, xp, RULES, off), off)
    assert plan_summary(same) == plan_summary(plan)
    # A frame with minutes, with the minutes features off, plans as the baseline frame.
    minutes = with_minutes(xp, 0.5, e_minutes=0.0)
    off = replace(params, bench_from_minutes=False, min_minutes=0.0)
    ignored = solve_plan(PlanInput.from_context(state, pool, minutes, RULES, off), off)
    assert plan_summary(ignored) == plan_summary(plan)
    assert repr(plan.objective) == GOLDEN_OBJECTIVE
    assert [
        (tuple((t.out_key, t.in_key) for t in g.transfers), g.bench, repr(g.bench_xp))
        for g in plan.gws
    ] == GOLDEN_GWS


def test_per_gw_weights_reach_the_objective() -> None:
    state, pool, xp = instance(4, n_gws=2)
    squad = squad_of(state)
    owned = {k for k, _ in squad}
    pool = pool[pool["player_key"].isin(owned)].reset_index(drop=True)  # no transfers
    xp = xp[xp["player_key"].isin(owned)].reset_index(drop=True)
    p = {k: [0.9, 0.6] for k in owned}
    frame = with_minutes(xp, p)
    params = replace(EXACT, horizon=2)
    problem = PlanInput.from_context(state, pool, frame, RULES, params)
    weights = problem.bench_weights
    assert weights is not None and weights[0] != weights[1]
    assert weights[0] == pytest.approx((0.1, *poisson_binomial_tail([0.1] * 10, 3)))
    assert weights[1] == pytest.approx((0.4, *poisson_binomial_tail([0.4] * 10, 3)))
    plan = solve_plan(problem, params)
    et = dict(squad)
    for t, g in enumerate(plan.gws):
        target = frame[frame["horizon"] == t]
        values = dict(zip(target["player_key"], target["xp"], strict=True))
        expected = sum(w * values[k] for w, k in zip(weights[t], g.bench, strict=True))
        assert g.bench_xp == pytest.approx(expected)
        # The MILP's lineup is the best for this GW's weights.
        best = best_lineup_value(list(owned), values, et, weights[t])
        assert g.xp + g.bench_xp == pytest.approx(best)
    # Different weights than the fixed ones: the plan values the bench differently.
    fixed = solve_plan(PlanInput.from_context(state, pool, xp, RULES, params), params)
    assert fixed.objective != pytest.approx(plan.objective)


def test_bench_boost_weights_stay_one() -> None:
    state, pool, xp = instance(5, n_gws=2)
    frame = with_minutes(xp, 0.8)
    params = replace(EXACT, horizon=2)
    problem = PlanInput.from_context(state, pool, frame, RULES, params)
    assert problem.bench_weights is not None
    plan = solve_plan(problem, params, chips={0: "bboost"})
    g = plan.gws[0]
    target = frame[frame["horizon"] == 0]
    values = dict(zip(target["player_key"], target["xp"], strict=True))
    assert g.chip == "bboost"
    assert g.bench_xp == pytest.approx(sum(values[k] for k in g.bench))
    # GW 1 has no chip: minutes-based weights.
    later = plan.gws[1]
    target = frame[frame["horizon"] == 1]
    values = dict(zip(target["player_key"], target["xp"], strict=True))
    expected = sum(
        w * values[k] for w, k in zip(problem.bench_weights[1], later.bench, strict=True)
    )
    assert later.bench_xp == pytest.approx(expected)
    # The Bench Boost bound of the chip search uses the GW's own weights (GW 1: decayed).
    w = problem.bench_weights[1]
    keepers = max(abs(p.xp[1]) for p in problem.players if p.element_type == 1)
    outfield = sorted((abs(p.xp[1]) for p in problem.players if p.element_type != 1), reverse=True)
    slots = sorted((abs(1.0 - x) for x in w[1:]), reverse=True)
    expected = (1.0 - w[0]) * keepers + sum(a * x for a, x in zip(slots, outfield, strict=False))
    assert chip_extra(problem, params, 1, "bboost") == pytest.approx(params.decay * expected)
    fixed = replace(problem, bench_weights=None)
    assert chip_extra(fixed, params, 1, "bboost") != pytest.approx(params.decay * expected)


# --- minutes floor ------------------------------------------------------------------------


def test_floor_drops_only_non_owned_low_minute_players() -> None:
    state, pool, xp = instance(6, n_gws=2)
    owned = {h.player_key for h in state.holdings}
    others = sorted(set(pool["player_key"]) - owned)
    low_owned = sorted(owned)[:3]
    low_others = others[:5]
    # Expected minutes per GW: 20 for the "low" players (40 over the horizon), 80 for the rest.
    minutes = {
        k: [20.0, 20.0] if k in set(low_owned) | set(low_others) else [80.0, 80.0]
        for k in pool["player_key"]
    }
    frame = with_minutes(xp, 1.0, e_minutes=minutes)
    params = replace(EXACT, horizon=2, min_minutes=60.0)
    kept = {p.player_key for p in PlanInput.from_context(state, pool, frame, RULES, params).players}
    assert set(low_owned) <= kept  # owned players are always candidates
    assert not set(low_others) & kept
    assert set(others[5:]) <= kept
    # No floor (0), a floor below their minutes, or a frame without e_minutes: all kept.
    for p, f in (
        (replace(params, min_minutes=0.0), frame),
        (replace(params, min_minutes=40.0), frame),
        (params, frame.drop(columns="e_minutes")),
    ):
        keys = {pl.player_key for pl in PlanInput.from_context(state, pool, f, RULES, p).players}
        assert keys == set(pool["player_key"])


def test_floor_runs_before_top_n() -> None:
    """A floored player doesn't take a top-N slot: the next best player gets it."""
    state, pool, xp = instance(7, n_gws=1)
    owned = {h.player_key for h in state.holdings}
    fwds = pool[(pool["element_type"] == 4) & ~pool["player_key"].isin(owned)]
    values = dict(zip(xp["player_key"], xp["xp"], strict=True))
    ranked = sorted(fwds["player_key"], key=lambda k: (-values[k], k))
    best = ranked[0]
    minutes = {k: [0.0 if k == best else 90.0] for k in pool["player_key"]}
    frame = with_minutes(xp, 1.0, e_minutes=minutes)
    params = replace(EXACT, horizon=1, prune_n={1: 0, 2: 0, 3: 0, 4: 1}, min_minutes=30.0)
    kept = {p.player_key for p in PlanInput.from_context(state, pool, frame, RULES, params).players}
    assert best not in kept
    assert ranked[1] in kept


def test_min_minutes_validation() -> None:
    with pytest.raises(ValueError, match="min_minutes"):
        OptimizerParams(min_minutes=-1.0)
    assert OptimizerParams(bench_from_minutes=0).bench_from_minutes is False
    assert OptimizerParams().min_minutes == 180.0  # Task 8 benchmark


def test_floor_is_pro_rata_on_a_short_horizon() -> None:
    """The floor is for a full horizon: with 2 of 6 GWs left, 180 becomes 60."""
    state, pool, xp = instance(8, n_gws=2)
    owned = {h.player_key for h in state.holdings}
    others = sorted(set(pool["player_key"]) - owned)
    # 25 or 35 expected minutes per GW: 50 / 70 over the 2 GWs left.
    minutes = {k: [25.0, 25.0] if k in others[:4] else [35.0, 35.0] for k in pool["player_key"]}
    frame = with_minutes(xp, 1.0, e_minutes=minutes)
    params = replace(EXACT, horizon=6, min_minutes=180.0)
    kept = {p.player_key for p in PlanInput.from_context(state, pool, frame, RULES, params).players}
    assert not set(others[:4]) & kept
    assert set(others[4:]) | owned <= kept
