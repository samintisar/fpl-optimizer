"""Chip scenario search by bounds, the LP-relaxation bound, the MILP dual bound and the bulk
HiGHS hand-over (fplopt.optimize.search / model; Phase 4 plan, Task 3)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pulp")
pytest.importorskip("highspy")

import pulp  # noqa: E402
from optimize_fixtures import (  # noqa: E402
    RULES,
    correlated_xp,
    first_valid_squad,
    grid_pool,
    make_state,
)

import fplopt.optimize.model as model_mod  # noqa: E402
from fplopt.backtest.state import TRANSFER_CHIPS  # noqa: E402
from fplopt.optimize import OptimizerParams, PlanInput, optimize, scenarios  # noqa: E402
from fplopt.optimize.chips import NO_CHIP, make_scenario, terminal_value  # noqa: E402
from fplopt.optimize.model import relaxation_bound, solve_plan  # noqa: E402
from fplopt.optimize.search import SEARCHES, chip_extra, search  # noqa: E402

# Small candidate pools keep the exhaustive searches fast (synthetic instances are much
# harder for HiGHS than real ones); the properties tested don't depend on the pool.
SMALL = {1: 3, 2: 6, 3: 6, 4: 4}
EXACT = OptimizerParams(mip_gap=0.0, prune_n=SMALL)
# Terminal values that make chips worth playing inside a short horizon.
CHEAP_CHIPS = {"wildcard": 0.5, "freehit": 0.5, "bboost": 0.5, "3xc": 0.5}


def instance(seed: int, *, n_gws: int = 2, gw_index: int = 5, ft: int = 1, negative=False):
    """A seeded grid league (6 clubs) with the cheapest squad owned; `negative` shifts a
    seeded third of the xP below zero (blanks, red cards)."""
    pool = grid_pool(range(1, 7), {1: 2, 2: 4, 3: 4, 4: 3}, seed=seed)
    xp = correlated_xp(pool, n_gws, seed, gw_index=gw_index)
    if negative:
        rng = np.random.default_rng(seed)
        mask = rng.random(len(xp)) < 0.33
        xp.loc[mask, "xp"] = xp.loc[mask, "xp"] - 1.5
    state = make_state(pool, first_valid_squad(pool), ft=ft, gw_index=gw_index)
    return state, pool, xp


def problem_for(state, pool, xp, params):
    return PlanInput.from_context(state, pool, xp, RULES, params)


# --- bulk hand-over -----------------------------------------------------------------------


def test_bulk_handover_gives_pulps_plans(monkeypatch) -> None:
    """The bulk hand-over passes HiGHS the model PuLP's own one does: identical plans."""
    state, pool, xp = instance(3, n_gws=3)
    params = OptimizerParams(horizon=3, prune_n=SMALL)
    problem = problem_for(state, pool, xp, params)
    picked = [
        NO_CHIP,
        make_scenario(problem, {0: "freehit"}),
        make_scenario(problem, {1: "wildcard", 2: "bboost"}),
        make_scenario(problem, {0: "3xc", 2: "freehit"}),
    ]
    bulk = [solve_plan(problem, params, chips=s) for s in picked]
    monkeypatch.setattr(model_mod, "_BulkHiGHS", pulp.HiGHS)
    plain = [solve_plan(problem, params, chips=s) for s in picked]
    for a, b in zip(bulk, plain, strict=True):
        assert a.gws == b.gws
        assert a.objective == b.objective
        assert a.bound == pytest.approx(b.bound)


def test_bulk_handover_refuses_warm_starts() -> None:
    lp = pulp.LpProblem("x", pulp.LpMaximize)
    x = lp.add_variable("x", lowBound=0, upBound=1)
    lp += x <= 1
    lp.setObjective(x + 0)
    with pytest.raises(ValueError, match="warm starts"):
        lp.solve(model_mod._BulkHiGHS(msg=False, warmStart=True))


# --- bounds -------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_lp_and_dual_bounds_bound_every_scenario(seed: int) -> None:
    """For every scenario: MILP optimum ≤ its dual bound ≤ LP relaxation (gap 0)."""
    state, pool, xp = instance(seed, n_gws=2, negative=seed == 1)
    params = replace(EXACT, horizon=2, chip_value=CHEAP_CHIPS)
    problem = problem_for(state, pool, xp, params)
    for s in scenarios(problem, params):
        plan = solve_plan(problem, params, chips=s)
        lp = relaxation_bound(problem, params, s)
        assert plan.total_objective <= plan.bound + 1e-6, s.label
        assert plan.bound == pytest.approx(plan.total_objective, abs=1e-6), s.label
        assert plan.total_objective <= lp + 1e-9, s.label


@pytest.mark.parametrize("seed", [0, 1])
def test_chip_extras_bound_tc_and_bb(seed: int) -> None:
    """A scenario's optimum ≤ its base's (WC/FH chips only) optimum + Σ E + the terminal
    value change, the derived bound of the search (gap 0, negative xP in one instance)."""
    state, pool, xp = instance(seed, n_gws=2, negative=seed == 1)
    params = replace(EXACT, horizon=2, chip_value=CHEAP_CHIPS)
    problem = problem_for(state, pool, xp, params)
    totals = {}
    checked = 0
    for s in scenarios(problem, params):
        extras = [(t, n) for t, n in s.chips if n not in TRANSFER_CHIPS]
        if not extras:
            continue
        base = make_scenario(problem, {t: n for t, n in s.chips if n in TRANSFER_CHIPS})
        for sc in (s, base):
            if sc.label not in totals:
                totals[sc.label] = solve_plan(problem, params, chips=sc).total_objective
        extra = sum(chip_extra(problem, params, t, n) for t, n in extras)
        derived = (
            totals[base.label]
            - terminal_value(problem, params, base)
            + extra
            + terminal_value(problem, params, s)
        )
        assert totals[s.label] <= derived + 1e-6, s.label
        checked += 1
    assert checked >= 10


def test_chip_extra_values() -> None:
    state, pool, xp = instance(4, n_gws=2)
    params = OptimizerParams(horizon=2, decay=0.5)
    problem = problem_for(state, pool, xp, params)
    x1 = [p.xp[1] for p in problem.players]
    assert chip_extra(problem, params, 1, "3xc") == pytest.approx(0.5 * max(x1))
    gk = max(p.xp[1] for p in problem.players if p.element_type == 1)
    out = sorted((p.xp[1] for p in problem.players if p.element_type != 1), reverse=True)
    w = params.bench_weights  # (0.03, 0.21, 0.06, 0.002): |1 − w| descending 0.998, 0.94, 0.79
    expected = 0.5 * ((1 - w[0]) * gk + (1 - w[3]) * out[0] + (1 - w[2]) * out[1])
    expected += 0.5 * (1 - w[1]) * out[2]
    assert chip_extra(problem, params, 1, "bboost") == pytest.approx(expected)
    assert chip_extra(problem, params, 0, "wildcard") == float("inf")


# --- the search ---------------------------------------------------------------------------


CASES = [
    # (seed, n_gws, gw_index, ft, chip values, mip_gap, negative xP)
    (0, 2, 5, 1, None, 0.005, False),
    (1, 2, 5, 2, CHEAP_CHIPS, 0.005, False),
    (2, 2, 18, 1, None, 0.0, False),  # set-1 chips expire inside the horizon (gw_index 19)
    (3, 2, 19, 3, CHEAP_CHIPS, 0.005, False),  # both Free Hits in the horizon
    (4, 2, 1, 1, CHEAP_CHIPS, 0.005, False),  # GW1: unlimited free transfers
    (5, 2, 9, 1, None, 0.005, True),
    (6, 2, 30, 5, CHEAP_CHIPS, 0.0, True),
    (7, 2, 12, 1, {"wildcard": 0.0, "freehit": 9.0, "bboost": 0.0, "3xc": 9.0}, 0.005, False),
    (8, 1, 38, 1, CHEAP_CHIPS, 0.005, False),  # the season's last GW
    (9, 1, 20, 2, {"wildcard": 2.0, "freehit": 0.0, "bboost": 2.0, "3xc": 0.0}, 0.0, True),
]


@pytest.mark.parametrize("case", CASES, ids=[f"seed{c[0]}" for c in CASES])
def test_bound_search_returns_what_all_returns(case) -> None:
    """Soundness: on every instance the bound search returns exactly the exhaustive
    search's plan (same scenario, objective and first-GW decision) with fewer solves."""
    seed, n_gws, gw_index, ft, chip_value, gap, negative = case
    state, pool, xp = instance(seed, n_gws=n_gws, gw_index=gw_index, ft=ft, negative=negative)
    params = OptimizerParams(horizon=n_gws, mip_gap=gap, prune_n=SMALL)
    if chip_value is not None:
        params = replace(params, chip_value=chip_value)
    problem = problem_for(state, pool, xp, params)
    candidates = scenarios(problem, params)
    full = search(problem, params, candidates, "all")
    fast = search(problem, params, candidates, "bound")
    assert fast.scenario == full.scenario
    assert fast.plan.total_objective == full.plan.total_objective
    assert fast.plan.gws == full.plan.gws
    assert fast.stats.n_solves < full.stats.n_solves == len(candidates)
    # Solved scenarios are solved identically; skipped ones were provably worse.
    for label, value in fast.objectives.items():
        assert value == full.objectives[label]
    for label, bound in fast.bounds.items():
        assert full.objectives[label] <= bound + 1e-6, label
        if label not in fast.objectives:
            assert bound <= fast.plan.total_objective + 1e-6, label


def test_bound_search_solves_only_no_chip_when_chips_cannot_pay() -> None:
    state, pool, xp = instance(8, n_gws=3)
    expensive = {"wildcard": 500.0, "freehit": 500.0, "bboost": 500.0, "3xc": 500.0}
    params = OptimizerParams(horizon=3, chip_value=expensive, prune_n=SMALL)
    problem = problem_for(state, pool, xp, params)
    result = optimize(problem, params, top_k=1, roll=False)
    assert result.best.scenario == "none"
    assert result.search.n_solves == 1
    assert dict(result.scenario_objectives) == {"none": result.best.total_objective}
    assert list(result.scenario_bounds) == [s.label for s in scenarios(problem, params)]


def test_search_validation() -> None:
    state, pool, xp = instance(9, n_gws=1)
    params = OptimizerParams(horizon=1)
    problem = problem_for(state, pool, xp, params)
    candidates = scenarios(problem, params)
    assert SEARCHES == ("bound", "all")
    with pytest.raises(ValueError, match="scenario search"):
        search(problem, params, candidates, "fast")
    with pytest.raises(ValueError, match="no-chip"):
        search(problem, params, candidates[1:], "bound")
    with pytest.raises(ValueError, match="scenario search"):
        optimize(problem, params, scenario_search="none")


# --- real data ----------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# (season, gw_index, start, xP model): a template and a random start, both xP models, a
# 3-GW horizon (~60 scenarios, so the exhaustive search stays ~10-30 s per instance).
REAL_CASES = [(2022, 15, "random:0", "rolling"), (2024, 30, "template", "ep_next")]


@pytest.mark.realdata
@pytest.mark.parametrize("case", REAL_CASES, ids=[f"{c[0]}-gw{c[1]}" for c in REAL_CASES])
def test_bound_search_returns_what_all_returns_on_real_data(case) -> None:
    if not (DATA_DIR / "player_match.parquet").exists():
        pytest.skip("data/ not built (run `uv run fplopt build all`)")
    from fplopt.backtest.simulator import Caches
    from fplopt.features.store import DataStore
    from fplopt.optimize.bench import BenchCase, build_case

    state, pool, xp, rules = build_case(DataStore(DATA_DIR), Caches(), BenchCase(*case))
    params = OptimizerParams(horizon=3)
    problem = PlanInput.from_context(state, pool, xp, rules, params)
    candidates = scenarios(problem, params)
    full = search(problem, params, candidates, "all")
    fast = search(problem, params, candidates, "bound")
    assert fast.scenario == full.scenario
    assert fast.plan.total_objective == full.plan.total_objective
    assert fast.plan.gws == full.plan.gws
    assert fast.stats.n_solves < full.stats.n_solves
