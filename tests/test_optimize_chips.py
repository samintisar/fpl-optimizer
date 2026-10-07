"""Chip scenarios, chip terms in the MILP, top-k plans and the roll plan (fplopt.optimize;
Phase 4 plan, Task 2)."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

pytest.importorskip("pulp")
pytest.importorskip("highspy")

from optimize_fixtures import (  # noqa: E402
    RULES,
    correlated_xp,
    first_valid_squad,
    grid_pool,
    make_state,
    xp_frame,
)

from fplopt.backtest.state import (  # noqa: E402
    InvalidDecision,
    apply_decision,
    next_state,
    selling_price,
)
from fplopt.optimize import (  # noqa: E402
    OptimizerParams,
    PlanInput,
    optimize,
    scenarios,
    terminal_value,
)
from fplopt.optimize.chips import NO_CHIP, make_scenario  # noqa: E402
from fplopt.optimize.model import solve_plan  # noqa: E402

# 2026-27 rules (used for the backtests too): two sets of every chip, set 1 on gw_index
# 1/2..19, set 2 on 20..38; ids WC 1/2, FH 3/6, BB 4/7, TC 5/8; FTs `retain` after WC/FH.
WC1, WC2, FH1, BB1, TC1, FH2, BB2, TC2 = 1, 2, 3, 4, 5, 6, 7, 8
EXACT = OptimizerParams(mip_gap=0.0, prune_n=None, prune_dominated=False)
EXPENSIVE = {"wildcard": 100.0, "freehit": 100.0, "bboost": 100.0, "3xc": 100.0}


def chip_params(n_gws: int, **chip_value: float) -> OptimizerParams:
    """Exact params where every chip costs 100 to play (its terminal value) unless
    `chip_value` says otherwise, so one chip at a time can be made attractive."""
    return replace(EXACT, horizon=n_gws, chip_value={**EXPENSIVE, **chip_value})


def problem_at(gw_index: int, n_gws: int, *, chips_used=(), rules=RULES, horizon=None):
    """A PlanInput over an xP frame of `n_gws` GWs from `gw_index` (only the state and the
    horizon matter for scenario enumeration)."""
    pool = grid_pool(range(1, 7), {1: 2, 2: 4, 3: 4, 4: 3})
    state = make_state(pool, first_valid_squad(pool), gw_index=gw_index)
    state = replace(state, chips_used=tuple(chips_used))
    xp = correlated_xp(pool, n_gws, 0, gw_index=gw_index)
    params = replace(EXACT, horizon=horizon or n_gws)
    return PlanInput.from_context(state, pool, xp, rules, params)


def labels(problem, params=EXACT):
    return [s.label for s in scenarios(problem, params)]


def team(owned_xp, *, other_xp=None, special=None, ft=1, gw_index=5, purchase=None):
    """A grid pool (6 clubs, 13 players each) with the cheapest valid squad owned; owned
    players score `owned_xp`, the others `other_xp` (default 0) per GW, `special`
    overrides single players."""
    n_gws = len(owned_xp)
    pool = grid_pool(range(1, 7), {1: 2, 2: 4, 3: 4, 4: 3})
    owned = first_valid_squad(pool)
    other = list(other_xp) if other_xp is not None else [0.0] * n_gws
    values = {int(k): list(other) for k in pool["player_key"]}
    values.update({k: list(owned_xp) for k in owned})
    values.update(special or {})
    state = make_state(pool, owned, ft=ft, gw_index=gw_index, purchase=purchase)
    return state, pool, xp_frame(values, gw_index=gw_index), owned


def replay(state, plan, pool, rules=RULES):
    """Every GW of the plan through apply_decision/next_state; the plan's ft (before),
    hits, transfers, chip and bank (after) must match the rules engine. Returns the state
    after the horizon."""
    for i, g in enumerate(plan.gws):
        assert state.free_transfers == g.ft or state.gw_index == 1, (i, state.free_transfers, g)
        gw_state, record = apply_decision(state, g.decision(), pool, rules)
        assert (record.hits, record.n_transfers, record.chip) == (g.hits, g.n_transfers, g.chip)
        assert gw_state.bank == g.bank, (i, gw_state.bank, g.bank)
        nxt = plan.gws[i + 1].gw_index if i + 1 < len(plan.gws) else g.gw_index + 1
        state = next_state(gw_state, record, rules, nxt)
    return state


def squad_of(g) -> set[int]:
    return set(g.starters) | set(g.bench)


# --- scenario enumeration -----------------------------------------------------------------


def test_scenarios_mid_season_singles_and_pairs() -> None:
    problem = problem_at(5, 3)
    got = labels(problem)
    names = ["3xc", "bboost", "freehit", "wildcard"]
    singles = [f"{n}@gw{g}" for g in (5, 6, 7) for n in names]
    pairs = [
        f"{a}@gw{g1}+{b}@gw{g2}"
        for g1, g2 in ((5, 6), (5, 7), (6, 7))
        for a in names
        for b in names
        if a != b  # a second window of the same chip isn't open before GW20
    ]
    pairs.sort(key=lambda s: (int(s.split("@gw")[1].split("+")[0]), s))
    assert got[0] == "none"
    assert got[1:13] == singles
    assert sorted(got[13:]) == sorted(pairs) and len(got) == 1 + 12 + 36
    assert len(set(got)) == len(got)
    # Pairs come in (first position, first name, second position, second name) order.
    keys = [tuple(s.chips) for s in scenarios(problem, EXACT)[13:]]
    assert keys == sorted(keys)


def test_scenarios_skip_used_chips_and_record_chip_ids() -> None:
    problem = problem_at(5, 3, chips_used=((WC1, 3), (TC1, 4)))
    found = scenarios(problem, EXACT)
    assert [s.label for s in found[:7]] == [
        "none",
        "bboost@gw5",
        "freehit@gw5",
        "bboost@gw6",
        "freehit@gw6",
        "bboost@gw7",
        "freehit@gw7",
    ]
    assert len(found) == 1 + 6 + 6
    by_label = {s.label: s for s in found}
    assert by_label["freehit@gw5+bboost@gw7"].chip_ids == (FH1, BB1)
    assert by_label["freehit@gw5+bboost@gw7"].by_t == {0: "freehit", 2: "bboost"}


@pytest.mark.parametrize("consecutive", [False, True])
def test_free_hits_across_the_set_boundary(consecutive: bool) -> None:
    rules = replace(RULES, freehit_consecutive=consecutive)
    problem = problem_at(18, 4, rules=rules)  # GWs 18-21: set 1 ends at 19
    by_label = {s.label: s for s in scenarios(problem, EXACT)}
    assert by_label["freehit@gw18+freehit@gw20"].chip_ids == (FH1, FH2)
    assert ("freehit@gw19+freehit@gw20" in by_label) is consecutive
    assert by_label["wildcard@gw19+wildcard@gw20"].chip_ids == (WC1, WC2)
    assert "wildcard@gw18+wildcard@gw19" not in by_label  # one set-1 wildcard only
    # A Free Hit played the GW before the deadline blocks one now (unless allowed).
    after_fh = labels(problem_at(20, 2, chips_used=((FH1, 19),), rules=rules))
    assert ("freehit@gw20" in after_fh) is consecutive
    assert "freehit@gw21" in after_fh
    assert "freehit@gw20+freehit@gw21" not in after_fh  # one set-2 Free Hit


def test_make_scenario_rejects_what_the_rules_forbid() -> None:
    problem = problem_at(5, 3, chips_used=((WC1, 3),))
    with pytest.raises(InvalidDecision, match="already used"):
        make_scenario(problem, {0: "wildcard"})
    with pytest.raises(InvalidDecision, match="already used"):
        make_scenario(problem, {0: "bboost", 2: "bboost"})
    with pytest.raises(ValueError, match="outside the horizon"):
        make_scenario(problem, {3: "bboost"})
    assert make_scenario(problem, {}) == NO_CHIP


def test_terminal_values() -> None:
    params = replace(EXACT, chip_value={"wildcard": 6.0, "freehit": 4.0, "bboost": 2.0})
    early = problem_at(5, 3)  # GWs 5-7: every window extends past GW7
    assert terminal_value(early, params, NO_CHIP) == 2 * (6 + 4 + 2 + 0)
    assert terminal_value(early, params, make_scenario(early, {1: "bboost"})) == 24 - 2
    # GWs 17-19: set 1 closes inside the horizon → worth 0 whether played or not.
    closing = problem_at(17, 3)
    assert terminal_value(closing, params, NO_CHIP) == 12
    assert terminal_value(closing, params, make_scenario(closing, {0: "wildcard"})) == 12
    # The state's used chips don't count.
    used = problem_at(5, 3, chips_used=((WC1, 2), (WC2, 21)))
    assert terminal_value(used, params, NO_CHIP) == 24 - 12


def test_horizon_near_season_end() -> None:
    problem = problem_at(36, 3, horizon=6)  # only GWs 36-38 in the xP frame
    assert [g.gw_index for g in problem.gws] == [36, 37, 38]
    found = scenarios(problem, EXACT)
    assert all(terminal_value(problem, OptimizerParams(), s) == 0 for s in found)
    singles = [s for s in found if len(s.chips) == 1]
    assert len(singles) == 12
    assert {s.chip_ids[0] for s in singles} == {WC2, FH2, BB2, TC2}


# --- each chip on a hand-built instance ----------------------------------------------------


def test_bench_boost_when_the_bench_scores() -> None:
    state, pool, xp, owned = team([2.0, 6.0])
    params = chip_params(2, bboost=1.0)
    best = optimize(state_problem(state, pool, xp, params), params, top_k=1).best
    assert best.scenario == "bboost@gw6"
    g = best.gws[1]
    assert g.chip == "bboost" and g.bench_xp == pytest.approx(4 * 6.0)
    assert best.terminal_value == pytest.approx(6 * 100.0 + 1.0)  # BB1 used, BB2 held
    replay(state, best, pool)
    # Worth 0.85 · 24 = 20.4 now; holding the chip is worth more → no chip.
    params = chip_params(2, bboost=25.0)
    assert optimize(state_problem(state, pool, xp, params), params, top_k=1).best.scenario == (
        "none"
    )


def test_triple_captain_on_the_big_gw() -> None:
    state, pool, xp, owned = team([2.0, 2.0])
    star = owned[-1]  # a forward
    xp.loc[xp["player_key"] == star, "xp"] = [5.0, 15.0]
    params = chip_params(2, **{"3xc": 3.0})
    problem = state_problem(state, pool, xp, params)
    best = optimize(problem, params, top_k=1).best
    assert best.scenario == "3xc@gw6"
    g = best.gws[1]
    assert g.captain == star and g.xp == pytest.approx(10 * 2.0 + 3 * 15.0)
    no_chip = solve_plan(problem, params)
    # TC adds the captain's xP once more, decayed: 0.85 · 15; the terminal value drops by 3.
    assert best.objective - no_chip.objective == pytest.approx(0.85 * 15.0)
    assert no_chip.terminal_value - best.terminal_value == pytest.approx(3.0)
    replay(state, best, pool)
    params = chip_params(2, **{"3xc": 13.0})
    assert optimize(state_problem(state, pool, xp, params), params, top_k=1).best.scenario == (
        "none"
    )


def test_wildcard_rebuilds_for_free_and_keeps_the_fts() -> None:
    state, pool, xp, owned = team([1.0, 1.0], other_xp=[5.0, 5.0], ft=2)
    params = chip_params(2, wildcard=6.0)
    problem = state_problem(state, pool, xp, params)
    plans = optimize(problem, params, top_k=1)
    best = plans.best
    assert best.scenario == "wildcard@gw5"
    first = best.first
    assert first.chip == "wildcard" and first.hits == 0 and first.n_transfers >= 10
    assert best.gws[1].ft == 2  # chip_week_ft "retain": the 2 FTs carry over
    end = replay(state, best, pool)
    assert end.chips_used == ((WC1, 5),)
    # The roll plan doesn't waste the wildcard on no transfers.
    assert plans.roll.first.n_transfers == 0 and plans.roll.first.chip is None
    assert plans.best.gain_vs_roll > 0
    # Holding the wildcard is worth more than the rebuild → no chip, hits instead.
    params = chip_params(2, wildcard=200.0)
    assert optimize(state_problem(state, pool, xp, params), params, top_k=1).best.scenario == (
        "none"
    )


def test_wildcard_under_retain_plus_one() -> None:
    rules = replace(RULES, chip_week_ft="retain_plus_one")
    state, pool, xp, _ = team([1.0, 1.0], other_xp=[5.0, 5.0], ft=2)
    params = chip_params(2, wildcard=0.0)
    problem = PlanInput.from_context(state, pool, xp, rules, params)
    plan = solve_plan(problem, params, chips={0: "wildcard"})
    assert plan.first.hits == 0 and plan.gws[1].ft == 3
    replay(state, plan, pool, rules)


def test_free_hit_for_one_big_gw_reverts() -> None:
    """Owned players score 1; the market scores 8 in GW6 only, and one MID is worth buying
    for good in GW5. Some owned players were bought cheaper (selling price < price)."""
    pool_probe = grid_pool(range(1, 7), {1: 2, 2: 4, 3: 4, 4: 3})
    owned_probe = first_valid_squad(pool_probe)
    full = {c for c, n in Counter(k // 100 for k in owned_probe).items() if n >= RULES.team_limit}
    keeper = next(
        k
        for k in pool_probe["player_key"]
        if k not in owned_probe and k // 10 % 10 == 3 and k // 100 not in full
    )  # a MID from a club with room: one FT buys him
    purchase = {k: 40 for k in owned_probe[::4]}  # bought at 4.0: they rose, sell lower
    state, pool, xp, owned = team(
        [1.0, 1.0, 1.0],
        other_xp=[0.0, 8.0, 0.0],
        special={keeper: [10.0, 10.0, 10.0]},
        purchase=purchase,
    )
    params = chip_params(3, freehit=4.0)
    problem = state_problem(state, pool, xp, params)
    plans = optimize(problem, params, top_k=1)
    best = plans.best
    assert best.scenario == "freehit@gw6"
    g0, g1, g2 = best.gws
    assert g0.transfers_in == (keeper,) and g0.chip is None
    assert g1.chip == "freehit" and g1.hits == 0 and g1.n_transfers >= 10
    assert g1.ft == g2.ft == 1  # "retain": the FT used in GW5 regenerates once, FH keeps it
    # The squad comes back after the Free Hit, and so does the bank.
    assert squad_of(g2) == squad_of(g0) and g2.n_transfers == 0
    assert g2.bank == g0.bank
    # FH sales: never-sold owned players at their selling price, the GW5 buy at his price.
    price = pool.set_index("player_key")["price"]
    value = {h.player_key: selling_price(h, RULES.sell_on_fee) for h in state.holdings}
    value[keeper] = int(price[keeper])
    assert any(value[k] < price[k] for k in g1.transfers_out)
    assert g1.bank == g0.bank + sum(value[k] for k in g1.transfers_out) - sum(
        int(price[k]) for k in g1.transfers_in
    )
    end = replay(state, best, pool)
    assert end.chips_used == ((FH1, 6),)
    assert end.bank == g2.bank and set(end.player_keys) == squad_of(g2)


def test_free_hit_in_the_first_gw() -> None:
    state, pool, xp, owned = team([1.0, 1.0], other_xp=[8.0, 0.0])
    params = chip_params(2, freehit=4.0)
    problem = state_problem(state, pool, xp, params)
    plans = optimize(problem, params, top_k=2)
    best = plans.best
    assert best.scenario == "freehit@gw5"
    decision = best.decision()
    assert decision.chip == "freehit" and len(decision.transfers) >= 10
    gw_state, record = apply_decision(state, decision, pool, RULES)
    assert gw_state.freehit_backup == state.holdings
    after = next_state(gw_state, record, RULES, 6)
    assert after.holdings == state.holdings and after.bank == state.bank
    assert squad_of(best.gws[1]) == set(state.player_keys)
    # Plan #2 is another Free Hit squad (cuts act on the FH transfers); the roll plays no FH.
    assert plans.plans[1].scenario == "freehit@gw5"
    assert plans.plans[1].first.transfer_set != best.first.transfer_set
    assert plans.roll.first.n_transfers == 0 and plans.roll.first.chip is None
    replay(state, best, pool)


def test_a_chip_pair() -> None:
    state, pool, xp, owned = team([2.0, 6.0])
    star = owned[-1]
    xp.loc[xp["player_key"] == star, "xp"] = [15.0, 6.0]
    params = chip_params(2, bboost=1.0, **{"3xc": 1.0})
    best = optimize(state_problem(state, pool, xp, params), params, top_k=1).best
    assert best.scenario == "3xc@gw5+bboost@gw6"
    assert [g.chip for g in best.gws] == ["3xc", "bboost"]
    end = replay(state, best, pool)
    assert end.chips_used == ((TC1, 5), (BB1, 6))


def test_chip_on_gw1_keeps_the_gw1_rules() -> None:
    state, pool, xp, _ = team([2.0, 6.0], gw_index=1)
    params = chip_params(2, bboost=0.0)
    problem = state_problem(state, pool, xp, params)
    plan = solve_plan(problem, params, chips={0: "bboost"})
    assert plan.first.hits == 0 and plan.gws[1].ft == 1
    replay(state, plan, pool)


def state_problem(state, pool, xp, params, rules=RULES):
    return PlanInput.from_context(state, pool, xp, rules, params)


# --- top-k, roll, determinism --------------------------------------------------------------


def medium(seed: int, *, ft: int = 2, n_gws: int = 2):
    pool = grid_pool(range(1, 7), {1: 2, 2: 4, 3: 4, 4: 3}, seed=seed)
    xp = correlated_xp(pool, n_gws, seed)
    state = make_state(pool, first_valid_squad(pool), ft=ft)
    return state, pool, xp


@pytest.mark.parametrize("seed", [0, 1])
def test_top3_distinct_first_gw_transfers_and_roll(seed: int) -> None:
    state, pool, xp = medium(seed)
    params = replace(EXACT, horizon=2)
    problem = state_problem(state, pool, xp, params)
    result = optimize(problem, params, top_k=3, chips=False)
    assert len(result.plans) == 3
    sets = [p.first.transfer_set for p in result.plans]
    assert len(set(sets)) == 3
    totals = [p.total_objective for p in result.plans]
    assert all(a >= b - 1e-9 for a, b in zip(totals, totals[1:], strict=False))
    assert all(p.scenario == "none" for p in result.plans)
    assert result.roll.first.n_transfers == 0
    assert result.best.gain_vs_roll >= -1e-9
    for p in (*result.plans, result.roll):
        assert p.gain_vs_roll == pytest.approx(p.total_objective - result.roll.total_objective)
        apply_decision(state, p.decision(), pool, RULES)
    assert dict(result.scenario_objectives) == {"none": pytest.approx(totals[0])}


def test_optimize_with_chips_records_every_scenario() -> None:
    state, pool, xp = medium(2, n_gws=2)
    params = OptimizerParams(horizon=2)
    problem = state_problem(state, pool, xp, params)
    result = optimize(problem, params, top_k=3, scenario_search="all")
    assert list(result.scenario_objectives) == labels(problem, params)
    assert list(result.scenario_bounds) == labels(problem, params)
    assert result.search.n_solves == len(labels(problem, params))
    best_label = max(result.scenario_objectives, key=result.scenario_objectives.get)
    assert result.best.total_objective == pytest.approx(result.scenario_objectives[best_label])
    assert all(p.scenario == result.best.scenario for p in result.plans)
    assert result.best.gain_vs_roll >= -1e-9
    assert len({p.first.transfer_set for p in result.plans}) == len(result.plans)
    apply_decision(state, result.decision(), pool, RULES)


def test_roll_is_plan_one_when_plan_one_rolls() -> None:
    state, pool, xp = medium(3, n_gws=2)
    xp = xp.assign(xp=0.0)  # nothing worth a transfer
    params = replace(EXACT, horizon=2)
    result = optimize(state_problem(state, pool, xp, params), params, top_k=2, chips=False)
    assert result.best.first.n_transfers == 0
    assert result.roll.gws == result.best.gws and result.best.gain_vs_roll == 0
    assert result.plans[1].first.n_transfers > 0 and result.plans[1].gain_vs_roll <= 1e-9


def test_optimize_is_deterministic() -> None:
    state, pool, xp = medium(4, n_gws=1)
    params = OptimizerParams(horizon=1)
    a = state_problem(state, pool, xp, params)
    b = state_problem(
        state, pool.sample(frac=1, random_state=3), xp.sample(frac=1, random_state=4), params
    )
    p, q = optimize(a, params), optimize(b, params)
    assert [x.gws for x in p.plans] == [x.gws for x in q.plans]
    assert p.roll.gws == q.roll.gws
    assert dict(p.scenario_objectives) == dict(q.scenario_objectives)


def test_top_k_validation_and_no_roll() -> None:
    state, pool, xp = medium(5, n_gws=1)
    params = replace(EXACT, horizon=1)
    problem = state_problem(state, pool, xp, params)
    with pytest.raises(ValueError, match="top_k"):
        optimize(problem, params, top_k=0)
    result = optimize(problem, params, top_k=1, chips=False, roll=False)
    assert result.roll is None and result.best.gain_vs_roll is None
