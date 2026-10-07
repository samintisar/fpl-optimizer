"""The MILP planner without chips (fplopt.optimize; Phase 4 plan, Task 1)."""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import replace

import pytest

pytest.importorskip("pulp")
pytest.importorskip("highspy")

from optimize_fixtures import (  # noqa: E402
    RULES,
    best_lineup_value,
    correlated_xp,
    first_valid_squad,
    grid_pool,
    key,
    make_pool,
    make_state,
    random_xp,
    xp_frame,
)

from fplopt.backtest.state import (  # noqa: E402
    Decision,
    Holding,
    InvalidDecision,
    Transfer,
    apply_decision,
    next_state,
    selling_price,
)
from fplopt.optimize import OptimizerParams, PlanInput, optimize  # noqa: E402
from fplopt.optimize.model import solve_plan  # noqa: E402

EXACT = OptimizerParams(mip_gap=0.0, prune_n=None, prune_dominated=False)


def plan_for(state, pool, xp, params=EXACT, rules=RULES, **kwargs):
    problem = PlanInput.from_context(state, pool, xp, rules, params)
    return problem, solve_plan(problem, params, **kwargs)


def replay(state, plan, pool, rules=RULES):
    """Run every GW of the plan through apply_decision/next_state; assert the plan's
    per-GW ft (before), hits, transfers and bank (after) match the rules engine."""
    for i, g in enumerate(plan.gws):
        assert state.free_transfers == g.ft or state.gw_index == 1, (i, state.free_transfers, g)
        gw_state, record = apply_decision(state, g.decision(), pool, rules)
        assert record.hits == g.hits, (i, record, g)
        assert record.n_transfers == g.n_transfers
        assert gw_state.bank == g.bank, (i, gw_state.bank, g.bank)
        nxt = plan.gws[i + 1].gw_index if i + 1 < len(plan.gws) else g.gw_index + 1
        state = next_state(gw_state, record, rules, nxt)
    return state


# --- brute force ------------------------------------------------------------------------------

# 15 owned players (3 per club 1-5) plus 6 extras: clubs 1 and 6, so buying a club-1
# extra needs a club-1 sale (the club cap binds).
OWNED = (
    [key(c, 1, 0) for c in (1, 2)]
    + [key(c, 2, 0) for c in (1, 2, 3, 4, 5)]
    + [key(c, 3, 0) for c in (1, 3, 4, 5, 2)]
    + [key(c, 4, 0) for c in (3, 4, 5)]
)
EXTRAS = [key(6, 1, 0), key(1, 2, 1), key(6, 2, 1), key(6, 3, 1), key(1, 3, 1), key(6, 4, 1)]


def tiny_pool(seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for k in OWNED + EXTRAS:
        et, club = k // 10 % 10, k // 100
        rows.append((k, et, club, 40 + 5 * et + int(rng.integers(0, 25))))
    pool = make_pool(rows)
    values = {int(k): [round(float(rng.uniform(0, 9)), 2)] for k in pool["player_key"]}
    return pool, xp_frame(values)


def brute_force_one_gw(state, pool, xp, params, rules=RULES):
    """Max objective over every transfer set (any size, the pool is tiny), 1-GW horizon."""
    et = dict(zip(pool["player_key"], pool["element_type"], strict=True))
    club = dict(zip(pool["player_key"], pool["team_key"], strict=True))
    price = dict(zip(pool["player_key"], pool["price"], strict=True))
    xs = dict(zip(xp["player_key"], xp["xp"], strict=True))
    owned = {h.player_key: h for h in state.holdings}
    sell = {k: selling_price(h, rules.sell_on_fee) for k, h in owned.items()}
    extras = [k for k in pool["player_key"] if k not in owned]
    best = -float("inf")
    hit = rules.hit_cost + params.hit_margin
    for r in range(len(extras) + 1):
        for ins in itertools.combinations(extras, r):
            need = Counter(et[k] for k in ins)
            choices = [
                itertools.combinations(sorted(k for k in owned if et[k] == p), n)
                for p, n in sorted(need.items())
            ]
            for outs_by_pos in itertools.product(*choices):
                outs = [k for group in outs_by_pos for k in group]
                squad = [k for k in owned if k not in outs] + list(ins)
                clubs = Counter(club[k] for k in squad)
                if any(clubs[club[k]] > rules.team_limit for k in ins):
                    continue
                bank = state.bank + sum(sell[k] for k in outs) - sum(price[k] for k in ins)
                if bank < 0:
                    continue
                n = len(ins)
                hits = 0 if state.gw_index == 1 else max(n - state.free_transfers, 0)
                value = (
                    best_lineup_value(squad, xs, et, params.bench_weights)
                    + params.itb_value * bank / 10
                    + params.ft_state_value(max(state.free_transfers, 0))
                    - hit * hits
                )
                best = max(best, value)
    return best


@pytest.mark.parametrize(
    ("seed", "ft", "margin"),
    [(0, 1, 0.0), (1, 1, 0.0), (2, 2, 0.0), (3, 1, 1.5), (4, 3, 0.0), (5, 1, 0.0)],
)
def test_one_gw_matches_brute_force(seed: int, ft: int, margin: float) -> None:
    pool, xp = tiny_pool(seed)
    state = make_state(pool, OWNED, ft=ft, bank=15, purchase={k: 45 for k in OWNED[::3]})
    params = replace(EXACT, hit_margin=margin)
    problem, plan = plan_for(state, pool, xp, params)
    expected = brute_force_one_gw(problem.state, pool, xp, params)
    assert plan.objective == pytest.approx(expected, abs=1e-6)
    apply_decision(state, plan.decision(), pool, RULES)


def test_gw1_squad_choice_matches_brute_force() -> None:
    """gw_index 1: unlimited free transfers, so the plan picks the best squad outright."""
    import numpy as np

    rng = np.random.default_rng(7)
    # 3 GK, 6 DEF, 6 MID, 4 FWD over 6 clubs (the club cap binds).
    rows, n = [], {1: 3, 2: 6, 3: 6, 4: 4}
    for et, count in n.items():
        for i in range(count):
            club = 1 + (i + 2 * et) % 6
            rows.append((key(club, et, i), et, club, 35 + 5 * et + int(rng.integers(0, 30))))
    pool = make_pool(rows)
    values = {int(k): [round(float(rng.uniform(0, 9)), 2)] for k in pool["player_key"]}
    xp = xp_frame(values, gw_index=1)
    owned = first_valid_squad(pool)
    state = make_state(pool, owned, gw_index=1, ft=1)
    problem, plan = plan_for(state, pool, xp)

    et = dict(zip(pool["player_key"], pool["element_type"], strict=True))
    club = dict(zip(pool["player_key"], pool["team_key"], strict=True))
    price = dict(zip(pool["player_key"], pool["price"], strict=True))
    by_pos = {p: sorted(k for k in et if et[k] == p) for p in n}
    budget = state.bank + sum(price[k] for k in owned)
    best = -float("inf")
    for parts in itertools.product(
        *(itertools.combinations(by_pos[p], RULES.squad_select[p]) for p in sorted(n))
    ):
        squad = [k for part in parts for k in part]
        if max(Counter(club[k] for k in squad).values()) > RULES.team_limit:
            continue
        bank = budget - sum(price[k] for k in squad)
        if bank < 0:
            continue
        value = best_lineup_value(squad, values_flat(values), et, EXACT.bench_weights)
        best = max(best, value + EXACT.itb_value * bank / 10 + EXACT.ft_state_value(1))
    assert plan.objective == pytest.approx(best, abs=1e-6)
    assert plan.first.hits == 0
    apply_decision(state, plan.decision(), pool, RULES)


def values_flat(values):
    return {k: v[0] for k, v in values.items()}


# --- decisions and replay -----------------------------------------------------------------


def medium_instance(seed: int, *, ft: int = 1, gw_index: int = 5, n_gws: int = 3):
    pool = grid_pool(range(1, 9), {1: 2, 2: 4, 3: 4, 4: 3}, seed=seed)
    xp = correlated_xp(pool, n_gws, seed, gw_index=gw_index)
    state = make_state(pool, first_valid_squad(pool), ft=ft, gw_index=gw_index)
    return state, pool, xp


@pytest.mark.parametrize("seed", [0, 1])
def test_multi_gw_plan_replays_through_the_rules_engine(seed: int) -> None:
    state, pool, xp = medium_instance(seed, ft=1, n_gws=4)
    params = OptimizerParams(horizon=4)
    _, plan = plan_for(state, pool, xp, params)
    assert len(plan.gws) == 4
    assert plan.status == "Optimal"
    assert any(g.n_transfers for g in plan.gws)
    replay(state, plan, pool)


def test_gw1_plan_replays_with_one_ft_after() -> None:
    state, pool, xp = medium_instance(3, ft=1, gw_index=1, n_gws=3)
    _, plan = plan_for(state, pool, xp, OptimizerParams(horizon=3))
    assert plan.first.hits == 0 and plan.first.n_transfers > 1
    assert plan.gws[1].ft == 1
    replay(state, plan, pool)


def test_ft_topups_follow_the_rules() -> None:
    rules = replace(RULES, ft_topups=((6, 2),))
    state, pool, xp = medium_instance(4, ft=1, n_gws=3)
    xp = xp.assign(xp=0.0)  # nothing worth a transfer: the FTs just bank
    _, plan = plan_for(state, pool, xp, OptimizerParams(horizon=3), rules=rules)
    assert [g.ft for g in plan.gws] == [1, 4, 5]
    assert all(g.n_transfers == 0 for g in plan.gws)
    replay(state, plan, pool, rules)


def test_first_gw_decision_passes_apply_decision_and_matches_best_lineup() -> None:
    state, pool, xp = medium_instance(2, ft=2)
    _, plan = plan_for(state, pool, xp, OptimizerParams())
    gw_state, _ = apply_decision(state, plan.decision(), pool, RULES)
    lineup = plan.first.lineup
    target = xp[xp["horizon"] == 0]
    xs = dict(zip(target["player_key"], target["xp"], strict=True))
    assert xs[lineup.captain] == max(xs[k] for k in lineup.starters)
    others = [k for k in lineup.starters if k != lineup.captain]
    assert lineup.vice == min(others, key=lambda k: (-xs[k], k))
    assert set(lineup.starters) | set(lineup.bench) == set(gw_state.player_keys)


# --- money --------------------------------------------------------------------------------


def churn_instance():
    """A owned (bought 50, now 60: sells for 55 first); C costs 50. A scores in GW 1 only,
    C in GWs 0 and 2, so the plan churns A → C → A → C, all on free transfers."""
    clubs = range(1, 6)
    rows = [
        (key(c, et, i), et, c, 45)
        for c in clubs
        for et, n in {1: 1, 2: 1, 3: 1, 4: 1}.items()
        for i in range(n)
    ]
    rows += [(key(c, 3, 1), 3, c, 45) for c in clubs]  # more MIDs
    rows += [(key(c, 2, 1), 2, c, 45) for c in clubs]
    rows += [(key(c, 4, 1), 4, c, 45) for c in (1, 2)]
    rows += [(key(c, 1, 1), 1, c, 45) for c in (1,)]
    a, c_key = key(6, 3, 0), key(7, 3, 0)
    rows += [(a, 3, 6, 60), (c_key, 3, 7, 50)]
    pool = make_pool(rows)
    owned = [key(1, 1, 0), key(2, 1, 0)]
    owned += [key(c, 2, 0) for c in clubs]
    owned += [key(c, 3, 0) for c in (1, 2, 3, 4)] + [a]
    owned += [key(c, 4, 0) for c in (1, 2, 3)]
    values = {int(k): [0.5, 0.5, 0.5] for k in pool["player_key"]}
    values.update({k: [1.0, 1.0, 1.0] for k in owned})  # no other move is worth making
    values[a] = [0.0, 20.0, 0.0]
    values[c_key] = [10.0, 0.0, 10.0]
    xp = xp_frame(values)
    state = make_state(pool, owned, ft=2, bank=5, purchase={a: 50})
    return state, pool, xp, a, c_key


def test_first_sale_at_selling_price_later_sales_at_buy_price() -> None:
    state, pool, xp, a, c = churn_instance()
    params = replace(EXACT, horizon=3, ft_value={}, itb_value=0.0)
    problem, plan = plan_for(state, pool, xp, params)
    assert problem.player_by_key[a].sell_price == 55
    moves = [(g.transfers_out, g.transfers_in) for g in plan.gws]
    assert moves == [((a,), (c,)), ((c,), (a,)), ((a,), (c,))]
    # first sale +55 −50, then −60 +50 (C at its buy price), then +60 (A at buy price) −50.
    assert [g.bank for g in plan.gws] == [10, 0, 10]
    replay(state, plan, pool)


def test_departed_player_is_sold_at_last_price_and_never_bought() -> None:
    pool = grid_pool(range(1, 7), {1: 2, 2: 4, 3: 4, 4: 2})
    owned = first_valid_squad(pool)
    gone_def = next(k for k in owned if k // 10 % 10 == 2)
    owned.remove(gone_def)
    departed = Holding(9921, 2, 9, purchase_price=56, price=60)  # sells for 58
    values = {int(k): [2.0, 2.0] for k in pool["player_key"]}
    values[key(6, 2, 3)] = [6.0, 6.0]
    state = make_state(pool, owned, ft=1, extra=[departed], bank=100)
    xp = xp_frame(values)
    problem, plan = plan_for(state, pool, xp, replace(EXACT, horizon=2))
    player = problem.player_by_key[9921]
    assert (player.owned, player.buyable, player.sell_price, player.xp) == (
        True,
        False,
        58,
        (0.0, 0.0),
    )
    assert plan.first.transfers_out == (9921,)
    assert all(9921 not in g.transfers_in for g in plan.gws)
    assert plan.first.bank == 100 + 58 - 50
    replay(state, plan, pool)


# --- free transfers and hits -------------------------------------------------------------


def upgrade_instance(*, ft: int = 1, n_gws: int = 1, gains=(6.0, 5.0)):
    """All owned score 2 (a FWD 20, the captain); two non-owned MIDs score 2 + gains."""
    pool = grid_pool(range(1, 9), {1: 2, 2: 5, 3: 5, 4: 3})
    owned = first_valid_squad(pool)
    values = {int(k): [0.0] * n_gws for k in pool["player_key"]}
    for k in owned:
        values[k] = [2.0] * n_gws
    captain = next(k for k in owned if k // 10 % 10 == 4)
    values[captain] = [20.0] * n_gws
    ups = [key(8, 3, 0), key(8, 3, 1)]  # club 8 has room
    assert not set(ups) & set(owned) and all(k // 100 != 8 for k in owned)
    for k, gain in zip(ups, gains, strict=True):
        values[k] = [2.0 + gain] * n_gws
    state = make_state(pool, owned, ft=ft)
    return state, pool, xp_frame(values), ups


@pytest.mark.parametrize(("margin", "n_transfers", "hits"), [(0.0, 2, 1), (1.5, 1, 0)])
def test_hit_margin(margin: float, n_transfers: int, hits: int) -> None:
    state, pool, xp, ups = upgrade_instance()
    _, plan = plan_for(state, pool, xp, replace(EXACT, horizon=1, hit_margin=margin))
    assert (plan.first.n_transfers, plan.first.hits) == (n_transfers, hits)
    assert ups[0] in plan.first.transfers_in
    replay(state, plan, pool)


def test_high_ft_value_rolls_zero_ft_value_transfers() -> None:
    state, pool, xp, ups = upgrade_instance(n_gws=2, gains=(1.0, 0.5))
    _, rolled = plan_for(state, pool, xp, replace(EXACT, horizon=2, ft_value={2: 10.0}))
    assert rolled.first.n_transfers == 0 and rolled.gws[1].ft == 2
    _, moved = plan_for(state, pool, xp, replace(EXACT, horizon=2, ft_value={}))
    assert moved.first.transfers_in == (ups[0],)


def test_fts_bank_to_the_cap() -> None:
    state, pool, xp, _ = upgrade_instance(ft=4, n_gws=3, gains=(0.0, 0.0))
    _, plan = plan_for(state, pool, xp, replace(EXACT, horizon=3))
    assert [g.ft for g in plan.gws] == [4, 5, 5]
    assert all(g.n_transfers == 0 for g in plan.gws)
    # The gain in FT value (V(5) − V(4) = 1.1) is credited in GW 1, decayed.
    assert plan.gws[1].objective - 0.85 * (plan.gws[1].xp + plan.gws[1].bench_xp) == pytest.approx(
        0.85 * (1.1 + EXACT.itb_value * plan.gws[1].bank / 10)
    )


def test_hits_never_buy_free_transfers() -> None:
    """With a huge FT value, paying a hit for a transfer you had an FT for would raise next
    GW's FTs under a ≤ formulation; the exact dynamics forbid it."""
    state, pool, xp, _ = upgrade_instance(ft=2, n_gws=2, gains=(0.0, 0.0))
    _, plan = plan_for(state, pool, xp, replace(EXACT, horizon=2, ft_value={3: 50.0}))
    assert plan.first.hits == 0 and plan.gws[1].ft == 3
    replay(state, plan, pool)


# --- club cap -----------------------------------------------------------------------------


def moved_club_instance(star_xp: float):
    """Club 1 has 4 owned players after a club change; a club-1 star is on the market."""
    pool = grid_pool(range(1, 9), {1: 2, 2: 5, 3: 5, 4: 3})
    owned = first_valid_squad(pool)
    clubs = Counter(k // 100 for k in owned)
    assert clubs[1] == 3
    mover = next(k for k in owned if k // 100 != 1 and k // 10 % 10 == 3)
    pool.loc[pool["player_key"] == mover, "team_key"] = 1  # moved to club 1
    star = key(1, 3, 4)
    assert star not in owned
    values = {int(k): [1.0] for k in pool["player_key"]}
    values.update({k: [2.0] for k in owned})
    values[star] = [star_xp]
    state = make_state(pool, owned, ft=5, bank=50)
    return state, pool, xp_frame(values), star, mover


def test_club_over_the_cap_must_shed_two_to_buy_one() -> None:
    state, pool, xp, star, mover = moved_club_instance(40.0)
    _, plan = plan_for(state, pool, xp, replace(EXACT, horizon=1))
    assert star in plan.first.transfers_in
    gw_state, _ = apply_decision(state, plan.decision(), pool, RULES)
    assert Counter(h.team_key for h in gw_state.holdings)[1] <= RULES.team_limit
    # Selling just one club-1 player for the star is what the rules forbid:
    bad = Decision((Transfer(mover, star),), plan.first.lineup)
    with pytest.raises(InvalidDecision, match="club 1"):
        apply_decision(state, bad, pool, RULES)


def test_club_over_the_cap_is_tolerated_without_buys() -> None:
    state, pool, xp, _, _ = moved_club_instance(0.0)
    _, plan = plan_for(state, pool, xp, replace(EXACT, horizon=1))
    assert plan.first.n_transfers == 0
    apply_decision(state, plan.decision(), pool, RULES)


# --- horizon, hooks, determinism ----------------------------------------------------------


def test_horizon_shortens_near_season_end() -> None:
    state, pool, xp = medium_instance(5, gw_index=36, n_gws=3)
    problem, plan = plan_for(state, pool, xp, OptimizerParams(horizon=6))
    assert [g.gw_index for g in problem.gws] == [36, 37, 38]
    assert len(plan.gws) == 3
    short = PlanInput.from_context(state, pool, xp, RULES, OptimizerParams(horizon=2))
    assert [g.horizon for g in short.gws] == [0, 1]


def test_xp_frame_must_start_at_the_states_gw() -> None:
    state, pool, xp = medium_instance(5, gw_index=7)
    with pytest.raises(ValueError, match="gw_index 8"):
        PlanInput.from_context(replace(state, gw_index=8), pool, xp, RULES, OptimizerParams())


def test_missing_xp_counts_zero() -> None:
    state, pool, xp = medium_instance(6, n_gws=2)
    dropped = int(pool["player_key"].iloc[-1])
    xp = xp[xp["player_key"] != dropped]
    xp.loc[xp.index[0], "xp"] = float("nan")
    problem = PlanInput.from_context(state, pool, xp, RULES, EXACT)
    assert problem.player_by_key[dropped].xp == (0.0, 0.0)
    assert problem.players[0].xp[0] == 0.0


def test_fix_and_exclude_first_gw() -> None:
    state, pool, xp = medium_instance(7, ft=2)
    params = OptimizerParams()
    problem = PlanInput.from_context(state, pool, xp, RULES, params)
    best = solve_plan(problem, params)
    assert best.first.n_transfers > 0
    roll = solve_plan(problem, params, fix_first_gw=(frozenset(), frozenset()))
    assert roll.first.n_transfers == 0
    assert roll.objective <= best.objective + 1e-6
    first = (frozenset(best.first.transfers_out), frozenset(best.first.transfers_in))
    second = solve_plan(problem, params, exclude_first_gw=[first])
    assert (frozenset(second.first.transfers_out), frozenset(second.first.transfers_in)) != first
    assert second.objective <= best.objective * (1 + params.mip_gap) + 1e-6
    with pytest.raises(NotImplementedError):
        solve_plan(problem, params, chips={0: "bboost"})


def test_same_input_same_plan() -> None:
    state, pool, xp = medium_instance(8, ft=1, n_gws=3)
    params = OptimizerParams()
    a = PlanInput.from_context(state, pool, xp, RULES, params)
    b = PlanInput.from_context(
        state, pool.sample(frac=1, random_state=1), xp.sample(frac=1, random_state=2), RULES, params
    )
    assert a == b
    p, q = optimize(a, params), optimize(b, params)
    assert p.gws == q.gws and p.objective == q.objective


def test_plan_reports() -> None:
    state, pool, xp = medium_instance(9, ft=1, n_gws=2)
    _, plan = plan_for(state, pool, xp, OptimizerParams(horizon=2))
    assert plan.build_seconds > 0 and plan.solve_seconds > 0
    assert 0 <= plan.mip_gap <= 0.005 + 1e-9
    assert plan.objective == pytest.approx(sum(g.objective for g in plan.gws))
    assert plan.total_xp == pytest.approx(sum(g.xp for g in plan.gws))
    for g in plan.gws:
        assert g.chip is None and len(g.starters) == 11 and len(g.bench) == 4
        assert g.vice != g.captain and g.captain in g.starters and g.vice in g.starters


def test_random_xp_instance_replays() -> None:
    """A churny instance (independent xP per GW) still replays exactly."""
    pool = grid_pool(range(1, 7), {1: 2, 2: 4, 3: 4, 4: 3}, seed=11)
    xp = random_xp(pool, 2, seed=11)
    state = make_state(pool, first_valid_squad(pool), ft=1)
    _, plan = plan_for(state, pool, xp, OptimizerParams(horizon=2))
    replay(state, plan, pool)
