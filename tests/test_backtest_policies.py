"""Decision policies (fplopt.backtest.policies): best_lineup, roll, greedy."""

from __future__ import annotations

import itertools
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.backtest.gw_score import validate_lineup
from fplopt.backtest.policies import (
    DecisionContext,
    GreedyPolicy,
    OptimizerPolicy,
    RollPolicy,
    best_lineup,
    horizon_xp,
)
from fplopt.backtest.rules import backtest_rules, load_rules
from fplopt.backtest.start_states import random_state, template_state
from fplopt.backtest.state import (
    Holding,
    SquadState,
    Transfer,
    apply_decision,
    next_state,
    refresh,
)
from fplopt.features.baseline import player_pool
from fplopt.features.store import DataStore
from fplopt.models import MODELS
from fplopt.models.fitted import FittedModel
from fplopt.optimize import OptimizerParams

RULES = load_rules("2026-27")

# --- hand-made pool (as tests/test_backtest_state.py) -------------------------------------
# key = club * 100 + element_type * 10 + i; 8 clubs, each with 2 GK, 5 DEF, 5 MID, 3 FWD.
BASE_PRICE = {1: 45, 2: 50, 3: 70, 4: 75}
N_PER_POS = {1: 2, 2: 5, 3: 5, 4: 3}
CLUBS = range(1, 9)


def make_pool(overrides: Mapping[int, int] | None = None) -> pd.DataFrame:
    rows = [
        (club * 100 + et * 10 + i, et, club, BASE_PRICE[et])
        for club in CLUBS
        for et, n in N_PER_POS.items()
        for i in range(n)
    ]
    pool = pd.DataFrame(rows, columns=["player_key", "element_type", "team_key", "price"])
    for key, price in (overrides or {}).items():
        pool.loc[pool["player_key"] == key, "price"] = price
    return pool.astype("int64")


POOL = make_pool()
# 3 players from each of clubs 1-5: GK 2, DEF 5, MID 5, FWD 3. Cost 915 → bank 85.
BASE_KEYS = (110, 120, 121, 210, 220, 221, 320, 330, 331, 430, 431, 440, 530, 540, 541)


def make_state(keys=BASE_KEYS, bank=85, ft=1, gw_index=5, pool=POOL) -> SquadState:
    rows = pool.set_index("player_key")
    holdings = tuple(
        Holding(k, int(rows.at[k, "element_type"]), int(rows.at[k, "team_key"]), p, p)
        for k in keys
        for p in [int(rows.at[k, "price"])]
    )
    return SquadState(2023, gw_index, holdings, bank, ft)


def xp_frame(values: Mapping[int, float] | None = None, pool=POOL, per_horizon=None):
    """xP frame for every pool player over 6 horizons: `values` per player (default 1.0),
    the same in every GW unless `per_horizon` maps key -> list of 6 values."""
    values = values or {}
    per_horizon = per_horizon or {}
    rows = []
    for key in pool["player_key"]:
        for h in range(6):
            xp = per_horizon[key][h] if key in per_horizon else values.get(key, 1.0)
            rows.append((key, 10 + h, 10 + h, h, float(xp)))
    frame = pd.DataFrame(rows, columns=["player_key", "gw", "gw_index", "horizon", "xp"])
    return frame.astype({"player_key": "int64", "gw": "int64", "gw_index": "int64"})


def ctx(state, xp, pool=POOL, rules=RULES):
    return DecisionContext(view=None, state=state, rules=rules, pool=pool, xp=xp)


def hxp_weight(decay=0.85, horizon=6):
    return sum(decay**h for h in range(horizon))


# --- best_lineup --------------------------------------------------------------------------


def all_valid_xis(squad: pd.DataFrame, rules=RULES):
    positions = dict(zip(squad["player_key"], squad["element_type"], strict=True))
    for xi in itertools.combinations(sorted(positions), rules.squad_play):
        counts = Counter(positions[k] for k in xi)
        if all(rules.play_min[et] <= counts[et] <= rules.play_max[et] for et in rules.play_min):
            yield xi


def random_squad(rng: random.Random) -> pd.DataFrame:
    keys = rng.sample(range(1, 1000), 15)
    ets = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3
    return pd.DataFrame({"player_key": keys, "element_type": ets}, dtype="int64")


@pytest.mark.parametrize("seed", range(25))
def test_best_lineup_is_optimal_against_brute_force(seed):
    rng = random.Random(seed)
    squad = random_squad(rng)
    # Small integers make ties common; some players have no xP at all (count as 0).
    xp = {int(k): float(rng.randint(0, 6)) for k in squad["player_key"] if rng.random() > 0.1}
    lineup = best_lineup(squad, xp, RULES)
    positions = dict(zip(squad["player_key"], squad["element_type"], strict=True))
    validate_lineup(lineup, positions, RULES)
    best = max(sum(xp.get(k, 0.0) for k in xi) for xi in all_valid_xis(squad))
    assert sum(xp.get(k, 0.0) for k in lineup.starters) == best
    # Captain/vice: the two best starters, ties by key.
    ranked = sorted(lineup.starters, key=lambda k: (-xp.get(k, 0.0), k))
    assert (lineup.captain, lineup.vice) == tuple(ranked[:2])
    # Bench: the GK first, then outfield by xP descending (ties by key).
    assert positions[lineup.bench[0]] == 1
    outfield = list(lineup.bench[1:])
    assert outfield == sorted(outfield, key=lambda k: (-xp.get(k, 0.0), k))


def test_best_lineup_breaks_ties_by_player_key_and_is_deterministic():
    squad = random_squad(random.Random(3))
    xp = {}  # everyone 0
    lineup = best_lineup(squad, xp, RULES)
    assert lineup == best_lineup(squad.iloc[::-1].reset_index(drop=True), xp, RULES)
    by_pos = squad.sort_values("player_key").groupby("element_type")["player_key"].apply(list)
    gk = by_pos[1][0]
    assert gk in lineup.starters and lineup.bench[0] == by_pos[1][1]
    # Minimums by key (3 DEF, 2 MID, 1 FWD), then the 4 smallest remaining outfield keys.
    starters = {gk, *by_pos[2][:3], *by_pos[3][:2], *by_pos[4][:1]}
    rest = sorted(set(squad["player_key"]) - starters - set(by_pos[1]))
    assert set(lineup.starters) == starters | set(rest[:4])
    assert lineup.captain == min(lineup.starters) and lineup.vice == sorted(lineup.starters)[1]


def test_best_lineup_respects_the_formation_maximums():
    # 5 great DEF, poor everyone else: all 5 DEF start (max 5), never a second GK.
    squad = random_squad(random.Random(1))
    xp = {int(k): (10.0 if et == 2 else 9.0 if et == 1 else 0.0) for k, et in squad.values}
    lineup = best_lineup(squad, xp, RULES)
    positions = dict(squad.values)
    formation = Counter(positions[k] for k in lineup.starters)
    assert formation[1] == 1 and formation[2] == 5


def test_best_lineup_ignores_nan_xp():
    squad = random_squad(random.Random(2))
    xp = {int(k): float("nan") for k in squad["player_key"]}
    assert best_lineup(squad, xp, RULES) == best_lineup(squad, {}, RULES)


# --- horizon xP -------------------------------------------------------------------------------


def test_horizon_xp_discounts_and_truncates():
    xp = xp_frame(per_horizon={110: [1, 2, 3, 4, 5, 6]})
    hxp = horizon_xp(xp, horizon=3, decay=0.5)
    assert hxp[110] == pytest.approx(1 + 2 * 0.5 + 3 * 0.25)
    assert hxp[111] == pytest.approx(1.75)
    assert horizon_xp(xp, horizon=6, decay=1.0)[110] == pytest.approx(21)


# --- roll -------------------------------------------------------------------------------------


def test_roll_never_transfers_and_plays_the_best_lineup():
    state = make_state()
    xp = xp_frame({121: 9.0, 330: 8.0, 811: 50.0})  # 811 not held
    decision = RollPolicy().decide(ctx(state, xp))
    assert decision.transfers == () and decision.chip is None
    assert (decision.lineup.captain, decision.lineup.vice) == (121, 330)
    apply_decision(state, decision, POOL, RULES)


# --- greedy -----------------------------------------------------------------------------------


def test_greedy_picks_the_max_gain_affordable_transfer():
    # Bank 85: a DEF (50) can be replaced by any DEF up to 135. 620 is the best affordable;
    # 720 is better but costs 160 > 85 + 50.
    pool = make_pool({720: 160})
    state = make_state(pool=pool)
    xp = xp_frame({620: 6.0, 720: 9.0, 621: 5.0}, pool=pool)
    decision = GreedyPolicy().decide(ctx(state, xp, pool=pool))
    # Sell the DEF with the smallest xP; all held DEFs have 1.0 → smallest key (120).
    assert decision.transfers == (Transfer(120, 620),)
    gw_state, record = apply_decision(state, decision, pool, RULES)
    assert record.hits == 0 and gw_state.bank == 85
    assert decision.lineup.captain == 620


def test_greedy_respects_the_threshold():
    gain = 1.0 / hxp_weight()  # per-GW xP difference giving a horizon gain of exactly 1.0
    state = make_state()
    exactly = xp_frame({620: 1.0 + gain})
    assert GreedyPolicy(threshold=1.0).decide(ctx(state, exactly)).transfers == ()
    above = xp_frame({620: 1.0 + gain * 1.01})
    assert GreedyPolicy(threshold=1.0).decide(ctx(state, above)).transfers == (Transfer(120, 620),)
    assert GreedyPolicy(threshold=2.0).decide(ctx(state, above)).transfers == ()


def test_greedy_needs_a_free_transfer_and_never_takes_a_hit():
    xp = xp_frame({620: 9.0, 621: 8.0, 622: 7.0})
    assert GreedyPolicy().decide(ctx(make_state(ft=0), xp)).transfers == ()
    two = GreedyPolicy(max_transfers=2)
    assert len(two.decide(ctx(make_state(ft=1), xp)).transfers) == 1
    decision = two.decide(ctx(make_state(ft=3), xp))
    assert decision.transfers == (Transfer(120, 620), Transfer(121, 621))
    _, record = apply_decision(make_state(ft=3), decision, POOL, RULES)
    assert record.hits == 0
    assert GreedyPolicy(max_transfers=0).decide(ctx(make_state(ft=5), xp)).transfers == ()


def test_greedy_respects_the_budget():
    # Bank 0: only players no dearer than the one sold (DEFs cost 50).
    pool = make_pool({620: 51, 621: 50})
    state = make_state(bank=0, pool=pool)
    xp = xp_frame({620: 9.0, 621: 5.0}, pool=pool)
    decision = GreedyPolicy().decide(ctx(state, xp, pool=pool))
    assert decision.transfers == (Transfer(120, 621),)
    apply_decision(state, decision, pool, RULES)


def test_greedy_uses_the_selling_price():
    # The DEF 320 rose 50 -> 60: he sells for 55 (half the rise), so bank 0 + 55 buys 55
    # but not 56.
    pool = make_pool({320: 60, 620: 56, 621: 55})
    rows = POOL.set_index("player_key")
    holdings = [
        Holding(k, int(rows.at[k, "element_type"]), int(rows.at[k, "team_key"]), p, p)
        for k in BASE_KEYS
        for p in [int(rows.at[k, "price"])]
    ]
    state = SquadState(2023, 5, tuple(holdings), 0, 1)
    # 320 has the lowest xP so he is the one sold.
    xp = xp_frame({320: 0.0, 620: 9.0, 621: 5.0}, pool=pool)
    decision = GreedyPolicy().decide(ctx(state, xp, pool=pool))
    assert decision.transfers == (Transfer(320, 621),)
    gw_state, _ = apply_decision(state, decision, pool, RULES)
    assert gw_state.bank == 0


def test_greedy_respects_the_club_cap():
    # Clubs 1-5 have 3 players each; club 1 holds GK 110 and DEFs 120, 121 (no MID).
    state = make_state()
    xp = xp_frame({130: 9.0, 630: 3.0})
    decision = GreedyPolicy().decide(ctx(state, xp))
    # 130 (club 1, MID) would make 4 from club 1 for any MID out: not allowed. 630 instead.
    assert decision.transfers == (Transfer(330, 630),)
    # A club-1 DEF (122) can replace a club-1 DEF (120): the club stays at 3.
    xp = xp_frame({122: 9.0, 120: 0.0})
    decision = GreedyPolicy().decide(ctx(state, xp))
    assert decision.transfers == (Transfer(120, 122),)
    apply_decision(state, decision, POOL, RULES)


def test_greedy_tolerates_a_club_already_over_the_cap():
    # 4 club-1 players (a held player moved club): transfers that don't add to club 1 are
    # fine; buying another club-1 player is not.
    pool = POOL.copy()
    pool.loc[pool["player_key"] == 220, "team_key"] = 1
    state = make_state()
    xp = xp_frame({131: 9.0, 620: 5.0}, pool=pool)
    decision = GreedyPolicy().decide(ctx(state, xp, pool=pool))
    assert decision.transfers == (Transfer(120, 620),)
    apply_decision(state, decision, pool, RULES)


def test_greedy_breaks_ties_by_in_key_then_out_key():
    state = make_state()
    xp = xp_frame({621: 9.0, 620: 9.0, 622: 9.0})
    assert GreedyPolicy().decide(ctx(state, xp)).transfers == (Transfer(120, 620),)
    # Two outs with the same (lowest) xP: the smaller out_key.
    xp = xp_frame({620: 9.0, 221: 0.0, 220: 0.0})
    assert GreedyPolicy().decide(ctx(state, xp)).transfers == (Transfer(220, 620),)


def test_greedy_at_gw1_keeps_transferring_while_the_gain_beats_the_threshold():
    state = make_state(gw_index=1, ft=0)
    # Club 6-8 DEFs and MIDs are much better than the held ones, but the club cap allows
    # only 3 per club: 9 transfers (equal prices, so the budget never binds).
    good = [k for k in POOL["player_key"] if k // 100 in (6, 7, 8) and k % 100 // 10 in (2, 3)]
    xp = xp_frame(dict.fromkeys(good, 5.0))
    decision = GreedyPolicy().decide(ctx(state, xp))
    assert len(decision.transfers) == 9
    assert Counter(t.in_key // 100 for t in decision.transfers) == {6: 3, 7: 3, 8: 3}
    # Ties: smallest in_key first, each for the smallest out_key of its position.
    assert decision.transfers[:3] == (Transfer(120, 620), Transfer(121, 621), Transfer(220, 622))
    gw_state, record = apply_decision(state, decision, POOL, RULES)
    assert record.hits == 0 and record.n_transfers == 9
    # Never more than a whole squad.
    everyone = xp_frame({k: 5.0 for k in POOL["player_key"] if k not in BASE_KEYS})
    many = GreedyPolicy(threshold=0.0).decide(ctx(state, everyone)).transfers
    assert len(many) <= RULES.squad_size
    apply_decision(state, GreedyPolicy(threshold=0.0).decide(ctx(state, everyone)), POOL, RULES)


def test_greedy_can_sell_a_player_who_left_the_game():
    pool = POOL[POOL["player_key"] != 320].reset_index(drop=True)
    state = make_state()
    xp = xp_frame({620: 3.0}, pool=pool)  # the DEF 320 has no xP any more: 0
    decision = GreedyPolicy().decide(ctx(state, xp, pool=pool))
    assert decision.transfers == (Transfer(320, 620),)
    apply_decision(state, decision, pool, RULES)


def test_policies_are_frozen_named_and_check_their_model():
    assert GreedyPolicy().name == "greedy(rolling,t=1.0)"
    assert GreedyPolicy("ep_next", threshold=2).name == "greedy(ep_next,t=2.0)"
    assert GreedyPolicy(horizon=3, decay=0.9, max_transfers=2).name == (
        "greedy(rolling,t=1.0,h=3,d=0.9,n=2)"
    )
    assert RollPolicy("ep_next").name == "roll(ep_next)"
    with pytest.raises(FrozenInstanceError):
        GreedyPolicy().threshold = 2.0
    with pytest.raises(ValueError, match="unknown xp_model"):
        GreedyPolicy("nope")
    with pytest.raises(ValueError, match="unknown xp_model"):
        RollPolicy("nope")
    assert GreedyPolicy() == GreedyPolicy() and hash(GreedyPolicy()) == hash(GreedyPolicy())


# --- synthetic league: every decision is valid -------------------------------------------------


@pytest.fixture(scope="module")
def league():
    """2023 of a two-season synthetic league (with a blank and a double): per gw_index the
    view, pool and both models' xP frames, computed once."""
    tables = synthetic_tables(seasons=(2022, 2023), blank=(2023, 10, 1), double=(2023, 11, 4))
    store = DataStore(tables=tables)
    gameweeks = tables["gameweek"]
    gameweeks = gameweeks[gameweeks["season"] == 2023].sort_values("gw_index")
    out = {}
    for gw_index, deadline in zip(gameweeks["gw_index"], gameweeks["deadline_time"], strict=True):
        view = store.as_of(deadline)
        # Every model except the walk-forward fitted ones (v1 would refit at every
        # deadline here: too slow for this fixture; the backtest smoke test covers it).
        xp = {
            name: model(view)
            for name, model in MODELS.items()
            if not isinstance(model, FittedModel)
        }
        out[int(gw_index)] = (view, player_pool(view), xp)
    return out


POLICIES = (
    RollPolicy("rolling"),
    GreedyPolicy("rolling"),
    GreedyPolicy("ep_next", threshold=0.5, max_transfers=2),
    GreedyPolicy("rolling", threshold=0.0, horizon=3, decay=1.0, max_transfers=5),
)


@pytest.mark.parametrize("policy", POLICIES, ids=lambda p: p.name)
def test_every_decision_passes_apply_decision_along_a_season(league, policy):
    """Run each policy GW by GW (decision -> apply -> next_state) from a template and two
    random start states: every decision is valid, deterministic and never takes a hit."""
    rules = backtest_rules(2023)
    first = league[1][0]
    starts = [
        template_state(first, rules),
        random_state(first, rules, 1),
        random_state(league[16][0], rules, 7),
    ]
    transfers = 0
    for state in starts:
        for gw_index in range(state.gw_index, max(league) + 1):
            assert state.gw_index == gw_index
            view, pool, xp = league[gw_index]
            state = refresh(state, pool)
            context = DecisionContext(view, state, rules, pool, xp[policy.xp_model])
            decision = policy.decide(context)
            assert decision == policy.decide(context)
            gw_state, record = apply_decision(state, decision, pool, rules)
            assert record.hits == 0 and decision.chip is None
            transfers += record.n_transfers
            if gw_index < max(league):
                state = next_state(gw_state, record, rules, gw_index + 1)
    if isinstance(policy, RollPolicy):
        assert transfers == 0
    else:
        assert transfers > 0


SMALL = OptimizerParams(horizon=3, prune_n={1: 4, 2: 10, 3: 10, 4: 6})


@pytest.mark.parametrize(
    ("policy", "runs"),
    [
        (OptimizerPolicy("ep_next_fade", replace(SMALL, max_hits=0)), ((1, 1, 4), (16, 7, 21))),
        (  # chip scenarios: a slower solve per GW
            OptimizerPolicy("rolling", replace(SMALL, horizon=2, max_hits=1), chips=True),
            ((16, 7, 19),),
        ),
    ],
    ids=["no-chips", "chips"],
)
def test_optimizer_decisions_pass_apply_decision(league, policy, runs):
    """The optimizer GW by GW from random starts (GW1: a squad build): valid,
    deterministic, within max_hits; a chip only when the policy plays chips."""
    pytest.importorskip("highspy")
    rules = backtest_rules(2023)
    hits = transfers = 0
    chips = []
    for first, seed, last in runs:
        state = random_state(league[first][0], rules, seed)
        for gw_index in range(first, last + 1):
            view, pool, xp = league[gw_index]
            state = refresh(state, pool)
            context = DecisionContext(view, state, rules, pool, xp[policy.xp_model])
            decision = policy.decide(context)
            assert decision == policy.decide(context)
            gw_state, record = apply_decision(state, decision, pool, rules)
            assert decision.solver_status == "Optimal" and decision.mip_gap is not None
            assert record.hits <= policy.params.max_hits
            hits, transfers = hits + record.hits, transfers + record.n_transfers
            chips += [decision.chip] if decision.chip else []
            state = next_state(gw_state, record, rules, gw_index + 1)
    assert transfers > 0
    assert policy.chips or not chips
    if policy.params.max_hits == 0:
        assert hits == 0


def test_optimizer_policy_records_a_limited_solve(league, monkeypatch, caplog):
    """A solve a limit stopped is executed (the best plan found) but never silently: the
    decision carries the status and gap, and a warning is logged."""
    pytest.importorskip("highspy")
    import fplopt.backtest.policies as policies_mod

    honest = policies_mod.optimize

    def limited(problem, params, **kwargs):
        plans = honest(problem, params, **kwargs)
        best = replace(plans.best, status="SolutionLimit", mip_gap=0.2)
        return replace(plans, plans=(best, *plans.plans[1:]))

    monkeypatch.setattr(policies_mod, "optimize", limited)
    rules = backtest_rules(2023)
    view, pool, xp = league[16]
    state = refresh(random_state(view, rules, 3), pool)
    policy = OptimizerPolicy("rolling", replace(SMALL, horizon=2))
    with caplog.at_level("WARNING"):
        decision = policy.decide(DecisionContext(view, state, rules, pool, xp["rolling"]))
    assert decision.solver_status == "SolutionLimit" and decision.mip_gap == 0.2
    assert "solver stopped at a limit (SolutionLimit" in caplog.text
    apply_decision(state, decision, pool, rules)


def test_optimizer_policy_is_frozen_named_and_picklable():
    import pickle

    default = OptimizerPolicy()
    assert default.name == "optimizer(ep_next,mh=0,m=0.0)"
    params = OptimizerParams(horizon=4, decay=0.9, max_hits=1, hit_margin=2, itb_value=0.08)
    policy = OptimizerPolicy("rolling", params, chips=True)
    assert policy.name == "optimizer(rolling,mh=1,m=2.0,h=4,d=0.9,itb=0.08,chips)"
    small = OptimizerPolicy(params=SMALL).name
    assert small == "optimizer(ep_next,mh=0,m=0.0,h=3,prune=1:4/2:10/3:10/4:6)"
    # Different positions pruned to the same N get different names.
    gk = OptimizerPolicy(params=OptimizerParams(prune_n={1: 20})).name
    fwd = OptimizerPolicy(params=OptimizerParams(prune_n={4: 20})).name
    assert gk == "optimizer(ep_next,mh=0,m=0.0,prune=1:20)" and gk != fwd
    assert pickle.loads(pickle.dumps(policy)) == policy
    with pytest.raises(FrozenInstanceError):
        policy.chips = False
    with pytest.raises(ValueError, match="unknown xp_model"):
        OptimizerPolicy("nope")
    with pytest.raises(TypeError, match="OptimizerParams"):
        OptimizerPolicy("rolling", {"horizon": 3})


def test_policies_refuse_horizons_beyond_the_xp_frame():
    """xP frames cover MAX_HORIZON GWs, so a longer horizon would silently plan over fewer
    GWs under a name that says otherwise."""
    from fplopt.models import MAX_HORIZON

    assert OptimizerPolicy(params=OptimizerParams(horizon=MAX_HORIZON)).params.horizon == 6
    assert GreedyPolicy(horizon=MAX_HORIZON).horizon == MAX_HORIZON
    with pytest.raises(ValueError, match=r"horizon must be in 1\.\.6"):
        OptimizerPolicy(params=OptimizerParams(horizon=MAX_HORIZON + 1))
    with pytest.raises(ValueError, match=r"horizon must be in 1\.\.6"):
        GreedyPolicy(horizon=MAX_HORIZON + 1)
    with pytest.raises(ValueError, match=r"horizon must be in 1\.\.6"):
        GreedyPolicy(horizon=0)


def test_greedy_decisions_are_valid_from_many_random_states(league):
    rules = backtest_rules(2023)
    rng = np.random.default_rng(0)
    policy = GreedyPolicy("rolling", threshold=0.0, max_transfers=5)
    for gw_index in sorted(rng.choice(sorted(league), 8, replace=False)):
        view, pool, xp = league[int(gw_index)]
        for seed in range(4):
            state = random_state(view, rules, seed)
            ft, bank = int(rng.integers(0, 6)), int(rng.integers(0, 30))
            state = replace(state, free_transfers=ft, bank=bank)
            decision = policy.decide(DecisionContext(view, state, rules, pool, xp["rolling"]))
            _, record = apply_decision(state, decision, pool, rules)
            assert record.hits == 0
            assert record.n_transfers <= (15 if gw_index == 1 else min(5, ft))
