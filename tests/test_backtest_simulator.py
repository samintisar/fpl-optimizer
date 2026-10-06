"""The season simulator (fplopt.backtest.simulator) on the synthetic league."""

from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.backtest.gw_score import score_gameweek
from fplopt.backtest.policies import (
    DecisionContext,
    GreedyPolicy,
    RollPolicy,
    best_lineup,
    greedy_transfers,
    horizon_xp,
    target_xp,
)
from fplopt.backtest.rules import backtest_rules
from fplopt.backtest.scoring import score_matches
from fplopt.backtest.simulator import (
    GW_COLUMNS,
    Caches,
    HoldoutError,
    decide_step,
    hindsight_points,
    read_outcomes,
    season_schedule,
    simulate,
)
from fplopt.backtest.start_states import random_state, template_state
from fplopt.backtest.state import Decision, apply_decision, next_state, refresh
from fplopt.features.baseline import player_pool
from fplopt.features.leakcheck import corrupt_future, truncate_future
from fplopt.features.store import DataStore

SEASON = 2023
RULES = backtest_rules(SEASON)
BLANK = (SEASON, 10, 1)  # club 1's GW10 fixture moves to GW12
DOUBLE = (SEASON, 14, 4)  # club 4's GW16 fixture moves to GW14
ROLL = RollPolicy("rolling")
GREEDY = GreedyPolicy("rolling")


def gameweek(tables, season, gw_index):
    gameweeks = tables["gameweek"]
    rows = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw_index"] == gw_index)]
    return rows.iloc[0]


@pytest.fixture(scope="module")
def league():
    """A two-season league (history for the rolling model) with a blank and a double in
    2023, its store, one shared Caches and the 2023 template start state."""
    tables = synthetic_tables(seasons=(2022, SEASON), blank=BLANK, double=DOUBLE)
    store = DataStore(tables=tables)
    start = template_state(store.as_of(gameweek(tables, SEASON, 1)["deadline_time"]), RULES)
    return tables, store, Caches(), start


@pytest.fixture(scope="module")
def runs(league):
    tables, store, caches, start = league
    return {p.name: simulate(store, RULES, p, start, caches=caches) for p in (ROLL, GREEDY)}


# --- basics -------------------------------------------------------------------------------


@pytest.mark.parametrize("policy", [ROLL, GREEDY], ids=lambda p: p.name)
def test_full_season_is_deterministic(league, runs, policy):
    """Two runs (the second on a fresh store and cache) are identical."""
    tables, _, _, start = league
    run = runs[policy.name]
    again = simulate(DataStore(tables=tables), RULES, policy, start)
    pd.testing.assert_frame_equal(run.gws, again.gws)
    assert run.decisions == again.decisions
    assert run.states == again.states
    assert run.final_state == again.final_state
    assert run.total == again.total


@pytest.mark.parametrize("policy", [ROLL, GREEDY], ids=lambda p: p.name)
def test_gw_rows_and_totals(runs, policy):
    run = runs[policy.name]
    gws = run.gws
    assert list(gws.columns) == [name for name, _ in GW_COLUMNS]
    assert gws["gw_index"].tolist() == list(range(1, 39))
    assert run.policy == policy.name
    assert run.total == gws["net_points"].sum()
    assert (gws["net_points"] == gws["points"] - gws["hit_points"]).all()
    assert (gws["hit_points"] == 0).all() and gws["chip"].isna().all()
    assert (gws["captain_regret"] >= 0).all()
    assert gws["xi_regret"].notna().all() and (gws["xi_regret"] >= 0).all()
    assert gws["xg_points"].notna().all()  # synthetic data has every xG source
    assert len(run.decisions) == len(run.states) == 38
    assert run.final_state.gw_index == 39
    transfers = [json.loads(t) for t in gws["transfers"]]
    assert [len(t) for t in transfers] == gws["n_transfers"].tolist()
    if policy is ROLL:
        assert gws["n_transfers"].sum() == 0
    else:
        assert gws["n_transfers"].sum() > 0


@pytest.mark.parametrize("policy", [ROLL, GREEDY], ids=lambda p: p.name)
def test_states_bank_and_free_transfers_follow_the_state_module(league, runs, policy):
    """Replaying the recorded decisions with refresh/apply_decision/next_state reproduces
    every state, the bank and FT columns and the final state."""
    tables, store, _, start = league
    run = runs[policy.name]
    state = start
    schedule = season_schedule(store, SEASON, 1)
    for i, row in enumerate(schedule.itertuples()):
        pool = player_pool(store.as_of(row.deadline_time))
        state = refresh(state, pool)
        assert state == run.states[i]
        gw_state, record = apply_decision(state, run.decisions[i], pool, RULES)
        assert run.gws["bank"].iloc[i] == gw_state.bank
        assert run.gws["free_transfers"].iloc[i] == state.free_transfers
        state = next_state(gw_state, record, RULES, row.gw_index + 1)
    assert state == run.final_state
    if policy is ROLL:  # FTs bank from 0 at GW1 (unlimited) to the cap of 5
        assert run.gws["free_transfers"].tolist()[:7] == [0, 1, 2, 3, 4, 5, 5]


def test_mid_season_start(league):
    tables, store, caches, _ = league
    view = store.as_of(gameweek(tables, SEASON, 20)["deadline_time"])
    start = random_state(view, RULES, seed=3)
    run = simulate(store, RULES, GREEDY, start, caches=caches)
    assert run.gws["gw_index"].tolist() == list(range(20, 39))
    assert run.gws["free_transfers"].iloc[0] == 1
    assert run.states[0].holdings == refresh(start, player_pool(view)).holdings


def test_end_gw_index_stops_early_and_matches_the_full_run(league, runs):
    _, store, caches, start = league
    run = simulate(store, RULES, GREEDY, start, end_gw_index=10, caches=caches)
    full = runs[GREEDY.name]
    pd.testing.assert_frame_equal(run.gws, full.gws.iloc[:10])
    assert run.decisions == full.decisions[:10]
    assert run.final_state.gw_index == 11
    assert refresh(run.final_state, player_pool(run_view(league, 11))) == full.states[10]


def run_view(league, gw_index):
    tables, store, _, _ = league
    return store.as_of(gameweek(tables, SEASON, gw_index)["deadline_time"])


def test_holdout_season_is_refused():
    tables = synthetic_tables(seasons=(2025,))
    store = DataStore(tables=tables)
    rules = backtest_rules(2025)
    start = random_state(store.as_of(gameweek(tables, 2025, 1)["deadline_time"]), rules, 0)
    with pytest.raises(HoldoutError, match="holdout"):
        simulate(store, rules, ROLL, start)
    run = simulate(store, rules, ROLL, start, end_gw_index=2, allow_holdout=True)
    assert len(run.gws) == 2


def test_caches_belong_to_one_store(league):
    tables, _, caches, start = league
    with pytest.raises(ValueError, match="another DataStore"):
        simulate(DataStore(tables=tables), RULES, ROLL, start, end_gw_index=1, caches=caches)


def test_a_gameweek_without_outcomes_ends_the_run(league, caplog):
    """The live season: GWs after the last lockdown have no outcomes yet."""
    tables, _, _, start = league
    cut = gameweek(tables, SEASON, 6)["lockdown_time"]
    store = DataStore(tables=truncate_future(tables, cut))
    run = simulate(store, RULES, ROLL, start)
    assert run.gws["gw_index"].tolist() == [1, 2, 3, 4, 5]
    assert "no outcomes" in caplog.text


# --- outcomes, doubles and blanks -------------------------------------------------------------


def expected_outcomes(tables, season, gw):
    """player_key -> (Σ total_points, Σ minutes) straight from the synthetic player_match
    (whose total_points follow the same scoring as the backtest rules for these players)."""
    rows = tables["player_match"]
    rows = rows[(rows["season"] == season) & (rows["gw"] == gw)]
    sums = rows.groupby("player_key")[["total_points", "minutes"]].sum()
    return {int(k): (int(p), int(m)) for k, p, m in sums.itertuples()}


def test_synthetic_total_points_match_the_backtest_rules(league):
    tables = league[0]
    rows = tables["player_match"]
    rows = rows[rows["season"] == SEASON]
    positions = tables["player_season"][["player_key", "season", "element_type"]]
    rows = rows.merge(positions, on=["player_key", "season"])
    assert (score_matches(rows, RULES)["points"] == rows["total_points"]).all()


def test_outcomes_sum_doubles_and_omit_blanks(league):
    tables, store, _, _ = league
    for gw in (9, 10, 12, 14, 16):
        lockdown = gameweek(tables, SEASON, gw)["lockdown_time"]
        outcomes = read_outcomes(store, RULES, SEASON, gw, lockdown)
        assert outcomes.realized == expected_outcomes(tables, SEASON, gw)
        assert set(outcomes.xg) == set(outcomes.realized)
    rows = tables["player_match"]
    rows = rows[rows["season"] == SEASON]
    club1 = rows[rows["team_key"] == 1]
    assert not (club1["gw"] == 10).any()  # blank
    assert club1.groupby("gw")["fixture_key"].nunique()[12] == 2  # double
    club4 = rows[rows["team_key"] == 4].groupby("gw")["fixture_key"].nunique()
    assert club4[14] == 2 and 16 not in club4
    gw10 = read_outcomes(store, RULES, SEASON, 10, gameweek(tables, SEASON, 10)["lockdown_time"])
    assert not any(k // 100 % 1000 == 1 for k in gw10.realized)  # club 1's players blank


def element_types(tables):
    rows = tables["player_season"]
    rows = rows[rows["season"] == SEASON]
    return dict(zip(rows["player_key"], rows["element_type"], strict=True))


def recomputed_points(tables, decision, gw):
    """score_gameweek of a decision on outcomes taken straight from player_match."""
    lineup = decision.lineup
    types = element_types(tables)
    positions = {k: types[k] for k in (*lineup.starters, *lineup.bench)}
    outcomes = expected_outcomes(tables, SEASON, gw)
    return score_gameweek(lineup, outcomes, positions, RULES, decision.chip)


@pytest.mark.parametrize("policy", [ROLL, GREEDY], ids=lambda p: p.name)
def test_every_gw_scores_like_an_independent_recomputation(league, runs, policy):
    """Points per GW = score_gameweek on outcomes taken straight from player_match (sums
    over a double's two fixtures; a blank = did not play)."""
    tables = league[0]
    run = runs[policy.name]
    for i, row in enumerate(run.gws.itertuples()):
        score = recomputed_points(tables, run.decisions[i], row.gw)
        assert (row.points, row.bench_points) == (score.points, score.bench_points)
        assert json.loads(row.autosubs) == [list(pair) for pair in score.autosubs]


def test_blank_and_double_gameweeks_in_a_run(league):
    """A squad holding club 1 and club 4 players (rolled, so they stay): club 1's players
    don't play in GW10 and play twice in GW12, club 4's twice in GW14."""
    tables, store, caches, _ = league
    view = store.as_of(gameweek(tables, SEASON, 9)["deadline_time"])
    state = None
    for seed in range(200):
        candidate = random_state(view, RULES, seed)
        clubs = Counter(h.team_key for h in candidate.holdings)
        if clubs[1] >= 2 and clubs[4] >= 2:
            state = candidate
            break
    assert state is not None
    run = simulate(store, RULES, ROLL, state, end_gw_index=14, caches=caches)
    gws = run.gws.set_index("gw")
    held = {club: {h.player_key for h in state.holdings if h.team_key == club} for club in (1, 4)}
    assert not held[1] & set(expected_outcomes(tables, SEASON, 10))
    rows = tables["player_match"]
    rows = rows[rows["season"] == SEASON]
    for club, gw in ((1, 12), (4, 14)):
        counts = rows[(rows["gw"] == gw) & rows["player_key"].isin(held[club])]
        assert (counts.groupby("player_key").size() == 2).all()
    for i, gw in enumerate(gws.index):
        assert gws.loc[gw, "points"] == recomputed_points(tables, run.decisions[i], gw).points


# --- chips and hits ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForcedPolicy:
    """Test-only: greedy-ranked transfers with any gain (`plan[gw_index] = (chip, n)`), a
    chip, then the best lineup; other GWs roll."""

    plan: tuple[tuple[int, str | None, int], ...]
    xp_model: str = "rolling"

    @property
    def name(self) -> str:
        return "forced"

    def decide(self, ctx: DecisionContext) -> Decision:
        plan = {gw: (chip, n) for gw, chip, n in self.plan}
        chip, n = plan.get(ctx.state.gw_index, (None, 0))
        state = refresh(ctx.state, ctx.pool)
        hxp = horizon_xp(ctx.xp, 6, 0.85)
        transfers = greedy_transfers(state, ctx.pool, hxp, ctx.rules, -math.inf, n)
        assert len(transfers) == n
        outs = {t.out_key for t in transfers}
        pool = ctx.pool.set_index("player_key")
        keys = [h.player_key for h in state.holdings if h.player_key not in outs]
        keys += [t.in_key for t in transfers]
        types = {h.player_key: h.element_type for h in state.holdings}
        types |= {t.in_key: int(pool.loc[t.in_key, "element_type"]) for t in transfers}
        squad = pd.DataFrame(
            {"player_key": keys, "element_type": [types[k] for k in keys]}, dtype="int64"
        )
        return Decision(tuple(transfers), best_lineup(squad, target_xp(ctx.xp), ctx.rules), chip)


def test_forced_chips_and_hits_end_to_end(league):
    tables, store, caches, start = league
    plan = (
        (3, None, 3),  # 2 FTs banked: one hit
        (4, "wildcard", 4),  # free
        (6, "freehit", 3),  # free, reverted in GW7
        (8, "3xc", 0),
        (22, "bboost", 0),
    )
    run = simulate(store, RULES, ForcedPolicy(plan), start, caches=caches)
    gws = run.gws.set_index("gw_index")
    # GW3: 3 transfers with 2 FTs -> a 4-point hit.
    assert gws.loc[3, "free_transfers"] == 2
    assert (gws.loc[3, "n_transfers"], gws.loc[3, "hits"], gws.loc[3, "hit_points"]) == (3, 1, 4)
    assert gws.loc[3, "net_points"] == gws.loc[3, "points"] - 4
    # Wildcard: free; next GW's FTs per chip_week_ft ("retain": stays at 1 after a WC).
    assert (gws.loc[4, "chip"], gws.loc[4, "hits"]) == ("wildcard", 0)
    assert gws.loc[5, "free_transfers"] == gws.loc[4, "free_transfers"]
    # Free Hit: free, and the squad and bank come back in GW7.
    assert (gws.loc[6, "chip"], gws.loc[6, "hits"], gws.loc[6, "n_transfers"]) == ("freehit", 0, 3)
    before, after = run.states[5], run.states[6]
    assert before.player_keys == after.player_keys
    assert before.bank == after.bank
    assert after.freehit_backup is None
    # Triple Captain: the captain's points count three times.
    outcomes = read_outcomes(
        store, RULES, SEASON, 8, gameweek(tables, SEASON, 8)["lockdown_time"]
    ).realized
    captain = gws.loc[8, "captain_used"]
    assert not pd.isna(captain)
    plain = score_gameweek(run.decisions[7].lineup, outcomes, positions_of(run, 7), RULES)
    assert gws.loc[8, "points"] == plain.points + outcomes[int(captain)][0]
    assert pd.isna(gws.loc[8, "xi_regret"])
    # Bench Boost: all 15 count, no autosubs, no bench points.
    outcomes = read_outcomes(
        store, RULES, SEASON, 22, gameweek(tables, SEASON, 22)["lockdown_time"]
    ).realized
    lineup = run.decisions[21].lineup
    squad = (*lineup.starters, *lineup.bench)
    expected = sum(outcomes.get(k, (0, 0))[0] for k in squad)
    captain = gws.loc[22, "captain_used"]
    expected += 0 if pd.isna(captain) else outcomes[int(captain)][0]
    assert gws.loc[22, "points"] == expected
    assert gws.loc[22, "bench_points"] == 0 and gws.loc[22, "autosubs"] == "[]"
    assert gws["chip"].notna().sum() == 4
    assert {cid for cid, _ in run.final_state.chips_used} == {1, 3, 5, 7}


def positions_of(run, i):
    return {h.player_key: h.element_type for h in run.states[i].holdings}


def test_regrets_on_a_hand_made_gameweek(league):
    """captain_regret = best counted player's points − the armband player's; xi_regret = the
    hindsight best XI + captain − actual points."""
    _, _, _, start = league
    squad = pd.DataFrame(
        {
            "player_key": [h.player_key for h in start.holdings],
            "element_type": [h.element_type for h in start.holdings],
        }
    )
    points = {k: (i % 7, 90) for i, k in enumerate(squad["player_key"])}
    lineup = best_lineup(squad, {k: v[0] for k, v in points.items()}, RULES)
    best = sum(points[k][0] for k in lineup.starters) + max(points[k][0] for k in lineup.starters)
    assert hindsight_points(squad, points, RULES) == best
    positions = dict(zip(squad["player_key"], squad["element_type"], strict=True))
    assert score_gameweek(lineup, points, positions, RULES).points == best


# --- leakage ----------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["corrupted", "truncated"])
def test_decisions_do_not_depend_on_data_after_the_deadline(league, runs, variant):
    """Corrupt (or delete) every row available at or after GW t's deadline: the decisions
    and scores of GWs < t and the decision at t are unchanged."""
    tables, _, _, start = league
    clean = runs[GREEDY.name]
    t = 12
    deadline = gameweek(tables, SEASON, t)["deadline_time"]
    if variant == "corrupted":
        altered = corrupt_future(tables, deadline, seed=5)
    else:
        altered = truncate_future(tables, deadline)
    store = DataStore(tables=altered)
    caches = Caches()
    run = simulate(store, RULES, GREEDY, start, end_gw_index=t - 1, caches=caches)
    assert run.decisions == clean.decisions[: t - 1]
    pd.testing.assert_frame_equal(run.gws, clean.gws.iloc[: t - 1])
    step = decide_step(store, RULES, GREEDY, run.final_state, deadline, caches)
    assert step.state == clean.states[t - 1]
    assert step.decision == clean.decisions[t - 1]


# --- xG points ----------------------------------------------------------------------------------


def null_xg(tables, keys, gw):
    """A copy of `tables` with every player xG source null for `keys` in (SEASON, gw)."""
    tables = dict(tables)
    rows = tables["player_match"].copy()
    mask = (rows["season"] == SEASON) & (rows["gw"] == gw) & rows["player_key"].isin(keys)
    for column in ("us_xg", "us_xa", "fpl_xg", "fpl_xa"):
        rows.loc[mask, column] = pd.NA
    tables["player_match"] = rows
    return tables


def test_xg_points_are_null_when_a_counted_player_has_no_xg(league, runs):
    tables, _, _, start = league
    clean = runs[ROLL.name]
    gw = 5
    outcomes = expected_outcomes(tables, SEASON, gw)
    lineup = clean.decisions[gw - 1].lineup
    counted = [k for k in lineup.starters if outcomes.get(k, (0, 0))[1] > 0]
    unused = [k for k in lineup.bench if outcomes.get(k, (0, 0))[1] > 0]
    autosubs = {pair[1] for pair in json.loads(clean.gws["autosubs"].iloc[gw - 1])}
    unused = [k for k in unused if k not in autosubs]
    run = simulate(DataStore(tables=null_xg(tables, counted[:1], gw)), RULES, ROLL, start)
    assert pd.isna(run.gws["xg_points"].iloc[gw - 1])
    assert pd.isna(run.gws["xg_net_points"].iloc[gw - 1])
    others = run.gws.drop(index=gw - 1)
    pd.testing.assert_frame_equal(others, clean.gws.drop(index=gw - 1))
    assert run.gws["points"].tolist() == clean.gws["points"].tolist()
    if unused:  # an unused bench player's missing xG doesn't matter
        run = simulate(DataStore(tables=null_xg(tables, unused[:1], gw)), RULES, ROLL, start)
        pd.testing.assert_frame_equal(run.gws, clean.gws)


def test_player_xg_is_null_if_any_of_his_double_gw_rows_is(league):
    tables, _, _, _ = league
    rows = tables["player_match"]
    double = rows[(rows["season"] == SEASON) & (rows["gw"] == 12) & (rows["team_key"] == 1)]
    double = double[double["minutes"] > 0]
    key = int(double["player_key"].value_counts().idxmax())
    first = double[double["player_key"] == key].index[0]
    altered = dict(tables)
    altered["player_match"] = rows.copy()
    for column in ("us_xg", "us_xa", "fpl_xg", "fpl_xa"):
        altered["player_match"].loc[first, column] = pd.NA
    lockdown = gameweek(tables, SEASON, 12)["lockdown_time"]
    clean = read_outcomes(DataStore(tables=tables), RULES, SEASON, 12, lockdown)
    outcomes = read_outcomes(DataStore(tables=altered), RULES, SEASON, 12, lockdown)
    assert not math.isnan(clean.xg[key][0])
    assert math.isnan(outcomes.xg[key][0]) and outcomes.xg[key][1] == clean.xg[key][1]
    assert outcomes.realized == clean.realized


def test_free_hit_state_is_not_a_valid_start(league):
    _, store, _, start = league
    bad = dataclasses.replace(start, freehit_backup=start.holdings, freehit_bank=start.bank)
    with pytest.raises(ValueError, match="Free Hit"):
        simulate(store, RULES, ROLL, bad)
