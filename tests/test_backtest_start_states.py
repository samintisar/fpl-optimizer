"""Start states (fplopt.backtest.start_states): template and random squads."""

from __future__ import annotations

import dataclasses
from collections import Counter

import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.backtest.rules import backtest_rules
from fplopt.backtest.start_states import (
    MAX_BANK,
    StartStateError,
    check_coverage,
    ownership,
    random_state,
    target_gameweek,
    template_state,
)
from fplopt.features.baseline import player_pool
from fplopt.features.store import DataStore

RULES = backtest_rules(2023)


def deadline(tables, season, gw):
    gameweeks = tables["gameweek"]
    row = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)]
    return row["deadline_time"].iloc[0]


@pytest.fixture(scope="module")
def league():
    tables = synthetic_tables()
    return tables, DataStore(tables=tables)


@pytest.fixture(scope="module")
def no_snapshots():
    tables = synthetic_tables(snapshots=False)
    return tables, DataStore(tables=tables)


def view_at(world, season, gw):
    tables, store = world
    return store.as_of(deadline(tables, season, gw))


def assert_valid(state, view, rules=RULES):
    pool = player_pool(view).set_index("player_key")
    season, _, gw_index = target_gameweek(view)
    assert (state.season, state.gw_index) == (season, gw_index)
    assert len(state.holdings) == rules.squad_size == 15
    assert Counter(h.element_type for h in state.holdings) == dict(rules.squad_select)
    assert max(Counter(h.team_key for h in state.holdings).values()) <= rules.team_limit
    for h in state.holdings:
        assert h.player_key in pool.index
        row = pool.loc[h.player_key]
        assert (h.element_type, h.team_key) == (row["element_type"], row["team_key"])
        assert h.purchase_price == h.price == row["price"]
    cost = sum(h.price for h in state.holdings)
    assert state.bank == rules.budget - cost >= 0
    assert state.free_transfers == (1 if gw_index > 1 else 0)
    assert state.chips_used == () and state.freehit_backup is None


@pytest.mark.parametrize("gw", [1, 2, 20, 38])
def test_start_states_are_valid(league, gw):
    view = view_at(league, 2023, gw)
    assert_valid(template_state(view, RULES), view)
    for seed in range(5):
        assert_valid(random_state(view, RULES, seed), view)


def test_random_states_are_deterministic_per_seed_and_differ_across_seeds(league):
    view = view_at(league, 2023, 5)
    states = [random_state(view, RULES, seed) for seed in range(6)]
    assert states[0] == random_state(view_at(league, 2023, 5), RULES, 0)
    assert len({s.player_keys for s in states}) == len(states)


@pytest.mark.parametrize("gw", [1, 10, 30])
def test_random_states_spend_realistically(league, gw):
    """Minimum spend: at most MAX_BANK (£2m) left, for every seed (on real data the
    price-weighted draws alone left £16–31m on average)."""
    view = view_at(league, 2023, gw)
    banks = [random_state(view, RULES, seed).bank for seed in range(20)]
    assert max(banks) <= MAX_BANK


def test_random_state_spend_bound_is_a_parameter(league):
    view = view_at(league, 2023, 10)
    for seed in range(5):
        state = random_state(view, RULES, seed, max_bank=5)
        assert_valid(state, view)
        assert state.bank <= 5


def test_template_is_the_most_owned_feasible_squad(league):
    view = view_at(league, 2023, 10)
    state = template_state(view, RULES)
    assert state == template_state(view_at(league, 2023, 10), RULES)
    owned = ownership(view)
    pool = player_pool(view)
    pool = pool.assign(owned=pool["player_key"].map(owned).fillna(0.0))
    keys = set(state.player_keys)
    # The three most-owned players are in (nothing can block them yet).
    top = pool.sort_values(["owned", "player_key"], ascending=[False, True]).head(3)
    assert set(top["player_key"]) <= keys
    # Owned well above the pool average (the budget forces some cheap, unowned fillers).
    assert pool.loc[pool["player_key"].isin(keys), "owned"].mean() > 2 * pool["owned"].mean()


def test_template_follows_ownership(league):
    """Hand-check the greedy rule: with ownership = a reversed key order the template takes
    the highest keys of each position that fit (prices permitting)."""
    tables, _ = league
    tables = {name: df.copy() for name, df in tables.items()}
    snaps = tables["player_snapshot"]
    tables["player_snapshot"] = snaps.assign(
        selected_by_percent=pd.array(snaps["player_key"] / 1000.0, dtype="Float64")
    )
    view = DataStore(tables=tables).as_of(deadline(tables, 2023, 10))
    rules = dataclasses.replace(RULES, budget=10_000)  # money no object
    state = template_state(view, rules)
    pool = player_pool(view).sort_values("player_key", ascending=False)
    expected, clubs = [], Counter()
    slots = Counter(dict(rules.squad_select))
    for key, et, team in pool[["player_key", "element_type", "team_key"]].itertuples(index=False):
        if slots[et] and clubs[team] < rules.team_limit:
            expected.append(key)
            slots[et] -= 1
            clubs[team] += 1
    assert sorted(state.player_keys) == sorted(expected)


def test_ownership_comes_from_the_snapshot_else_player_gw_ownership(league, no_snapshots):
    view = view_at(league, 2023, 10)
    owned = ownership(view)
    tables, _ = league
    snaps = tables["player_snapshot"]
    newest = snaps[snaps["snapshot_at"] < view.deadline]
    newest = newest[newest["snapshot_at"] == newest["snapshot_at"].max()]
    expected = newest.set_index("player_key")["selected_by_percent"].astype("float64")
    pd.testing.assert_series_equal(
        owned.sort_index(), expected.sort_index(), check_names=False, check_index_type=False
    )
    # Without snapshots: GW9's `selected` (available at GW9's deadline) is the newest at GW10.
    tables, _ = no_snapshots
    owned = ownership(view_at(no_snapshots, 2023, 10))
    rows = tables["player_gw_ownership"]
    gw9 = rows[(rows["season"] == 2023) & (rows["gw"] == 9)].set_index("player_key")["selected"]
    assert owned.sort_index().to_dict() == gw9.astype("float64").sort_index().to_dict()
    assert_valid(
        template_state(view_at(no_snapshots, 2023, 2), RULES), view_at(no_snapshots, 2023, 2)
    )


def test_template_is_refused_without_visible_ownership(no_snapshots):
    view = view_at(no_snapshots, 2023, 1)
    assert ownership(view) is None
    with pytest.raises(StartStateError, match="no ownership"):
        template_state(view, RULES)
    assert_valid(random_state(view, RULES, 0), view)  # random starts need no ownership


def test_start_states_are_refused_on_a_coverage_gap(no_snapshots):
    """Club 1 without player_gw rows at a GW1 without snapshots (like 2020/21 GW1): the pool
    lacks the club entirely."""
    tables, _ = no_snapshots
    tables = {name: df.copy() for name, df in tables.items()}
    rows = tables["player_gw"]
    club1 = (rows["season"] == 2023) & (rows["gw"] == 1) & (rows["team_key"] == 1)
    tables["player_gw"] = rows[~club1].reset_index(drop=True)
    view = DataStore(tables=tables).as_of(deadline(tables, 2023, 1))
    with pytest.raises(StartStateError, match=r"coverage gap .*\[1\]"):
        check_coverage(view)
    with pytest.raises(StartStateError, match="coverage gap"):
        random_state(view, RULES, 0)
    with pytest.raises(StartStateError, match="coverage gap"):
        template_state(view, RULES)
    check_coverage(view_at(no_snapshots, 2023, 1))  # the intact league is fine


def test_start_states_are_refused_when_the_pool_cannot_form_a_squad(league):
    view = view_at(league, 2023, 5)
    poor = dataclasses.replace(RULES, budget=300)
    with pytest.raises(StartStateError, match="cannot form a valid squad"):
        random_state(view, poor, 0)
    with pytest.raises(StartStateError, match="cannot form a valid squad"):
        template_state(view, poor)
    # The cheapest valid squad exactly: both still succeed.
    pool = player_pool(view)
    cheapest = sum(
        int(pool.loc[pool["element_type"] == et, "price"].nsmallest(n).sum())
        for et, n in RULES.squad_select.items()
    )
    tight = dataclasses.replace(RULES, budget=cheapest)
    for state in (random_state(view, tight, 3), template_state(view, tight)):
        assert state.bank == 0
        assert_valid(state, view, tight)
