"""The chip study's code (docs/superpowers/plans/2026-10-11-chip-policy.md): the gain probe
(S0), the rule baseline H, the terminal value estimator, the full-run window t-test and its
size, and the CLI specs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt import cli
from fplopt.backtest.chip_values import estimate_terminal_values
from fplopt.backtest.evaluate import full_run_window_test
from fplopt.backtest.policies import (
    CHIP_DIAGNOSTICS,
    ChipGainProbe,
    ChipRulePolicy,
    DecisionContext,
    OptimizerPolicy,
)
from fplopt.backtest.rules import backtest_rules
from fplopt.backtest.start_states import random_state
from fplopt.backtest.state import apply_decision, next_state, refresh
from fplopt.features.baseline import player_pool
from fplopt.features.store import DataStore
from fplopt.models import MODELS
from fplopt.optimize import OptimizerParams

pytest.importorskip("highspy")

SMALL = OptimizerParams(horizon=2, max_hits=1, prune_n={1: 4, 2: 10, 3: 10, 4: 6})
BLANK_GW, BLANK_CLUB = 10, 1


@pytest.fixture(scope="module")
def league():
    """2023 of the synthetic league with club 1 blank in GW10: per gw_index the view, pool
    and rolling xP, GWs 8-12."""
    tables = synthetic_tables(seasons=(2022, 2023), blank=(2023, BLANK_GW, BLANK_CLUB))
    store = DataStore(tables=tables)
    gameweeks = tables["gameweek"]
    gameweeks = gameweeks[gameweeks["season"] == 2023].sort_values("gw_index")
    out = {}
    for gw_index, deadline in zip(gameweeks["gw_index"], gameweeks["deadline_time"], strict=True):
        if 8 <= gw_index <= 12:
            view = store.as_of(deadline)
            out[int(gw_index)] = (view, player_pool(view), MODELS["rolling"](view))
    return out


def contexts(league, policy, first=8, last=12, seed=3):
    """Run `policy` GW by GW; yields (context, decision)."""
    rules = backtest_rules(2023)
    state = random_state(league[first][0], rules, seed)
    for gw_index in range(first, last + 1):
        view, pool, xp = league[gw_index]
        state = refresh(state, pool)
        context = DecisionContext(view, state, rules, pool, xp)
        decision = policy.decide(context)
        yield context, decision
        gw_state, record = apply_decision(state, decision, pool, rules)
        state = next_state(gw_state, record, rules, gw_index + 1)


def test_gain_probe_decides_like_s0_and_records_the_gains(league):
    probe = ChipGainProbe("rolling", SMALL)
    s0 = OptimizerPolicy("rolling", SMALL)
    assert probe.name.startswith("chipgains(rolling,") and probe.name != s0.name
    for context, decision in contexts(league, probe):
        assert decision == s0.decide(context) and decision.chip is None
        gains = decision.diagnostics
        assert set(gains) == set(CHIP_DIAGNOSTICS)
        assert gains["gain_tc"] == gains["captain_xp"] > 0  # TC playable all of set 1
        assert gains["bench_xp"] >= gains["gain_bb"] >= 0
        assert gains["gain_fh"] is not None and gains["gain_fh"] >= -1e-6
        assert gains["gain_wc"] is not None and gains["gain_wc"] >= -1e-6
        held = sum(1 for h in context.state.holdings if h.team_key == BLANK_CLUB)
        expected = held if context.state.gw_index == BLANK_GW else 0
        assert gains["blanks"] == expected


def test_rule_policy_choose(league):
    view, pool, xp = league[8]
    rules = backtest_rules(2023)
    state = refresh(random_state(view, rules, 3), pool)
    policy = ChipRulePolicy("rolling", SMALL, t_tc=10, t_bb=12, k_fh=3, t_wc=10)
    context = DecisionContext(view, state, rules, pool, xp)
    quiet = {"captain_xp": 9.0, "bench_xp": 11.0, "gain_wc": 9.0, "blanks": 2.0}
    assert policy.choose(context, quiet) is None
    assert policy.choose(context, {**quiet, "captain_xp": 10.0}) == "3xc"
    assert policy.choose(context, {**quiet, "captain_xp": 10.0, "bench_xp": 12.0}) == "bboost"
    assert policy.choose(context, {**quiet, "bench_xp": 20, "gain_wc": 10.0}) == "wildcard"
    assert policy.choose(context, {**quiet, "gain_wc": 50, "blanks": 3.0}) == "freehit"
    assert policy.choose(context, {**quiet, "gain_wc": None}) is None
    # Last chance: 4 set-1 chips unused with 2 GWs left in the window (18, 19).
    late = DecisionContext(view, replace(state, gw_index=18), rules, pool, xp)
    assert policy.choose(late, quiet) == "freehit"
    # With 3 of them played, one left with 2 GWs to go: not forced yet.
    played = tuple((w.chip_id, 5) for w in rules.chips if w.stop == 19 and w.name != "3xc")
    alone = DecisionContext(view, replace(state, gw_index=18, chips_used=played), rules, pool, xp)
    assert policy.choose(alone, quiet) is None
    last = DecisionContext(view, replace(state, gw_index=19, chips_used=played), rules, pool, xp)
    assert policy.choose(last, quiet) == "3xc"


def test_rule_policy_plays_the_chip_with_a_valid_plan(league):
    policy = ChipRulePolicy("rolling", SMALL, t_tc=0.0, t_bb=1e9, k_fh=99, t_wc=1e9)
    assert policy.name.startswith("chiprules(rolling,") and "tc=0.0" in policy.name
    chips = []
    for context, decision in contexts(league, policy, last=9):
        apply_decision(context.state, decision, context.pool, context.rules)  # valid
        assert set(decision.diagnostics) == set(CHIP_DIAGNOSTICS)
        chips.append(decision.chip)
    assert chips == ["3xc", None]  # TC once (set 1), then no TC left


def test_simulator_records_the_diagnostics():
    from fplopt.backtest.simulator import simulate

    store = DataStore(tables=synthetic_tables(seasons=(2022, 2023)))
    rules = backtest_rules(2023)
    from fplopt.backtest.simulator import season_schedule

    deadline = season_schedule(store, 2023, 17)["deadline_time"].iloc[0]
    start = random_state(store.as_of(deadline), rules, 1)
    run = simulate(store, rules, ChipGainProbe("rolling", SMALL), start, end_gw_index=18)
    assert list(run.gws["gw_index"]) == [17, 18]
    assert run.gws["gain_tc"].notna().all() and run.gws["blanks"].notna().all()
    s0 = simulate(store, rules, OptimizerPolicy("rolling", SMALL), start, end_gw_index=18)
    assert s0.gws["gain_tc"].isna().all()
    assert list(s0.gws["net_points"]) == list(run.gws["net_points"])


# --- terminal values --------------------------------------------------------------------------


def gains_frame(season=2020, values=None, start_id="t"):
    """One run's GW rows: TC gain = gw_index (or `values`), others 1.0."""
    rows = []
    for gw in range(1, 39):
        tc = float(gw) if values is None else values(gw)
        rows.append(
            {
                "season": season,
                "policy": "chipgains(x)",
                "start_id": start_id,
                "gw_index": gw,
                "gain_tc": tc,
                "gain_bb": 1.0,
                "gain_fh": None if gw == 1 else 1.0,
                "gain_wc": None if gw == 1 else 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_estimate_terminal_values():
    table = estimate_terminal_values(gains_frame())
    assert set(table) == {"3xc", "bboost", "freehit", "wildcard"}
    tc = table["3xc"]
    assert len(tc) == 19
    # L GWs left: set 1 -> best of GW 20-L..19 = 19; set 2 -> best of 39-L..38 = 38.
    assert tc[0] == pytest.approx((19 + 38) / 2) and tc[18] == pytest.approx((19 + 38) / 2)
    assert table["bboost"] == (1.0,) * 19
    # Free Hit / Wildcard: set 1 is GW2-19 (18 GWs), set 2 GW20-38 (19).
    assert len(table["freehit"]) == 19 and all(v == pytest.approx(1.0) for v in table["freehit"])
    # A decreasing gain: the best of the last L GWs grows with L.
    falling = estimate_terminal_values(gains_frame(values=lambda gw: float(40 - gw)))
    assert list(falling["3xc"]) == sorted(falling["3xc"])
    # Two runs average.
    two = pd.concat([gains_frame(), gains_frame(values=lambda gw: 0.0, start_id="r")])
    assert estimate_terminal_values(two)["3xc"][0] == pytest.approx((19 + 38) / 4)


def test_estimate_terminal_values_refuses_validate_seasons_and_missing_gains():
    with pytest.raises(ValueError, match="develop seasons only"):
        estimate_terminal_values(gains_frame(season=2023))
    empty = gains_frame().assign(gain_tc=None, gain_bb=None, gain_fh=None, gain_wc=None)
    with pytest.raises(ValueError, match="no chip gains"):
        estimate_terminal_values(empty)


def test_estimate_terminal_values_by_window():
    table = estimate_terminal_values(gains_frame(), split="window")
    assert set(table) == {
        f"{c}@{k}" for c in ("3xc", "bboost", "freehit", "wildcard") for k in (1, 2)
    }
    # TC gain = gw_index: set 1's best of its last L GWs is 19, set 2's is 38.
    assert table["3xc@1"] == (19.0,) * 19
    assert table["3xc@2"] == (38.0,) * 19
    assert len(table["freehit@1"]) == 18 and len(table["freehit@2"]) == 19
    # The pooled table is unchanged by default.
    assert estimate_terminal_values(gains_frame())["3xc"][0] == pytest.approx((19 + 38) / 2)


def test_estimate_terminal_values_split_by_known_doubles():
    # 2019 gains rise with the GW (best of the window's last L GWs = 19 in set 1), 2020's are 0.
    frames = pd.concat(
        [gains_frame(season=2019, start_id=str(i)) for i in range(30)]
        + [gains_frame(season=2020, values=lambda gw: 0.0, start_id=str(i)) for i in range(30)]
    )  # 30 runs each = MIN_CELL
    # Set 1 (GWs 1/2-19), L = 5 -> decision GW 19-5-6+1 = 9: a double in GW15 is in (14, 19].
    # L = 4 -> decision GW 10, where GW15 is not in (15, 19]. Known only for 2019, only at 9 and 10.
    doubles = {(2019, 9): frozenset({15}), (2019, 10): frozenset({15})}
    table = estimate_terminal_values(frames, split="double", doubles=doubles, horizon=6)
    assert table["3xc@1+1"][4] == 19.0  # 2019 runs only
    assert table["3xc@1+0"][4] == 0.0  # 2020 runs only
    # L = 4: every run is d = 0 (n = 60), the d = 1 cell is empty and takes the window value.
    assert table["3xc@1+0"][3] == pytest.approx(9.5)
    assert table["3xc@1+1"][3] == table["3xc@1"][3] == pytest.approx(9.5)
    # Set 2 never has a double ahead: its d = 1 cells are empty.
    assert table["3xc@2+1"] == table["3xc@2"]
    with pytest.raises(ValueError, match="doubles"):
        estimate_terminal_values(frames, split="double")


def test_estimate_terminal_values_sparse_cells_fall_back_for_both_flags():
    # 20 runs each: at L = 4 the 2020 runs have a double ahead (d = 1), the 2019 runs not, so
    # both d cells hold 20 < MIN_CELL runs and both take the window value (19 and 0 -> 9.5).
    frames = pd.concat(
        [gains_frame(season=2019, start_id=str(i)) for i in range(20)]
        + [gains_frame(season=2020, values=lambda gw: 0.0, start_id=str(i)) for i in range(20)]
    )
    doubles = {(2020, 10): frozenset({16})}
    table = estimate_terminal_values(frames, split="double", doubles=doubles, horizon=6)
    assert table["3xc@1+0"][3] == table["3xc@1+1"][3] == pytest.approx(9.5)


def test_estimate_loso_leaves_each_season_out():
    from fplopt.backtest.chip_values import estimate_loso

    two = pd.concat([gains_frame(season=2019), gains_frame(season=2020, values=lambda gw: 0.0)])
    tables = estimate_loso(two, split="window")
    assert set(tables) == {2019, 2020}
    assert tables[2019]["3xc@1"][0] == 0.0  # estimated on 2020 only
    assert tables[2020]["3xc@1"][0] == 19.0  # on 2019 only
    with pytest.raises(ValueError, match="two seasons"):
        estimate_loso(gains_frame(), split="window")


def test_load_loso_values(tmp_path):
    from fplopt.backtest.chip_values import load_loso_values, load_terminal_values

    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "format": 2,
                "split": "window",
                "values": {"3xc@1": [2.0]},
                "loso": {"2019": {"3xc@1": [1.0]}, "2020": {"3xc@1": [3.0]}},
            }
        ),
        encoding="utf-8",
    )
    assert load_loso_values(0.5, path) == {2019: {"3xc@1": (0.5,)}, 2020: {"3xc@1": (1.5,)}}
    assert load_terminal_values(1.0, path) == {"3xc@1": (2.0,)}
    plain = tmp_path / "p.json"
    plain.write_text(json.dumps({"values": {"3xc": [1.0]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no leave-one-season-out"):
        load_loso_values(1.0, plain)


# --- the full-run window t-test ---------------------------------------------------------------


def test_full_run_window_test_cells():
    # Two seasons, one start: GW diffs 1..38; 8-GW cells (the last 6 GWs).
    rows = [(s, "t", gw, float(gw)) for s in (2023, 2024) for gw in range(1, 39)]
    paired = pd.DataFrame(rows, columns=["season", "start_id", "gw_index", "diff"])
    result = full_run_window_test(paired)
    assert result.n_cells == 10
    cells = [np.mean(range(a, min(a + 8, 39))) for a in (1, 9, 17, 25, 33)] * 2
    assert result.mean == pytest.approx(np.mean(cells))
    assert full_run_window_test(paired, window=38).n_cells == 2
    with pytest.raises(ValueError, match="window"):
        full_run_window_test(paired, window=0)


@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_full_run_window_test_size_on_two_seasons(rho):
    """A/A placebo for the chip study's statistic: two seasons of per-GW differences, three
    starts sharing nothing, no effect, AR(1) per-GW correlation `rho` (measured full-run
    autocorrelations are ~0). Rejects at nominal 0.10 about 10% of the time; the GW-block
    bootstrap's percentile p rejects ~12.5% here (test_backtest_evaluate)."""
    rng = np.random.default_rng(11 if rho == 0 else 12)
    n_rep, rejected = 3000, 0
    for _ in range(n_rep):
        rows = []
        for season in (2023, 2024):
            for start in range(3):
                x = np.empty(38)
                e = rng.normal(0, 1, 38)
                x[0] = e[0]
                for i in range(1, 38):
                    x[i] = rho * x[i - 1] + np.sqrt(1 - rho**2) * e[i]
                rows += [(season, start, gw + 1, x[gw]) for gw in range(38)]
        paired = pd.DataFrame(rows, columns=["season", "start_id", "gw_index", "diff"])
        rejected += full_run_window_test(paired).p_one_sided < 0.10
    assert 0.08 <= rejected / n_rep <= 0.12


def test_cli_window_constant_matches():
    from fplopt.backtest.evaluate import FULL_RUN_WINDOW

    assert cli.FULL_RUN_WINDOW == FULL_RUN_WINDOW


# --- CLI specs and `fplopt chips values` ------------------------------------------------------


def test_chip_policy_specs(tmp_path, monkeypatch):
    rules = cli.parse_policy_spec("chiprules:v1:horizon=6,decay=0.75,max_hits=1,t_tc=9,k_fh=3")
    policy = rules.build()
    assert isinstance(policy, ChipRulePolicy)
    assert (policy.t_tc, policy.k_fh, policy.params.decay) == (9.0, 3, 0.75)
    probe = cli.parse_policy_spec("chipgains:v1:horizon=6,max_hits=1").build()
    assert isinstance(probe, ChipGainProbe) and probe.params.max_hits == 1
    with pytest.raises(Exception, match="no parameter 'chips'"):
        cli.parse_policy_spec("chipgains:v1:chips=1")
    # chip_shrink: the estimated table times k; needs chips=1 and the table.
    table = tmp_path / "terminal-values.json"
    monkeypatch.setattr(cli, "CHIP_VALUES_FILE", table)
    with pytest.raises(Exception, match="no estimated chip values"):
        cli.parse_policy_spec("optimizer:v1:chips=1,chip_shrink=0.5")
    table.write_text(json.dumps({"values": {"bboost": [2.0, 4.0]}}), encoding="utf-8")
    built = cli.parse_policy_spec("optimizer:v1:chips=1,chip_shrink=0.5").build()
    assert built.chips and built.params.chip_value_left == {"bboost": (1.0, 2.0)}
    assert "cvl=bboost:1.00-2.00" in built.name
    with pytest.raises(Exception, match="chip_shrink needs chips=1"):
        cli.parse_policy_spec("optimizer:v1:chip_shrink=0.5")
    with pytest.raises(Exception, match="chip_shrink"):
        cli.parse_policy_spec("optimizer:v1:chips=1,chip_shrink=-1")


def test_chips_values_command(tmp_path, capsys):
    run = tmp_path / "gains"
    run.mkdir()
    gains_frame().to_parquet(run / "gws.parquet", index=False)
    out = tmp_path / "cfg" / "terminal-values.json"
    settings = cli.Settings(
        raw_dir=tmp_path, odds_api_key=None, telegram_bot_token=None, telegram_admin_chat_id=None
    )
    argv = ["chips", "values", "--gains", str(run), "--out", str(out)]
    assert cli.main(argv, settings=settings) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["seasons"] == ["2020-21"] and len(payload["values"]["3xc"]) == 19
    assert "Written to" in capsys.readouterr().out


# --- review fixes -----------------------------------------------------------------------------


def test_chip_policies_survive_pickling_into_workers():
    """The backtester pickles policies into worker processes: the estimated terminal
    values and every name must survive (they once dropped `chip_value_left`)."""
    import pickle

    table = {"bboost": (1.0, 2.5), "3xc": (0.5,)}
    params = replace(SMALL, chip_value_left=table)
    policies = [
        OptimizerPolicy("v1", params, chips=True),
        OptimizerPolicy(
            "v1",
            replace(params, exclude_chips=("3xc",)),
            chips=True,
            season_tables={2022: table, 2023: {"wildcard": (2.0, 1.0)}},
            double_lead=10,
            tc_rule=8.5,
        ),
        ChipRulePolicy("v1", params, t_tc=9.0, t_bb=11.0, k_fh=2, t_wc=7.0),
        ChipGainProbe("v1", params),
    ]
    for policy in policies:
        back = pickle.loads(pickle.dumps(policy))
        assert back == policy and back.name == policy.name
        assert back.params.chip_value_left == {"bboost": (1.0, 2.5), "3xc": (0.5,)}
    assert pickle.loads(pickle.dumps(OptimizerParams())).chip_value_left is None


def test_the_name_pins_the_whole_table():
    a = OptimizerPolicy("v1", replace(SMALL, chip_value_left={"bboost": (1.0, 2.0, 3.0)}), True)
    b = OptimizerPolicy("v1", replace(SMALL, chip_value_left={"bboost": (1.0, 2.5, 3.0)}), True)
    assert a.name != b.name  # same first and last value, different table


def test_chip_policy_minutes_keys_need_a_minutes_model():
    with pytest.raises(Exception, match="only apply to xP models with minutes"):
        cli.parse_policy_spec("chipgains:rolling:min_minutes=100")
    with pytest.raises(Exception, match="only apply to xP models with minutes"):
        cli.parse_policy_spec("chiprules:ep_next:bench_from_minutes=0")


def test_window_test_null_size():
    from fplopt.backtest.evaluate import window_test_null_size

    rng = np.random.default_rng(3)
    rows = [(s, "t", gw, rng.normal()) for s in range(2016, 2023) for gw in range(1, 39)]
    paired = pd.DataFrame(rows, columns=["season", "start_id", "gw_index", "diff"])
    size = window_test_null_size(paired, n_rep=4000, seed=1)
    assert size.n_cells == 35
    assert 0.07 <= size.rates[0.10] <= 0.13 and size.rates[0.05] <= 0.08
    assert abs(size.critical_p[0.05] - 0.05) < 0.03
    # Spiky, skewed cells (a big chip GW now and then): the rate and the calibrated p
    # are reported, whatever they are.
    spiky = paired.assign(diff=rng.exponential(1, len(paired)) ** 3)
    skewed = window_test_null_size(spiky, n_rep=4000, seed=1)
    assert set(skewed.rates) == {0.05, 0.10} and 0 < skewed.critical_p[0.05] < 1
    with pytest.raises(ValueError, match="2 cells"):
        window_test_null_size(paired[paired["gw_index"] <= 8].head(1))


def test_chips_size_command(tmp_path, capsys):
    rows = [(s, "t", gw, float(gw % 3), float(gw % 2)) for s in (2020, 2021) for gw in range(1, 39)]
    paired = pd.DataFrame(rows, columns=["season", "start_id", "gw_index", "diff", "diff_xg"])
    path = tmp_path / "paired.parquet"
    paired.to_parquet(path, index=False)
    settings = cli.Settings(
        raw_dir=tmp_path, odds_api_key=None, telegram_bot_token=None, telegram_admin_chat_id=None
    )
    assert cli.main(["chips", "size", "--paired", str(path)], settings=settings) == 0
    printed = capsys.readouterr().out
    assert "realized: 10 develop cells; alpha 0.05: rate" in printed and "xg:" in printed
    paired.assign(season=2023).to_parquet(path, index=False)
    assert cli.main(["chips", "size", "--paired", str(path)], settings=settings) == 1


def test_backtest_config_logs_the_estimated_table(tmp_path, monkeypatch):
    import argparse

    table = tmp_path / "terminal-values.json"
    table.write_text(json.dumps({"source": "run", "values": {"bboost": [2.0]}}), encoding="utf-8")
    monkeypatch.setattr(cli, "CHIP_VALUES_FILE", table)
    specs = [
        cli.parse_policy_spec("optimizer:v1:chips=1,chip_shrink=0.5"),
        cli.parse_policy_spec("optimizer:v1"),
    ]
    args = argparse.Namespace(
        seasons=(2020,), starts="template@1", policy_specs=specs, jobs=1, family="f", command="run"
    )
    config = cli._backtest_config(args, tmp_path / "out")
    assert config["chip_values"] == {"0": {"source": "run", "values": {"bboost": [2.0]}}}
    args.policy_specs = specs[1:]
    assert "chip_values" not in cli._backtest_config(args, tmp_path / "out")


def test_free_hit_gain_with_an_expected_minutes_floor(league):
    """With a minutes model (`e_minutes`) the 1-GW Free Hit plan scales the floor to one
    GW; unscaled (180 minutes in one GW) it had no candidates, and every Free Hit gain was
    'keep the squad' (found in the develop run, before any tuning)."""
    from fplopt.backtest.policies import chip_gains, one_gw_params
    from fplopt.optimize import PlanInput, optimize

    assert one_gw_params(SMALL).min_minutes == pytest.approx(SMALL.min_minutes / SMALL.horizon)
    view, pool, xp = league[9]
    rules = backtest_rules(2023)
    state = refresh(random_state(view, rules, 5), pool)
    xp = xp.assign(e_minutes=95.0)  # 190 over the 2-GW horizon: above the 180 floor
    context = DecisionContext(view, state, rules, pool, xp)
    problem = PlanInput.from_context(state, pool, xp, rules, SMALL)
    plans = optimize(problem, SMALL, top_k=1, chips=False, roll=False)
    gains = chip_gains(context, problem, SMALL, plans.best)
    assert gains["gain_fh"] > 1.0  # a random squad: a Free Hit rebuilds it


# --- chip policy 2 (docs/superpowers/plans/2026-10-30-chip-policy-2.md) ----------------------


def test_optimizer_policy_uses_the_seasons_loso_table(league):
    view, pool, xp = league[8]
    rules = backtest_rules(2023)
    state = random_state(view, rules, 3)
    ctx = DecisionContext(view, state, rules, pool, xp)
    params = replace(SMALL, chip_value={})
    # A Wildcard held is worth 0 here: the optimizer plays it at GW8 ...
    zero = {2023: {"wildcard": (0.0,)}}
    free = OptimizerPolicy("rolling", params, chips=True, season_tables=zero)
    assert free.decide(ctx).chip == "wildcard"
    # ... but 500 in 2023's table: never played (2022's table, with 0, is not used).
    tables = {2023: {"wildcard": (500.0,)}, 2022: {"wildcard": (0.0,)}}
    policy = OptimizerPolicy("rolling", params, chips=True, season_tables=tables)
    assert "loso#" in policy.name and policy.name != free.name
    assert policy.decide(ctx).chip != "wildcard"
    # The same decision as the optimizer with 2023's table as `chip_value_left`.
    direct = OptimizerPolicy("rolling", replace(params, chip_value_left=tables[2023]), True)
    assert policy.decide(ctx) == direct.decide(ctx)
    missing = OptimizerPolicy("rolling", params, chips=True, season_tables={2022: {}})
    with pytest.raises(ValueError, match="no leave-one-season-out table for season 2023"):
        missing.decide(ctx)


def test_optimizer_policy_loso_name_pins_every_table():
    def name(tables):
        return OptimizerPolicy("v1", SMALL, chips=True, season_tables=tables).name

    a = name({2022: {"3xc": (1.0, 2.0, 3.0)}, 2023: {"3xc": (1.0,)}})
    assert a == name({2023: {"3xc": (1,)}, 2022: {"3xc": [1, 2, 3]}})  # canonical
    assert a != name({2022: {"3xc": (1.0, 2.5, 3.0)}, 2023: {"3xc": (1.0,)}})
    assert a != name({2021: {"3xc": (1.0, 2.0, 3.0)}, 2023: {"3xc": (1.0,)}})
    # a short hash, not the tables: 7 seasons x 24 keys would run to kilobytes
    big = {s: {f"k{i}": tuple(range(38)) for i in range(24)} for s in range(2016, 2023)}
    assert len(name(big)) == len(a) < 120
    with pytest.raises(ValueError, match="finite values"):
        name({2022: {"3xc": (-1.0,)}})


def test_optimizer_policy_rule_triple_captain(league):
    from fplopt.backtest.policies import _target_xp_of
    from fplopt.optimize import PlanInput, optimize

    rules = backtest_rules(2023)
    params = replace(SMALL, exclude_chips=("3xc",))
    always = OptimizerPolicy("rolling", params, chips=True, tc_rule=0.0)
    never = OptimizerPolicy("rolling", params, chips=True, tc_rule=1e9)
    assert "tcr=0.0" in always.name and "tcr=1000000000.0" in never.name
    run = list(contexts(league, never))
    plan_chips = [d.chip for _, d in run]
    assert "3xc" not in plan_chips and None in plan_chips
    # TC at the first GW plan #1 plays no chip (TC playable all of set 1), never again.
    first = plan_chips.index(None)
    # The threshold is inclusive: TC at exactly plan #1's captain's xP, not just above.
    context = run[first][0]
    assert context.state.gw_index < 19  # not TC's last GW
    problem = PlanInput.from_context(context.state, context.pool, context.xp, rules, params)
    plans = optimize(problem, params, top_k=1, chips=True, roll=False)
    assert plans.decision().chip is None
    value = _target_xp_of(context.xp, plans.best.first.captain)
    at = OptimizerPolicy("rolling", params, chips=True, tc_rule=value)
    above = OptimizerPolicy("rolling", params, chips=True, tc_rule=math.nextafter(value, math.inf))
    assert at.decide(context).chip == "3xc"
    assert above.decide(context).chip is None
    expected = plan_chips[:first] + ["3xc"] + plan_chips[first + 1 :]
    played = []
    for context, decision in contexts(league, always):
        apply_decision(context.state, decision, context.pool, context.rules)  # valid
        played.append(decision.chip)
    assert played == expected
    with pytest.raises(ValueError, match="tc_rule needs"):
        OptimizerPolicy("rolling", SMALL, chips=True, tc_rule=8.3)  # 3xc not excluded
    with pytest.raises(ValueError, match="tc_rule needs"):
        OptimizerPolicy("rolling", params, chips=False, tc_rule=8.3)


def test_optimizer_policy_rule_triple_captain_in_its_windows_last_gw():
    """Below the threshold, TC is still played in the last GW of its window (19), not
    before; the other set-1 chips are used, so plan #1 plays none."""
    from fplopt.backtest.simulator import season_schedule

    store = DataStore(tables=synthetic_tables(seasons=(2022, 2023)))
    rules = backtest_rules(2023)
    others = tuple((w.chip_id, 5) for w in rules.chips if w.stop == 19 and w.name != "3xc")
    never = OptimizerPolicy("rolling", replace(SMALL, exclude_chips=("3xc",)), True, tc_rule=1e9)
    for gw_index, chip in ((18, None), (19, "3xc")):
        view = store.as_of(season_schedule(store, 2023, gw_index)["deadline_time"].iloc[0])
        pool = player_pool(view)
        state = replace(refresh(random_state(view, rules, 3), pool), chips_used=others)
        context = DecisionContext(view, state, rules, pool, MODELS["rolling"](view))
        decision = never.decide(context)
        apply_decision(state, decision, pool, rules)  # valid
        assert decision.chip == chip


def test_optimizer_policy_fills_known_doubles(league, monkeypatch):
    from fplopt.backtest import policies

    view, pool, xp = league[8]
    rules = backtest_rules(2023)
    ctx = DecisionContext(view, random_state(view, rules, 3), rules, pool, xp)
    doubles = frozenset({ctx.state.gw_index + 8})
    seen, planned = [], []

    def fake_known_doubles(view, season, gw_index, lead):
        seen.append((season, gw_index, lead))
        return doubles

    def recording_optimize(problem, *args, **kwargs):
        planned.append(problem.doubles)
        return real_optimize(problem, *args, **kwargs)

    real_optimize = policies.optimize
    monkeypatch.setattr(policies, "known_doubles", fake_known_doubles)
    monkeypatch.setattr(policies, "optimize", recording_optimize)
    policy = OptimizerPolicy("rolling", SMALL, chips=True, double_lead=10)
    assert "dl=10" in policy.name
    policy.decide(ctx)
    assert seen == [(ctx.state.season, ctx.state.gw_index, 10)]
    assert planned == [doubles]


# --- chip policy 2: CLI (docs/superpowers/specs/2026-10-30-chip-policy-2-design.md) ---------

V_BASE = "horizon=6,decay=0.75,max_hits=1,hit_margin=2,chips=1"
SETTINGS = {"odds_api_key": None, "telegram_bot_token": None, "telegram_admin_chat_id": None}


def _v2_table(path, split, **extra):
    """A format-2 table file: values 4.0, season 2019's leave-one-out table 2.0."""
    keys = ("3xc@1", "3xc@1+0", "3xc@1+1") if split == "double" else ("3xc@1",)
    payload = {
        "source": "s0",
        "seasons": ["2019-20", "2020-21"],
        "format": 2,
        "split": split,
        **extra,
        "values": {k: [4.0] for k in keys},
        "loso": {"2019": {k: [2.0] for k in keys}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def v2_tables(tmp_path, monkeypatch):
    window = _v2_table(tmp_path / "w.json", "window")
    double = _v2_table(tmp_path / "d.json", "double", lead=10, horizon=6)
    monkeypatch.setitem(cli.CHIP_TABLE_FILES, 1, window)
    monkeypatch.setitem(cli.CHIP_TABLE_FILES, 2, double)
    return window, double


def test_cli_chip_splits_match_the_estimator():
    from fplopt.backtest.chip_values import DOUBLE_VALUES_FILE, SPLITS, WINDOW_VALUES_FILE

    assert cli.CHIP_SPLITS == SPLITS
    assert cli.CHIP_TABLE_FILES == {1: WINDOW_VALUES_FILE, 2: DOUBLE_VALUES_FILE}


def test_cli_chip_policy_2_variant_specs_build(v2_tables):
    """The spec's Variants V1-V6, verbatim (k* = 0.8 for V4-V6)."""
    for k in (0.6, 0.8, 1.0):  # V1-V3
        spec = f"optimizer:v1:{V_BASE},chip_shrink={k},chip_split=1,loso=1"
        policy = cli.parse_policy_spec(spec).build()
        assert policy.season_tables == {2019: {"3xc@1": (2.0 * k,)}}
        assert policy.double_lead is None and policy.tc_rule is None
        assert policy.params.exclude_chips == () and policy.params.fh_scale == 1.0
    star = f"optimizer:v1:{V_BASE},chip_shrink=0.8"
    v4 = cli.parse_policy_spec(f"{star},chip_split=1,loso=1,tc_rule=8.3").build()
    v5 = cli.parse_policy_spec(f"{star},chip_split=2,loso=1").build()
    v6 = cli.parse_policy_spec(f"{star},chip_split=1,loso=1,fh_scale=0.4").build()
    assert all(p.season_tables is not None for p in (v4, v5, v6))
    assert (v4.tc_rule, v4.params.exclude_chips, v4.double_lead) == (8.3, ("3xc",), None)
    assert v5.double_lead == 10 and v5.tc_rule is None and v5.params.exclude_chips == ()
    assert v5.season_tables[2019]["3xc@1+1"] == pytest.approx((1.6,))
    assert v6.params.fh_scale == 0.4 and v6.double_lead is None and v6.tc_rule is None
    assert len({p.name for p in (v4, v5, v6)}) == 3


def test_cli_chip_split_without_loso_uses_the_files_full_table(v2_tables):
    policy = cli.parse_policy_spec(f"optimizer:v1:{V_BASE},chip_shrink=0.5,chip_split=1").build()
    assert policy.season_tables is None
    assert policy.params.chip_value_left == {"3xc@1": (2.0,)}


def test_cli_chip_policy_2_bad_specs(v2_tables):
    for bad in (
        "chip_split=1",
        "loso=1",
        "chip_split=3,chip_shrink=1",
        "chip_split=-1,chip_shrink=1",
    ):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_policy_spec(f"optimizer:v1:chips=1,{bad}")
    with pytest.raises(argparse.ArgumentTypeError, match="tc_rule needs chips"):
        cli.parse_policy_spec("optimizer:v1:tc_rule=8.3")
    with pytest.raises(argparse.ArgumentTypeError, match="fh_scale"):
        cli.parse_policy_spec(f"optimizer:v1:{V_BASE},fh_scale=2")
    for name in ("chiprules", "chipgains"):  # the chip keys are the optimizer's only
        for key in ("chip_split=1", "loso=1", "tc_rule=8.3", "fh_scale=0.4"):
            with pytest.raises(argparse.ArgumentTypeError, match="has no parameter"):
                cli.parse_policy_spec(f"{name}:v1:{key}")


def test_cli_double_split_table_must_match_the_policys_lead(v2_tables, tmp_path, monkeypatch):
    spec = f"optimizer:v1:{V_BASE},chip_shrink=0.8,chip_split=2,loso=1"
    for name, extra in (("lead8", {"lead": 8, "horizon": 6}), ("nolead", {"horizon": 6})):
        table = _v2_table(tmp_path / f"{name}.json", "double", **extra)
        monkeypatch.setitem(cli.CHIP_TABLE_FILES, 2, table)
        with pytest.raises(argparse.ArgumentTypeError, match="known-double lead"):
            cli.parse_policy_spec(spec)
    table = _v2_table(tmp_path / "h4.json", "double", lead=10, horizon=4)
    monkeypatch.setitem(cli.CHIP_TABLE_FILES, 2, table)
    with pytest.raises(argparse.ArgumentTypeError, match="horizon"):
        cli.parse_policy_spec(spec)
    monkeypatch.setitem(cli.CHIP_TABLE_FILES, 2, v2_tables[0])  # a window table
    with pytest.raises(argparse.ArgumentTypeError, match="not a double-split table"):
        cli.parse_policy_spec(spec)


def test_cli_pooled_chip_shrink_is_unchanged(tmp_path, monkeypatch, v2_tables):
    """`chip_shrink=k` alone: the pooled table, the same policy (and name) as before."""
    from fplopt.backtest.chip_values import load_terminal_values

    table = tmp_path / "terminal-values.json"
    table.write_text(json.dumps({"values": {"bboost": [2.0, 4.0]}}), encoding="utf-8")
    monkeypatch.setattr(cli, "CHIP_VALUES_FILE", table)
    params = "horizon=6,decay=0.75,max_hits=1,hit_margin=2,chips=1,chip_shrink=0.5"
    built = cli.parse_policy_spec(f"optimizer:v1:{params}").build()
    explicit = cli.parse_policy_spec(f"optimizer:v1:{params},chip_split=0,loso=0").build()
    chip_value_left = load_terminal_values(0.5, table)
    expected = OptimizerPolicy(
        "v1",
        OptimizerParams(
            horizon=6, decay=0.75, max_hits=1, hit_margin=2.0, chip_value_left=chip_value_left
        ),
        chips=True,
    )
    assert built == expected and built.name == expected.name
    assert built.season_tables is None and built.double_lead is None and built.tc_rule is None
    assert explicit == expected


def test_cli_chips_values_loso(tmp_path):
    run = tmp_path / "s0"
    run.mkdir()
    gws = pd.concat([gains_frame(season=2019), gains_frame(season=2020)])
    gws.to_parquet(run / "gws.parquet", index=False)
    out = tmp_path / "t.json"
    settings = cli.Settings(raw_dir=tmp_path, **SETTINGS)
    argv = ["chips", "values", "--gains", str(run), "--split", "window", "--loso"]
    assert cli.main([*argv, "--out", str(out)], settings=settings) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == 2 and payload["split"] == "window"
    assert "lead" not in payload and "horizon" not in payload
    assert set(payload["loso"]) == {"2019", "2020"} and "3xc@1" in payload["values"]
    assert set(payload["values"]) == set(payload["loso"]["2019"])
    # the default (--split pooled, no --loso) writes the old format
    pooled = tmp_path / "p.json"
    argv = ["chips", "values", "--gains", str(run), "--out", str(pooled)]
    assert cli.main(argv, settings=settings) == 0
    payload = json.loads(pooled.read_text(encoding="utf-8"))
    assert "format" not in payload and "loso" not in payload and "3xc" in payload["values"]


def test_cli_chips_values_double_writes_the_lead(tmp_path, monkeypatch):
    from fplopt.backtest import chip_doubles

    run = tmp_path / "s0"
    run.mkdir()
    gws = pd.concat([gains_frame(season=2019), gains_frame(season=2020)])
    gws.to_parquet(run / "gws.parquet", index=False)
    calls = []

    def fake_doubles(store, seasons, lead):
        calls.append((store, list(seasons), lead))
        return {}

    monkeypatch.setattr(cli, "open_data_store", lambda data_dir: ("store", data_dir))
    monkeypatch.setattr(chip_doubles, "known_doubles_by_gw", fake_doubles)
    out = tmp_path / "d.json"
    settings = cli.Settings(raw_dir=tmp_path, data_dir=tmp_path / "data", **SETTINGS)
    argv = ["chips", "values", "--gains", str(run), "--split", "double", "--loso"]
    assert cli.main([*argv, "--out", str(out)], settings=settings) == 0
    assert calls == [(("store", tmp_path / "data"), [2019, 2020], 10)]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert (payload["split"], payload["lead"], payload["horizon"]) == ("double", 10, 6)
    assert "3xc@1+1" in payload["values"] and "3xc@1+0" in payload["loso"]["2020"]
    # the file it wrote builds V5
    monkeypatch.setitem(cli.CHIP_TABLE_FILES, 2, out)
    spec = f"optimizer:v1:{V_BASE},chip_shrink=0.8,chip_split=2,loso=1"
    assert cli.parse_policy_spec(spec).build().double_lead == 10


def test_backtest_config_logs_chip_policy_2_tables_by_hash(tmp_path, monkeypatch, v2_tables):
    pooled = tmp_path / "terminal-values.json"
    pooled.write_text(json.dumps({"source": "run", "values": {"bboost": [2.0]}}), encoding="utf-8")
    monkeypatch.setattr(cli, "CHIP_VALUES_FILE", pooled)
    star = f"optimizer:v1:{V_BASE},chip_shrink=0.8"
    specs = [
        cli.parse_policy_spec(f"{star},chip_split=1,loso=1"),
        cli.parse_policy_spec(f"{star},chip_split=2,loso=1"),
        cli.parse_policy_spec(star),
        cli.parse_policy_spec("chiprules:v1:t_tc=8.3"),
    ]
    args = argparse.Namespace(
        seasons=(2019,), starts="template@1", policy_specs=specs, jobs=1, family="f", command="run"
    )
    logged = cli._backtest_config(args, tmp_path / "out")["chip_values"]
    assert set(logged) == {"0", "1", "2"}
    assert logged["0"] == {"source": "run", "values": {"bboost": [2.0]}}
    window, _ = v2_tables
    assert logged["1"]["sha1"] == hashlib.sha1(window.read_bytes()).hexdigest()
    assert logged["1"]["file"] == str(window) and logged["1"]["split"] == "window"
    assert (logged["2"]["lead"], logged["2"]["horizon"]) == (10, 6)
    assert "values" not in logged["1"] and "loso" not in logged["2"]
