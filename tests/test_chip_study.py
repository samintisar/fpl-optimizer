"""The chip study's code (docs/superpowers/plans/2026-10-11-chip-policy.md): the gain probe
(S0), the rule baseline H, the terminal value estimator, the full-run window t-test and its
size, and the CLI specs."""

from __future__ import annotations

import json
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
    assert config["chip_values"] == {"source": "run", "values": {"bboost": [2.0]}}
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
