"""Paired evaluation (fplopt.backtest.evaluate): start specs, grids, paired and
per-decision differences, the block bootstrap and the experiment log."""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.backtest.evaluate import (
    EXPERIMENT_COLUMNS,
    StartSpec,
    block_bootstrap,
    build_start_states,
    log_experiment,
    paired_full_run,
    parse_start_specs,
    per_decision,
    run_grid,
    summarize,
)
from fplopt.backtest.policies import GreedyPolicy, RollPolicy
from fplopt.backtest.rules import backtest_rules
from fplopt.backtest.simulator import Caches, HoldoutError, simulate
from fplopt.features.store import DataStore

SEASON = 2023
RULES = backtest_rules(SEASON)
ROLL = RollPolicy("rolling")
GREEDY = GreedyPolicy("rolling")
STARTS = "template@1,random:2@1,random@30"


@pytest.fixture(scope="module")
def league():
    tables = synthetic_tables(seasons=(2022, SEASON))
    return tables, DataStore(tables=tables), Caches()


@pytest.fixture(scope="module")
def grid(league):
    _, store, caches = league
    return run_grid(store, [SEASON], STARTS, [ROLL, GREEDY], caches=caches)


# --- start specs ------------------------------------------------------------------------------


def test_parse_start_specs():
    specs = parse_start_specs("template@1, random:3@1,random:2@20,random@5")
    assert specs == (
        StartSpec("template", 1),
        StartSpec("random", 1, 0),
        StartSpec("random", 1, 1),
        StartSpec("random", 1, 2),
        StartSpec("random", 20, 0),
        StartSpec("random", 20, 1),
        StartSpec("random", 5, 0),
    )
    assert [s.start_id() for s in specs[:2]] == ["template@1", "random0@1"]
    assert parse_start_specs(["template@2"]) == (StartSpec("template", 2),)


@pytest.mark.parametrize(
    "text", ["", "template", "template@0", "random:0@1", "best@1", "template@1,template@1"]
)
def test_parse_start_specs_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_start_specs(text)


def test_template_at_gw1_without_ownership_falls_back_to_gw2(caplog):
    """Without snapshots no ownership is visible at GW1 (as before 2021/22): template@1
    starts at GW2 and says so; random squads still start at GW1."""
    tables = synthetic_tables(snapshots=False)
    store = DataStore(tables=tables)
    with caplog.at_level("INFO"):
        states = build_start_states(store, SEASON, "template@1,random@1,template@2", RULES)
    assert [start_id for start_id, _ in states] == ["template@2", "random0@1"]
    assert [state.gw_index for _, state in states] == [2, 1]
    assert "starting at gw_index 2" in caplog.text
    assert "duplicate start id" in caplog.text


def test_refused_start_states_are_skipped(caplog):
    """A coverage gap (a club with fixtures and no pool players) refuses the start: skipped
    with a warning (at GW1, retried at GW2 first)."""
    tables = synthetic_tables(snapshots=False)
    gaps = dict(tables)
    gw = tables["player_gw"]
    gaps["player_gw"] = gw[~((gw["team_key"] == 3) & (gw["gw"].isin([1, 2, 10])))]
    store = DataStore(tables=gaps)
    states = build_start_states(store, SEASON, "random@1,random@10,random@11", RULES)
    assert [start_id for start_id, _ in states] == ["random0@11"]
    assert caplog.text.count("skipped") == 2


# --- grid and full runs -----------------------------------------------------------------------


def test_run_grid_runs_every_policy_from_every_start(league, grid):
    _, store, _ = league
    assert list(grid.columns[:3]) == ["season", "start_id", "policy"]
    for column in ("gw_index", "net_points", "points", "hit_points", "xg_points"):
        assert column in grid.columns
    for column in ("n_transfers", "captain_regret", "xi_regret"):
        assert column in grid.columns
    sizes = grid.groupby(["start_id", "policy"]).size()
    assert set(sizes.index.get_level_values("start_id")) == {
        "template@1",
        "random0@1",
        "random1@1",
        "random0@30",
    }
    assert sizes.loc[("random0@30", ROLL.name)] == 9
    assert sizes.loc[("template@1", GREEDY.name)] == 38
    # The grid's rows are exactly the single runs'.
    (start_id, state), *_ = build_start_states(store, SEASON, "random@1", RULES)
    run = simulate(store, RULES, GREEDY, state)
    part = grid[(grid["start_id"] == start_id) & (grid["policy"] == GREEDY.name)]
    assert part["net_points"].tolist() == run.gws["net_points"].tolist()


def test_run_grid_refuses_holdout_seasons():
    tables = synthetic_tables(seasons=(2025,))
    with pytest.raises(HoldoutError):
        run_grid(DataStore(tables=tables), [2025], "random@1", [ROLL])


def test_run_grid_needs_unique_policy_names(league):
    _, store, _ = league
    with pytest.raises(ValueError, match="unique"):
        run_grid(store, [SEASON], "random@30", [ROLL, RollPolicy("rolling")])


def test_paired_full_run(grid):
    pairs = paired_full_run(grid, GREEDY, ROLL.name)
    assert len(pairs) == len(grid) // 2
    a = grid[grid["policy"] == GREEDY.name].set_index(["start_id", "gw_index"])
    b = grid[grid["policy"] == ROLL.name].set_index(["start_id", "gw_index"])
    row = pairs.iloc[5]
    key = (row["start_id"], row["gw_index"])
    assert row["diff"] == a.loc[key, "net_points"] - b.loc[key, "net_points"]
    xg = a.loc[key, "xg_net_points"] - b.loc[key, "xg_net_points"]
    assert row["diff_xg"] == pytest.approx(xg)
    same = paired_full_run(grid, ROLL, ROLL)
    assert (same["diff"] == 0).all() and (same["diff_xg"] == 0).all()
    with pytest.raises(ValueError, match="no results"):
        paired_full_run(grid, "nobody", ROLL)


def test_paired_full_run_xg_diff_is_null_if_either_arm_is():
    rows = []
    for policy, xg in (("a", [1.0, None]), ("b", [0.5, 2.0])):
        for gw, value in enumerate(xg, start=1):
            rows.append(
                {
                    "season": 2023,
                    "start_id": "s",
                    "policy": policy,
                    "gw": gw,
                    "gw_index": gw,
                    "net_points": 10 * gw,
                    "xg_net_points": value,
                }
            )
    results = pd.DataFrame(rows).astype({"xg_net_points": "Float64"})
    pairs = paired_full_run(results, "a", "b")
    assert pairs["diff"].tolist() == [0, 0]
    assert pairs["diff_xg"].iloc[0] == 0.5 and pd.isna(pairs["diff_xg"].iloc[1])


# --- per-decision -----------------------------------------------------------------------------


def test_per_decision_with_identical_policies_is_all_zero(league):
    _, store, caches = league
    out = per_decision(store, [SEASON], "random@30", GREEDY, GREEDY, ROLL, k=3, caches=caches)
    assert len(out) == 9
    assert out["same_decision"].all()
    assert (out["diff"] == 0).all() and (out["diff_xg"] == 0).all()
    assert out["k"].tolist() == [3] * 7 + [2, 1]  # truncated at the season's end


def test_per_decision_arms_differ_only_in_the_decision_at_t(league):
    """Reference = B = roll and continuation = roll: arm B is the reference run itself, so
    its window points are the roll run's; arm A only adds A's transfers at t."""
    _, store, caches = league
    k = 4
    out = per_decision(store, [SEASON], "template@1", GREEDY, ROLL, ROLL, k=k, caches=caches)
    (_, start), *_ = build_start_states(store, SEASON, "template@1", RULES)
    ref = simulate(store, RULES, ROLL, start, caches=caches)
    net = ref.gws["net_points"].to_numpy()
    xg = ref.gws["xg_net_points"].to_numpy(dtype="float64")
    for i, row in enumerate(out.itertuples()):
        assert row.points_b == net[i : i + k].sum()
        assert row.xg_b == pytest.approx(xg[i : i + k].sum())
        assert row.transfers_b == 0
        assert row.diff == row.points_a - row.points_b
        if row.same_decision:
            assert row.diff == 0 and row.transfers_a == 0
        else:
            assert row.transfers_a > 0
    assert (~out["same_decision"]).sum() > 10
    # With reference = A, the states come from greedy's own run: arm A at t is greedy's run
    # for one GW, then roll.
    out_a = per_decision(
        store, [SEASON], "template@1", GREEDY, ROLL, ROLL, k=1, reference="a", caches=caches
    )
    greedy_run = simulate(store, RULES, GREEDY, start, caches=caches)
    assert out_a["points_a"].tolist() == greedy_run.gws["net_points"].tolist()


def test_per_decision_checks_its_arguments(league):
    _, store, _ = league
    with pytest.raises(ValueError, match="k must"):
        per_decision(store, [SEASON], "random@30", ROLL, ROLL, ROLL, k=0)
    with pytest.raises(ValueError, match="reference"):
        per_decision(store, [SEASON], "random@30", ROLL, ROLL, ROLL, reference="c")


# --- bootstrap --------------------------------------------------------------------------------


def diffs_frame(values_by_season: dict[int, list[float]], starts: int = 1) -> pd.DataFrame:
    rows = [
        {"season": season, "start_id": f"s{s}", "gw_index": gw, "diff": value}
        for season, values in values_by_season.items()
        for gw, value in enumerate(values, start=1)
        for s in range(starts)
    ]
    return pd.DataFrame(rows)


def test_bootstrap_of_a_constant_collapses():
    result = block_bootstrap(diffs_frame({2021: [2.0] * 38, 2022: [2.0] * 30}), n_boot=500)
    assert (result.mean, result.ci_low, result.ci_high) == pytest.approx((2.0, 2.0, 2.0))
    assert result.p_one_sided == 0.0
    assert (result.n_seasons, result.n_gws) == (2, 68)
    negative = block_bootstrap(diffs_frame({2021: [-1.0] * 38}), n_boot=200)
    assert negative.p_one_sided == 1.0


def test_bootstrap_of_zero_mean_noise_has_p_near_one_half():
    rng = np.random.default_rng(1)
    values = {season: rng.normal(0, 10, 38) for season in range(2016, 2025)}
    centred = {season: list(v - v.mean()) for season, v in values.items()}
    result = block_bootstrap(diffs_frame(centred), n_boot=4000)
    assert result.mean == pytest.approx(0.0, abs=1e-9)
    assert result.p_one_sided == pytest.approx(0.5, abs=0.05)
    assert result.ci_low < 0 < result.ci_high


def test_bootstrap_is_deterministic_per_seed():
    rng = np.random.default_rng(2)
    diffs = diffs_frame({2021: list(rng.normal(1, 5, 38)), 2022: list(rng.normal(1, 5, 38))})
    assert block_bootstrap(diffs, seed=3) == block_bootstrap(diffs, seed=3)
    assert block_bootstrap(diffs, seed=3) != block_bootstrap(diffs, seed=4)


def test_bootstrap_blocks_stay_within_seasons():
    """Season A is +1 every GW, season B −1: resampling GW blocks within each season keeps
    every season's sum, so every bootstrap mean is exactly the observed one."""
    result = block_bootstrap(diffs_frame({2021: [1.0] * 10, 2022: [-1.0] * 30}), n_boot=500)
    assert (result.mean, result.ci_low, result.ci_high) == pytest.approx((-0.5, -0.5, -0.5))
    assert result.p_one_sided == 1.0


def test_bootstrap_averages_over_starts_and_drops_nulls():
    diffs = diffs_frame({2021: [1.0, 2.0, 3.0]}, starts=2)
    diffs.loc[(diffs["start_id"] == "s1"), "diff"] += 2  # per GW: 1 and 3 -> mean 2 ...
    diffs.loc[len(diffs)] = {"season": 2021, "start_id": "s2", "gw_index": 4, "diff": None}
    result = block_bootstrap(diffs, n_boot=100)
    assert result.mean == pytest.approx(3.0)  # (2 + 3 + 4) / 3
    assert result.n_gws == 3
    empty = block_bootstrap(diffs.assign(diff=None), n_boot=10)
    assert np.isnan(empty.mean) and empty.n_gws == 0


def test_bootstrap_ci_level():
    rng = np.random.default_rng(5)
    diffs = diffs_frame({2021: list(rng.normal(0, 5, 38)), 2022: list(rng.normal(0, 5, 38))})
    wide = block_bootstrap(diffs, ci=0.99)
    narrow = block_bootstrap(diffs, ci=0.5)
    assert wide.ci_low < narrow.ci_low < narrow.ci_high < wide.ci_high


# --- summaries and the experiment log ---------------------------------------------------------


def test_summarize(league, grid):
    _, store, caches = league
    decisions = per_decision(store, [SEASON], "random@30", GREEDY, ROLL, ROLL, caches=caches)
    summary = summarize(grid, [(GREEDY, ROLL)], decisions, n_boot=200)
    totals = summary.season_totals.set_index("policy")
    runs = grid.groupby(["policy", "start_id"])["net_points"].sum()
    assert totals.loc[ROLL.name, "total"] == pytest.approx(runs[ROLL.name].mean())
    assert totals.loc[ROLL.name, "n_starts"] == 4
    assert totals.loc[ROLL.name, "n_transfers"] == 0
    assert set(summary.policies["policy"]) == {ROLL.name, GREEDY.name}
    comparisons = summary.comparisons
    assert len(comparisons) == 4
    assert set(zip(comparisons["method"], comparisons["metric"], strict=True)) == {
        ("full_run", "realized"),
        ("full_run", "xg"),
        ("per_decision", "realized"),
        ("per_decision", "xg"),
    }
    full = comparisons[
        (comparisons["method"] == "full_run") & (comparisons["metric"] == "realized")
    ]
    expected = block_bootstrap(paired_full_run(grid, GREEDY, ROLL), n_boot=200)
    assert full["mean"].iloc[0] == pytest.approx(expected.mean)
    assert full["ci_low"].iloc[0] == pytest.approx(expected.ci_low)
    json.dumps(summary.to_dict())


def test_log_experiment_appends_with_one_header(tmp_path):
    path = tmp_path / "results" / "experiments.csv"
    log_experiment(path, "fplopt backtest run", {"policy": "greedy"}, {"total": 1.5}, 1)
    log_experiment(path, "fplopt backtest compare", {"a": "x"}, {"diff": -2}, 3)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert path.read_text(encoding="utf-8").count("timestamp") == 1
    assert list(rows[0]) == list(EXPERIMENT_COLUMNS)
    assert [r["command"] for r in rows] == ["fplopt backtest run", "fplopt backtest compare"]
    assert json.loads(rows[1]["config"]) == {"a": "x"}
    assert json.loads(rows[1]["metrics"]) == {"diff": -2}
    assert [r["n_variants"] for r in rows] == ["1", "3"]
    assert rows[0]["git_sha"] and rows[0]["timestamp"]
