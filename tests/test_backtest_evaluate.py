"""Paired evaluation (fplopt.backtest.evaluate): start specs, grids, paired and
per-decision differences, the block bootstrap and the experiment log."""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.backtest import evaluate
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
from fplopt.backtest.simulator import Caches, HoldoutError, season_schedule, simulate
from fplopt.features.leakcheck import corrupt_future
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


def refuse_start_states(*args, **kwargs):
    raise AssertionError("start states built before the holdout check")


def test_holdout_seasons_are_refused_before_anything_runs(league, monkeypatch):
    """A holdout season anywhere in the list fails up front, even after a valid season and
    on a store without its data."""
    _, store, _ = league
    monkeypatch.setattr(evaluate, "build_start_states", refuse_start_states)
    with pytest.raises(HoldoutError, match="2025-26 is a holdout season"):
        run_grid(store, iter([SEASON, 2025]), "random@30", [ROLL])
    with pytest.raises(HoldoutError, match="2025-26 is a holdout season"):
        per_decision(store, [SEASON, 2025], "random@30", GREEDY, ROLL, ROLL)


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
    out = per_decision(
        store, [SEASON], "random@30", GREEDY, GREEDY, ROLL, k=3, stride=1, caches=caches
    )
    assert len(out) == 9
    assert out["same_decision"].all()
    assert (out["diff"] == 0).all() and (out["diff_xg"] == 0).all()
    assert out["k"].tolist() == [3] * 7 + [2, 1]  # truncated at the season's end


def test_per_decision_arms_differ_only_in_the_decision_at_t(league):
    """Reference = B = roll and continuation = roll: arm B is the reference run itself, so
    its window points are the roll run's; arm A only adds A's transfers at t."""
    _, store, caches = league
    k = 4
    out = per_decision(
        store, [SEASON], "template@1", GREEDY, ROLL, ROLL, k=k, stride=1, caches=caches
    )
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


@pytest.fixture(scope="module")
def every_gw(league):
    """Per-decision greedy vs roll from template@1 and random@30 at every GW (stride 1)."""
    _, store, caches = league
    starts = "template@1,random@30"
    return per_decision(store, [SEASON], starts, GREEDY, ROLL, ROLL, stride=1, caches=caches)


def test_per_decision_windows_tile_the_run_without_overlap(league, every_gw):
    """Default stride = k: decisions at gw_index 1, 1 + k, 1 + 2k, ... for every start (a
    start off that grid waits for its next point), so the windows of all starts coincide,
    don't overlap and cover every GW from the first decision on; each row equals the
    every-GW row of the same decision GW."""
    _, store, caches = league
    out = per_decision(store, [SEASON], "template@1,random@30", GREEDY, ROLL, ROLL, caches=caches)
    assert (out["stride"] == 4).all() and (every_gw["stride"] == 1).all()
    for start_id, first in (("template@1", 1), ("random0@30", 33)):
        part = out[out["start_id"] == start_id]
        assert part["gw_index"].tolist() == list(range(first, 39, 4))
        covered = [g for r in part.itertuples() for g in range(r.gw_index, r.gw_index + r.k)]
        assert covered == list(range(first, 39))
    assert out.loc[out["start_id"] == "template@1", "k"].tolist() == [4] * 9 + [2]
    assert out.loc[out["start_id"] == "random0@30", "k"].tolist() == [4, 2]
    windows = out.groupby("start_id")[["gw_index", "k"]].apply(lambda f: set(map(tuple, f.values)))
    assert windows["random0@30"] <= windows["template@1"]  # one grid: same cells
    same = every_gw.merge(out[["start_id", "gw_index"]], on=["start_id", "gw_index"])
    columns = [c for c in out.columns if c != "stride"]
    pd.testing.assert_frame_equal(out[columns], same[columns])


def test_per_decision_stride_two(league, every_gw):
    _, store, caches = league
    out = per_decision(store, [SEASON], "random@30", GREEDY, ROLL, ROLL, stride=2, caches=caches)
    assert out["gw_index"].tolist() == [31, 33, 35, 37]  # gw_index 1 + 2j
    assert out["k"].tolist() == [4, 4, 4, 2]
    expected = every_gw[every_gw["gw_index"].isin(out["gw_index"])]
    expected = expected[expected["start_id"] == "random0@30"].reset_index(drop=True)
    pd.testing.assert_frame_equal(out.drop(columns="stride"), expected.drop(columns="stride"))


def test_per_decision_decisions_do_not_depend_on_data_after_the_deadline(league):
    """Corrupt every row available at or after GW t's deadline: windows starting at or
    before t keep their decisions (the reference states and each arm's decision at its
    start only see earlier data); windows that end before t keep everything."""
    tables, store, caches = league
    t = 13
    deadline = season_schedule(store, SEASON, t)["deadline_time"].iloc[0]
    clean = per_decision(store, [SEASON], "template@1", GREEDY, ROLL, ROLL, caches=caches)
    # Seed 3 leaves some (corrupted) GW13 outcome rows, so the reference run goes on past t
    # and the window starting at t is evaluated (other seeds can empty GW13 and stop it).
    altered = DataStore(tables=corrupt_future(tables, deadline, seed=3))
    out = per_decision(altered, [SEASON], "template@1", GREEDY, ROLL, ROLL)
    decision = ["season", "start_id", "gw", "gw_index", "stride", "same_decision"]
    decision += ["transfers_a", "transfers_b"]
    early = out["gw_index"] <= t
    assert out.loc[early, "gw_index"].tolist() == [1, 5, 9, 13]
    pd.testing.assert_frame_equal(
        out.loc[early, decision], clean.loc[clean["gw_index"] <= t, decision]
    )
    assert (~clean.loc[clean["gw_index"] <= t, "same_decision"]).any()  # not vacuous
    done = clean["gw_index"] + clean["k"] - 1 < t
    assert done.sum() == 3
    pd.testing.assert_frame_equal(out[done], clean[done])


def test_per_decision_checks_its_arguments(league):
    _, store, _ = league
    with pytest.raises(ValueError, match="k must"):
        per_decision(store, [SEASON], "random@30", ROLL, ROLL, ROLL, k=0)
    with pytest.raises(ValueError, match="reference"):
        per_decision(store, [SEASON], "random@30", ROLL, ROLL, ROLL, reference="c")
    with pytest.raises(ValueError, match="stride must"):
        per_decision(store, [SEASON], "random@30", ROLL, ROLL, ROLL, stride=0)
    with pytest.raises(ValueError, match="continuation"):
        per_decision(store, [SEASON], "random@30", ROLL, ROLL, "mine")


def test_per_decision_own_continuation(league):
    """continuation="own": each arm keeps playing its own policy after t, so arm A's window
    is A's own run from the reference state; with reference B = roll, arm B is the roll run
    itself. Same decisions at t no longer imply equal windows."""
    _, store, caches = league
    k = 4
    out = per_decision(store, [SEASON], "template@1", GREEDY, ROLL, "own", k=k, caches=caches)
    assert (out["continuation"] == "own").all() and (out["reference"] == "b").all()
    (_, start), *_ = build_start_states(store, SEASON, "template@1", RULES)
    ref = simulate(store, RULES, ROLL, start, caches=caches)
    net = ref.gws["net_points"].to_numpy()
    for row in out.itertuples():
        i = row.gw_index - 1
        assert row.points_b == net[i : i + k].sum()
        own = simulate(
            store, RULES, GREEDY, ref.states[i], end_gw_index=row.gw_index + k - 1, caches=caches
        )
        assert row.points_a == own.total
    shared = per_decision(store, [SEASON], "template@1", GREEDY, ROLL, ROLL, k=k, caches=caches)
    assert (shared["continuation"] == ROLL.name).all()
    assert (out["points_a"] != shared["points_a"]).any()  # follow-up moves now count
    # reference="a" with own continuation: arm A is A's own run, window by window.
    out_a = per_decision(
        store, [SEASON], "template@1", GREEDY, ROLL, "own", k=k, reference="a", caches=caches
    )
    greedy_run = simulate(store, RULES, GREEDY, start, caches=caches)
    points = greedy_run.gws["net_points"].to_numpy()
    assert out_a["points_a"].tolist() == [
        points[g - 1 : g - 1 + k].sum() for g in out_a["gw_index"]
    ]


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


def no_effect_diffs(rng, n_seasons, window, gws=38):
    """A/A data: iid N(0, 1) per-GW effects summed over non-overlapping windows of `window`
    GWs (the last one shorter), one row per window per season; window 1 = full-run GWs."""
    rows = []
    for season in range(n_seasons):
        effects = rng.normal(0, 1, gws)
        for start in range(0, gws, window):
            rows.append((season, start + 1, effects[start : start + window].sum()))
    return pd.DataFrame(rows, columns=["season", "gw_index", "diff"])


def rejection_rate(n_seasons, window, block_length, n_rep=400, n_boot=400):
    rng = np.random.default_rng(n_seasons * 100 + window)
    rejected = 0
    for rep in range(n_rep):
        diffs = no_effect_diffs(rng, n_seasons, window)
        result = block_bootstrap(diffs, block_length=block_length, n_boot=n_boot, seed=rep)
        rejected += result.p_one_sided < 0.10
    return rejected / n_rep


@pytest.mark.parametrize(
    ("design", "n_seasons", "window", "block_length", "limit"),
    [
        ("per-decision", 4, 4, 1, 0.16),
        ("per-decision", 1, 4, 1, 0.20),
        ("full-run", 4, 1, 4, 0.16),
        ("full-run", 1, 1, 4, 0.16),
    ],
)
def test_bootstrap_size_under_no_effect(design, n_seasons, window, block_length, limit):
    """A/A placebo: with no effect, p < 0.10 should happen about 10% of the time. The
    designs are what `summarize` bootstraps: per-decision k = 4 windows (stride 4, so a
    4-GW block is one window) and full-run GWs (4-GW blocks).

    Measured 2026-10-06 with 2000 replicates and n_boot 2000 (SE ~0.007-0.009):
    per-decision (block 1 window) 0.141 for 1 season, 0.122 for 4, 0.122 for 9;
    full run (block 4 GWs) 0.131 / 0.124 / 0.122. Blocks of 4 *windows* (16 GWs) would give
    0.182 / 0.167 / 0.163: with ~10 windows a season, the circular block bootstrap
    understates the variance by about (n − b)/(n − 1). One season is anti-conservative
    (percentile bootstrap on ~10 cells)."""
    rate = rejection_rate(n_seasons, window, block_length)
    assert 0.04 <= rate <= limit, f"{design}, {n_seasons} season(s): size {rate:.3f}"


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
    assert len(comparisons) == 12  # 2023/24 is a validate season: splits all + validate
    assert set(comparisons["split"]) == {"all", "validate"}
    assert set(zip(comparisons["method"], comparisons["metric"], strict=True)) == {
        (method, metric)
        for method in ("full_run", "per_decision")
        for metric in ("realized", "realized@xg", "xg")
    }
    full = comparisons[
        (comparisons["method"] == "full_run")
        & (comparisons["metric"] == "realized")
        & (comparisons["split"] == "all")
    ]
    # One season: no season-level t-test; one variant: no deflation.
    assert full["p_season_t"].isna().all()
    assert (full["deflated_mean"] == full["mean"]).all()
    expected = block_bootstrap(paired_full_run(grid, GREEDY, ROLL), n_boot=200)
    assert full["mean"].iloc[0] == pytest.approx(expected.mean)
    assert full["ci_low"].iloc[0] == pytest.approx(expected.ci_low)
    json.dumps(summary.to_dict())


def test_summarize_reports_transfer_gains(grid):
    """Predicted vs realized transfer gain per policy over GWs after GW1 with transfers:
    means, hits and the least-squares slope of realized on predicted."""
    summary = summarize(grid)
    gains = summary.transfer_gains.set_index("policy")
    assert list(summary.transfer_gains.columns) == list(evaluate.GAIN_COLUMNS)
    rows = grid[(grid["policy"] == GREEDY.name) & (grid["n_transfers"] > 0)]
    rows = rows[rows["gw_index"] > 1]
    pred, real = rows["pred_gain"].astype(float), rows["real_gain"].astype(float)
    greedy = gains.loc[GREEDY.name]
    assert greedy["n_decisions"] == len(rows) > 20
    assert greedy["pred"] == pytest.approx(pred.mean())
    assert greedy["real"] == pytest.approx(real.mean())
    assert greedy["hit_points"] == 0 and greedy["real_net"] == pytest.approx(real.mean())
    slope, intercept = np.polyfit(pred, real, 1)
    assert greedy["slope"] == pytest.approx(slope)
    assert greedy["intercept"] == pytest.approx(intercept)
    assert gains.loc[ROLL.name, "n_decisions"] == 0
    assert np.isnan(gains.loc[ROLL.name, "slope"])
    assert "transfer_gains" in json.loads(json.dumps(summary.to_dict()))
    # Hand-made: real = 0.5 * pred - 1 exactly; hits are subtracted in the net means.
    frame = pd.DataFrame(
        {
            "policy": "p",
            "gw_index": [1, 2, 3, 4, 5],
            "n_transfers": [15, 1, 2, 0, 1],
            "hit_points": [0, 0, 4, 0, 0],
            "pred_gain": [50.0, 2.0, 6.0, None, 10.0],
            "real_gain": [9.0, 0.0, 2.0, None, 4.0],
        }
    ).astype({"pred_gain": "Float64", "real_gain": "Float64"})
    (row,) = evaluate.transfer_gain_summary(frame).to_dict("records")
    assert row["n_decisions"] == 3 and row["transfers"] == pytest.approx(4 / 3)
    assert (row["slope"], row["intercept"]) == (pytest.approx(0.5), pytest.approx(-1.0))
    assert row["pred"] == pytest.approx(6.0) and row["real"] == pytest.approx(2.0)
    assert row["pred_net"] == pytest.approx(6.0 - 4 / 3)
    assert row["real_net"] == pytest.approx(2.0 - 4 / 3)


# --- parallel runs ----------------------------------------------------------------------------


def test_parallel_grid_and_per_decision_equal_serial(league, grid):
    """jobs=2 (spawned workers, each with its own store over a copy of the tables and its
    own caches) gives exactly the serial results, in the same order."""
    tables, store, caches = league
    parallel = run_grid(store, [SEASON], STARTS, [ROLL, GREEDY], jobs=2)
    pd.testing.assert_frame_equal(parallel, grid)
    serial = per_decision(store, [SEASON], "random:2@25", GREEDY, ROLL, ROLL, caches=caches)
    two = per_decision(store, [SEASON], "random:2@25", GREEDY, ROLL, ROLL, jobs=2)
    pd.testing.assert_frame_equal(two, serial)
    assert len(serial) > 0


def test_parallel_optimizer_with_chips_equals_serial(league):
    """The optimizer policy (with chip scenarios) through the spawn path: workers rebuild
    params and policies from pickles and solve deterministically, so jobs=2 equals jobs=1,
    and every decision records an Optimal solve."""
    pytest.importorskip("highspy")
    from fplopt.backtest.policies import OptimizerPolicy
    from fplopt.optimize import OptimizerParams

    _, store, _ = league
    params = OptimizerParams(horizon=2, prune_n={1: 3, 2: 6, 3: 6, 4: 4})
    policy = OptimizerPolicy("rolling", params, chips=True)
    serial = run_grid(store, [SEASON], "random:2@34", [policy, GREEDY], caches=Caches())
    parallel = run_grid(store, [SEASON], "random:2@34", [policy, GREEDY], jobs=2)
    pd.testing.assert_frame_equal(parallel, serial)
    mine = serial[serial["policy"] == policy.name]
    assert (mine["solver_status"] == "Optimal").all() and mine["mip_gap"].notna().all()
    assert serial.loc[serial["policy"] == GREEDY.name, "solver_status"].isna().all()
    one = per_decision(store, [SEASON], "random:2@35", policy, GREEDY, "own", k=2)
    two = per_decision(store, [SEASON], "random:2@35", policy, GREEDY, "own", k=2, jobs=2)
    pd.testing.assert_frame_equal(two, one)
    assert len(one) > 0


def _die_once(store, caches, rules, unit, payload):
    """A unit that kills its worker process (like a worker crashing at start-up): only the
    first time any unit runs in a worker (a marker file), or every time with `always`; in
    the parent process it always works."""
    import os
    from pathlib import Path

    directory, parent_pid, always = payload
    marker = Path(directory) / "died"
    if os.getpid() != parent_pid and (always or not marker.exists()):
        marker.touch()
        os._exit(3)
    return (unit[1], os.getpid() == parent_pid)


def _poison(store, caches, rules, unit, payload):
    """Unit `payload[0]` kills every worker process it runs in; the others work."""
    import os

    bad, parent_pid = payload
    if unit[1] == bad and os.getpid() != parent_pid:
        os._exit(5)
    return unit[1]


def _fail_unit(store, caches, rules, unit, payload):
    raise ValueError(f"unit {unit[1]} is bad")


def test_dead_workers_are_replaced_by_their_peers_then_this_process(league, tmp_path, caplog):
    """A worker dying (as one did on Windows under heavy CPU load, see `_run_pool`) does
    not fail the run: its unit goes to the other workers; if every worker dies the units
    left go to a fresh pool, and after POOL_ATTEMPTS dead pools to this process. Results
    come back in unit order; a unit's own exception propagates with its type."""
    import os

    _, store, _ = league
    units = [(SEASON, f"u{i}", None) for i in range(4)]
    once = tmp_path / "once"
    once.mkdir()
    with caplog.at_level("WARNING"):
        out = evaluate.run_units(
            store, Caches(), backtest_rules, _die_once, units, (str(once), os.getpid(), False), 2
        )
    assert out == [(f"u{i}", False) for i in range(4)]  # all done by worker processes
    assert "died (exit code 3)" in caplog.text and "re-queued" in caplog.text
    assert "remaining unit(s) in this process" not in caplog.text
    caplog.clear()
    always = tmp_path / "always"
    always.mkdir()
    with caplog.at_level("WARNING"):
        out = evaluate.run_units(
            store, Caches(), backtest_rules, _die_once, units, (str(always), os.getpid(), True), 2
        )
    assert out == [(f"u{i}", True) for i in range(4)]  # fell back to this process
    assert caplog.text.count("worker pool broke") == evaluate.POOL_ATTEMPTS
    assert "remaining unit(s) in this process" in caplog.text
    # Every unit fails; the first worker to report decides which one is named.
    with pytest.raises(ValueError, match=r"unit u\d is bad") as info:
        evaluate.run_units(store, Caches(), backtest_rules, _fail_unit, units, None, 2)
    assert any("in a worker process" in note for note in info.value.__notes__)


def test_a_unit_that_keeps_killing_workers_stops_the_run(league, caplog):
    """A unit that crashes its worker every time (a native crash, the OOM killer) is not
    passed from worker to worker and then run in this process, which it would crash too:
    after UNIT_DEATHS deaths the run fails naming it."""
    import os

    _, store, _ = league
    units = [(SEASON, f"u{i}", None) for i in range(4)]
    with caplog.at_level("WARNING"), pytest.raises(evaluate.UnitCrashed, match="u1 killed 3"):
        evaluate.run_units(store, Caches(), backtest_rules, _poison, units, ("u1", os.getpid()), 3)
    assert "remaining unit(s) in this process" not in caplog.text


def test_store_spec_and_jobs_validation(league, tmp_path):
    tables, store, caches = league
    spec = evaluate.store_spec(store)
    assert set(spec) == set(tables) and spec["gameweek"] is tables["gameweek"]
    assert evaluate.store_spec(DataStore(tmp_path)) == tmp_path
    with pytest.raises(ValueError, match="jobs"):
        run_grid(store, [SEASON], "random@30", [ROLL], jobs=0)


def test_season_t_test_and_t_distribution():
    """One-sided t-test over the per-season means (cells averaged over starts first)."""
    from statistics import mean, stdev

    from fplopt.backtest.evaluate import season_t_test, t_sf

    assert t_sf(0.0, 5) == pytest.approx(0.5)
    assert t_sf(1.0, 1) == pytest.approx(0.25)  # Cauchy
    assert t_sf(-1.0, 1) == pytest.approx(0.75)
    assert t_sf(2.015048, 5) == pytest.approx(0.05, abs=1e-6)  # t_{0.95, 5}
    assert t_sf(1.372184, 10) == pytest.approx(0.10, abs=1e-6)  # t_{0.90, 10}
    frame = diffs_frame({2016: [1.0, 3.0], 2017: [0.0, 1.0], 2018: [2.0, 2.0]}, starts=2)
    p, n = season_t_test(frame)
    means = [2.0, 0.5, 2.0]
    t = mean(means) / (stdev(means) / 3**0.5)
    assert n == 3 and p == pytest.approx(t_sf(t, 2))
    assert np.isnan(season_t_test(diffs_frame({2016: [1.0, 2.0]}))[0])


def test_deflated_mean_haircut():
    """mean − SE·sqrt(2 ln N), SE from the 80% CI width (2 × 1.2816 SE)."""
    from fplopt.backtest.evaluate import deflated_mean

    se, z = 0.5, 1.2815515655446004
    low, high = 1.0 - z * se, 1.0 + z * se
    assert deflated_mean(1.0, low, high, 1, 0.8) == 1.0
    expected = 1.0 - se * (2 * np.log(8)) ** 0.5
    assert deflated_mean(1.0, low, high, 8, 0.8) == pytest.approx(expected)


def test_summarize_splits_develop_and_validate():
    """Every comparison row is reported for all seasons, develop (2016/17-2022/23) and
    validate (2023/24-2024/25); `n_variants` deflates each split's mean."""
    rng = np.random.default_rng(3)
    rows = []
    for season, shift in ((2021, 2.0), (2022, 1.0), (2023, -1.0), (2024, 0.0)):
        for gw in range(1, 39):
            value = shift + rng.normal(0, 3)
            rows.append(
                {"season": season, "start_id": "s", "gw": gw, "gw_index": gw}
                | {"diff": value, "diff_xg": value}
            )
    diffs = pd.DataFrame(rows).astype({"diff_xg": "Float64"})
    out = evaluate._comparison(
        diffs, "a", "b", "full_run", block_length=4, n_boot=200, seed=0, ci=0.8, n_variants=8
    )
    frame = pd.DataFrame(out).set_index(["split", "metric"])
    assert set(frame.index.get_level_values("split")) == {"all", "develop", "validate"}
    develop = frame.loc[("develop", "realized")]
    validate = frame.loc[("validate", "realized")]
    assert develop["n_seasons"] == 2 and validate["n_seasons"] == 2
    assert develop["mean"] == pytest.approx(diffs.loc[diffs["season"] <= 2022, "diff"].mean())
    assert validate["mean"] == pytest.approx(diffs.loc[diffs["season"] >= 2023, "diff"].mean())
    assert (frame["deflated_mean"] < frame["mean"]).all()
    assert 0 <= frame.loc[("all", "realized"), "p_season_t"] <= 1


def test_default_ci_is_80_percent():
    """Two-sided 80% = the one-sided α = 0.10 bound of the go-live gate."""
    rng = np.random.default_rng(6)
    diffs = diffs_frame({2021: list(rng.normal(0, 5, 38)), 2022: list(rng.normal(0, 5, 38))})
    assert block_bootstrap(diffs) == block_bootstrap(diffs, ci=0.8)


def decision_frame(rows, stride):
    """A per-decision-like frame: (season, gw_index, k, diff, diff_xg) per row."""
    frame = pd.DataFrame(rows, columns=["season", "gw_index", "k", "diff", "diff_xg"])
    return frame.assign(start_id="s", policy_a="a", policy_b="b", stride=stride).astype(
        {"diff_xg": "Float64"}
    )


def tiny_results():
    """A two-policy, two-season grid: policy a has a null xG GW in 2021, b in 2022."""
    rows = []
    for season in (2021, 2022, 2023):
        for policy in ("a", "b"):
            for gw in (1, 2):
                null = (policy, season, gw) in {("a", 2021, 2), ("b", 2022, 1)}
                rows.append(
                    {
                        "season": season,
                        "start_id": "s",
                        "policy": policy,
                        "gw": gw,
                        "gw_index": gw,
                        "net_points": 10 * gw + (policy == "a"),
                        "xg_net_points": None if null else 10.0 * gw + (season - 2020),
                        "n_transfers": 0,
                        "hit_points": 0,
                        "captain_regret": 0,
                        "xi_regret": 0,
                    }
                )
    return pd.DataFrame(rows).astype({"xg_net_points": "Float64"})


def test_summarize_per_decision_season_points_and_xg_sample():
    """Non-overlapping windows: season_diff = Σ window diffs per season / seasons. Overlapping
    (stride 1): each window's diff is divided by its k. `realized@xg` uses only the rows
    with an xG difference, the same sample as `xg`; per-decision blocks are in GWs."""
    results = tiny_results()
    tiling = decision_frame(
        [
            (2021, 1, 4, 8, 2.0),
            (2021, 5, 4, -4, None),
            (2021, 9, 2, 2, 1.0),
            (2022, 1, 4, 6, 3.0),
            (2022, 5, 4, 0, 1.0),
        ],
        stride=4,
    )
    summary = summarize(results, per_decision_diffs=tiling, n_boot=50)
    rows = summary.comparisons.query("split == 'all'").set_index("metric")
    assert rows.loc["realized", "season_diff"] == pytest.approx((8 - 4 + 2 + 6 + 0) / 2)
    assert rows.loc["realized", "mean"] == pytest.approx(12 / 5)
    assert rows.loc["realized@xg", "season_diff"] == pytest.approx((8 + 2 + 6 + 0) / 2)
    assert rows.loc["realized@xg", "n_gws"] == rows.loc["xg", "n_gws"] == 4
    assert rows.loc["xg", "season_diff"] == pytest.approx((2 + 1 + 3 + 1) / 2)
    expected = block_bootstrap(tiling, block_length=1, n_boot=50)  # 4 GWs = 1 window
    assert rows.loc["realized", "ci_low"] == pytest.approx(expected.ci_low)
    overlapping = decision_frame(
        [(2021, 1, 2, 4, 1.0), (2021, 2, 2, 6, 1.0), (2021, 3, 1, 3, 1.0)], stride=1
    )
    summary = summarize(results, per_decision_diffs=overlapping, n_boot=50)
    rows = summary.comparisons.query("split == 'all'").set_index("metric")
    assert rows.loc["realized", "season_diff"] == pytest.approx(4 / 2 + 6 / 2 + 3 / 1)
    assert rows.loc["realized", "mean"] == pytest.approx(13 / 3)
    expected = block_bootstrap(overlapping, block_length=4, n_boot=50)
    assert rows.loc["realized", "ci_low"] == pytest.approx(expected.ci_low)


def test_summarize_full_run_realized_on_the_xg_sample():
    results = tiny_results()
    summary = summarize(results, [("a", "b")], n_boot=50)
    rows = summary.comparisons.query("split == 'all'").set_index("metric")
    assert rows.loc["realized", "n_gws"] == 6
    assert rows.loc["realized@xg", "n_gws"] == rows.loc["xg", "n_gws"] == 4
    assert rows.loc["realized@xg", "season_diff"] == pytest.approx(4 / 3)  # 1 per GW


def test_policy_xg_means_use_the_seasons_common_to_all_policies():
    """a's 2021 xG total is null and b's 2022: both policies' xG means are over 2023 only,
    and `xg_seasons` says so; realized means are over every season."""
    summary = summarize(tiny_results(), n_boot=10)
    totals = summary.season_totals.set_index(["season", "policy"])
    assert pd.isna(totals.loc[(2021, "a"), "xg_total"])
    assert pd.isna(totals.loc[(2022, "b"), "xg_total"])
    policies = summary.policies.set_index("policy")
    assert policies.loc["a", "xg_total"] == policies.loc["b", "xg_total"] == pytest.approx(36.0)
    assert (policies["xg_seasons"] == 1).all() and (policies["n_seasons"] == 3).all()
    assert policies.loc["a", "total"] == pytest.approx(32.0)


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
    assert b"\r" not in path.read_bytes()


def test_log_experiment_writes_strict_json_numbers(tmp_path):
    """NaN/inf become null and numpy scalars numbers (not strings)."""
    path = tmp_path / "experiments.csv"
    metrics = {
        "mean": np.float64(1.5),
        "n": np.int64(3),
        "missing": float("nan"),
        "rows": [{"p": np.float32(0.25), "ci": [np.float64("nan"), float("inf")]}],
        "flag": np.bool_(True),
    }
    log_experiment(path, "cmd", {"k": np.int64(4), "path": tmp_path}, metrics, np.int64(2))
    with path.open(encoding="utf-8", newline="") as handle:
        (row,) = list(csv.DictReader(handle))
    assert "NaN" not in row["metrics"] and "Infinity" not in row["metrics"]
    assert json.loads(row["metrics"]) == {
        "flag": True,
        "mean": 1.5,
        "missing": None,
        "n": 3,
        "rows": [{"ci": [None, None], "p": 0.25}],
    }
    assert json.loads(row["config"]) == {"k": 4, "path": str(tmp_path)}
    assert row["n_variants"] == "2"


def test_experiment_family_counts_and_an_old_log_is_migrated(tmp_path):
    """n_variants is cumulative per family (`family_variants` + this run); a log written
    before the `family` column existed is rewritten with it (old rows: empty family)."""
    from fplopt.backtest.evaluate import family_variants, read_experiments

    path = tmp_path / "experiments.csv"
    old = ["timestamp", "git_sha", "command", "config", "metrics", "n_variants"]
    path.write_text(",".join(old) + "\nt0,sha,cmd0,{},{},2\n", encoding="utf-8", newline="\n")
    assert family_variants(path, "greedy:rolling 2016-2024") == 0
    for _ in range(2):
        n = family_variants(path, "greedy:rolling 2016-2024") + 1
        log_experiment(path, "cmd", {}, {}, n, "greedy:rolling 2016-2024")
    log_experiment(path, "cmd", {}, {}, family_variants(path, "other") + 1, "other")
    rows = read_experiments(path)
    assert [(r["command"], r["n_variants"], r["family"]) for r in rows] == [
        ("cmd0", "2", ""),
        ("cmd", "1", "greedy:rolling 2016-2024"),
        ("cmd", "2", "greedy:rolling 2016-2024"),
        ("cmd", "1", "other"),
    ]
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(EXPERIMENT_COLUMNS)
    assert family_variants(tmp_path / "missing.csv", "x") == 0


def test_no_semaphore_handle_is_handed_to_a_starting_worker(league, monkeypatch):
    """The root cause of the flaky parallel runs (Phase 4 review): semaphore handles that
    the parent duplicates into a worker while it starts were found closed in the worker on
    Windows under load. The worker pool must not pickle any SemLock into a worker."""
    import multiprocessing.synchronize

    def refuse(self):
        raise AssertionError("a SemLock was handed to a worker process")

    monkeypatch.setattr(multiprocessing.synchronize.SemLock, "__getstate__", refuse)
    _, store, _ = league
    units = [(SEASON, f"u{i}", None) for i in range(3)]
    out = evaluate.run_units(store, Caches(), backtest_rules, _unit_name, units, None, 2)
    assert out == ["u0", "u1", "u2"]


def _unit_name(store, caches, rules, unit, payload):
    return unit[1]


_CALLS: dict[str, int] = {}


class _Counting:
    """A policy wrapper counting `decide` calls per policy name."""

    def __init__(self, inner):
        self.inner, self.name, self.xp_model = inner, inner.name, inner.xp_model

    def decide(self, ctx):
        _CALLS[self.name] = _CALLS.get(self.name, 0) + 1
        return self.inner.decide(ctx)


@pytest.mark.parametrize("continuation", ["own", "fixed"])
def test_per_decision_makes_each_decision_once(league, continuation):
    """The reference arm's decision is the reference run's, the other arm decides once per
    window (no separate decide for the same-decision check), and with own continuation the
    reference arm's window is read back from the reference run: B (the reference) decides
    only in its own run, plus the fixed continuation's GWs if there is one."""
    _, store, _ = league
    a, b = _Counting(GREEDY), _Counting(ROLL)
    cont = "own" if continuation == "own" else _Counting(GreedyPolicy("rolling", threshold=0.5))
    _CALLS.clear()
    out = per_decision(store, [SEASON], "random@25", a, b, cont, k=3, caches=Caches())
    rules = backtest_rules(SEASON)
    ((_, start),) = build_start_states(store, SEASON, "random@25", rules)
    n_gws = len(simulate(store, rules, ROLL, start, caches=Caches()).gws)
    assert len(out) > 0
    if continuation == "own":
        assert _CALLS[a.name] == out["k"].sum()  # its decision + its own k − 1 GWs
        assert _CALLS[b.name] == n_gws  # the reference run only
    else:
        assert _CALLS[a.name] == len(out)  # one decision per window
        assert _CALLS[b.name] == n_gws
        differ = out[~out["same_decision"]]  # arm B is re-simulated only where they differ
        assert _CALLS[cont.name] == (out["k"] - 1).sum() + (differ["k"] - 1).sum()
